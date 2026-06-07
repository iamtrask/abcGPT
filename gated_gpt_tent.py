"""GatedGPT: single transformer with per-neuron mask-based specialty routing.

The third design (after dual-slot and BTM) for the dual-source slider.

Three mask groups, each fixed at init from Beta(0.5, 0.5) (arcsine — tail-
heavy U-shape, ~20% mass near 0, ~20% near 1, the rest in middle):

  - M_embd (n_embd,):       GLOBAL per-residual-stream-channel mask.
                            Applied at: wte+wpe output, every attn c_proj
                            output, every mlp c_proj output, and ln_f
                            output before lm_head. The residual stream is
                            implicitly per-channel-gated because every
                            write to it is gated. lm_head sees only the
                            active channels at any given alpha.
  - M_mlp[i] (4*n_embd,):   per-layer MLP-inner-unit mask. Applied to the
                            post-GELU hidden activations of each block.
  - M_attn[i] (n_head,):    per-layer attention-head mask. Applied to each
                            head's output before c_proj.

During forward, a scalar alpha gates each unit by a tent/bell function
peaked at alpha=m with adaptive span:

    span(m) = max(m, 1 - m)                       # 1 for specialists, 0.5 for halfsies
    gate(unit, alpha) = cos^2(pi/2 * |alpha - m| / span(m))

For specialists (m=0 or m=1) span=1 gives a smooth ramp from 0 at the
opposite corner to 1 at the matched corner. For halfsies (m=0.5) span=0.5
gives a bell peaked at alpha=0.5 with EXACTLY zero at alpha=0 and alpha=1.
For m in between, an asymmetric bell peaks at alpha=m with exactly zero at
the further endpoint. At the extremes alpha=0 and alpha=1, ONLY the
respective specialists fire; halfsies (and all blended neurons) are
smoothly turned off.

LayerNorms stay un-gated by design: per-channel gating BEFORE LN is
partially undone by LN's mean subtraction and variance renormalization,
so we gate AFTER LN where it matters (residual-stream gating is on the
things that WRITE to the residual stream, not on LN's output that goes
into Q/K/V inside the block).

During training, alpha is sampled per iter from Beta(0.5, 0.5) (concentrating
on corners), and the corpus the batch is drawn from is chosen by Bernoulli(alpha)
— high alpha favors shake batches, low favors ts. Specialists end up trained
mostly on their own corpus, halfsies on both.

This is closer to fixed-routing Mixture-of-Experts than to weight averaging:
one model, one set of parameters, alpha modulates which subnetwork is active.
The slider exposes that routing continuously to the user.
"""

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Mask sampling — Beta(0.5, 0.5) via inverse CDF (sin^2(pi*u/2))
# ============================================================================

def sample_beta_half_mask(shape, seed, alpha=0.5):
    """Sample a symmetric Beta(α, α)-distributed tensor of the given shape.

    For α=0.5 (default): the arcsine distribution, U-shaped, sampled via the
    closed-form inverse CDF sin²(πU/2). Preserves the exact seed→sample map
    used by every nano-2 run prior to the perturbation sweep.

    For α≠0.5: symmetric Beta sampled via the ratio of two i.i.d. Gamma(α)
    samples. Uses the global RNG (save+restore) since torch's Beta dist
    sampler doesn't accept a per-call generator.
    """
    if alpha == 0.5:
        g = torch.Generator()
        g.manual_seed(seed)
        u = torch.rand(shape, generator=g)
        return torch.sin(math.pi / 2 * u) ** 2
    saved = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        a = torch.tensor(float(alpha))
        samples = torch.distributions.Beta(a, a).sample(shape if isinstance(shape, tuple) else (shape,))
    finally:
        torch.random.set_rng_state(saved)
    return samples


# ============================================================================
# Corpus-routed gate (straight-through estimator)
# ============================================================================
#
# Forward uses the smooth alpha-gate (so the model is exposed to mid-alpha
# behavior at training time). Backward routes gradient through the *hard*
# corpus-based gate: M for shake batches, 1-M for ts batches. Specialists
# (m=1 for shake) get full gradient from their own corpus regardless of
# alpha, and zero from the other corpus. Halfsies (m=0.5) get half from
# either. Cross-corpus leakage on specialists is plugged.
#
# At eval time (no backward, or no corpus passed), the gate is just a
# scalar multiply with the smooth alpha-gate — no autograd indirection.

def _smooth_tent(alpha, M, narrowness=1.0):
    """Tent/bell gate centered at α=m with adaptive span = max(m, 1-m) · narrowness.

    `narrowness=1.0` (default): each neuron fires across the full α∈[0,1]:
      - m=0 / m=1 (specialists): span=1, smooth ramp 0→1 from opposite to matched corner.
      - m=0.5 (halfsies): span=0.5, bell peaked at α=0.5, exactly 0 at both endpoints.

    `narrowness < 1`: each neuron's fire zone is correspondingly narrow.
      - At narrowness=0.25, m=1 fires only at α > ~0.83 (only the strong specialists
        fire near the corners); halfsies fire only in α ∈ [0.4, 0.6]; etc.
      - This produces the "more neurons at the corners" property when combined
        with the Beta(0.5,0.5) mask distribution: at α=1, only m>0.8-ish units
        fire (~30% of total per Beta(0.5,0.5)); at α=0.5, only m≈0.5 fires (~13%).

    Uses cos²(π/2 · |α-m|/span) for C^∞ smoothness; outside the support the rel
    is clamped to 1 so cos²(π/2) = 0 (exactly off).
    """
    span = torch.maximum(M, 1.0 - M) * narrowness
    rel = ((alpha - M).abs() / span).clamp(max=1.0)
    return torch.cos(math.pi * 0.5 * rel).pow(2)


class _CorpusRoutedGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, M, alpha, corpus, narrowness):
        # corpus: 1.0 for shake, 0.0 for ts. Stored as a Python float on ctx.
        ctx.save_for_backward(M)
        ctx.alpha = float(alpha)
        ctx.corpus = float(corpus)
        ctx.narrowness = float(narrowness)
        return x * _smooth_tent(alpha, M, narrowness)

    @staticmethod
    def backward(ctx, grad_output):
        M, = ctx.saved_tensors
        if ctx.corpus >= 0.5:
            # shake: only m near 1 get gradient
            gate_bwd = M
        else:
            # ts: only m near 0 get gradient (i.e., 1 - M near 1)
            gate_bwd = 1.0 - M
        # backward gate is independent of narrowness (it's the hard corpus mask,
        # not the smooth forward shape)
        return grad_output * gate_bwd, None, None, None, None


def gate_with_corpus(x, M, alpha, corpus, narrowness=1.0, anneal_t=1.0):
    """Smooth tent-gate in forward, corpus-routed gate in backward.

    `anneal_t` ∈ [0, 1] interpolates between "no gating" and "full tent gating":
        gate_effective = (1 - anneal_t) + anneal_t * tent(α, M, narrowness)
    - anneal_t = 0: gate = 1 everywhere → every neuron fully active for every α.
                    Model behaves like a non-gated GPT during this phase, so all
                    weights learn from both cohorts. (Mask gradient via STE is
                    still nonzero through the tent contribution, so scores can
                    accumulate ranking signal even during the "ungated" warmup.)
    - anneal_t = 1: gate = tent — full specialization committed.
    Used by the learn-assign mechanism to ramp specialization in gradually
    instead of starting committed. For non-learn-assign variants, callers leave
    anneal_t=1.0 and the blend is a no-op.

    If `corpus is None` (or x doesn't need grad), reduces to a plain multiply
    with the smooth tent-gate (used at eval time so we don't pay for
    autograd-function overhead).

    If `M.requires_grad` is True (trainable_masks or learned_assignment), use
    the plain smooth gate so gradient flows through BOTH x AND M. The
    corpus-routed straight-through estimator was designed for buffered masks
    that don't need data-loss gradient — using it for learnable masks would
    silently zero their gradient (M's backward returns None there).
    """
    t = anneal_t.item() if torch.is_tensor(anneal_t) else anneal_t

    def _gate(M_, learnable):
        tent = _smooth_tent(alpha, M_, narrowness)
        if t >= 1.0:
            return tent
        gate_fwd = (1.0 - t) + t * tent
        if not learnable:
            # No assignment-learning to support; just use the blend directly.
            return gate_fwd
        # STE trick: forward = linear blend (ungated when t=0); backward = full
        # tent gradient flows to M. Scores learn the assignment from iter 0 at
        # FULL strength even though the model trains like ungated during warmup.
        return tent + (gate_fwd - tent).detach()

    if corpus is None or not torch.is_grad_enabled():
        # eval path: M might be a buffer (boundary_mode / fixed-mn) or a learn-assign tensor.
        # Either way, no backward — just use the forward blend.
        return x * _gate(M, learnable=False)
    if M.requires_grad:
        return x * _gate(M, learnable=True)
    # _CorpusRoutedGate path is for buffer-only masks (fixed-mn, boundary_mode)
    # — they don't use the anneal blend.
    return _CorpusRoutedGate.apply(x, M, alpha, corpus, narrowness)


# ============================================================================
# Model
# ============================================================================

@dataclass
class GatedGPTConfig:
    vocab_size: int = 75
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.2
    bias: bool = False
    mask_seed: int = 1337       # fixed seed for reproducible mask sampling
    tent_narrowness: float = 1.0  # 1.0 = full-width tent; <1 narrows each neuron's fire-zone
    trainable_masks: bool = False  # when True, M_embd / M_head / M_inner become nn.Parameter
                                   # via sigmoid(logits) so they stay in [0,1] under gradient updates.
                                   # Caller is expected to put mask logits in a separate optimizer
                                   # param group with a much smaller LR (see GatedGPT.m_n_logits()).
    boundary_mode: bool = False    # when True, each neuron has a permanent `rank` (Beta-sampled at init)
                                   # and m_n is computed every forward as sigmoid((rank - boundary) * sharpness).
                                   # A single mutable scalar `boundary` controls the cohort split for each mask.
                                   # Used by the "boundary-migration" variant: control-loop training adjusts
                                   # `boundary` based on per-cohort loss imbalance, NOT via gradient. Neurons
                                   # near the boundary cross back and forth as it moves; their identity is
                                   # preserved (same rank forever). This is the consistency-preserving version
                                   # of trainable_masks — same neurons cross the boundary in either direction.
    boundary_sharpness: float = 10.0  # sharpness of sigmoid transition. higher = sharper specialist/halfsie split.
    rank_beta_alpha: float = 0.5      # symmetric Beta(α, α) shape parameter for rank/mask sampling.
                                      # 0.5 = U-shape (default), 1.0 = uniform, 2.0 = bell.
    learn_assign_method: str = 'ste'  # 'ste' = hard argsort + straight-through (current default).
                                       # 'softsort' = differentiable approximation via soft permutation matrix
                                       # at temperature tau. Higher tau = smoother; lower → hard.
                                       # SoftSort has O(n²) compute vs STE's O(n log n) but gives unbiased
                                       # gradient (vs STE's identity proxy).
    learn_assign_softsort_tau: float = 1.0  # Temperature for soft permutation matrix when method='softsort'.
                                             # Smaller → sharper assignment (closer to hard argsort).
                                             # Larger → smoother → easier optimization but more averaged m_n.
    learned_assignment: bool = False  # When True: masks via hard-assignment-with-STE. Sample target_values
                                      # (sorted Beta(α, α)) and maintain per-neuron learnable scores. Forward:
                                      # neuron with i-th rank score gets i-th target value (distribution
                                      # exactly preserved). Backward: STE — ∂L/∂m flows to scores as identity.
                                      # Lets the model decide which neurons go where while the overall
                                      # mask distribution stays fixed at the target shape. Mutually
                                      # exclusive with boundary_mode and trainable_masks.


def softsort_assign_mask(scores, target_values, tau=1.0):
    """Differentiable approximation of learn_assign_mask via SoftSort.

    Builds a soft permutation matrix P (n × n) at temperature tau:
        P[i, j] = softmax_j(-(sort(scores)[i] - scores[j])² / tau)
    P[i, j] ≈ "probability that neuron j has rank i."

    Then m[j] = sum_i P[i, j] * target_values[i] = (Pᵀ @ target_values)[j].

    As tau → 0, P → hard permutation matrix and m → STE result.
    As tau → ∞, P → uniform 1/n and m → mean(target_values) for every neuron.

    Reference: Prillo & Eisenschlos, "SoftSort: A Continuous Relaxation
    for the argsort Operator" (ICML 2020).

    Cost: O(n²) — for our largest mask (n_inner=1536), ~2.4M ops per call
    per layer × 6 layers ≈ ~15M extra ops per forward. Manageable.
    """
    with torch.amp.autocast(device_type=scores.device.type, enabled=False):
        scores_f = scores.float()
        tv_f = target_values.float()
        n = scores_f.shape[0]
        # sort scores ascending so rank-0 = smallest score = target_values[0] (smallest target)
        s_sorted, _ = torch.sort(scores_f, descending=False)
        # Pairwise: how close is each neuron's score to each rank's order statistic
        # shape (n, n): row i = rank i, col j = neuron j
        sim = -((s_sorted.unsqueeze(1) - scores_f.unsqueeze(0)) ** 2)
        P = torch.softmax(sim / max(tau, 1e-6), dim=-1)        # (n, n), rows sum to 1
        # m[j] = sum_i P[i, j] * tv_f[i]   →   m = Pᵀ @ tv_f
        m = P.transpose(0, 1) @ tv_f
        return m


def learn_assign_mask(scores, target_values):
    """Hard distribution-preserving assignment with straight-through estimator.

    Forward:
        Neuron at the i-th lowest score gets target_values[i]. The resulting
        m_n distribution is EXACTLY target_values — every value appears once.
        Which neuron gets which value is determined by `scores`.

    Backward:
        Hard assignment is non-differentiable (argsort). STE: ∂L/∂m flows to
        scores as if the assignment were identity. The optimizer treats the
        mask-value gradient as a "push my score up/down" signal on each neuron.

    Note: this used to also take an `anneal_t` arg that collapsed target_values
    toward 0.5 early in training. That was the WRONG end to anneal — at
    narrowness=1, m≈0.5 gives a tent that fires only at α=0.5, so cohort-
    endpoint training (α=0 or 1) saw ~zero neuron activation and couldn't
    learn. Annealing now happens on GATE STRENGTH instead (see gate_with_corpus
    anneal_t arg), which starts the model in "pure ungated" mode and gradually
    introduces specialization. Mask distribution stays at full target shape.

    Args:
        scores:        (n,) learnable tensor. Higher score = higher m_n.
        target_values: (n,) sorted, fixed. The committed distribution shape.

    Returns:
        m: (n,) the mask values. Has the gradient of scores via STE.
    """
    # Force fp32 + disable autocast for the STE math. Under bfloat16 autocast on
    # CUDA, the combination of argsort + in-place permutation assignment +
    # STE arithmetic crashed the container (observed: pods cycled, log only
    # contained config record). Doing this in fp32 is correct anyway since
    # the mask values are scalars in [0, 1] and argsort cares about ordering
    # not magnitude.
    with torch.amp.autocast(device_type=scores.device.type, enabled=False):
        scores_f = scores.float()
        tv_f = target_values.float()
        sorted_idx = torch.argsort(scores_f)             # rank ordering
        m_hard = torch.empty_like(tv_f)
        m_hard[sorted_idx] = tv_f                        # neuron at rank-i gets target[i]
        return scores_f + (m_hard - scores_f).detach()   # STE: forward=m_hard, grad=scores


def _la_dispatch(scores, target, method, tau):
    """Single dispatch point for the two learn-assign methods."""
    if method == 'softsort':
        return softsort_assign_mask(scores, target, tau)
    return learn_assign_mask(scores, target)  # 'ste' (default)


def _beta_to_logit(m, eps=1e-4):
    """Inverse sigmoid. If m = sigmoid(l), this returns l.

    Used to initialize trainable mask logits so sigmoid(logits) equals the
    Beta(0.5, 0.5) sample we'd otherwise store as a buffer — keeping the
    "fixed init" identical between fixed and trainable modes.
    """
    m = m.clamp(eps, 1 - eps)
    return torch.log(m / (1 - m))


class GatedSelfAttention(nn.Module):
    def __init__(self, config: GatedGPTConfig, layer_idx: int, M_embd_or_logits_or_ranks: torch.Tensor,
                 M_embd_boundary: torch.Tensor = None, M_embd_target: torch.Tensor = None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.narrowness = config.tent_narrowness
        self.trainable_masks = config.trainable_masks
        self.boundary_mode = config.boundary_mode
        self.boundary_sharpness = config.boundary_sharpness
        self.learned_assignment = config.learned_assignment
        head_init = sample_beta_half_mask((config.n_head,), seed=config.mask_seed + 100 * layer_idx + 1, alpha=config.rank_beta_alpha)
        if config.boundary_mode:
            # Each block's head mask has its own rank vector and its own boundary scalar.
            # Boundaries can be tied across layers from the outside (see GatedGPT.__init__).
            self.register_buffer('M_head_ranks', head_init)
            self.register_buffer('M_head_boundary', torch.tensor(0.5))
            self.register_buffer('M_embd_ranks', M_embd_or_logits_or_ranks)  # shared from root
            # M_embd_boundary is shared with root via direct attribute (registered as buffer at root only)
            self.M_embd_boundary = M_embd_boundary
        elif config.learned_assignment:
            # Per-block head: sort the Beta-sampled head_init to make it the target distribution,
            # then learnable scores assign which head gets which slot in that distribution.
            self.register_buffer('M_head_target', torch.sort(head_init).values)
            self.M_head_scores = nn.Parameter(torch.randn(config.n_head) * 0.01)
            # M_embd scores: shared with root via nn.Parameter (Module.__setattr__ auto-registers
            # Parameters; both root and block see the same Parameter object).
            self.M_embd_scores = M_embd_or_logits_or_ranks  # nn.Parameter from root
            # M_embd target: must register_buffer to get .to(device) movement. Plain attribute
            # assignment of a Tensor (`self.M_embd_target = M_embd_target`) silently fails to
            # move to CUDA — the attribute stays pointing at the OLD CPU tensor — causing
            # "Expected all tensors on same device" at first forward. (The boundary_mode code
            # has the same pattern with a 0-dim scalar tensor that PyTorch implicitly broadcasts
            # across devices, so it gets away with it.)
            self.register_buffer('M_embd_target', M_embd_target)
            # anneal_t: per-module buffer, updated together via GatedGPT.set_learn_assign_anneal()
            self.register_buffer('learn_assign_anneal_t', torch.tensor(1.0))
            self.learn_assign_method = config.learn_assign_method
            self.learn_assign_softsort_tau = config.learn_assign_softsort_tau
        elif config.trainable_masks:
            self.M_head_logits = nn.Parameter(_beta_to_logit(head_init))
            self.M_embd_logits = M_embd_or_logits_or_ranks
        else:
            self.register_buffer('M_head', head_init)
            self.register_buffer('M_embd', M_embd_or_logits_or_ranks)

    def _M_head(self):
        if self.boundary_mode:
            return torch.sigmoid((self.M_head_ranks - self.M_head_boundary) * self.boundary_sharpness)
        if self.learned_assignment:
            return _la_dispatch(self.M_head_scores, self.M_head_target, self.learn_assign_method, self.learn_assign_softsort_tau)
        if self.trainable_masks:
            return torch.sigmoid(self.M_head_logits)
        return self.M_head

    def _M_embd(self):
        if self.boundary_mode:
            return torch.sigmoid((self.M_embd_ranks - self.M_embd_boundary) * self.boundary_sharpness)
        if self.learned_assignment:
            return _la_dispatch(self.M_embd_scores, self.M_embd_target, self.learn_assign_method, self.learn_assign_softsort_tau)
        if self.trainable_masks:
            return torch.sigmoid(self.M_embd_logits)
        return self.M_embd

    def forward(self, x: torch.Tensor, alpha: float, corpus) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )  # (B, H, T, hd)
        anneal_t = self.learn_assign_anneal_t if self.learned_assignment else 1.0
        y = gate_with_corpus(y, self._M_head().view(1, self.n_head, 1, 1), alpha, corpus, self.narrowness, anneal_t=anneal_t)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(y)
        out = gate_with_corpus(out, self._M_embd(), alpha, corpus, self.narrowness, anneal_t=anneal_t)
        return self.resid_dropout(out)


class GatedMLP(nn.Module):
    def __init__(self, config: GatedGPTConfig, layer_idx: int, M_embd_or_logits_or_ranks: torch.Tensor,
                 M_embd_boundary: torch.Tensor = None, M_embd_target: torch.Tensor = None):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.narrowness = config.tent_narrowness
        self.trainable_masks = config.trainable_masks
        self.boundary_mode = config.boundary_mode
        self.boundary_sharpness = config.boundary_sharpness
        self.learned_assignment = config.learned_assignment
        inner_init = sample_beta_half_mask((4 * config.n_embd,), seed=config.mask_seed + 100 * layer_idx + 2, alpha=config.rank_beta_alpha)
        if config.boundary_mode:
            self.register_buffer('M_inner_ranks', inner_init)
            self.register_buffer('M_inner_boundary', torch.tensor(0.5))
            self.register_buffer('M_embd_ranks', M_embd_or_logits_or_ranks)
            self.M_embd_boundary = M_embd_boundary
        elif config.learned_assignment:
            self.register_buffer('M_inner_target', torch.sort(inner_init).values)
            self.M_inner_scores = nn.Parameter(torch.randn(4 * config.n_embd) * 0.01)
            self.M_embd_scores = M_embd_or_logits_or_ranks  # nn.Parameter from root (auto-shares)
            # See GatedSelfAttention for why this MUST be register_buffer, not attr-assign.
            self.register_buffer('M_embd_target', M_embd_target)
            self.register_buffer('learn_assign_anneal_t', torch.tensor(1.0))
            self.learn_assign_method = config.learn_assign_method
            self.learn_assign_softsort_tau = config.learn_assign_softsort_tau
        elif config.trainable_masks:
            self.M_inner_logits = nn.Parameter(_beta_to_logit(inner_init))
            self.M_embd_logits = M_embd_or_logits_or_ranks
        else:
            self.register_buffer('M_inner', inner_init)
            self.register_buffer('M_embd', M_embd_or_logits_or_ranks)

    def _M_inner(self):
        if self.boundary_mode:
            return torch.sigmoid((self.M_inner_ranks - self.M_inner_boundary) * self.boundary_sharpness)
        if self.learned_assignment:
            return _la_dispatch(self.M_inner_scores, self.M_inner_target, self.learn_assign_method, self.learn_assign_softsort_tau)
        if self.trainable_masks:
            return torch.sigmoid(self.M_inner_logits)
        return self.M_inner

    def _M_embd(self):
        if self.boundary_mode:
            return torch.sigmoid((self.M_embd_ranks - self.M_embd_boundary) * self.boundary_sharpness)
        if self.learned_assignment:
            return _la_dispatch(self.M_embd_scores, self.M_embd_target, self.learn_assign_method, self.learn_assign_softsort_tau)
        if self.trainable_masks:
            return torch.sigmoid(self.M_embd_logits)
        return self.M_embd

    def forward(self, x: torch.Tensor, alpha: float, corpus) -> torch.Tensor:
        h = F.gelu(self.c_fc(x))                                                            # (B, T, 4D)
        anneal_t = self.learn_assign_anneal_t if self.learned_assignment else 1.0
        h = gate_with_corpus(h, self._M_inner(), alpha, corpus, self.narrowness, anneal_t=anneal_t)  # MLP-inner gate
        h = self.c_proj(h)
        h = gate_with_corpus(h, self._M_embd(), alpha, corpus, self.narrowness, anneal_t=anneal_t)   # residual-stream gate
        return self.dropout(h)


class GatedBlock(nn.Module):
    def __init__(self, config: GatedGPTConfig, layer_idx: int, M_embd, M_embd_boundary=None, M_embd_target=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = GatedSelfAttention(config, layer_idx, M_embd, M_embd_boundary, M_embd_target)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = GatedMLP(config, layer_idx, M_embd, M_embd_boundary, M_embd_target)

    def forward(self, x: torch.Tensor, alpha: float, corpus) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), alpha, corpus)
        x = x + self.mlp(self.ln_2(x), alpha, corpus)
        return x


class GatedGPT(nn.Module):
    def __init__(self, config: GatedGPTConfig):
        super().__init__()
        self.config = config
        self.narrowness = config.tent_narrowness
        self.trainable_masks = config.trainable_masks

        self.boundary_mode = config.boundary_mode
        self.boundary_sharpness = config.boundary_sharpness
        self.learned_assignment = config.learned_assignment

        # Global per-channel mask for the residual-stream (n_embd) dimension.
        # Sampled once with a layer-independent seed so re-runs reproduce it.
        # Shared across every module that touches the residual stream
        # (embeddings, every block's c_proj output, and ln_f output before lm_head).
        M_embd_init = sample_beta_half_mask((config.n_embd,), seed=config.mask_seed, alpha=config.rank_beta_alpha)
        shared_boundary = None
        shared_target = None
        if config.boundary_mode:
            self.register_buffer('M_embd_ranks', M_embd_init)
            self.register_buffer('M_embd_boundary', torch.tensor(0.5))
            shared = M_embd_init
            shared_boundary = self.M_embd_boundary  # tensor reference shared with blocks
        elif config.learned_assignment:
            # Sorted target distribution = the committed mask shape. Learnable scores
            # let the model assign each embd channel to a slot in that distribution.
            self.register_buffer('M_embd_target', torch.sort(M_embd_init).values)
            self.M_embd_scores = nn.Parameter(torch.randn(config.n_embd) * 0.01)
            shared = self.M_embd_scores
            shared_target = self.M_embd_target  # buffer ref shared with blocks
            # Anneal scalar — multiplier on deviation of target_values from 0.5.
            # t=1 → full committed distribution; t=0 → all targets collapse to 0.5
            # (every neuron fully mixed). Schedule t over training to "force unmixing."
            self.register_buffer('learn_assign_anneal_t', torch.tensor(1.0))
            self.learn_assign_method = config.learn_assign_method
            self.learn_assign_softsort_tau = config.learn_assign_softsort_tau
        elif config.trainable_masks:
            self.M_embd_logits = nn.Parameter(_beta_to_logit(M_embd_init))
            shared = self.M_embd_logits
        else:
            self.register_buffer('M_embd', M_embd_init)
            shared = M_embd_init  # buffer tensor reference passed down

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([GatedBlock(config, i, shared, shared_boundary, shared_target) for i in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        # Initialize linear/embedding weights, but DON'T overwrite mask logits.
        # _init_weights would normally re-init nn.Parameter tensors; we apply it
        # to modules and rely on the fact that mask logits are not nn.Linear/Embedding.
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _M_embd(self):
        if self.boundary_mode:
            return torch.sigmoid((self.M_embd_ranks - self.M_embd_boundary) * self.boundary_sharpness)
        if self.learned_assignment:
            return _la_dispatch(self.M_embd_scores, self.M_embd_target, self.learn_assign_method, self.learn_assign_softsort_tau)
        if self.trainable_masks:
            return torch.sigmoid(self.M_embd_logits)
        return self.M_embd

    def set_learn_assign_anneal(self, t: float):
        """Set the anneal_t scalar on root + every block (kept in sync).

        t ∈ [0, 1]: 0 = all targets collapse to 0.5 (full mix); 1 = full target shape.
        Train-time scheduler calls this each iter (or every K iters) to anneal
        the target distribution from mixed → committed over training.
        """
        if not self.learned_assignment:
            return
        with torch.no_grad():
            self.learn_assign_anneal_t.fill_(t)
            for block in self.transformer.h:
                block.attn.learn_assign_anneal_t.fill_(t)
                block.mlp.learn_assign_anneal_t.fill_(t)

    def shift_all_boundaries(self, delta: float):
        """Move ALL mask boundaries (M_embd, every layer's M_head and M_inner) by `delta`.

        Use in the boundary-migration training loop: when shake_loss > ts_loss, call with
        delta < 0 (boundary moves down → more neurons end up above boundary → more shake-spec).
        Reverse sign when ts is the harder cohort.
        """
        if not self.boundary_mode:
            raise RuntimeError("shift_all_boundaries called but model is not in boundary_mode")
        with torch.no_grad():
            self.M_embd_boundary += delta  # also seen by blocks since they share the reference
            for block in self.transformer.h:
                block.attn.M_head_boundary += delta
                block.mlp.M_inner_boundary += delta

    def get_boundaries(self):
        """Return a dict of current boundary values (for logging)."""
        if not self.boundary_mode:
            return None
        out = {'M_embd': float(self.M_embd_boundary.item())}
        for i, block in enumerate(self.transformer.h):
            out[f'layer_{i}_M_head']  = float(block.attn.M_head_boundary.item())
            out[f'layer_{i}_M_inner'] = float(block.mlp.M_inner_boundary.item())
        return out


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, alpha: float, targets=None, corpus=None):
        """Forward pass.

        `corpus` controls the backward routing: 1.0 = shake (specialists with
        m near 1 receive gradient), 0.0 = ts (specialists with m near 0
        receive gradient), None = eval / smooth gate everywhere (no routing
        autograd overhead).
        """
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        # Compute the sigmoid-decoded M_embd once per forward (trainable case);
        # for fixed buffers _M_embd() returns the buffer directly.
        M_embd = self._M_embd()
        anneal_t = self.learn_assign_anneal_t if self.learned_assignment else 1.0
        # Gate the initial residual stream (embedding output)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = gate_with_corpus(x, M_embd, alpha, corpus, self.narrowness, anneal_t=anneal_t)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, alpha, corpus)
        # ln_f normalizes the residual stream; gate again so lm_head only sees
        # the active channels at this alpha
        x = self.transformer.ln_f(x)
        x = gate_with_corpus(x, M_embd, alpha, corpus, self.narrowness, anneal_t=anneal_t)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, alpha: float, max_new_tokens: int,
                 temperature: float = 1.0, top_k=None):
        # eval / generation: corpus=None so the gate is just a multiply (no autograd routing)
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond, alpha, corpus=None)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def mask_summary(self):
        """Quick per-mask histogram for sanity-checking the Beta(0.5,0.5) sample.

        For trainable masks, returns the *sigmoid-decoded* current values
        (i.e., the actual mask values flowing through the gate), not the raw logits.
        """
        def stats(M):
            return {
                'mean': float(M.mean()),
                'shake_specialist (m>0.9)': int((M > 0.9).sum()),
                'ts_specialist (m<0.1)':    int((M < 0.1).sum()),
                'halfsies (0.1..0.9)':      int(((M >= 0.1) & (M <= 0.9)).sum()),
                'total': int(M.numel()),
            }
        with torch.no_grad():
            out = {'M_embd (global residual-stream)': stats(self._M_embd())}
            for i, block in enumerate(self.transformer.h):
                out[f'layer_{i}_attn_heads'] = stats(block.attn._M_head())
                out[f'layer_{i}_mlp_inner']  = stats(block.mlp._M_inner())
        return out

    def mask_logit_params(self):
        """Iterate over all mask learnable Parameters (for separate optimizer LR group).

        Covers both `trainable_masks` (sigmoid(logits)) and `learned_assignment`
        (per-neuron scores driving the hard-assignment STE). Returns an empty
        iterator if neither mode is active.

        M_embd is owned by GatedGPT and SHARED across blocks via weight-tying.
        Yielding it from here only (not from each block's attn/mlp) avoids
        double-counting in the optimizer.
        """
        if self.trainable_masks:
            yield self.M_embd_logits
            for block in self.transformer.h:
                yield block.attn.M_head_logits
                yield block.mlp.M_inner_logits
        elif self.learned_assignment:
            yield self.M_embd_scores
            for block in self.transformer.h:
                yield block.attn.M_head_scores
                yield block.mlp.M_inner_scores

    def mask_variance_regularizer(self, gamma=0.25):
        """Anti-collapse penalty on the population standard deviation of each mask.

        Returns sum over all masks of ReLU(gamma - std(sigmoid(logits)))**2.
        Zero if std(M) >= gamma everywhere. Larger gamma forces wider spread.

        Recommended: γ=0.25 keeps the m_n distribution close to uniform-spread
        on [0,1] (uniform[0,1] has std ≈ 0.289; Beta(0.5,0.5) has std ≈ 0.354).
        Combine with a `lambda_var` weight at the call site.
        """
        if not self.trainable_masks:
            return torch.tensor(0.0)
        # Compute per-mask penalty
        device = self.M_embd_logits.device
        total = torch.tensor(0.0, device=device)
        masks = [self._M_embd()]
        for block in self.transformer.h:
            masks.append(block.attn._M_head())
            masks.append(block.mlp._M_inner())
        for M in masks:
            std = M.std()
            total = total + F.relu(gamma - std) ** 2
        return total

    def m_n_drift_from_init(self, init_M_embd):
        """L2 distance between current M_embd and a captured init snapshot.

        For diagnostic logging only. init_M_embd should be a tensor on the
        same device as the model's M_embd.
        """
        cur = self._M_embd()
        return float(((cur - init_M_embd.to(cur.device)) ** 2).sum().sqrt())


# ============================================================================
# Training
# ============================================================================

def train_gated(model, get_shake_batch, get_ts_batch, n_iters,
                lr=1e-3, warmup=100, lr_decay_iters=None, min_lr=1e-4,
                beta2=0.99, weight_decay=0.1, grad_clip=1.0,
                alpha_dist='beta_half',
                log_interval=100, eval_interval=500, eval_iters=200,
                get_shake_val=None, get_ts_val=None,
                device='cuda', amp_dtype=torch.bfloat16):
    """Train GatedGPT.

    Each iter:
      1. Sample alpha (Beta(0.5,0.5) by default; 'uniform' also supported).
      2. Choose corpus by Bernoulli(alpha): shake with prob alpha, else ts.
      3. Forward through the gated model with this alpha.
      4. CE loss, backward, AdamW step.

    Specialists end up trained mostly on their own corpus (the gate keeps the
    off-specialty unit's gradient at near zero when alpha is at the off-corner,
    and the corpus matches the gate), and halfsies are trained on both.
    """
    if lr_decay_iters is None:
        lr_decay_iters = n_iters

    decay_params = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{'params': decay_params, 'weight_decay': weight_decay},
         {'params': nodecay_params, 'weight_decay': 0.0}],
        lr=lr, betas=(0.9, beta2), fused=(device == 'cuda'),
    )

    def get_lr(it):
        if it < warmup:
            return lr * (it + 1) / warmup
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup) / (lr_decay_iters - warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (lr - min_lr)

    def sample_alpha() -> float:
        if alpha_dist == 'beta_half':
            u = torch.rand(1).item()
            return math.sin(math.pi / 2 * u) ** 2
        return torch.rand(1).item()

    @torch.no_grad()
    def estimate_val(get_batch_fn, alpha):
        if get_batch_fn is None:
            return None
        model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_fn()
            with torch.amp.autocast(device_type=device, dtype=amp_dtype):
                _, loss = model(X, alpha, Y, corpus=None)  # eval: smooth gate, no routing
            losses[k] = loss.item()
        model.train()
        return losses.mean().item()

    # Counters for diagnostics
    n_shake = 0
    n_ts = 0

    model.train()
    t_start = time.time()
    t_log = t_start
    for it in range(n_iters):
        cur_lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        alpha = sample_alpha()
        use_shake = torch.rand(1).item() < alpha
        if use_shake:
            X, Y = get_shake_batch(); n_shake += 1
        else:
            X, Y = get_ts_batch();    n_ts += 1

        with torch.amp.autocast(device_type=device, dtype=amp_dtype):
            _, loss = model(X, alpha, Y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if (it + 1) % log_interval == 0:
            dt = time.time() - t_log
            print(f"iter {it+1:5d} | alpha {alpha:.2f} | {'shake' if use_shake else '   ts'} | "
                  f"loss {loss.item():.4f} | lr {cur_lr:.6f} | dt {dt:.1f}s", flush=True)
            t_log = time.time()

        if (it + 1) % eval_interval == 0:
            v_sh_1 = estimate_val(get_shake_val, alpha=1.0)
            v_ts_0 = estimate_val(get_ts_val,    alpha=0.0)
            v_sh_5 = estimate_val(get_shake_val, alpha=0.5)
            v_ts_5 = estimate_val(get_ts_val,    alpha=0.5)
            print(f"  >>> step {it+1}: shake@a=1.0={v_sh_1:.3f}  ts@a=0.0={v_ts_0:.3f}  "
                  f"shake@a=0.5={v_sh_5:.3f}  ts@a=0.5={v_ts_5:.3f}  "
                  f"corpus split so far: {n_shake} shake / {n_ts} ts", flush=True)

    print(f"total training time: {time.time() - t_start:.1f}s  "
          f"final split: {n_shake} shake / {n_ts} ts", flush=True)
    return model
