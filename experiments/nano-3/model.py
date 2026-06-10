"""nano-3 N-cohort GatedGPT with three variants.

Same transformer arch as nano-2 (n_layer=6, n_head=6, n_embd=384, block_size=256)
but generalized to N cohorts via α ∈ simplex^N.

Variants:
  - 'ungated':    vanilla GPT; α is ignored. Capacity ceiling.
  - 'per_weight': W_eff = W * (Σ_c α_c · scale_c)  with per-cohort scales
                  (out × in) on each FFN projection. Stratified-init recommended.
                  Memory: O(L · N · 4 · n_embd²). Infeasible past N ≈ 100.
  - 'hypernet':   W_eff = W * sigmoid(scale_bias + U(e_α) @ V(e_α)^T) * 2
                  where e_α = α @ cohort_embeddings.
                  Memory: O(L · (out + in) · r · d_embed) + O(N · d_embed).
                  Scales to N >> 1M with d_embed=64, r=16.

Attention QKV is shared across cohorts (slider only modulates FFN). This matches
the tiny-3-source design that validated on real text and the
[[abcgpt-dual-source-training]] / [[hypernet-validated]] toy results.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NanoGPTConfig:
    vocab_size: int = 103
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False
    n_cohorts: int = 3
    variant: str = "ungated"          # 'ungated' | 'per_weight' | 'hypernet' | 'lora'
    d_embed: int = 8                  # cohort-embedding dim (hypernet)
    rank: int = 16                    # rank of the factorized hypernet
    cohort_names: List[str] = field(default_factory=lambda: ["shake", "ts", "code"])
    # "Gate everything" knobs — default False keeps prior nano-3 results valid
    gate_attention: bool = False      # gate c_attn (Q/K/V) + attn.c_proj
    gate_embedding: bool = False      # gate the tied wte/lm_head matrix
    # Phase 1.5: capacity asymmetry. If > 0, factorize each gated linear's
    # base weight as A @ B.T (rank=base_rank) instead of a full (out × in)
    # matrix. Forces more representation capacity into per-cohort deltas.
    base_rank: int = -1               # -1 = full rank (no change)


# ---------------------------------------------------------------------------
# Linear factory: returns the right kind of (possibly-gated) Linear for cfg.variant
# ---------------------------------------------------------------------------
def make_linear(cfg, in_features, out_features, gated):
    """Return Linear/PerWeightGatedLinear/HypernetGatedLinear depending on cfg.variant
    AND whether this projection should be gated. Used to "wrap" attention/FFN/etc
    projections so we can selectively enable gating on different parts of the model."""
    if not gated or cfg.variant == "ungated":
        return nn.Linear(in_features, out_features, bias=cfg.bias)
    if cfg.variant == "per_weight":
        return PerWeightGatedLinear(in_features, out_features, cfg.n_cohorts, bias=cfg.bias)
    if cfg.variant == "hypernet":
        return HypernetGatedLinear(in_features, out_features, cfg.n_cohorts,
                                     cfg.d_embed, cfg.rank, bias=cfg.bias)
    if cfg.variant == "lora":
        return LoRAAdditiveLinear(in_features, out_features, cfg.n_cohorts,
                                    rank=cfg.rank, bias=cfg.bias,
                                    base_rank=cfg.base_rank)
    raise ValueError(f"unknown variant: {cfg.variant}")


def _call_linear(layer, x, alpha, cohort_embeddings):
    """Forward through layer regardless of whether it's plain Linear or a gated one."""
    if isinstance(layer, PerWeightGatedLinear):
        return layer(x, alpha)
    if isinstance(layer, HypernetGatedLinear):
        return layer(x, alpha, cohort_embeddings)
    if isinstance(layer, LoRAAdditiveLinear):
        return layer(x, alpha)
    return layer(x)


# ---------------------------------------------------------------------------
# Attention (gated optionally on c_attn + c_proj)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: NanoGPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.cfg = cfg
        self.c_attn = make_linear(cfg, cfg.n_embd, 3 * cfg.n_embd, gated=cfg.gate_attention)
        self.c_proj = make_linear(cfg, cfg.n_embd, cfg.n_embd, gated=cfg.gate_attention)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

    def forward(self, x, alpha=None, cohort_embeddings=None):
        B, T, C = x.size()
        qkv = _call_linear(self.c_attn, x, alpha, cohort_embeddings)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(_call_linear(self.c_proj, y, alpha, cohort_embeddings))


# ---------------------------------------------------------------------------
# FFN variants
# ---------------------------------------------------------------------------
class UngatedFFN(nn.Module):
    def __init__(self, cfg: NanoGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, alpha=None, cohort_embeddings=None):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class PerWeightGatedLinear(nn.Module):
    """W_eff = W * mix where mix = (α.view(-1,1,1) * scales).sum(0).

    `scales` is (n_cohorts, out, in). Default init is ones; the trainer will
    overwrite with stratified-init samples.
    """
    def __init__(self, in_features, out_features, n_cohorts, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_cohorts = n_cohorts
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.scales = nn.Parameter(torch.ones(n_cohorts, out_features, in_features))

    def forward(self, x, alpha):
        mix = (alpha.view(-1, 1, 1) * self.scales).sum(dim=0)
        W_eff = self.weight * mix
        return F.linear(x, W_eff, self.bias)


class PerWeightFFN(nn.Module):
    def __init__(self, cfg: NanoGPTConfig):
        super().__init__()
        self.c_fc = PerWeightGatedLinear(cfg.n_embd, 4 * cfg.n_embd, cfg.n_cohorts, bias=cfg.bias)
        self.c_proj = PerWeightGatedLinear(4 * cfg.n_embd, cfg.n_embd, cfg.n_cohorts, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, alpha, cohort_embeddings=None):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x, alpha)), alpha))


class FactorizedHypernet(nn.Module):
    """g(e) = sigmoid(scale_bias + U(e) @ V(e)^T) * 2  bounded in [0, 2]."""

    def __init__(self, d_embed, out_features, in_features, rank):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.rank = rank
        self.U = nn.Linear(d_embed, out_features * rank, bias=False)
        self.V = nn.Linear(d_embed, in_features * rank, bias=False)
        self.scale_bias = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.normal_(self.U.weight, std=0.01)
        nn.init.normal_(self.V.weight, std=0.01)

    def forward(self, e):
        if e.dim() == 1:
            U = self.U(e).view(self.out_features, self.rank)
            V = self.V(e).view(self.in_features, self.rank)
            raw = self.scale_bias + U @ V.T
        else:
            B = e.shape[0]
            U = self.U(e).view(B, self.out_features, self.rank)
            V = self.V(e).view(B, self.in_features, self.rank)
            raw = self.scale_bias.unsqueeze(0) + torch.bmm(U, V.transpose(1, 2))
        return torch.sigmoid(raw) * 2.0


class HypernetGatedLinear(nn.Module):
    def __init__(self, in_features, out_features, n_cohorts, d_embed, rank, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_cohorts = n_cohorts
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.hypernet = FactorizedHypernet(d_embed, out_features, in_features, rank)

    def forward(self, x, alpha, cohort_embeddings):
        e_alpha = alpha @ cohort_embeddings  # (d_embed,)
        mix = self.hypernet(e_alpha)
        W_eff = self.weight * mix
        return F.linear(x, W_eff, self.bias)


# ---------------------------------------------------------------------------
# Gated embeddings (per_weight and hypernet variants) for wte/lm_head
# ---------------------------------------------------------------------------
class PerWeightGatedEmbedding(nn.Module):
    """Gated embedding: weight is (vocab_size, n_embd); scales is (n_cohorts, vocab_size, n_embd).
    Used for both wte (lookup) and lm_head (matmul) — tied via shared `weight`.

    Exposes `out_features` (= vocab_size) and `in_features` (= n_embd) aliases so
    the per-weight init + anchor regularizer code that walks `gated_layers()`
    treats this layer uniformly with PerWeightGatedLinear.
    """
    def __init__(self, vocab_size, n_embd, n_cohorts):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.out_features = vocab_size
        self.in_features = n_embd
        self.n_cohorts = n_cohorts
        self.weight = nn.Parameter(torch.empty(vocab_size, n_embd))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.scales = nn.Parameter(torch.ones(n_cohorts, vocab_size, n_embd))

    def _W_eff(self, alpha):
        mix = (alpha.view(-1, 1, 1) * self.scales).sum(dim=0)
        return self.weight * mix

    def embed(self, idx, alpha):
        return F.embedding(idx, self._W_eff(alpha))

    def project(self, x, alpha):
        return F.linear(x, self._W_eff(alpha))


class HypernetGatedEmbedding(nn.Module):
    """Hypernet-gated embedding: shared base weight + per-cohort multiplicative
    modulation via a factorized hypernet of the cohort embeddings.

    Same `out_features` / `in_features` aliases as PerWeightGatedEmbedding.
    """
    def __init__(self, vocab_size, n_embd, n_cohorts, d_embed, rank):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.out_features = vocab_size
        self.in_features = n_embd
        self.n_cohorts = n_cohorts
        self.weight = nn.Parameter(torch.empty(vocab_size, n_embd))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.hypernet = FactorizedHypernet(d_embed, vocab_size, n_embd, rank=rank)

    def _W_eff(self, alpha, cohort_embeddings):
        e_alpha = alpha @ cohort_embeddings
        mix = self.hypernet(e_alpha)
        return self.weight * mix

    def embed(self, idx, alpha, cohort_embeddings):
        return F.embedding(idx, self._W_eff(alpha, cohort_embeddings))

    def project(self, x, alpha, cohort_embeddings):
        return F.linear(x, self._W_eff(alpha, cohort_embeddings))


class HypernetFFN(nn.Module):
    def __init__(self, cfg: NanoGPTConfig):
        super().__init__()
        self.c_fc = HypernetGatedLinear(cfg.n_embd, 4 * cfg.n_embd, cfg.n_cohorts,
                                          cfg.d_embed, cfg.rank, bias=cfg.bias)
        self.c_proj = HypernetGatedLinear(4 * cfg.n_embd, cfg.n_embd, cfg.n_cohorts,
                                            cfg.d_embed, cfg.rank, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, alpha, cohort_embeddings):
        h = self.c_fc(x, alpha, cohort_embeddings)
        return self.dropout(self.c_proj(F.gelu(h), alpha, cohort_embeddings))


# ---------------------------------------------------------------------------
# LoRA-Additive variant (Phase 1 of LORA_RESEARCH_PLAN.md)
#
# Mechanism per gated linear:
#   W_eff = W_shared + Σ_c α_c · (U_c @ V_c^T)
# Each cohort gets its own (U_c, V_c) low-rank factor pair. LoRA-standard
# init (U ~ N(0, 0.02), V = 0) means ΔW = 0 at start → model behaves as
# ungated baseline → deltas grow during training.
#
# No warmstart needed. No anchor reg required by default (LoRA-additive
# has no uniformity-collapse failure mode).
# ---------------------------------------------------------------------------
class LoRAAdditiveLinear(nn.Module):
    """W_eff = W_shared + Σ_c α_c · (U_c @ V_c^T). Per-cohort low-rank deltas.

    Phase 1.5: optionally make W_shared itself low-rank (base_A @ base_B^T,
    rank = base_rank). Forces more representation capacity into per-cohort
    deltas — tests the "non-slider capacity overflow" hypothesis.
    """

    def __init__(self, in_features, out_features, n_cohorts, rank=16, bias=False,
                  base_rank=-1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_cohorts = n_cohorts
        self.rank = rank
        # Cap base_rank at the natural rank limit; -1 = full rank
        if base_rank <= 0 or base_rank >= min(in_features, out_features):
            self.base_rank = -1
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.base_A = None
            self.base_B = None
        else:
            # Low-rank base: W_shared = base_A @ base_B^T
            self.base_rank = base_rank
            self.base_A = nn.Parameter(torch.empty(out_features, base_rank))
            self.base_B = nn.Parameter(torch.empty(in_features, base_rank))
            nn.init.kaiming_uniform_(self.base_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.base_B, a=math.sqrt(5))
            self.weight = None
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # Per-cohort low-rank factors
        self.U = nn.Parameter(torch.empty(n_cohorts, out_features, rank))
        self.V = nn.Parameter(torch.zeros(n_cohorts, in_features, rank))
        nn.init.normal_(self.U, std=0.02)

    def _W_shared(self):
        if self.base_rank > 0:
            return self.base_A @ self.base_B.T
        return self.weight

    def forward(self, x, alpha):
        delta = torch.einsum('c,cor,cir->oi', alpha, self.U, self.V)
        return F.linear(x, self._W_shared() + delta, self.bias)


class LoRAAdditiveFFN(nn.Module):
    def __init__(self, cfg: NanoGPTConfig):
        super().__init__()
        self.c_fc = LoRAAdditiveLinear(cfg.n_embd, 4 * cfg.n_embd, cfg.n_cohorts,
                                          rank=cfg.rank, bias=cfg.bias)
        self.c_proj = LoRAAdditiveLinear(4 * cfg.n_embd, cfg.n_embd, cfg.n_cohorts,
                                            rank=cfg.rank, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, alpha, cohort_embeddings=None):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x, alpha)), alpha))


class LoRAAdditiveEmbedding(nn.Module):
    """LoRA-additive on the tied wte/lm_head matrix.

    Same `out_features`/`in_features` aliases as the other gated embeddings so
    the init code in train.py walks all gated layers uniformly. (Though
    LoRA-additive doesn't actually need stratified init.)
    """
    def __init__(self, vocab_size, n_embd, n_cohorts, rank=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.out_features = vocab_size
        self.in_features = n_embd
        self.n_cohorts = n_cohorts
        self.rank = rank
        self.weight = nn.Parameter(torch.empty(vocab_size, n_embd))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.U = nn.Parameter(torch.empty(n_cohorts, vocab_size, rank))
        self.V = nn.Parameter(torch.zeros(n_cohorts, n_embd, rank))
        nn.init.normal_(self.U, std=0.02)

    def _W_eff(self, alpha):
        delta = torch.einsum('c,cor,cir->oi', alpha, self.U, self.V)
        return self.weight + delta

    def embed(self, idx, alpha):
        return F.embedding(idx, self._W_eff(alpha))

    def project(self, x, alpha):
        return F.linear(x, self._W_eff(alpha))


# ---------------------------------------------------------------------------
# Block + Model
# ---------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class Block(nn.Module):
    def __init__(self, cfg: NanoGPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.variant = cfg.variant
        if cfg.variant == "ungated":
            self.ffn = UngatedFFN(cfg)
        elif cfg.variant == "per_weight":
            self.ffn = PerWeightFFN(cfg)
        elif cfg.variant == "hypernet":
            self.ffn = HypernetFFN(cfg)
        elif cfg.variant == "lora":
            self.ffn = LoRAAdditiveFFN(cfg)
        else:
            raise ValueError(f"unknown variant: {cfg.variant}")

    def forward(self, x, alpha=None, cohort_embeddings=None):
        x = x + self.attn(self.ln1(x), alpha, cohort_embeddings)
        h = self.ln2(x)
        if self.variant == "ungated":
            return x + self.ffn(h)
        elif self.variant == "per_weight":
            return x + self.ffn(h, alpha)
        elif self.variant == "lora":
            return x + self.ffn(h, alpha)
        else:  # hypernet
            return x + self.ffn(h, alpha, cohort_embeddings)


class NanoGPT(nn.Module):
    def __init__(self, cfg: NanoGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        # wte / lm_head: optionally gated (tied weight either way)
        self.gate_embedding = cfg.gate_embedding and cfg.variant != "ungated"
        if self.gate_embedding:
            if cfg.variant == "per_weight":
                self.gated_embed = PerWeightGatedEmbedding(cfg.vocab_size, cfg.n_embd, cfg.n_cohorts)
            elif cfg.variant == "hypernet":
                self.gated_embed = HypernetGatedEmbedding(cfg.vocab_size, cfg.n_embd, cfg.n_cohorts,
                                                            cfg.d_embed, cfg.rank)
            elif cfg.variant == "lora":
                self.gated_embed = LoRAAdditiveEmbedding(cfg.vocab_size, cfg.n_embd, cfg.n_cohorts,
                                                          rank=cfg.rank)
            else:
                raise ValueError(f"gate_embedding=True with unsupported variant {cfg.variant}")
            self.wte = None  # not used; gated_embed.embed/project replace it
            self.lm_head = None
        else:
            self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
            self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
            self.wte.weight = self.lm_head.weight  # tied
            self.gated_embed = None
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        if cfg.variant == "hypernet":
            self.cohort_embeddings = nn.Parameter(torch.randn(cfg.n_cohorts, cfg.d_embed) * 0.1)
        else:
            self.cohort_embeddings = None
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if not isinstance(module, (PerWeightGatedLinear, HypernetGatedLinear)) \
               and not (hasattr(module, 'weight') and module.weight.dim() == 4):
                # Standard linear init only on plain Linear layers (kaiming already
                # set on the gated ones; hypernet U/V already normal_(0, 0.01))
                if not hasattr(module, '_skip_default_init'):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, alpha=None, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
        if self.gated_embed is not None:
            if self.cfg.variant in ("per_weight", "lora"):
                tok = self.gated_embed.embed(idx, alpha)
            else:  # hypernet
                tok = self.gated_embed.embed(idx, alpha, self.cohort_embeddings)
        else:
            tok = self.wte(idx)
        x = self.drop(tok + self.wpe(pos))
        for block in self.blocks:
            x = block(x, alpha, self.cohort_embeddings)
        x = self.ln_f(x)
        if self.gated_embed is not None:
            if self.cfg.variant in ("per_weight", "lora"):
                logits = self.gated_embed.project(x, alpha)
            else:  # hypernet
                logits = self.gated_embed.project(x, alpha, self.cohort_embeddings)
        else:
            logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1,
        )
        return logits, loss

    def gated_layers(self):
        """Return list of all per-cohort-gated submodules (PerWeightGatedLinear /
        HypernetGatedLinear / PerWeightGatedEmbedding / HypernetGatedEmbedding) — used
        for stratified init + anchor regularization."""
        out = []
        # Attention layers (if gated)
        for block in self.blocks:
            for proj in (block.attn.c_attn, block.attn.c_proj):
                if isinstance(proj, (PerWeightGatedLinear, HypernetGatedLinear, LoRAAdditiveLinear)):
                    out.append(proj)
        # FFN layers
        for block in self.blocks:
            if isinstance(block.ffn, (PerWeightFFN, HypernetFFN, LoRAAdditiveFFN)):
                out.append(block.ffn.c_fc)
                out.append(block.ffn.c_proj)
        # Embedding (if gated)
        if self.gated_embed is not None:
            out.append(self.gated_embed)
        return out

    def num_params(self, exclude_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n -= self.wpe.weight.numel()
        return n
