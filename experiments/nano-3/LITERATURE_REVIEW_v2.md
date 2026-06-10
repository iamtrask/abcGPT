# Prior-work scan v2 — Hypernet weight-gating specifically

(2026-06-10; focused agent search complementing LITERATURE_REVIEW.md which
focused on multi-LoRA blending. This v2 focuses on the hypernet-style
multiplicative weight gating mechanism in nano-3.)

## The mechanism being scanned

Per gated linear in nano-3 hypernet variant:
```
W_eff = W_shared * sigmoid(scale_bias + U(e_α) @ V(e_α)^T) * 2
```
- W_shared: shared weight matrix
- e_α = α @ E (cohort embeddings × slider α)
- U, V: low-rank factorization of the hypernet
- gate values in [0, 2], centered at 1 at init
- Trained from scratch jointly; stratified-singleton init for warmstart;
  anchor regularization during main training
- Continuous α slider, user-controlled at inference

## Headline

The full combination is **novel as an integration**, not in any single
component. Per-element multiplicative gating on shared LLM weights via a
cohort-conditioned hypernet, trained from scratch with a continuous
user-controlled α slider — this specific assembly does not have a clean
prior paper, but several adjacent papers cover individual ingredients.

## Closest analogue per axis

### Axis 1 — Per-weight multiplicative gating via small generator networks

- **HyperFormer / HyperFormer++** (Karimi Mahabadi et al., ACL 2021;
  arxiv 2106.04489). Shared hypernet generates ADD-ON adapter weights per
  task (down/up projections). Backbone frozen. **Critical: generates
  additive adapters, not gates on shared backbone.**
- **Compacter** (Karimi Mahabadi et al., NeurIPS 2021). Same family —
  Kronecker-product hypernet for adapter weights. Same difference: generates
  added adapters, not multiplicative gates.
- **WeightNet** (Ma et al., ECCV 2020). Generates conv kernel weights via
  grouped FC on attention vector; conditioned on input not task/cohort.
- **CondConv** (Yang NeurIPS 2019), **Dynamic Convolution** (Chen CVPR 2020).
  Kernel = Σ α_i(x) · K_i where α_i are sigmoid coefficients. Per-input
  mixing of N full kernels, not per-weight gating of one shared kernel.
- **HYWA / Hypernetwork Weight Adapting** (2025, arxiv 2510.12947).
  Personalized VAD; hypernet adapts weights conditioned on speaker
  embedding. **Closest in spirit to nano-3's mechanism**, but small-model
  (VAD), used for fine-tuning not joint pretraining. Worth reading before
  paper draft.

### Axis 2 — Stratified-singleton initialization for cohort-conditional gates

Searched: "stratified initialization," "singleton ownership," "exclusive
mask warm-start." **Nothing direct.**

- **Piggyback** (Mallya ECCV 2018). Real-valued mask weights initialized to
  positive constant → all-ones mask at init. Global ownership, not partitioned.
- **HAT — Hard Attention to the Task** (Serra ICML 2018). Sigmoid pseudo-gate
  per task with temperature annealing; gates LEARN which neurons each task owns.
  No stratified partition at init.
- **PackNet** (Mallya & Lazebnik CVPR 2018), **Supermasks in Superposition**
  (Wortsman NeurIPS 2020). Disjoint subsets per task, but AFTER training
  (prune-then-assign).
- **Exclusive Supermask Subnetwork** (EMNLP findings 2022). Exclusive
  non-overlapping masks per task for continual learning, but exclusivity
  enforced as TRAINING constraint, not initialization scheme.

**Verdict: stratified-singleton init followed by continued soft-gate training
appears specific to nano-3. Plausibly novel; moderate confidence (area is
broad, could hide under different keyword).**

### Axis 3 — Trained from scratch jointly with gating mechanism present

- **DEMix layers** (Gururangan NAACL 2022). Domain-conditional FFN experts
  trained from scratch with hard routing at training, soft mixture at
  inference. Closest in spirit (multi-source/joint/continuous-mixing-at-
  inference). Different mechanism: separate FFN per expert, no shared
  backbone gating.
- **Switch Transformer / GShard / ST-MoE**. Sparse routing per token. Joint
  from-scratch. Token-level routing, not cohort-conditional weight gating.
- **Branch-Train-Merge (BTM)** (Li et al. 2022) and **BTX / Branch-Train-MiX**
  (Sukhbaatar 2024). Separate domain experts, then merge. Explicit forking
  — no shared backbone during training.
- **Task-conditioned hypernetwork continual-learning** (von Oswald ICLR 2020).
  Hypernet outputs all weights per task, sequential training. Not from-scratch
  joint multi-cohort.
- **SMEAR** (Muqeeth ICLR 2024, arxiv 2306.03745). Soft merging of N expert
  parameter blocks with adaptive routing, differentiable end-to-end.
  **Closest "continuous weight-space mixing" precedent.** Differences: mixes
  whole expert blocks (not per-weight gates on one shared backbone); routing
  is input-adaptive (not user-controlled cohort α).

### Axis 4 — Continuous α as user-controlled knob over multi-cohort gates

- **Concept Sliders** (Gandikota ECCV 2024; sliders.baulab.info,
  github.com/rohitgandikota/sliders). User-controlled α on LoRA delta.
  Additive, single-direction; not multiplicative gating, not N-cohort simplex.
- **Text Slider** (2025, arxiv 2509.18831). Same family.
- **Task Arithmetic / TIES-Merging / DARE / Model Soups**. Linear arithmetic
  on weights or task vectors. Continuous α exists but post-hoc.
- **SMEAR** again — continuous routing, input-adaptive (not user-facing).

**Verdict: user-facing continuous α over multi-cohort gates implemented via
gate hypernet on a cohort-embedding convex combination — closer to novel
than known.**

### Axis 5 — Bounded multiplicative gates (sigmoid·2) specifically

The bounded multiplicative gate is well-established:

- **Highway Networks** (Srivastava 2015). Transform/carry gates in [0,1].
- **LSTM/GRU**. Sigmoid gates in [0,1]. Bound prevents multiplicative
  explosion and gradient issues.
- **Squeeze-and-Excitation** (Hu CVPR 2018). Channel-wise sigmoid gate on
  activations.
- **Recent attention gating** (Qwen NeurIPS 2025 best paper). Sigmoid-bounded
  gates dominate empirically.

The specific [0,2]-centered-at-1 expansion (instead of [0,1]) makes the gate
identity-at-init (sigmoid(0)·2 = 1), preserving optimizer landscape.
Folklore in conditioning-layer literature (StyleGAN-2 modulation, some FiLM
variants).

**Verdict: bounded-multiplicative-gate mechanism is well-known; the [0,2]
centered-at-1 specifically is folklore but not load-bearingly novel; applying
this to shared weights (not activations) per cohort is the distinctive part,
and closest precedent is Piggyback/HAT for binary/per-neuron versions.**

## Specific named comparisons (questions Andrew asked)

- **IA³** (Liu NeurIPS 2022, arxiv 2205.05638). Three learned vectors `l_k,
  l_v, l_ff` rescale activations element-wise. **Activation scaling, not
  weight scaling.** Per-task vectors stored separately (no hypernet). When
  gating non-square matrices, weight-side gating is strictly more expressive
  than activation-side per-row scaling. **Not a precedent for our mechanism.**
- **HyperFormer / Compacter** — covered above. Generate parallel adapter
  weights, do NOT gate shared backbone.
- **FiLM** (Perez AAAI 2018). γ·activation + β. Activation-side affine.
  "FiLM on weights" specifically — searched, no canonical paper. Per-element
  on weights (vs per-row of FiLM) is strictly more expressive.
- **Concept Sliders** — additive LoRA delta with user α. Multiplicative-gate
  version is not in literature.
- **MoE / Switch Transformer / GShard** — discrete expert routing per token.
  **SMEAR** is the cleanest continuous-routing-in-weight-space precedent.

## Novel-vs-known scorecard

| Component | Verdict | Confidence |
|---|---|---|
| Hypernet outputs per-weight gates on shared weights | **Uncommon but not strictly novel** — HYWA (2025) does it for VAD; no clean LLM precedent | Moderate |
| Multiplicative-bounded gating on shared weights (sigmoid·2 centered at 1) | **Known mechanism, novel-ish target.** Bounded mult-gates: long-known. Centered-at-1 expansion: folklore. Applied to shared weights per cohort: HAT/CAT/Piggyback (per-neuron, no hypernet) | High |
| Stratified-singleton warmstart for cohort gates | **Plausibly novel.** | Moderate |
| Joint from-scratch training with continuous α slider over cohorts | **Novel as combination.** | Moderate-to-high |
| The full combination | **Novel.** | High |

## Risks to novelty claim

Reviewers will likely bring up:

1. **HYWA (2510.12947)** — recent (late 2025), targets VAD, uses hypernet to
   adapt weights conditioned on identity embedding. Read carefully before any
   paper draft.
2. **SMEAR (ICLR 2024)** — "isn't this just SMEAR with per-element instead
   of per-expert mixing?" Response: SMEAR mixes N separate experts' full
   parameter blocks; ours gates a single shared backbone with N
   cohort-specific gates derived from a shared low-rank hypernet. Storage
   (O(d_embed × N) vs O(N × |W|)) and parameter sharing structure are
   qualitatively different.
3. **HAT / CAT** (continual learning) — strongest "we already do bounded
   multiplicative gating per task" objection. Differences: per-neuron not
   per-weight; no hypernet (per-task masks stored directly); sequential
   continual learning not joint multi-cohort; no user-controlled α slider.

## Useful repos to clone for comparison

- HyperFormer: github.com/rabeehk/hyperformer
- Concept Sliders: github.com/rohitgandikota/sliders
- SMEAR: github.com/r-three/smear
- Piggyback: github.com/arunmallya/piggyback
- DEMix: github.com/kernelmachine/demix

## Sources

- HyperFormer (arxiv 2106.04489)
- Compacter (Semantic Scholar)
- IA³ "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper"
  (arxiv 2205.05638)
- FiLM (arxiv 1709.07871)
- Concept Sliders (arxiv 2311.12092)
- Text Slider (arxiv 2509.18831)
- Piggyback (arxiv 1801.06519)
- Diff Pruning (arxiv 2012.07463)
- DEMix layers (arxiv 2108.05036)
- SMEAR (arxiv 2306.03745)
- WeightNet (arxiv 2007.11823)
- HYWA hypernet weight adapting (arxiv 2510.12947)
- CondConv (arxiv 1904.04971)
- Feature-wise transformations (Distill)
- Hyper-Modulation conditional GAN (arxiv 2112.02219)
- Many-Task Learning with Task Routing (arxiv 1903.12117)
