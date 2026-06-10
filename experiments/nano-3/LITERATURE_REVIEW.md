# Prior-work scan: continuous-α multi-cohort LoRA with joint pretraining

(2026-06-10; ~15-min web/arxiv search. Confidence: moderate on completeness;
low confidence specifically on Phase 1.5 territory.)

## Headline

**No prior work matches the full mechanism.** Closest analogues:
- **Concept Sliders** (continuous α, but binary axis, post-hoc, diffusion-only)
- **Rewarded Soups** (continuous α between fully-trained models, post-hoc, no LoRA)

The combination of (a) joint pretraining of base + N cohort LoRAs from scratch,
(b) continuous-α inference, and (c) one-hot → Dirichlet curriculum appears to
be novel.

## Our novel surface area

1. Joint pretraining of `W_shared + N` cohort LoRAs from scratch (everyone in
   the MoE-LoRA literature freezes a pretrained base)
2. One-hot → Dirichlet α curriculum (independently novel as far as can be told)
3. Deliberately low-rank base to force capacity into cohort deltas (Phase 1.5)
   — genuinely unexplored intersection
4. Per-step α-routing during CoT (Phase 5 extension) — token-level routing
   exists (X-LoRA) but reasoning-step granularity is open

## Closest prior work by axis

### Axis 1 — Multi-LoRA blending / weighted-LoRA composition at inference

- **LoRAHub** (Huang et al., COLM 2024). N independently-trained LoRAs +
  CMA-ES-learned scalar weights per LoRA. Our mechanism but post-hoc.
  Composition coherent up to ~20 LoRAs, plateaus past ~5. Clean repo.
- **LoRA Soups** (2024). Concat-then-finetune beats simple weighted averaging
  on practical multi-skill tasks. Useful negative result: pure linear
  blending suboptimal when skills interact non-linearly.

### Axis 2 — Branch-Train-Merge

- **BTM** (Li, Gururangan et al., 2022). N independent full models per domain,
  weight-averaged at inference. Holds up to ~64 domains in 22B regime —
  encouraging for N≥10.
- **TIES-Merging** (NeurIPS 2023) and **DARE** (ICML 2024). Resolve
  sign/magnitude interference in vanilla averaging of independent task vectors.
  **Important warning**: these exist because vanilla averaging suffers
  destructive interference at modest N. **We should baseline our jointly-trained
  cohort LoRAs vs TIES post-merge of the same** — if TIES helps significantly,
  our curriculum isn't preventing delta interference.
- **Model Soups** (ICML 2022). Foundation of linear-mode-connectivity for
  weight averaging.
- **Rewarded Soups** (NeurIPS 2023). **Closest user-facing α slider precedent.**
  Train one model per reward, linear interpolation at inference. Proves
  continuous α is Pareto-efficient. Post-hoc, no LoRA, no joint training.

### Axis 3 — Mixture of LoRAs / continuous routing

- **MoLE — Mixture of LoRA Experts** (Wu et al., ICLR 2024). **Critical for us:**
  per-layer learnable α composition over N independent LoRAs. Their per-layer
  vs global α ablation: per-layer beats global by 3-5 pts on V&L. **Their
  N>8 gating collapse without regularization is the most relevant scaling
  data we have** — possibly explains our N=5 failure mode.
- **X-LoRA** (Buehler et al., 2024). Token-level, layer-level gating over
  pretrained LoRA experts. Hidden-state router. Two-stage.
- **LoRAMoE** (ACL 2024). Frozen base + LoRA experts + router for SFT
  forgetting mitigation.
- **AdaMix** (EMNLP 2022). **Closest spiritual match to our curriculum.**
  Stochastic random routing during training + consistency reg, then averaged
  to a single adapter at inference. Joint training of LoRAs from start
  (base frozen). The random-routing-then-average is closer in spirit to our
  one-hot-to-Dirichlet schedule than anything else in the space.
- **LD-MoLE, LoRA-Mixer, HDMoLE, DynMoLE**. All variants of "learnable routing
  over LoRA experts." None train base + cohorts jointly with α curriculum.

### Axis 4 — Adapter sliders / continuous control

- **Concept Sliders** (Gandikota et al., ECCV 2024). **Closest "user picks α"
  precedent.** Train a LoRA so scaling its α dial moves a target attribute.
  Diffusion-specific. Their key trick — contrastive objective enforcing α=+1
  vs α=−1 behaviors — should be **adapted into our one-hot phase** as a
  matched-vs-opposite α contrastive loss. Clean PyTorch repo.
- **Text Slider** (2025). Plug-and-play continuous concept control via LoRA
  for image/video. Diffusion only.
- **AdapterFusion / AdapterSoup** (EACL 2021, 2023). Two-stage train task
  adapters + attention/similarity-based fusion. Post-hoc.

### Axis 5 — Low-rank base + LoRA deltas (our Phase 1.5)

**Thinnest axis — possibly genuinely unexplored.**

- **ReLoRA** (2023). Pretrains via repeated LoRA + merge cycles. Base ends
  full-rank — opposite of what we want.
- **LOST** (2025). Low-rank + sparse pretraining. Doesn't pair with adapter
  deltas, but factorization scheme reusable.
- **PreLoRA** (2025). ViT warmup with full-rank + LoRA, then freeze full-rank.
  Opposite goal.
- **NoRA** (2024). Nested low-rank adaptation. Inner factorization on the
  adapter, not the base. Math reusable for hierarchical extension.

**No documented base-capacity vs cohort-capacity tradeoff study.** This is
genuinely open territory.

### Axis 6 — Hierarchical / nested adapter structures

- **NoRA** (2024). Two-level low-rank: outer SVD basis + inner trainable delta.
- **HiLo / Rank Also Matters** (2025). Hierarchical rank allocation for
  MoE-LoRA across layers + experts.
- **HDMoLE** (2024). **Closest match to our cluster+cohort hierarchy.**
  Top-level domain gate, bottom-level expert gate within domain. Read carefully
  before Phase 3.

### Axis 7 — LoRA defaults

- Apply LoRA to all attention proj (q,k,v,o) + FFN (gate/up/down). Modern
  default — we already do this via gate_attention + gate_embedding.
- Init: A ~ Kaiming, B = 0. **We follow this.**
- Scaling: ΔW = (α_lora / r) · BA where α_lora is hyperparam (typically r or 2r).
  This is a learning-rate trick on the LoRA branch. **We DON'T currently do
  this** — should adopt as easy 1-2 pt improvement.
- **DoRA** (ICML 2024). Magnitude/direction decomposition. Drop-in replacement
  for U V^T. Consistent 1-2 pt gains over plain LoRA.
- **rsLoRA** (2023). Replaces α/r with α/√r for stable high-rank training.
  Use if going past r=64.

## Ranked recommended reads

1. **Concept Sliders** (Gandikota et al., ECCV 2024) — sliders.baulab.info,
   github.com/rohitgandikota/sliders. The only prior work that trains-for-α-
   control with a user dial. Steal their training objective.
2. **MoLE** (arxiv 2404.13628). Per-layer α ablation + N>8 gating collapse.
3. **Rewarded Soups** (arxiv 2306.04488). User-controlled continuous α
   between full models. Clean repo.
4. **AdaMix** (arxiv 2205.12410). Closest curriculum precedent.
5. **HDMoLE** (arxiv 2409.19878). Hierarchical routing; read before Phase 3.

## Baselines we should add (based on what we learned)

1. **Joint-trained LoRAs + TIES post-merge vs without** — sanity check that
   our curriculum produces non-interfering deltas. If TIES helps a lot, our
   joint training isn't preventing interference.
2. **Per-layer α vs single global α** (MoLE ablation) — could be our weak-
   contrast bottleneck. Add as Phase 1.6 if 1.5 doesn't recover contrast.
3. **Rewarded Soups full-model interpolation at fixed compute** — does the
   low-rank cohort structure actually help, or could we just train N small
   full models and Soup them?

## Sources

[Full URL list at end of agent report; see internal notes.]
