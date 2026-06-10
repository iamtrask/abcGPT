# LoRA-Additive Research Plan — N=3 → genuine N=1M

Architectural arc for a continuous-α slider mechanism built on additive low-rank
cohort deltas. Replaces the hypernet mechanism currently in nano-3 if Phase 1
validates.

The plan is staged with explicit decision gates. We only advance to the next
phase if the previous one validates. Total compute budget envelope for
research phases (1-6): **~$160 + ~6 weeks**. Phases 7-8 are infrastructure
deployment and order(months, $10K+).

---

## Mechanism summary

Each gated linear has:
- `W_shared ∈ R^{out × in}` — base weight, same for all cohorts
- `U_c ∈ R^{out × rank}`, `V_c ∈ R^{in × rank}` for each cohort c — per-cohort
  low-rank factors

At α (distribution over N cohorts):
```
ΔW = Σ_c α_c · (U_c V_c^T)
W_eff = W_shared + ΔW
y = W_eff @ x + b
```

Init: LoRA-standard (`U ~ N(0, 0.02)`, `V = 0` → ΔW=0 at start, model
behaves as ungated baseline; learn deltas from scratch).

No warmstart needed. No anchor reg required initially (no
uniformity-collapse pathology — if all V → 0, model is just ungated, valid).

Inference math:
- At α = one-hot(c): W_eff = W_shared + U_c V_c^T
- At α = mix: linear blend of deltas, smooth interpolation
- Each cohort's params are independent of other cohorts — adding a new
  cohort doesn't perturb existing ones

---

## Phase 1 — Validate vs hypernet at small N (1-2 days, ~$3)

**Goal**: prove LoRA-additive matches or beats hypernet at N=3 and N=5 on
multiple metrics at comparable parameter counts.

### Implementation (~3 hrs)

- `LoRAAdditiveLinear` class in `experiments/nano-3/model.py` (parallel to
  `PerWeightGatedLinear` and `HypernetGatedLinear`)
- `LoRAAdditiveEmbedding` class for the tied wte/lm_head
- Wire as `--variant lora` in train.py
- No warmstart phase; no anchor reg by default
- Smoke test locally

### Phase 1 sweep (~$0.50, 25 min)

5-6 variants:
1. `n3-lora-r16-full` — N=3, rank=16, full-gating (attn+embed+ffn).
   Rank-matched to hypernet-singletons-full's r=16.
2. `n3-lora-r64-full` — N=3, rank=64. Closer to hypernet's param count.
3. `n5-lora-r16-full` — N=5, rank=16. Direct comparison to hypernet-low_hamming-full.
4. `n5-lora-r32-full` — N=5, rank=32. More delta capacity.
5. `n3-lora-r16-ffn-only` — N=3 FFN-only (no attn/embed gating). Tests
   whether LoRA needs full coverage.
6. (Optional) `n3-lora-r16-anchor` — same as #1 with anchor reg added.
   Tests if anchor still helps even when not theoretically needed.

### Apples-to-apples metrics

| Metric | Hypernet baseline (we have) | LoRA target |
|---|---|---|
| N=3 diag sum | 3.42 (singletons-full) | match or beat |
| N=3 contrast sum | +0.51 (singletons-full) | match or beat |
| N=5 diag sum | 5.63 (low_hamming-full) | match or beat |
| N=5 contrast sum | +0.28 (low_hamming-full) | match or beat |
| edge-curve smoothness (X-shape) | mixed (asymmetry from cohort difficulty) | expect cleaner due to linear blending |
| qualitative prompt-override | strong | expect comparable or stronger |
| param count | 26M (N=3) / ~30M (N=5) | ~12M (r=16) / ~18M (r=64) at N=3 |

### Decision gate

If LoRA-additive matches or beats hypernet on **≥3 of 6 metrics at both N=3 AND
N=5**, advance to Phase 2. If LoRA underperforms consistently, document and
pivot back to hypernet refinement (e.g., better low_hamming variants).

### Expected outcome

LoRA wins on smoothness (linear arithmetic guarantees it) and possibly on
matched-corner quality (clean delta semantics). Might tie on contrast (sigmoid×2
gates may be slightly more expressive than rank-limited linear deltas).

---

## Phase 2 — Scale within the simple regime (1 week, ~$10)

**Only enter if Phase 1 validates.**

**Goal**: find the LoRA recipe's Pareto frontier across (N, model size) up to
N=20-30 cohorts.

### Required prep

- N=10 dataset: 10 maximally-distinct registers built from
  `data/100_regime_char/sources/`. Tentative pick: shake / ts / Python / SQL /
  markdown / Rust / Clojure / KJV-Bible / news articles / Wikipedia.
- N=20 dataset: extension to 20 cohorts (subselection from 100_regime).

### Experiments

- N=10 × {4L-256d, 6L-384d, 8L-512d, 12L-768d} with LoRA-additive r=16, r=32
- N=20 × same arch grid
- Measure: does the "smaller-model = stronger slider" anti-scaling we saw
  with hypernet hold for LoRA? Is the sweet spot bigger?
- Test arbitrary α blends (non-trained mixtures) qualitatively — does
  interpolation produce coherent text?

### Decision gate

Recipe stays clean (positive contrast on all cohorts, comparable diag to
ungated) up to N ≥ 15. If breaks below 15, we move to hierarchical (Phase 3)
sooner than expected.

---

## Phase 3 — Hierarchical structure (2 weeks, ~$30)

**Only enter if Phase 2 validates.**

**Goal**: prove that cluster-level + cohort-level decomposition reduces total
params at fixed quality AND enables scaling to N=100+.

### Architecture

```
W_shared            # universal base
ΔW_cluster_k        # for k in 1..K
ΔW_cohort_c         # for c in 1..N

W_eff_c = W_shared + ΔW_cluster(c) + ΔW_cohort_c
```

Forward at α:
```
delta_cluster_blend = Σ_k (Σ_{c in k} α_c) · ΔW_cluster_k
delta_cohort_blend  = Σ_c α_c · ΔW_cohort_c
W_eff = W_shared + delta_cluster_blend + delta_cohort_blend
```

### Implementation (~1 day)

- Extend LoRA-additive to accept multi-level adapter compositions
- Cluster assignments stored as either hard (1-hot) or soft (probability
  distribution over clusters per cohort)
- Initial training: clusters hand-defined from natural register groupings
  ("prose-cluster" = shake+ts+md+novels; "code-cluster" = python+sql+rust+
  clojure)

### Experiments

- N=10 hand-clustered hierarchical vs N=10 flat LoRA — does hierarchy save
  params at equal quality?
- N=50-100 with cluster assignments; flat already infeasible, only
  hierarchical viable
- Auto-discovery (Option A): train Phase-2 flat → k-means cluster the learned
  deltas → retrain hierarchical → compare to hand-cluster

### Decision gate

Hierarchy gives ≥30% parameter reduction at equal contrast+diag vs flat at
N=10. If not, the cluster level isn't pulling its weight and we need
different cluster discovery (soft membership / online clustering / metadata).

---

## Phase 4 — α-Router (1 week, ~$5)

**Goal**: prove an automatic α-router predicts useful α distributions from
prompts.

### Implementation (~1 day)

- Small router model: 2-3 layer attention encoder of prompt → linear → softmax
  to N cohorts (or hierarchical: cluster softmax × cohort-within-cluster softmax)
- Supervised training (Recipe A): training prompts labeled with source cohort
- Evaluation: held-out single-source prompts + hand-crafted mixed-register prompts

### Experiments

- Phase 1's N=3 + router: cohort-identification accuracy from prompts
- Phase 3's N=20 hierarchical + router: hierarchical routing (cluster then
  cohort)
- Mixed-register held-out: "explain SQL syntax in a Shakespeare style" — does
  router predict soft α blend?
- Ablation: router vs hand-picked α — which produces better text?

### Decision gate

Router achieves competitive performance with hand-picked α on diverse prompt
distribution. If not, we may need stronger meta-training signal (RLHF on
router decisions, etc.)

---

## Phase 5 — Test-time scaling: CoT with adapter routing (3-4 weeks, ~$80)

This is where Percy Liang's CS336 lessons become critical. We're training the
system to USE its capabilities, not just have them.

**Goal**: prove that multi-pass inference with adapter routing across passes
solves problems single-pass cannot.

### Required from CS336 / equivalent

- Curated CoT training data (MATH, GSM8K, MMLU, AIME problems with
  reasoning traces)
- DPO or PPO for refining the reasoning policy
- Tree-of-thoughts / Monte Carlo Tree Search for test-time search
- Synthetic data generation (self-generated CoT traces filtered by
  correctness)

### Implementation (~2 weeks)

- SFT base model on CoT traces (MATH, GSM8K subset)
- Train α-router to predict the **sequence** of α values across reasoning
  steps (not just one α per prompt)
- **This is the genuinely novel mechanism: routing-across-time within a chain
  of thought**
- DPO refinement of routing policy using reasoning-correctness as reward

### Experiments

- Single-pass baseline: prompt → final answer, one α blend, measure correctness
- CoT with fixed α: standard CoT, single α throughout, measure improvement
- **CoT with routed α** (the novel piece): different α per reasoning step,
  predicted by routing policy. Measure further improvement.

### Decision gate

Routing-across-time provides ≥5% absolute accuracy improvement on hard
reasoning tasks (AIME, MATH) vs single-α CoT.

**This is where the "make 250M behave like 250T" claim starts to make sense.**
Test-time scaling × adapter routing × CoT = effective per-query capability
scaling with compute budget.

---

## Phase 6 — Tool integration (1-2 weeks, ~$30)

**Goal**: LLM with tools + adapter routing + CoT exceeds the LLM-alone on
tasks requiring external capabilities.

### Required from CS336 + tool-use literature (Toolformer, Voyager, etc.)

- Tool API: calculator, Python execution, web search, document retrieval
- Tool-call training data
- Tool-selection policy training

### Experiments

- Math + calculator: does 250M + tools match 250T without?
- Code generation + execution: iterative refinement with code-execution beats
  single-shot generation?
- RAG via tool calls vs RAG via cohort routing — which is better when?

---

## Phase 7 — Real scale: federated / personalized inference (months, $500-5K)

**Goal**: deploy as a real personalized AI system with millions of users.

### Required from OpenMined infrastructure

- Federated training protocols (PySyft / PyVertical)
- Per-cohort differential privacy
- Adapter-versioning + serving infrastructure
- User-facing API (slider + router)

### Experiments

- 10K user simulation
- Per-user inference: load base + that user's adapter; measure cost
- Privacy attack red-teaming
- Federated update protocols

---

## Phase 8 — Real benchmark match (months, $10K-100K)

**Goal**: match GPT-4 / Claude / DeepSeek-class capability with ≤10B effective
model + adapters + CoT + tools.

### Benchmarks

- MMLU (knowledge): 80%+
- HumanEval (code): 80%+
- GSM8K + MATH + AIME (math): match o1 / DeepSeek-R1 levels
- LongBench (long context)
- HellaSwag / TruthfulQA (common sense, factuality)

### Required

- Real pretrain corpus (FineWeb subsets) — task #39 already in backlog
- Reasoning training (Phase 5 outputs)
- Tool training (Phase 6 outputs)
- Many adapters (Phases 2-3)
- Smart routing (Phase 4-5 outputs)

---

## Critical reading / dependencies

For Phases 5-6, we need:
- **CS336** (Stanford foundation models) syllabus material on:
  - Pretraining recipes, data curation, deduplication
  - SFT, RLHF, DPO
  - Reasoning training (STaR, ReST, self-improvement)
  - Tool use
- **DeepSeek-R1** paper (reasoning training)
- **o1** related papers (test-time compute scaling)
- **LongRoPE** (long context)
- Relevant LoRA literature: LoRA, QLoRA, LoRAHub, TIES, DARE, BTM

---

## Total budget envelope

| Phase | Time | Compute cost | Decision gate |
|---|---|---|---|
| 1 | 1-2 days | ~$3 | LoRA matches hypernet at N=3, N=5 |
| 2 | 1 week | ~$10 | Recipe scales clean to N≥15 |
| 3 | 2 weeks | ~$30 | Hierarchy reduces params at equal quality |
| 4 | 1 week | ~$5 | Router predicts useful α from prompts |
| 5 | 3-4 weeks | ~$80 | Routing-across-time beats single-pass on reasoning |
| 6 | 1-2 weeks | ~$30 | Tools + routing > pure LLM |
| 7 | months | $500-5K | Per-user inference bounded; federated viable |
| 8 | months | $10K-100K | Match GPT-4/Claude/DeepSeek class |

Phases 1-4 are inexpensive and answer the architectural question.
Phases 5-6 require real CS336-style work on CoT/RLHF/tools.
Phases 7-8 are infrastructure + scale.
