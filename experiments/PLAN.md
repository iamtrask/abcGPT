# abcGPT Scaling Plan

## Goal

Train a GPT-2-quality model (124M params, comparable to Karpathy's nanoGPT reference) with the per-neuron source-attribution gating mechanism prototyped in the original toy demo (`nano-2`), and prove the gated model matches a same-architecture *ungated* baseline on the same data — the "no quality loss for free attribution" claim.

Total budget before committing to the headline experiment (`mega`): **under ~$300**, about ~25% of `mega`'s own cost.

## The ladder

Two phases:

**Phase 1 — nano series**: hold model + tokenizer + data type constant; vary cohort count N. Pure ablation of the question "does the tent-gate generalize from N=1 (single source) to N=100?" nano-1 produces the per-source baseline numbers every later tier compares against in cohort-α mode.

**Phase 2 — scale tiers**: micro → mini → major → mega. Each adds one or two scale knobs (tokenizer → BPE, model size, data scale, cohort count). Each is the smallest test that de-risks specific failure modes for the next tier.

| tier | data | tokenizer | model | N (cohorts) | wall-clock | est. cost | de-risks |
|---|---|---|---|---|---|---|---|
| **nano-1** | each of the 100 sources separately | char | 10.7M × 100 | 1 (per model) | ~7 hr (4090, all 100) | ~$2-3 | reference baselines for every later tier |
| **nano-2** | tinyshakespeare + TinyStories | char (vocab=75) | 10.7M | 2 | ~90 min (T4) | free / Colab | the original toy. ✅ |
| **nano-5** | 100_simple_voices top 5 | char | 10.7M | 5 | ~30 min (4090) | ~$0.30 | first k-dim gate test |
| **nano-10** | 100_simple_voices top 10 | char | 10.7M | 10 | ~30 min | ~$0.30 | gate at moderate N |
| **nano-25** | 100_simple_voices top 25 | char | 10.7M | 25 | ~30 min | ~$0.30 | gate at large-N regime |
| **nano-100** | 100_simple_voices all 100 | char | 10.7M | 100 | ~30-45 min | ~$0.50 | gate at full-corpus N |
| **micro** | FineWeb-10BT × 1% (~100M tokens) | GPT-2 BPE | ~30M | 20 | ~2-4 hr (4090) | ~$10-20 | BPE-vs-char; real-data shift |
| **mini** | FineWeb-10BT × 5-10% (~1B tokens) | GPT-2 BPE | ~50-80M | 100 | ~6-12 hr (H100) | ~$50 | model + data + cohort scaling together |
| **major** | FineWeb-10BT × 20-50% | GPT-2 BPE | ~100M | 500 | ~24-48 hr | ~$200-300 | last cheap chance to find capacity smearing |
| **mega** | FineWeb-10BT full (10B tokens) | GPT-2 BPE | 124M (GPT-2 small spec) | 1000 | 4 days × 8×A100 | ~$1200 | **THE HEADLINE.** GPT-2-quality with per-source slider. |

## Pass criteria (same template at every tier)

Train two models per tier with identical compute on identical data:

1. **Ungated baseline** — vanilla architecture, no gating, trained on the union of all N cohorts.
2. **Gated model** — same architecture + the k-dim tent gate, trained with (α, cohort) joint sampling.

Then evaluate the gated model in two modes:

- **Uniform-α mode** (slider disabled, all neurons fire equally): val_loss within seed variance of the ungated baseline. This is the "no quality loss" claim. **Without this, no result is publishable.**
- **Cohort-α mode** (slider engaged to cohort c's center): val_loss on c's held-out subset substantially below the ungated baseline's val_loss on the same subset. This is the "slider does something useful" claim.

Failing either at any tier → stop. Each failed tier saves the next tier's budget. Failing at `micro` saves ~$1500; failing at `mini` saves ~$1450; etc.

## Cost / risk ladder

| running total | tier | what we've bought |
|---|---|---|
| $0 | nano-2 | the toy + visceral intuition for the mechanism |
| ~$3 | nano-1 | 100 reference val_loss numbers — the bar every gated tier needs to clear |
| ~$5 | nano-{5,10,25,100} | confidence the tent generalizes from N=2 to N=100 at constant scale |
| ~$20 | micro | confidence BPE + real web text don't break the gate |
| ~$70 | mini | confidence the technique works at ~real-LM scale |
| ~$300 | major | confidence the technique survives at near-GPT-2 capacity with N=500 |
| ~$1500 | mega | the GPT-2-quality + slider publication-grade result |

**Total to "high-confidence-go on the $1200 commit" ≈ $300 (~25% of mega's own cost).**

## Architectural open questions to resolve before nano-5

The `nano-2` tent gate is built around 1D `α ∈ [0,1]` and 1D `m ∈ [0,1]`. At N>2 we need a generalization. Three candidates:

1. **Low-dim cohort embedding (default proposal)**. Each cohort gets a frozen vector `m_c ∈ R^k` for small k (8-32). Each neuron also gets a frozen `m_n ∈ R^k`. Gate is `cos²(π/2 · ‖α − m_n‖ / span)` where α is now a point in R^k. Slider becomes navigation in a 2D-or-higher cohort space. Continuous, generalizes nano-2's design directly, preserves slider-interpolation property.

2. **One-hot cohort mask per neuron**. Each neuron hard-assigned to one cohort at init. Gate is `1` if α=my-cohort else `0`. Cleaner but loses continuous interpolation between cohorts.

3. **Random-projection routing**. Each cohort gets a random vector `v_c ∈ R^k`. Each neuron gets a frozen `v_n`. Gate is some monotone function of `cos_sim(v_α, v_n)`. Mathematically cute, dot-product based, fast.

**Default proposal**: Option 1. Most direct extension of nano-2. Tested at `nano-5`; if it fails, fall back to Option 2 before `nano-10`.

## Status (as of 2026-06-05)

| tier | status | next action |
|---|---|---|
| nano-1 | 🟡 ready to run | scripts in place at `data/100_simple_voices/baselines/`. One RunPod 4090, ~7 hr, ~$2-3 |
| nano-2 | ✅ done | (no action — exists at `notebooks/train_gated_tent_karpathy_sts.ipynb`) |
| nano-5 | ⏭ blocked on nano-1 | (a) generalize tent gate to k-dim (Option 1), (b) compare to nano-1's top-5 |
| nano-10 | ⏭ blocked on nano-5 | |
| nano-25 | ⏭ blocked on nano-10 | |
| nano-100 | ⏭ blocked on nano-25 | |
| micro | ⏭ blocked on nano-100 | needs FineWeb-10BT prep with URL→domain cohorts |
| mini | ⏭ blocked on micro | scale-up of micro |
| major | ⏭ blocked on mini | |
| mega | ⏭ blocked on major | 8×A100 for 4 days. Lambda Labs or GCP. |

## Supporting work (not in the tier ladder but related)

- **`owt_repro/`**: byte-deterministic Karpathy-style OWT prep. Verified Check 0 + Check 1. Now superseded by the FineWeb pivot — Karpathy's published 2.85 was on OWT specifically; for FineWeb tiers we train our own ungated baseline at each tier (the "pass criteria" pattern above).
- **`data/100_simple_voices/`**: 100-source corpus with per-source vocab statistics. Top-5 baselines (single-source training) at `data/100_simple_voices/baselines/`. Serves as the data for the entire nano series + per-cohort reference numbers.

## Layout

```
experiments/
├── PLAN.md          ← this file
├── README.md        ← quick-reference index
├── nano-1/          ← single-source baselines, one model per source (training infra at data/100_simple_voices/baselines/)
├── nano-2/          ← existing toy (links to root)
├── nano-5/
├── nano-10/
├── nano-25/
├── nano-100/
├── micro/
├── mini/
├── major/
├── mega/
└── owt_repro/       ← byte-deterministic OWT prep (supporting)
```

Each tier directory contains a `README.md` with: goal, dataset, model config, cost estimate, pass criteria, open questions, and (when work happens) the training scripts and results.
