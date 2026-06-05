# nano-5 — first N>>2 test of the k-dim tent gate

**Status**: 🟡 next. Cheapest informative experiment.

**One-line goal**: prove the k-dim generalization of the tent gate (Option 1 from `../PLAN.md`) produces clean per-cohort specialization at N=5 with no measurable cost on the uniform-α baseline.

## Setup

| field | value |
|---|---|
| data | top 5 of `data/100_simple_voices/sources/` (by demo_score): NBA play-by-play, Retrosheet baseball, PubMed abstracts, Fannie Farmer cookbook, CIA World Factbook |
| tokenizer | char-level, vocab = union of per-source vocabs (~110-130 chars total) |
| model | 10.7M params (same nano-2 architecture: 6L/6H/384, dropout 0.2, block_size 256) |
| cohorts (N) | 5 |
| gate | **k-dim tent, k=8 (or 4)** — first test of the generalization |
| training | 5000 iters (Karpathy's shakespeare_char config) |
| GPU | RTX 4090 spot (~$0.30/hr on runpod) |
| wall-clock | ~30 min |
| cost | ~$0.30 |

## Reference baselines

Single-source baselines for each of the 5 cohorts are scripted at [`../../data/100_simple_voices/baselines/`](../../data/100_simple_voices/baselines/) (task #37, pending GPU). Plus a combined-5-source ungated baseline.

| baseline | training data | what it gives us |
|---|---|---|
| 5 × single-source | one cohort each | per-cohort reference val_loss (for cohort-α mode) |
| 1 × combined ungated | all 5 cohorts | reference val_loss for the uniform-α mode comparison |

## Pass criteria

- **Uniform-α**: gated model's val_loss on union of 5 val splits within ~5% of the combined ungated baseline.
- **Cohort-α** (for each of 5 cohorts c): gated model with α = m_c hits c's val_loss within seed variance of c's single-source baseline.

Both pass → advance to `nano-10`. Either fails → fix the k-dim gate design before any more compute.

## Open questions to resolve in `design.md` first

- **How to sample `m_c`?** Options: random Gaussian in R^k? Uniform on the unit sphere? Vertices of a regular simplex (N points equidistant)? For N=5, k=8, a simplex actually fits (5 ≤ k+1).
- **How to sample `m_n` (per-neuron position)?** Beta(0.5, 0.5) per dim (corner-concentrating, generalizes nano-2)? Uniform-on-sphere? Mixture with some neurons placed near cohort centers?
- **How to define `span(m_n)`?** In 1D nano-2 it's `max(m, 1-m)`. In k-dim, options: distance to nearest cohort center? distance to nearest cohort vertex? fixed scalar?
- **What does (α, cohort) joint sampling look like?** Simplest: pick a random cohort c, set α = m_c (deterministic, not Bernoulli). Other option: α ~ Gaussian around m_c with small variance.

## Files this tier will produce

```
experiments/nano-5/
├── README.md                  ← this file
├── design.md                  ← architectural decisions for k-dim gate (write before code)
├── gated_gpt_tent_kdim.py     ← k-dim gate generalization
├── prepare.py                 ← char-level prep of N=5 sources with shared vocab
├── train.py                   ← parameterized training entry (also reused by nano-10/25/100)
├── config.py                  ← Karpathy's train_shakespeare_char.py config
└── results/
    ├── val_loss_per_cohort.csv    ← per-cohort val_loss at each α setting
    ├── slider_sweep.png           ← visualization
    └── summary.json               ← pass/fail vs criteria
```

## Decision after nano-5

- **Pass** → run nano-10 (same setup, more cohorts). Should be uneventful if nano-5 passed.
- **Fail uniform-α only** → the gate is too aggressive; back off the gating strength or increase k.
- **Fail cohort-α only** → the gate isn't specializing enough; check that per-cohort m positions are sufficiently distinguishable in k-dim space.
- **Fail both** → fundamental design issue. Switch to Option 2 (one-hot cohort mask) before further iteration.
