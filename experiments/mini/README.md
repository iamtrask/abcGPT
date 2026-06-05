# mini — real-LM-scale test

**Status**: ⏭ blocked on micro. First tier where the model is "real" sized.

**One-line goal**: prove the gated technique works at ~10% of FineWeb-10BT scale (~1B tokens) with a ~50-80M model and N=100 cohorts. This is the last cheap tier before things get expensive.

## Setup

| field | value |
|---|---|
| data | FineWeb sample-10BT × 5-10% (~500M-1B tokens) |
| tokenizer | GPT-2 BPE |
| model | ~50-80M params (configuration TBD; likely 8 layers, 8 heads, 512 embed) |
| cohorts (N) | 100 (top-100 domains in the slice) |
| gate | k-dim tent, k probably 32-64 |
| training | maybe 10K-20K iters depending on data fit |
| GPU | H100 spot or 4×A100 |
| wall-clock | 6-12 hours |
| cost | ~$50 |

## What it de-risks

Three things at once vs `micro`:
- **Cohort count**: 20 → 100. Tests whether mid-N specialization still produces visible slider effects.
- **Model size**: 30M → 50-80M. Tests whether the gating scales gracefully with capacity (or whether capacity smearing kicks in).
- **Data scale**: 100M → 500-1000M tokens. Tests whether the gate keeps holding as the model sees genuinely diverse web text rather than a tiny convenience-sampled slice.

The "real-LM-scale" framing is intentional: at this tier the architecture starts to look like an actual language model people would train, not a research toy.

## Reference baselines

- **Combined ungated baseline** at the same scale: ~$50 (matches mini's gated run). Doubles total mini cost to ~$100 — still cheap.
- **Per-domain single-source baselines**: at N=100, training 100 single-source baselines is expensive (~$500 if each is the full mini-scale). Don't do all of them. Train baselines for ~10 specifically-named domains and evaluate cohort-α only on those.

## Pass criteria

- **Uniform-α**: within ~10% of combined ungated baseline.
- **Cohort-α (10 named domains)**: ≥7/10 within seed variance of their single-source baseline.
- **Generation quality at named-cohort α**: subjectively recognizable as that domain's voice (nytimes vs wikipedia vs medium etc.).

## Files

```
experiments/mini/
├── README.md
├── prepare_fineweb_10pct.py
├── gated_gpt_tent_kdim_bpe.py  ← reuse from micro, maybe minor tweaks
├── train.py
├── config.py
└── results/
```

## Decision after mini

- **Pass** → `major` is mostly insurance. Could optionally skip-to-mega depending on confidence level, but the $200-300 of major buys a lot of insurance against the $1200 mega.
- **Fail uniform-α** → model size or training schedule mismatched for this data scale. Tune and retry, OR back off to a smaller N.
- **Fail cohort-α** → capacity smearing kicking in. Bump model size for `major`.
