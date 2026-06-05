# nano-100 — full-corpus stress test

**Status**: ⏭ blocked on nano-25 passing.

**One-line goal**: prove the k-dim gate at N=100 still produces meaningful per-cohort specialization on at least the most-distinctive subset of cohorts. This is the closest char-level analog of what `mega` will be doing at GPT-2 scale (N=1000 BPE), so its result is the strongest single signal for whether `mega` is worth the $1200.

## Setup

| field | value |
|---|---|
| data | all 100 of `data/100_simple_voices/sources/` |
| tokenizer | char-level, full union vocab (~250 chars expected) |
| model | 10.7M params (nano-2 architecture) |
| cohorts (N) | 100 |
| gate | k-dim tent, k=32 (or larger if nano-25 showed crowding) |
| training | 10K iters (longer than nano-{5,10,25} because more data: 100 MB total vs ~5-25 MB) |
| GPU | RTX 4090 spot |
| wall-clock | ~30-45 min |
| cost | ~$0.50 |

## Reference baselines

We don't need *all* 100 single-source baselines for the pass criterion (75 of them, beyond nano-25's first 25, would be ~$4 extra in baseline cost — affordable but unnecessary). Run:

- **Top-25 single-source baselines** (already from nano-25, reused).
- **Combined-100 ungated baseline** (new, ~$0.30).
- **Optional: 5 random additional baselines** from positions 26-100 to sanity-check the gated model on a sampled middle/tail of the cohort distribution.

The cost-effective evaluation: measure gated model's per-cohort val_loss for the top-25, report aggregate uniform-α loss vs combined-100, and sample-check the lower-ranked cohorts.

## Pass criteria

- **Uniform-α**: within ~10% of combined-100 ungated baseline (slightly relaxed from earlier tiers because the larger N is genuinely harder).
- **Cohort-α (top-25)**: ≥80% of top-25 cohorts within seed variance of their single-source baseline.
- **Cohort-α (sampled tail)**: 5 sampled cohorts from positions 26-100 — at least 3 show *visibly different* generations at their cohort-α setting vs. uniform-α (qualitative; just look at sampled outputs).

The tail criterion is intentionally qualitative because at N=100 with 10.7M params we expect *some* tail cohorts to fail specialization. The bar isn't "all 100 work" — it's "the top cohorts work and the demo is still visceral when you slide to a tail cohort."

## Open questions

- **Capacity floor**: 10.7M params / 100 cohorts ≈ 100K params per cohort if hard-partitioned. With soft routing this is more like ~300-500K effective per cohort, plus halfsies-shared capacity. Probably enough for a char-level model to specialize on distinctive cohorts; almost certainly NOT enough for indistinctive cohorts to specialize cleanly.
- **Slider UI**: at N=100 in k=32-dim space, the slider isn't a 1D drag anymore. UI needs to be either: (a) checkbox-list of named cohorts that the user picks one of, or (b) 2D UMAP of cohort positions in m-space that the user clicks. (b) is the cooler demo.

## Decision after nano-100

This is the major decision point of the nano series. Outcomes:

- **All criteria pass** → highest-confidence proceed to `micro`. The k-dim gate works at N=100 in the char-level regime; the question for `micro` is whether BPE + real web text changes anything.
- **Top-25 fine, tail mostly fails** → expected and acceptable. Document which cohorts work and which don't. Proceed to `micro` with cohort-count tuned to "where specialization breaks for THIS data."
- **Uniform-α fails** → 10.7M is genuinely too small for N=100. Either ship at N=25-50 as the "final char-level result" or bump model size — but the more useful move is probably to skip to `micro` (which scales model size anyway) and accept that pure char-level capped out at moderate N.
- **Top-25 mostly fails** → fundamental architecture issue at large N. Redesign before any BPE work.

## Files

Same template:

```
experiments/nano-100/
├── README.md
├── prepare.py    ← --n 100 (parameterized from nano-5)
├── train.py      ← reuses nano-5/gated_gpt_tent_kdim.py
├── config.py     ← maybe bumped iters / lr
├── results/
│   ├── val_loss_per_cohort.csv  (top-25 + 5 sampled tail)
│   ├── slider_sweep_umap.png    (2D UMAP of cohort m-positions, colored by val_loss)
│   ├── sample_generations.md    (qualitative tail-cohort outputs)
│   └── summary.json
```
