# nano-25 — large-N regime test

**Status**: ⏭ blocked on nano-10 passing.

**One-line goal**: stress-test the k-dim gate at N=25, where per-cohort capacity drops below ~1/20 of model and cohort-spacing in k-dim space becomes nontrivial.

## Setup

| field | value |
|---|---|
| data | top 25 of `data/100_simple_voices/sources/` |
| tokenizer | char-level, vocab union (~180-200 chars) |
| model | 10.7M params (nano-2 architecture) |
| cohorts (N) | 25 |
| gate | k-dim tent, k probably bumped to 16 (24 < 25 cohorts requires k≥24 for simplex-like geometry, but lower k with random positions also works) |
| training | 5000 iters, possibly 7500-10000 if loss is still descending |
| GPU | RTX 4090 spot |
| wall-clock | ~30-45 min |
| cost | ~$0.30-0.50 |

## Reference baselines

Need 15 more single-source baselines (positions 11-25) + a combined-25 ungated baseline. ~$1 extra in baseline cost.

## Pass criteria

- **Uniform-α**: within ~5% of combined-25 ungated baseline.
- **Cohort-α**: for *most* (≥80%) of the 25 cohorts, gated val_loss within seed variance of single-source. Tolerate up to ~5 cohorts being noticeably worse (this is the "smearing" risk: at high N, weaker / smaller / less-distinctive cohorts may lose specialization first).

The relaxed criterion is because at N=25 we expect *some* cohorts to be intrinsically harder for the gate to isolate cleanly, and what we want to know is whether the *median* cohort still specializes. If 80% works at N=25, we have high confidence ≥80% will work at N=100.

## Open questions

- **Capacity bottleneck**: at N=25, with k=16-dim m and 10.7M params, each cohort effectively gets ~400K "dedicated" params. Is that enough for distinct voices?
- **Cohort-spacing**: random Gaussian positions in R^16 for N=25 cohorts — distance distribution? Are any pairs too close?
- **Halfsies behavior**: in k-dim, what's "in the middle"? Centroid of all m_c? A sampled point? At α ≠ any m_c, what happens to output? (likely uninteresting noise, but worth checking)

## Decision after nano-25

- **Pass (≥80% cohorts)** → run nano-100. Probably the most informative single experiment in the nano series.
- **Fail (specific cohorts smearing)** → identify which cohorts smeared. If it's specific (e.g., the small-vocab sports cohorts that already dominate the baseline), document as a known limitation. If it's broad (random cohorts smear), redesign before nano-100.
- **Fail uniform-α** → capacity issue. Decide whether to ship a smaller-N final result or bump model size.
