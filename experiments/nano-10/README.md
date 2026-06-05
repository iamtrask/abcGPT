# nano-10 — moderate-N test of the k-dim tent gate

**Status**: ⏭ blocked on nano-5 passing.

**One-line goal**: confirm the k-dim gate scales from N=5 to N=10 without per-cohort quality degradation.

## Setup

| field | value |
|---|---|
| data | top 10 of `data/100_simple_voices/sources/` (by demo_score) |
| tokenizer | char-level, vocab = union of per-source vocabs (~150 chars expected) |
| model | 10.7M params (nano-2 architecture) |
| cohorts (N) | 10 |
| gate | k-dim tent, k same as nano-5 (probably 8 or 4) |
| training | 5000 iters |
| GPU | RTX 4090 spot |
| wall-clock | ~30 min |
| cost | ~$0.30 |

## Reference baselines

Need to expand the baselines run beyond the top 5 used in nano-5. 5 additional single-source baselines from positions 6-10 of `data/100_simple_voices/vocab_stats.json` (by demo_score): IETF RFCs, CFR Title 21, arXiv math abstracts, US Code Title 18, Federal Register rules (approximate order, verify against `vocab_stats.json` at run time). Plus a combined-10-source ungated baseline.

Each additional baseline ~$0.05 on a 4090, so ~$0.30 extra for the 5 new ones + 1 combined.

## Pass criteria

Same template as nano-5:
- **Uniform-α**: within ~5% of combined-10-source ungated baseline.
- **Cohort-α** (for each of 10 cohorts): within seed variance of single-source baseline on that cohort's val.

## Open questions

- **Per-cohort capacity at N=10**: each neuron's "tent peak" still hits 1.0 at α=m_n, but the population of neurons firing strongly at any given α is now ~1/10 of total instead of ~1/2. Is 10.7M params enough?
- **Cohort-distance in k-dim space**: at N=10, k=8, can we still place cohorts at near-equidistant positions? (10 vs k+1=9 — almost simplex-fitting; might want k=16.)

## Decision after nano-10

- **Pass** → run nano-25.
- **Fail uniform-α** → likely capacity issue at this N. Either bump model size for nano-25+ or accept the limit and document.
- **Fail cohort-α** → cohorts probably too close in k-dim space. Bump k for nano-25.

## Files

Same template as nano-5:

```
experiments/nano-10/
├── README.md
├── prepare.py             ← parameterized, --n 10 --source-list top10
├── train.py               ← reuses nano-5/gated_gpt_tent_kdim.py
├── config.py
└── results/
```

Probably most of the code is shared with nano-5 (only data prep and the N constant change). Concrete refactor decision: leave both nano-5 and nano-10 with their own README + config + results, but symlink or import the model + train.py from nano-5.
