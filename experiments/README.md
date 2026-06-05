# abcGPT experiments

Tier ladder for scaling abcGPT's per-neuron source-attribution gate from the original toy (`nano-2`) up to a GPT-2-quality model (`mega`), with cheap intermediate tiers that de-risk specific failure modes before committing to the headline cost.

See [`PLAN.md`](./PLAN.md) for the full strategy.

## Tiers

### nano series — cohort-count ablation at constant scale (10.7M char-level)

| tier | sources | N | cost | status |
|---|---|---|---|---|
| [nano-1](./nano-1/) | each of 100 sources, separately | 1 (per model, 100 models) | ~$2-3 | 🟡 next — reference baselines |
| [nano-2](./nano-2/) | shakespeare + TinyStories | 2 | free | ✅ done |
| [nano-5](./nano-5/) | 100_simple_voices top 5 | 5 | ~$0.30 | ⏭ blocked on nano-1 |
| [nano-10](./nano-10/) | 100_simple_voices top 10 | 10 | ~$0.30 | ⏭ blocked on nano-5 |
| [nano-25](./nano-25/) | 100_simple_voices top 25 | 25 | ~$0.30 | ⏭ blocked on nano-10 |
| [nano-100](./nano-100/) | all 100 of 100_simple_voices | 100 | ~$0.50 | ⏭ blocked on nano-25 |

### scale tiers — real data, BPE tokenization, growing model size

| tier | data | model | N | cost | status |
|---|---|---|---|---|---|
| [micro](./micro/) | FineWeb-10BT × 1% (~100M tok) | ~30M BPE | 20 | ~$15 | ⏭ blocked on nano-100 |
| [mini](./mini/) | FineWeb-10BT × ~10% (~1B tok) | ~50-80M BPE | 100 | ~$50 | ⏭ blocked on micro |
| [major](./major/) | FineWeb-10BT × ~50% | ~100M BPE | 500 | ~$200-300 | ⏭ blocked on mini |
| [mega](./mega/) | FineWeb-10BT full (10B tok) | 124M (GPT-2 small) | 1000 | ~$1200 | ⏭ blocked on major. **THE HEADLINE.** |

## Other work in this directory

- [`owt_repro/`](./owt_repro/) — byte-deterministic Karpathy-style OWT prep pipeline. Check 0 + Check 1 passed. Superseded by the FineWeb pivot for tier work.
