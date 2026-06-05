# major — last-stop insurance before mega

**Status**: ⏭ blocked on mini. The most expensive tier before the headline.

**One-line goal**: at ~70-80% of mega's compute footprint, prove the gated model still matches an ungated baseline. If major passes cleanly, mega is high-confidence; if major reveals any cracks, fix them at major-scale rather than at mega-scale.

## Setup

| field | value |
|---|---|
| data | FineWeb sample-10BT × 20-50% (~2-5B tokens) |
| tokenizer | GPT-2 BPE |
| model | ~100M params (closer to mega's 124M but still smaller). Likely 10 layers, 10 heads, 640-768 embed |
| cohorts (N) | 500 (top-500 domains) |
| gate | k-dim tent, k=64-128 |
| training | 30K-60K iters |
| GPU | H100 spot or 4×A100 |
| wall-clock | 24-48 hours |
| cost | ~$200-300 |

## What major adds beyond mini

- **Data scale**: 1B → 2-5B tokens. Last chance to find issues with longer training runs hitting capacity walls.
- **Cohort count**: 100 → 500. The biggest single-step jump. Tests whether the gate handles 5x the cohorts without smearing.
- **Model size**: 80M → 100M. Approaches mega's 124M, ensuring the gate-scaling behavior near GPT-2 size is observed before mega commits.

## Reference baselines

- **Combined ungated baseline**: same scale as major's gated run = ~$200-300. Doubles cost to ~$400-600 total for major. This is the most expensive baseline of the ladder but it's the most informative one.
- **Per-domain baselines**: not worth doing 500 separately. Pick ~15-20 representative domains and train single-source baselines at *mini scale* (not major scale) as proxy reference numbers. Acknowledge in the writeup that these are scale-mismatched and only serve as ballpark sanity checks.

## Pass criteria

- **Uniform-α**: within ~5% of combined ungated baseline. Tighter than earlier tiers because at this scale we expect the gate to be either clearly working or clearly broken.
- **Cohort-α (representative 15-20 domains)**: subjective + quantitative — gated generation at that domain's α should be visibly distinct from uniform-α generation, and val_loss should be lower than uniform-α's loss on that domain's val.

## Open questions

- **Whether to skip major and go direct to mega**: if mini is *very* clean (all criteria pass, with no quality gap), the case for skipping major is "we're paying $300 to learn what we already know." Counter-argument: scale interactions are real, mega is $1200, the $300 buys insurance even if expected outcome is "pass."
- **Where to run**: H100 spot on RunPod (~$3/hr) for 30-100 hours, or 4×A100 on GCP spot (~$3.60/hr). H100 is faster wall-clock per dollar.

## Files

```
experiments/major/
├── README.md
├── prepare_fineweb_20pct.py
├── gated_gpt_tent_kdim_bpe.py
├── train.py
├── config.py
└── results/
```

## Decision after major

- **Pass cleanly** → run mega. Highest possible confidence in mega's result.
- **Pass but with degradation creeping in** → fix at major scale (architectural tweak, bumped capacity, different k), re-run major (another $300), then mega.
- **Fail** → don't run mega. Likely outcome: ship the major-scale result as the final paper with the asterisk "we couldn't scale further."
