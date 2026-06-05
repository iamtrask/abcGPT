# mega — the headline

**Status**: ⏭ blocked on major. **The publication-grade experiment.**

**One-line goal**: train a 124M GatedGPT (architecturally GPT-2 small spec, exactly Karpathy's nanoGPT config) on the full FineWeb sample-10BT (10 billion tokens) with N=1000 cohorts, and demonstrate that the gated model matches a same-architecture ungated baseline on val_loss while providing a slider that dials into any of the 1000 cohorts' voices.

## The claim it supports

> *abcGPT-mega: per-source dials at GPT-2 quality.*
>
> *We train a 124M GatedGPT with per-domain cohort routing on FineWeb sample-10BT, partitioned into N=1000 publisher cohorts by URL domain. With the slider disabled (uniform α), the model matches a same-architecture ungated baseline on FineWeb-10BT val (val_loss X.XX vs Y.YY, within seed variance). With the slider engaged to a specific cohort, val_loss on that cohort drops by Z.ZZ nats below the all-data baseline on the same cohort. The slider is free at the corpus level and useful at the cohort level.*

That's the publication's first paragraph.

## Setup

| field | value |
|---|---|
| data | FineWeb sample-10BT full (10 billion GPT-2 BPE tokens) |
| tokenizer | GPT-2 BPE (tiktoken `gpt2` encoder) |
| model | **124M params, exactly GPT-2 small spec**: 12 layers, 12 heads, 768 embed dim, 1024 block size, dropout 0.1, vocab 50257 |
| cohorts (N) | 1000 (top-1000 domains in FineWeb-10BT) |
| gate | k-dim tent, k probably 64-128 (carried forward from major) |
| training | ~600K iters, AdamW β2=0.95, cosine LR 6e-4 → 6e-5, 2K warmup, grad clip 1.0 |
| GPU | 8 × A100 40GB or 8 × H100 |
| wall-clock | 4 days |
| cost | ~$1200 (8×A100 spot on Lambda Labs or GCP) |

These are Karpathy's published nanoGPT-on-OpenWebText settings, applied to FineWeb-10BT with the gating layer added.

## Reference baselines

- **Combined ungated baseline at mega scale**: ~$1200 additional. **This is necessary** — Karpathy's published 2.85 was on OpenWebText, not FineWeb-10BT; we need our own apples-to-apples number.
- **Total mega cost including baseline**: ~$2400.

Possible cost reduction: re-use the major-scale ungated baseline as a proxy, accepting that the comparison number isn't exactly at mega scale. Saves $1200 but weakens the publication claim. Default decision: pay the $1200 for the proper mega-scale ungated baseline.

## Pass criteria

- **Uniform-α**: within seed variance of the mega-scale ungated baseline. Concrete: |val_loss_gated − val_loss_ungated| < 0.02 nats.
- **Cohort-α (named domain set)**: train ~20 named-domain single-source baselines at mini scale as proxy references, evaluate gated model at each cohort's α on that cohort's val split, show substantially lower loss than uniform-α on that subset.
- **Sample generations**: continuations from named cohort α's (nytimes, reuters, wikipedia, etc.) should be recognizable as that publication's house style.
- **2D cohort-space UMAP**: visualization of all 1000 cohort positions in k-dim space, colored by something interesting (corpus loss, domain category, etc.).

## Compute infrastructure

Two viable hosts:

| host | 8×A100 spot price | reliability | notes |
|---|---|---|---|
| Lambda Labs | ~$10/hr | high | Karpathy's host of choice |
| GCP us-central1 | ~$12/hr spot | high | already have tooling from Check 1 |

Lambda Labs slightly cheaper but reliability matters when running 4-day jobs. GCP's spot pricing has more preemption risk but the spend-management story is tighter. Either is reasonable.

## Risk register

- **Spot preemption mid-run**: 4 days on spot has nonzero preemption probability. Mitigation: checkpoint every 1000 iters, restart from latest checkpoint on preemption. Standard nanoGPT pattern.
- **HF dataset drift**: FineWeb is occasionally re-uploaded. Pin a specific revision hash in the data-prep step. Already designed-in via Step 2' work.
- **Cohort imbalance**: top-1000 domains aren't uniformly sized. Largest domain (probably wikipedia.org) has 10× the docs of the 500th. The gated training loop needs balanced cohort sampling (`Bernoulli(α=uniform-over-cohorts)` rather than weighted-by-doc-count) to avoid the largest cohorts dominating gate signal.
- **k-dim scaling**: at k=128 and N=1000, cohort embedding table is 1000×128 floats = 500KB. Trivial. Per-neuron m_n also k-dim = 12K floats per layer × 12 layers = 144K floats total. Trivial.

## What gets published

- The repo (cleaned-up version of `gated_gpt_tent_kdim_bpe.py`).
- The trained checkpoint (upload to HuggingFace as `iamtrask/abcGPT-mega`).
- The prep pipeline (`prepare_fineweb_10bt.py`).
- The val_loss comparison table.
- A live slider demo (deploy on HF Spaces or a custom site).
- The writeup (probably blog post + arXiv preprint).

## Files

```
experiments/mega/
├── README.md
├── prepare_fineweb_10bt.py
├── gated_gpt_tent_kdim_bpe.py
├── train.py
├── config.py
├── checkpoint/                ← gitignored. Final model weights.
└── results/
    ├── val_loss_comparison.csv
    ├── cohort_space_umap.png
    ├── sample_generations.md
    └── summary.json
```

## Final note

Don't run mega until major has passed cleanly. The cost asymmetry between running-now-and-failing vs. waiting-for-major-result is roughly $1500 vs $300. Always pay the $300 first.
