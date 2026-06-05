# nano-2 — the original abcGPT toy

**Status**: ✅ done. This is what the public Twitter thread + repo README + Colab notebook are all about.

## What it is

A 10.65M-param char-level transformer (Karpathy's `train_shakespeare_char` architecture: 6 layers, 6 heads, 384 embed dim, 256 block size, dropout 0.2) trained jointly on tinyshakespeare + a 1.5MB slice of TinyStories. Per-neuron tent-gating with `m ~ Beta(0.5, 0.5)` produces specialists at each end of a single-axis slider plus halfsies in the middle.

| field | value |
|---|---|
| data | tinyshakespeare (~1.1 MB) + TinyStories slice (~1.5 MB) |
| tokenizer | char-level, vocab=75 |
| model | 10.65M params, 6L/6H/384 embed |
| cohorts (N) | 2 |
| gate | smooth tent, 1D α and 1D m, both ∈ [0, 1] |
| training | 10K iters on Colab T4, ~90 min |
| result | val_loss curves descend monotonically toward each corner; slider produces visibly different generations |

## Where the code lives

- **Model**: [`gated_gpt_tent.py`](../../gated_gpt_tent.py) at the repo root
- **Data prep**: [`data/shakespeare_tinystories_char/prepare.py`](../../data/shakespeare_tinystories_char/prepare.py)
- **Training notebook**: [`notebooks/train_gated_tent_karpathy_sts.ipynb`](../../notebooks/train_gated_tent_karpathy_sts.ipynb)
- **README walkthrough**: the public-facing repo [`README.md`](../../README.md)

These live at the repo root (not in `experiments/nano-2/`) because they predate this experiments directory and they're the public-facing artifacts. **Do not move them** — moving would break the Colab badge URL in the README.

## What it proves

- The tent gate produces clean per-corpus specialization at N=2 with 10.7M params and ~3M chars of training data.
- The (α, corpus) joint Bernoulli sampling during training is sufficient to make specialists barely-ever-fire on the wrong corpus without needing the hard backward mask.
- Halfsies (m≈0.5) learn to interpolate smoothly between corpora.
- One set of weights produces a sliderable model that morphs between two distinct voices.

## What it doesn't prove (and why we need the rest of the ladder)

- That the tent gate generalizes from N=2 to N>>2. The nano-{5,10,25,100} series tests this in isolation.
- That the technique works with BPE tokenization. `micro` tests this.
- That the technique survives at real-data scale. `mini` and `major` test this.
- That a 124M-param model trained this way matches a GPT-2-quality baseline. `mega` is the headline.

See [`../PLAN.md`](../PLAN.md) for the full ladder.
