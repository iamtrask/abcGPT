# nano-1 — single-source baselines

**Status**: 🟡 ready to run. Scripts in place, awaiting one GPU box for ~7 hours.

**One-line goal**: produce a per-source val_loss baseline for every one of the 100 sources in `100_simple_voices`, trained with Karpathy's exact `shakespeare_char` recipe. These are the reference numbers every later tier compares against in cohort-α mode.

## Why "nano-1"

In the nano series, N = number of cohorts jointly trained. nano-1 = N=1 = train one model per source independently. nano-2 = N=2 (the existing Shakespeare + TinyStories toy). nano-5 onwards = N=5, N=10, N=25, N=100.

nano-1 isn't a single experiment — it's 100 small experiments, each producing one val_loss number. Without these reference numbers, the "gated model matches single-source quality" claim in nano-5 and beyond is unfalsifiable.

## Setup

| field | value |
|---|---|
| data | each source in `data/100_simple_voices/sources/source_<NNN>_<name>.txt` |
| tokenizer | char-level, per-source vocab |
| model | 10.7M params (Karpathy's `shakespeare_char` spec: 6L/6H/384, dropout 0.2, block 256) |
| training | 5000 iters, AdamW β2=0.99, cosine LR 1e-3 → 1e-4, 100 warmup |
| cohorts (N) | 1 (per-source-trained, no joint training) |
| count | 100 separate trainings |
| GPU | RTX 4090 spot (RunPod) |
| wall-clock | ~7 hours total (sequential) |
| cost | ~$2-3 |

## Where the code lives

[`data/100_simple_voices/baselines/`](../../data/100_simple_voices/baselines/) — colocated with the data it consumes. Contains:

- `run_all.sh` — orchestrator (loops over all 100 sources, idempotent)
- `prepare_char.py` — per-source char-level tokenization
- `extract_result.py` — parses training.log + ckpt.pt
- `aggregate.py` — rolls results into one CSV, sorted by val_loss
- `README.md` — detailed protocol + Karpathy reference

The `run_all.sh` script uses Karpathy's `config/train_shakespeare_char.py` byte-for-byte, only overriding `--dataset` and `--out_dir` per source via CLI flags. No per-source config files needed.

## How to run efficiently (no Colab management)

```bash
# 1. Spin up a single GPU box. Recommended:
#    runpod.io → "Deploy" → RTX 4090 spot, 50 GB disk, PyTorch 2.x image.
#    Connect via SSH (the runpod web shell also works).

# 2. On the box:
git clone --depth 1 https://github.com/iamtrask/abcGPT.git
cd abcGPT

# 3. Sync the corpus (88 MB of source files, not committed to git).
#    Easiest: rsync from your laptop:
#       rsync -av ~/Desktop/abcGPT/data/100_simple_voices/sources/ \
#                 user@runpod-host:/workspace/abcGPT/data/100_simple_voices/sources/

# 4. Run all 100 baselines:
bash data/100_simple_voices/baselines/run_all.sh

# 5. Pull back the results directory:
rsync -av user@runpod-host:/workspace/abcGPT/data/100_simple_voices/baselines/results/ \
          ~/Desktop/abcGPT/data/100_simple_voices/baselines/results/

# 6. Tear down the pod (or let it stop itself).
```

That's it. One box, one command, ~7 hours wall-clock, ~$2-3 cost.

### Resuming after interruption

`run_all.sh` is idempotent: it skips any source whose `results/<src>/best_val_loss.txt` already exists. If the box gets preempted halfway, just rerun the same command on a new box — it picks up where it left off.

### Smoke-testing first

```bash
bash data/100_simple_voices/baselines/run_all.sh top5
```

Runs only the top-5 (by demo_score) — ~25 min wall-clock, ~$0.15. Use this to verify the box is set up correctly before kicking off the full 100.

## Pass criteria

nano-1 doesn't pass or fail — it's data collection. The output is a CSV with 100 (source, best_val_loss, vocab_size, train_tokens) rows. Likely shape of the data:

- Sources with **small distinctive vocab** (NBA play-by-play, baseball, patents, recipes) — very low val_loss (e.g. < 0.5), heavily compressed by the 10.7M model.
- Sources with **large diffuse vocab** (Wikipedia, news, modern English novels) — val_loss in the 1.0-2.0 range, similar to Karpathy's tinyshakespeare baseline of 1.47.
- Sources with **OCR cruft** (Lotus Sutra, Police Gazette, etc.) — slightly worse val_loss reflecting noise in the text.

If anything is *wildly* anomalous (val_loss > 4 or < 0.05), worth investigating before treating that source's number as a real reference.

## What nano-1 unlocks

Once `results/summary.csv` exists, nano-2 → nano-100 + micro → mega all have something concrete to compare against. The cohort-α pass criterion for those tiers becomes:

> For each cohort C in the gated model, val_loss(gated@α=m_C) on C's val should be within seed variance of val_loss(nano-1[C]).

Without nano-1, there's no apples-to-apples comparison.

## Files this tier will produce

```
data/100_simple_voices/baselines/results/
├── summary.csv                          ← 100 rows: source, best_val_loss, vocab, tokens, wall_s
└── <source>/                            ← one dir per source
    ├── best_val_loss.txt
    ├── val_loss_curve.csv
    └── meta.json
```

## What this tier will NOT do

- It will not train the existing nano-2 (Shakespeare + TinyStories) — that's a different toy that predates the 100_simple_voices corpus and already has its own published numbers via the public Twitter thread.
- It will not produce any gated models. Every nano-1 baseline is vanilla nanoGPT. The gated work starts at nano-5.
