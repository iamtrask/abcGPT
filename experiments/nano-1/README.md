# nano-1 — single-source baselines

**Status**: 🟡 in flight. Karpathy parity check running on Modal as of 2026-06-05; 100-baseline fan-out blocks on that passing.

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
| GPU | Modal T4 (serverless, parallel via `.map()`) |
| wall-clock | ~2.5 hr total (10× parallel fan-out, ~15 min/source) |
| cost | ~$15 (within Modal's $30/mo free credit) |

## Where the code lives

- [`./run.sh`](./run.sh) — Modal dispatcher (preflight, smoke, karpathy, all, status, logs, fetch, aggregate). The one place that knows `--detach` is non-optional.
- [`./modal_nano1.py`](./modal_nano1.py) — Modal app: `train_one(source_name)` and `train_karpathy_shakespeare()` functions plus local entrypoints.
- [`../../data/100_simple_voices/baselines/`](../../data/100_simple_voices/baselines/) — training scripts (colocated with the data they consume):
  - `run_all.sh` — single-box orchestrator (used by the legacy RunPod path, still works for any sequential GPU run)
  - `prepare_char.py` — per-source char-level tokenization
  - `extract_result.py` — parses training.log + ckpt.pt
  - `aggregate.py` — rolls results into one CSV, sorted by val_loss

Whether invoked via Modal or `run_all.sh`, training uses Karpathy's `config/train_shakespeare_char.py` byte-for-byte, only overriding `--dataset` and `--out_dir` per source via CLI flags. No per-source config files.

## How to run (Modal, serverless GPU)

Everything is wrapped in [`run.sh`](./run.sh) so individual `modal run --detach …` invocations can't be forgotten. **`--detach` is critical** — without it, any blip in your laptop's network (DNS hiccup, WiFi drop, sleep) terminates the in-flight container as a safety measure. With `--detach`, the container's lifetime is decoupled from your local CLI and runs to completion regardless.

Prerequisites: `pip install modal` (or `uv tool install modal`) and `modal token new` to authenticate.

```bash
cd experiments/nano-1

./run.sh preflight       # check modal CLI, auth, and that local git HEAD matches the pinned image commit
./run.sh karpathy        # ~12 min — validates pipeline reproduces Karpathy's published val_loss ≈ 1.4697
./run.sh all             # ~2.5 hr, ~$15 — fan-out across 100 sources + Karpathy parity (in parallel)
./run.sh status          # show running apps + last launched app ID
./run.sh logs            # stream logs from last launched app
./run.sh fetch           # download results from Modal volume to local results/
./run.sh aggregate       # roll up local results/ into summary.csv
./run.sh publish         # upload models to HF Hub + stage metric files for git commit
```

Run `./run.sh help` for the full list. The last-launched app ID is saved to `.last_app_id` (gitignored) so `status` and `logs` without args query the most recent run.

### Why Modal (not RunPod)

Originally the plan was RunPod 4090 spot @ ~$0.30/hr for ~7 hr serial. Modal is **~3-5× faster** (parallel fan-out across many T4 containers via `.map()`) at a similar total cost (~$0.59/hr × ~100 container-hours ÷ parallelism). Modal also has a $30 free monthly credit, which covers the full nano-1 run with margin to spare. The serverless model means no idle-time billing if a container finishes early.

### Resuming after interruption

`run.sh all` fans out via `train_one(source_name)` per source. Each source's result is written independently to `/vol/results/<src>/best_val_loss.txt`. If a container OOMs or a single source fails, re-running with `--only=<failed_sources>` picks up just those without re-doing the rest.

### Smoke-testing first

```bash
./run.sh smoke           # ~12 min, ~$0.12 — trains one source (077_nba-play-by-play) to validate the pipeline
./run.sh karpathy        # ~12 min, ~$0.12 — validates against Karpathy's published number
```

Always run `karpathy` before `all`. If our pipeline can't reproduce the one published reference number on Karpathy's exact recipe + data, the 100 baselines aren't worth committing to.

## RunPod fallback (legacy path)

If Modal is unavailable, the same training code runs on any single GPU box via `data/100_simple_voices/baselines/run_all.sh` (sequential, idempotent, ~7 hr on a 4090). See git history for the prior RunPod-based instructions if needed.

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
experiments/nano-1/results/              ← tier-centric layout: this tier's outputs colocated with its docs/code
├── summary.csv                          ← 100 rows: source, best_val_loss, vocab, tokens, wall_s
├── _karpathy-shakespeare-baseline/      ← Karpathy parity result (val_loss should be ≈ 1.4697)
│   ├── best_val_loss.txt
│   ├── training.log
│   └── ckpt.pt                          ← trained model weights (~80 MB, gitignored)
└── <source>/                            ← one dir per source (100 total)
    ├── best_val_loss.txt
    ├── val_loss_curve.csv
    ├── meta.json
    └── ckpt.pt                          ← trained model weights (~80 MB, gitignored; ~8 GB total across 100)
```

Total local disk after `./run.sh fetch`: ~8 GB (almost all of it is the per-source `ckpt.pt` files). We save the models so we can sample from them, compare against gated models at nano-5+, or compute out-of-source perplexity — re-training all 100 to recover them later would cost another $15 and 2.5 hr.

## Release pipeline (where the artifacts live)

Trained models and metrics are split across two places so each lives where it belongs:

- **GitHub (this repo)** — metrics only: `summary.csv`, `best_val_loss.txt`, `val_loss_curve.csv`, `meta.json`, `training.log`. Small, version-controlled, diffable. Anyone who clones the abcGPT repo can read the numbers without downloading 8 GB. The repo's top-level `.gitignore` already excludes `*.pt` so model weights never accidentally land in git.
- **HuggingFace Hub** ([`iamtrask/abcGPT-nano-1-baselines`](https://huggingface.co/iamtrask/abcGPT-nano-1-baselines)) — full snapshot including all 101 model checkpoints (~80 MB each, ~8 GB total). End users download individual baselines with one line:
  ```python
  from huggingface_hub import snapshot_download
  model_dir = snapshot_download("iamtrask/abcGPT-nano-1-baselines",
                                allow_patterns=["077_nba-play-by-play/*"])
  ```

The publish workflow:

```bash
./run.sh fetch     # local results/ now contains everything (incl. ckpt.pt)
./run.sh publish   # uploads results/ to HF + auto-generates model card + stages metrics for git
git commit -m "nano-1: add baseline metrics"
git push
```

`publish.py` requires `pip install huggingface_hub` + a one-time `huggingface-cli login`. Pass `--dry-run` to see what would happen without uploading, `--private` to create the HF repo as private (useful for staging before flipping to public).

## What this tier will NOT do

- It will not train the existing nano-2 (Shakespeare + TinyStories) — that's a different toy that predates the 100_simple_voices corpus and already has its own published numbers via the public Twitter thread.
- It will not produce any gated models. Every nano-1 baseline is vanilla nanoGPT. The gated work starts at nano-5.
