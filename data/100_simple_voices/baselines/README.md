# nano-1 baselines for `100_simple_voices`

One char-level baseline model per source in the 100-source corpus. Each model uses **Karpathy's `train_shakespeare_char` config byte-for-byte**, with only the input data swapped. This gives 100 directly-comparable val_loss numbers — the reference bar that every multi-source `nano-N` (and beyond) tier needs to match in cohort-α mode.

(This is the work referenced by [`experiments/nano-1/`](../../../experiments/nano-1/). The training infrastructure lives here because the data lives here.)

## Why nano-1 matters

For any multi-source gated model in the tier ladder, we need to answer:
"When you slide α to cohort C, does the gated model produce val_loss on C's data that's *as good as if you'd trained a dedicated model on just C*?"

That dedicated single-source val_loss is what nano-1 produces — for all 100 sources.

## Method

Per source:

1. **Char-level tokenization** identical to `nanoGPT/data/shakespeare_char/prepare.py`:
   - Per-source vocab from the unique chars in the source
   - 90/10 train/val split (sequential, no shuffle)
   - uint16 binary output: `train.bin`, `val.bin`, `meta.pkl`
2. **Model** = `nanoGPT/config/train_shakespeare_char.py` config byte-for-byte:
   - 6 layers, 6 heads, 384 embed dim, 256 block size, dropout 0.2
   - 5000 iters, AdamW (β2=0.99), cosine LR 1e-3 → 1e-4, 100 warmup, gradient clip 1
   - `bias=False`, `eval_interval=250`, `eval_iters=200`
   - **Only `dataset` and `out_dir` differ between runs** — passed via CLI override, not separate config files.
3. **Result** = best val_loss across all evaluation points during training.

## Layout

```
baselines/
├── README.md                  ← this file
├── prepare_char.py            ← parameterized char-level prep
├── extract_result.py          ← parses training.log + ckpt.pt for results
├── aggregate.py               ← rolls all 100 per-source results into results/summary.csv
├── run_all.sh                 ← orchestrator: prep + train + extract per source
├── .gitignore                 ← excludes regeneratable data/ and out/
├── data/                      ← gitignored. per-source {train,val}.bin + meta.pkl
├── out/                       ← gitignored. per-source ckpt.pt + training.log
└── results/                   ← committed. per-source result summaries
    ├── summary.csv            ← rolled up table of all sources, sorted by val_loss
    └── <source>/
        ├── best_val_loss.txt
        ├── val_loss_curve.csv ← (iter, train_loss, val_loss) per eval
        └── meta.json          ← vocab_size, tokens, wall-clock, etc.
```

## Reproducing

```bash
# Clone the abcGPT repo, then on any CUDA box with torch+numpy installed:
cd /path/to/abcGPT

# Default: train baselines for ALL 100 sources
bash data/100_simple_voices/baselines/run_all.sh

# Subset modes:
bash data/100_simple_voices/baselines/run_all.sh top5
bash data/100_simple_voices/baselines/run_all.sh 077_nba-play-by-play 076_retrosheet-baseball

# Idempotent — already-done sources are skipped. Safe to interrupt and resume.
```

## Cost and wall-clock

Per-source training on a single GPU (5000 iters of the Karpathy shakespeare_char config):

| GPU | min/source | total (100 sources) | spot cost | notes |
|---|---:|---:|---|---|
| T4 | 12-15 | 20-25 hr | ~$3 (GCP spot) | slow but cheapest |
| RTX 3090 | 4-6 | 7-10 hr | ~$2 (RunPod spot) | sweet spot |
| **RTX 4090** | **3-5** | **5-8 hr** | **~$2-3 (RunPod spot)** | **recommended** |
| A100 40GB | ~3 | ~5 hr | ~$5-6 (GCP spot) | overkill; A100s are model-bandwidth-limited for this tiny model |
| H100 | ~2 | ~3.5 hr | ~$10 (RunPod spot) | even more overkill |

The recommended workflow: spin up one RunPod RTX 4090 spot instance, clone the repo, run `bash data/100_simple_voices/baselines/run_all.sh`, wait ~7 hours, pull the `results/` directory back. Total cost ~$2-3. Single command, single box, no notebooks to manage.

## Reference: Karpathy's tinyshakespeare baseline

The same 6×6×384 model + same config trained on `tinyshakespeare` (~1.1 MB, vocab=65) reaches val_loss ≈ **1.4697** in ~3 minutes on A100. Our 100 baselines should land in a similar ballpark for similar-sized data; large divergences indicate either:
- a corpus quirk (very small vocab + tight repetition → very low val_loss / fast overfit)
- an OCR / format issue with the source
- a source that's intrinsically harder for a char-LM at this scale

The whole point is to *see* this variation across the 100 sources, so we have a per-source baseline to compare future gated models against.

## Results

(Populated by `aggregate.py` once `run_all.sh` completes. Numbers are best-val-loss across the 5000-iter training run, lower = better fit.)

See [`results/summary.csv`](results/summary.csv).
