# nano-1 RunPod orchestration

Drop-in replacement for the Modal fan-out at ~4× lower cost. Spins up N RunPod pods, chunks the source list across them, runs training, uploads results to HF Hub, terminates pods.

## Why this exists

Modal is ~4× more expensive per GPU-hour than RunPod spot (Modal T4 $0.59/hr vs RunPod 4090 spot $0.34/hr) AND its free-tier workspace concurrency cap (~10 containers) means we can't fan out widely even when we want to. For the nano-1 fan-out:

| | Modal | RunPod |
|---|---|---|
| Per-source compute | ~$0.57 | ~$0.14 |
| 100 sources total | ~$57 | ~$14 |
| Wall-clock (sensible parallelism) | ~9 hr (cap-limited) | ~4-8 hr (your choice) |

The savings compound at scale: mega tier on Modal would cost ~$5000+; on RunPod, ~$1200.

## One-time setup

1. **Create a RunPod account**: https://www.runpod.io/ — needs a payment method.
2. **Get an API key**: https://www.runpod.io/console/user/settings → "API Keys" → create one with write scope. Save it.
3. **Install the Python SDK**: `pip install runpod` (or `uv tool install runpod`).
4. **HF token** (you already have one from the Modal publish path): https://huggingface.co/settings/tokens — needs `write` scope on `iamtrask/abcGPT-nano-1-baselines`.

Set env vars in your shell rc:
```bash
export RUNPOD_API_KEY="rp_..."
export HF_TOKEN="hf_..."
```

## Usage

```bash
cd experiments/nano-1/runpod

# Dry-run (always do this first — shows plan + cost estimate without launching)
python runpod_fanout.py --dry-run

# Launch 5 boxes, full 100 sources, wait for completion
python runpod_fanout.py

# Train only specific sources (e.g. the ones Modal didn't finish)
python runpod_fanout.py --sources 050_xxx,051_yyy,052_zzz

# More boxes for faster wall-clock
python runpod_fanout.py --n-boxes 10

# Launch and detach (don't block waiting for completion)
python runpod_fanout.py --no-wait
```

After completion, pull results from HF:
```bash
huggingface-cli download iamtrask/abcGPT-nano-1-baselines \
  --local-dir experiments/nano-1/results
```

## How it works

```
runpod_fanout.py (on your laptop)
    ↓ (RunPod API: create_pod × N)
RunPod pod 0   pod 1   pod 2   ...   pod N-1
   ↓             ↓        ↓                ↓
   curl + bash on_box.sh on each pod, with SOURCES env set to its chunk
   ↓             ↓        ↓                ↓
   git clone abcGPT, install deps, run data/100_simple_voices/baselines/run_all.sh <sources>
   ↓             ↓        ↓                ↓
   upload results/ to HF Hub (iamtrask/abcGPT-nano-1-baselines)
   ↓             ↓        ↓                ↓
   pod exits → RunPod auto-terminates (no idle billing)
```

Each pod uploads independently; HF dedupes on content hash, so multi-box concurrent commits to the same repo are safe. The result is N commits to the HF repo, one per box. (We can squash later if it bothers anyone.)

## Cost model

- 4090 spot: ~$0.34/hr (varies by region/availability)
- Per source on 4090: ~25 min training (~2.5× faster than T4)
- 100 sources / 5 boxes = 20 sources/box × 25 min = ~8.3 hr per box
- Total: 5 × 8.3 × $0.34 = **~$14**

For faster wall-clock at proportionally similar total cost:
- 10 boxes: 10 sources/box × 25 min = ~4.2 hr per box. Total: 10 × 4.2 × $0.34 = ~$14.30
- 25 boxes: 4 sources/box × 25 min = ~1.7 hr per box. Total: 25 × 1.7 × $0.34 = ~$14.40
- 100 boxes (1 source each): ~25 min wall. Total: 100 × 0.42 × $0.34 = ~$14.30

Wall-clock scales as `sources_per_box × 25min`. Total cost is approximately constant — the variable is how many GPUs you tie up simultaneously. RunPod has no workspace concurrency cap (unlike Modal), so the only ceiling is GPU availability in the region.

## Spot vs on-demand

`runpod_fanout.py` defaults to `cloud_type="SECURE"` (on-demand). For ~30-40% additional savings switch to spot (`cloud_type="COMMUNITY"`):

```python
# In runpod_fanout.py launch_pod():
cloud_type="COMMUNITY",
```

Spot trade-off: pod can be preempted with ~30s notice. Since `run_all.sh` is per-source idempotent (skips sources whose `results/<src>/best_val_loss.txt` exists locally), a preempted box just resumes its remaining work on next boot. We don't have *intra-source* checkpointing (a preemption mid-training of a single source loses that source's 25 min of progress), but cost-weighted that's a fine trade.

## Comparison with Modal path

| | Modal (`./run.sh all`) | RunPod (`runpod_fanout.py`) |
|---|---|---|
| Setup effort | 5 min (`modal token new`) | 10 min (API key + `pip install runpod`) |
| Per-source $$$ | $0.57 | $0.14 |
| Concurrency cap | 10 (workspace) | none (GPU availability) |
| Result storage | Modal volume → laptop fetch | HF Hub directly |
| Idle billing | per-second only | per-second only |
| Detach support | built-in (`--detach`) | inherently detached (each pod is independent) |
| Spot recovery | n/a | per-source idempotency in `run_all.sh` |

## Failure modes + recovery

- **Pod failed to launch (capacity)**: RunPod's `create_pod` raises. `runpod_fanout.py` exits; re-run with fewer boxes or different region.
- **Pod ran but training crashed**: pod's HF upload step is skipped, results stay only on the (now-terminated) pod's ephemeral disk. Lost. Re-run with `--sources <failed_list>`.
- **HF upload failed**: results lost (pod terminated). Re-run.
- **HF rate-limited from concurrent commits**: HF generally tolerates this; multiple commits from multiple pods are fine. If you ever hit a wall, stagger pod launches by 30s.

The blast radius of any single-pod failure is bounded to whichever sources that pod was assigned. Worst case: 1/N of the run needs to be re-run.

## TODO / known limitations

- Doesn't yet auto-detect which sources are *already done* on HF and skip them. If you re-run, currently re-trains everything in `--sources`. Workaround: list missing sources manually.
- Spot mode commented but not default — uncomment when you trust it for your workload.
- No per-source intra-training checkpointing to durable storage. A preempted spot pod mid-source loses ~25 min of training for that source.
