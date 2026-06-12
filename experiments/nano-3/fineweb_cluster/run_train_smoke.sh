#!/usr/bin/env bash
# Pod entrypoint: Stage-4 tokenize a chunk of FineWeb into per-cluster BPE bins,
# then run a GPT-2-124M-scale slider TRAIN for ~60 micro-steps to (a) confirm the
# clustered token-level pipeline works and (b) MEASURE throughput (tokens/sec) so we
# can extrapolate full-run time/cost across GPUs. Uploads smoke_result.json + log.
set -uo pipefail
exec > >(tee -a /tmp/smoke.log) 2>&1
echo "=== gpt2-slider clustered smoke $(date) ==="
: "${HF_TOKEN:?need HF_TOKEN}"
HF_REPO="${HF_REPO:-iamtrask/abcGPT-nano-3}"
BRANCH="${BRANCH:-fineweb-cluster}"
N_DOCS="${N_DOCS:-500000}"
K="${K:-100}"
BATCH="${BATCH:-8}"
GA="${GA:-8}"
ITERS="${ITERS:-60}"

cd /workspace
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pip install -q --upgrade pip
pip install -q datasets tiktoken "huggingface_hub>=0.24" numpy

base="https://raw.githubusercontent.com/iamtrask/abcGPT/${BRANCH}/experiments/nano-3"
curl -fsSL "$base/fineweb_cluster/prepare_clustered.py" -o prepare_clustered.py
curl -fsSL "$base/model.py" -o model.py
curl -fsSL "$base/train.py" -o train.py
python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ.get("HF_REPO","iamtrask/abcGPT-nano-3"),
                    "fineweb_cluster/clusters.npz", repo_type="model", token=os.environ["HF_TOKEN"])
shutil.copy(p, "clusters.npz"); print("got clusters.npz")
PY

echo "=== STAGE 4: tokenize $N_DOCS docs -> per-cluster bins ==="
python prepare_clustered.py --n-docs "$N_DOCS" --clusters clusters.npz --out-dir data_clustered --k "$K"

echo "=== SMOKE TRAIN: GPT-2 124M slider (K=$K, batch=$BATCH, grad-accum=$GA, $ITERS iters) ==="
python train.py --variant lora --variant-name smoke-gpt2-clustered --data-dir data_clustered \
  --n-layer 12 --n-head 12 --n-embd 768 --block-size 1024 --dropout 0.0 \
  --batch-size "$BATCH" --grad-accum "$GA" --n-iters "$ITERS" --warmup 10 \
  --lr 6e-4 --beta2 0.95 --weight-decay 0.1 \
  --rank 16 --base-rank 64 --adaptive-capacity --bias-anchor --rslora \
  --gate-attention --gate-embedding --alpha-curriculum-until 0 \
  --no-eval --log-interval 5 --device cuda --amp-dtype bfloat16 --results-root /workspace/results || echo "TRAIN_FAILED"

echo "=== measure throughput from the train log ==="
BATCH="$BATCH" GA="$GA" python - <<'PY'
import json, os, glob, statistics
res = {"ok": False}
logs = glob.glob("/workspace/results/*/log.jsonl")
batch = int(os.environ["BATCH"]); toks_micro = batch * 1024
if logs:
    its = [json.loads(l) for l in open(logs[0]) if '"type": "iter"' in l]
    # each iter record = one log_interval(5) of micro-steps; dt_s is wall time for it.
    # drop the first record (kernel compile / warmup).
    dts = [r["dt_s"] for r in its[1:] if r.get("dt_s", 0) > 0]
    if dts:
        li = 5
        tps = sorted((li * toks_micro) / d for d in dts)
        med = statistics.median(tps)
        res = {"ok": True, "median_tok_per_s": med, "n_samples": len(tps),
               "batch": batch, "block": 1024, "grad_accum": int(os.environ["GA"]),
               "toks_per_microstep": toks_micro, "model": "gpt2-124M-slider-K100-r16"}
        print(f"THROUGHPUT: {med:,.0f} tok/s  (median of {len(tps)} samples)")
        for h in (1, 4):  # quick extrapolation hint at 10B tokens
            print(f"  -> 10B tokens / {h} GPU(s)-equiv: {10e9/(med*h)/3600:.1f} hr")
open("smoke_result.json", "w").write(json.dumps(res, indent=1))
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ.get("HF_REPO", "iamtrask/abcGPT-nano-3")
for src, dst in [("smoke_result.json", "fineweb_cluster/smoke_result.json"),
                 ("/tmp/smoke.log", "fineweb_cluster/smoke.log")]:
    try: api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo, repo_type="model")
    except Exception as e: print("upload err", e)
print("DONE_MARKER")
PY

echo "=== done $(date) — idling (launcher terminates) ==="
sleep infinity
