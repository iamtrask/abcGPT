#!/usr/bin/env bash
# Pod entrypoint: plain GPT-2 124M BASELINE on FineWeb-10BT (ungated, uniform sampling).
# Validates that our pipeline reproduces GPT-2 quality (val loss ~3.29) BEFORE we trust
# the slider. Tokenizes the full corpus, trains 1 epoch with periodic checkpoint + loss
# logging to HF (so a community-pod reclaim is recoverable + the loss curve is visible).
set -uo pipefail
exec > >(tee -a /tmp/base.log) 2>&1
echo "=== gpt2-124M BASELINE $(date) ==="
: "${HF_TOKEN:?need HF_TOKEN}"
HF_REPO="${HF_REPO:-iamtrask/abcGPT-nano-3}"
BRANCH="${BRANCH:-fineweb-cluster}"
N_DOCS="${N_DOCS:-0}"
BATCH="${BATCH:-16}"; GA="${GA:-30}"; NITERS="${NITERS:-610000}"; WARMUP="${WARMUP:-21000}"

cd /workspace
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pip install -q --upgrade pip
pip install -q datasets tiktoken "huggingface_hub>=0.24" numpy

base="https://raw.githubusercontent.com/iamtrask/abcGPT/${BRANCH}/experiments/nano-3"
curl -fsSL "$base/fineweb_cluster/prepare_full.py" -o prepare_full.py
curl -fsSL "$base/model.py" -o model.py
curl -fsSL "$base/train.py" -o train.py

# resume from HF ckpt if a prior (reclaimed) attempt left one
python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
try:
    p = hf_hub_download(os.environ["HF_REPO"], "baseline_gpt2/ckpt.pt", repo_type="model", token=os.environ["HF_TOKEN"])
    shutil.copy(p, "/workspace/resume.pt"); print("found resume ckpt on HF")
except Exception:
    print("no resume ckpt -> fresh start")
PY
RESUME=""; [ -f /workspace/resume.pt ] && RESUME="--resume-from /workspace/resume.pt"

# background: push log + ckpt to HF every 5 min. STARTS BEFORE tokenize so the long
# tokenization phase is visible too (not just training).
push() { python - <<'PY' 2>/dev/null
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ["HF_REPO"]
for s, d in [("/tmp/base.log", "baseline_gpt2/base.log"),
             ("results/baseline-gpt2/log.jsonl", "baseline_gpt2/log.jsonl"),
             ("results/baseline-gpt2/ckpt.pt", "baseline_gpt2/ckpt.pt")]:
    if os.path.exists(s):
        try: api.upload_file(path_or_fileobj=s, path_in_repo=d, repo_id=repo, repo_type="model")
        except Exception: pass
PY
}
( while true; do sleep 300; push; done ) &

get_cache() { python - <<'PY' 2>/dev/null
import os, sys, tarfile
from huggingface_hub import hf_hub_download
try:
    p = hf_hub_download(os.environ["HF_REPO"], os.environ["CKEY"] + "/bins.tar.gz",
                        repo_type="model", token=os.environ["HF_TOKEN"])
    os.makedirs(os.environ["COUT"], exist_ok=True)
    with tarfile.open(p) as t: t.extractall(os.environ["COUT"])
    sys.exit(0)
except Exception: sys.exit(1)
PY
}
put_cache() { python - <<'PY' 2>/dev/null
import os, glob, tarfile
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ["HF_REPO"]; key = os.environ["CKEY"]
if f"{key}/bins.tar.gz" in set(api.list_repo_files(repo)):
    print("cache already present; skip upload"); raise SystemExit
with tarfile.open("/tmp/cache.tar.gz", "w:gz") as tar:
    for f in glob.glob(os.environ["COUT"] + "/*"):
        tar.add(f, arcname=os.path.basename(f))
api.upload_file(path_or_fileobj="/tmp/cache.tar.gz", path_in_repo=f"{key}/bins.tar.gz", repo_id=repo, repo_type="model")
print("uploaded tokenization cache", key)
PY
}

if [ ! -f data_full/meta.pkl ]; then
  export CKEY="data_full_cache" COUT="data_full"
  if get_cache; then
    echo "=== downloaded PRE-TOKENIZED combined bin from cache (skipped tokenize) ==="
  else
    echo "=== no cache -> tokenize + upload for next time (~2.4 hr) ==="
    python prepare_full.py --n-docs "$N_DOCS" --out-dir data_full
    put_cache || echo "cache upload failed (non-fatal)"
  fi
fi

echo "=== TRAIN baseline GPT-2 124M (ungated, target val ~3.29) ==="
python train.py --variant ungated --variant-name baseline-gpt2 --data-dir data_full \
  --n-layer 12 --n-head 12 --n-embd 768 --block-size 1024 --dropout 0.0 \
  --batch-size "$BATCH" --grad-accum "$GA" --n-iters "$NITERS" --warmup "$WARMUP" \
  --lr 6e-4 --min-lr 6e-5 --beta2 0.95 --weight-decay 0.1 --grad-clip 1.0 \
  --eval-interval 30000 --eval-iters 100 --log-interval 300 --ckpt-every 15000 \
  --device cuda --amp-dtype bfloat16 --results-root /workspace/results $RESUME || echo "TRAIN_FAILED"

push
echo "=== done $(date) DONE_MARKER ==="
sleep infinity
