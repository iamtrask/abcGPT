#!/usr/bin/env bash
# Pod entrypoint: GPT-2 124M 100-CLUSTER SLIDER on FineWeb-10BT, 1 epoch. Parameterized
# by RUN_NAME (HF folder + variant) and COMMIT_FRAC (α regime: 0=Dirichlet interior,
# 1.0=one-hot corners, 0.98=mostly-committed). Tokenizes per-cluster bins, trains with
# periodic checkpoint + log push to HF. --no-eval (diag/contrast computed offline).
set -uo pipefail
exec > >(tee -a /tmp/full.log) 2>&1
echo "=== gpt2-124M SLIDER full run $(date) ==="
: "${HF_TOKEN:?need HF_TOKEN}"
export HF_REPO="${HF_REPO:-iamtrask/abcGPT-nano-3}"
export RUN_NAME="${RUN_NAME:-slider-gpt2}"
BRANCH="${BRANCH:-fineweb-cluster}"
N_DOCS="${N_DOCS:-0}"; K="${K:-100}"
# K-general: which cluster file / data dir / cache key this run uses (so K=1000 etc.
# don't collide with the K=100 artifacts). Defaults reproduce the original K=100 run.
export CLUSTERS_HF="${CLUSTERS_HF:-fineweb_cluster/clusters.npz}"
export DATA_DIR="${DATA_DIR:-data_clustered}"
export CKEY="${CKEY:-data_clustered_cache}"
BATCH="${BATCH:-8}"; GA="${GA:-64}"; NITERS="${NITERS:-1220000}"; WARMUP="${WARMUP:-44000}"; RANK="${RANK:-16}"
COMMIT_FRAC="${COMMIT_FRAC:-0}"
# base_rank: -1 = FULL base (a real GPT-2 backbone, deltas ride on top); >0 = low-rank A@B^T.
BASE_RANK="${BASE_RANK:-64}"
# offload: stream only the active cohort's delta to GPU (frees ~3.8GB at commit-100 -> bigger batch).
OFFLOAD_FLAG=""; [ -n "${OFFLOAD:-}" ] && OFFLOAD_FLAG="--offload-deltas"
# reserve_frac: reserve a fraction of base output neurons for per-cohort deltas to imprint (0 = off).
RESERVE_FRAC="${RESERVE_FRAC:-0}"
echo "RUN_NAME=$RUN_NAME  COMMIT_FRAC=$COMMIT_FRAC"

cd /workspace
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pip install -q --upgrade pip
pip install -q datasets tiktoken "huggingface_hub>=0.24" numpy

base="https://raw.githubusercontent.com/iamtrask/abcGPT/${BRANCH}/experiments/nano-3"
curl -fsSL "$base/fineweb_cluster/prepare_clustered.py" -o prepare_clustered.py
curl -fsSL "$base/model.py" -o model.py
curl -fsSL "$base/train.py" -o train.py
rm -f /workspace/NEED_CLUSTER
python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
try:
    p = hf_hub_download(os.environ["HF_REPO"], os.environ["CLUSTERS_HF"], repo_type="model", token=os.environ["HF_TOKEN"])
    shutil.copy(p, "clusters.npz"); print("got clusters.npz <-", os.environ["CLUSTERS_HF"])
except Exception:
    print("no precomputed", os.environ["CLUSTERS_HF"], "-> will cluster domains into K=" + os.environ["K"])
    open("/workspace/NEED_CLUSTER", "w").close()
try:
    r = hf_hub_download(os.environ["HF_REPO"], f"{os.environ['RUN_NAME']}/ckpt.pt", repo_type="model", token=os.environ["HF_TOKEN"])
    shutil.copy(r, "/workspace/resume.pt"); print("found resume ckpt")
except Exception: print("no resume ckpt -> fresh start")
PY
RESUME=""; [ -f /workspace/resume.pt ] && RESUME="--resume-from /workspace/resume.pt"

# One-time on-the-fly clustering when this K has no precomputed cluster file yet.
if [ -f /workspace/NEED_CLUSTER ]; then
  echo "=== clustering 4.46M domains -> K=$K balanced buckets (one-time, ~20-40 min) ==="
  pip install -q scikit-learn
  curl -fsSL "$base/fineweb_cluster/cluster_domains_balanced.py" -o cluster_domains_balanced.py
  mkdir -p clusterin
  python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ["HF_REPO"], "fineweb_cluster/domains.npz", repo_type="model", token=os.environ["HF_TOKEN"])
shutil.copy(p, "clusterin/domains.npz"); print("got domains.npz")
PY
  python cluster_domains_balanced.py --in-dir clusterin --out-dir clusterin --k "$K"
  cp clusterin/clusters.npz clusters.npz
  python - <<'PY'
import os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj="clusters.npz",
    path_in_repo=os.environ["CLUSTERS_HF"], repo_id=os.environ["HF_REPO"], repo_type="model")
print("uploaded", os.environ["CLUSTERS_HF"])
PY
fi

push() { python - <<'PY' 2>/dev/null
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ["HF_REPO"]; rn = os.environ["RUN_NAME"]
for s, d in [("/tmp/full.log", f"{rn}/full.log"),
             (f"results/{rn}/log.jsonl", f"{rn}/log.jsonl"),
             (f"results/{rn}/ckpt.pt", f"{rn}/ckpt.pt")]:
    if os.path.exists(s):
        try: api.upload_file(path_or_fileobj=s, path_in_repo=d, repo_id=repo, repo_type="model")
        except Exception: pass
PY
}
( while true; do sleep 300; push; done ) &   # starts BEFORE tokenize -> tokenize visible

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

if [ ! -f "$DATA_DIR/meta.pkl" ]; then
  export COUT="$DATA_DIR"
  if get_cache; then
    echo "=== downloaded PRE-TOKENIZED per-cluster bins from cache (skipped tokenize) ==="
  else
    echo "=== no cache -> tokenize + upload for next time (~2.4 hr) ==="
    python prepare_clustered.py --n-docs "$N_DOCS" --clusters clusters.npz --out-dir "$DATA_DIR" --k "$K"
    put_cache || echo "cache upload failed (non-fatal)"
  fi
fi

echo "=== TRAIN $RUN_NAME (K=$K, rank=$RANK, base_rank=$BASE_RANK, offload=${OFFLOAD:-0}, reserve_frac=$RESERVE_FRAC, commit_frac=$COMMIT_FRAC) ==="
python train.py --variant lora --variant-name "$RUN_NAME" --data-dir "$DATA_DIR" \
  --n-layer 12 --n-head 12 --n-embd 768 --block-size 1024 --dropout 0.0 \
  --batch-size "$BATCH" --grad-accum "$GA" --n-iters "$NITERS" --warmup "$WARMUP" \
  --lr 6e-4 --min-lr 6e-5 --beta2 0.95 --weight-decay 0.1 --grad-clip 1.0 \
  --rank "$RANK" --base-rank "$BASE_RANK" $OFFLOAD_FLAG --reserve-frac "$RESERVE_FRAC" --adaptive-capacity --bias-anchor --rslora \
  --gate-attention --gate-embedding --commit-frac "$COMMIT_FRAC" --alpha-curriculum-until 0 \
  --no-eval --log-interval 500 --ckpt-every 30000 \
  --device cuda --amp-dtype bfloat16 --results-root /workspace/results $RESUME || echo "TRAIN_FAILED"

push
echo "=== done $(date) DONE_MARKER ==="
sleep infinity
