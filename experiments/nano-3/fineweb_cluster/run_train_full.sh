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
BATCH="${BATCH:-8}"; GA="${GA:-64}"; NITERS="${NITERS:-1220000}"; WARMUP="${WARMUP:-44000}"; RANK="${RANK:-16}"
COMMIT_FRAC="${COMMIT_FRAC:-0}"
echo "RUN_NAME=$RUN_NAME  COMMIT_FRAC=$COMMIT_FRAC"

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
p = hf_hub_download(os.environ["HF_REPO"], "fineweb_cluster/clusters.npz", repo_type="model", token=os.environ["HF_TOKEN"])
shutil.copy(p, "clusters.npz"); print("got clusters.npz")
try:
    r = hf_hub_download(os.environ["HF_REPO"], f"{os.environ['RUN_NAME']}/ckpt.pt", repo_type="model", token=os.environ["HF_TOKEN"])
    shutil.copy(r, "/workspace/resume.pt"); print("found resume ckpt")
except Exception: print("no resume ckpt -> fresh start")
PY
RESUME=""; [ -f /workspace/resume.pt ] && RESUME="--resume-from /workspace/resume.pt"

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

if [ ! -f data_clustered/meta.pkl ]; then
  echo "=== tokenize full corpus -> per-cluster bins (~2.4 hr) ==="
  python prepare_clustered.py --n-docs "$N_DOCS" --clusters clusters.npz --out-dir data_clustered --k "$K"
fi

echo "=== TRAIN $RUN_NAME (K=$K, rank=$RANK, commit_frac=$COMMIT_FRAC) ==="
python train.py --variant lora --variant-name "$RUN_NAME" --data-dir data_clustered \
  --n-layer 12 --n-head 12 --n-embd 768 --block-size 1024 --dropout 0.0 \
  --batch-size "$BATCH" --grad-accum "$GA" --n-iters "$NITERS" --warmup "$WARMUP" \
  --lr 6e-4 --min-lr 6e-5 --beta2 0.95 --weight-decay 0.1 --grad-clip 1.0 \
  --rank "$RANK" --base-rank 64 --adaptive-capacity --bias-anchor --rslora \
  --gate-attention --gate-embedding --commit-frac "$COMMIT_FRAC" --alpha-curriculum-until 0 \
  --no-eval --log-interval 500 --ckpt-every 30000 \
  --device cuda --amp-dtype bfloat16 --results-root /workspace/results $RESUME || echo "TRAIN_FAILED"

push
echo "=== done $(date) DONE_MARKER ==="
sleep infinity
