#!/usr/bin/env bash
# Pod entrypoint: OFFLINE corner eval for a trained FineWeb K-cluster checkpoint.
# Computes diag / contrast / middle (see eval_corners.py) on the cohort val bins,
# uploads the result JSON to HF as ${RUN_NAME}/corner_metrics.json, then idles.
#
# env:
#   RUN_NAME     HF dir of the run (e.g. slider-gpt2-bRfull-commit100)   [required]
#   VARIANT      ungated | lora                                          [required]
#   RESERVE_FRAC reserve-frac the run was trained with (default 0)
#   K            number of cohorts (default 100)
#   DATA_CKEY    HF cache key for the tokenized bins (default data_clustered_cache)
#   CKPT_HF      HF path to the checkpoint (default $RUN_NAME/ckpt.pt)
#   EVAL_ITERS   val batches per eval point (default 80)
#   SAMPLE_OFFDIAG  random other-cohort subset for min off-diag (default 12)
#   HF_TOKEN     HF auth token                                           [required]
#   HF_REPO      default iamtrask/abcGPT-nano-3
#   BRANCH       default fineweb-cluster
set -uo pipefail
exec > >(tee -a /tmp/eval.log) 2>&1
echo "=== corner-eval $(date) ==="
: "${HF_TOKEN:?need HF_TOKEN}"
export RUN_NAME="${RUN_NAME:?need RUN_NAME}"
export VARIANT="${VARIANT:?need VARIANT (ungated|lora)}"
export HF_REPO="${HF_REPO:-iamtrask/abcGPT-nano-3}"
BRANCH="${BRANCH:-fineweb-cluster}"
export RESERVE_FRAC="${RESERVE_FRAC:-0}"
export K="${K:-100}"
export DATA_CKEY="${DATA_CKEY:-data_clustered_cache}"
export CKPT_HF="${CKPT_HF:-$RUN_NAME/ckpt.pt}"
export EVAL_ITERS="${EVAL_ITERS:-80}"
export SAMPLE_OFFDIAG="${SAMPLE_OFFDIAG:-12}"
export DATA_DIR="${DATA_DIR:-data_clustered}"
echo "RUN_NAME=$RUN_NAME VARIANT=$VARIANT K=$K RESERVE_FRAC=$RESERVE_FRAC CKPT_HF=$CKPT_HF"

cd /workspace
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pip install -q --upgrade pip
pip install -q torch tiktoken numpy "huggingface_hub>=0.24"

base="https://raw.githubusercontent.com/iamtrask/abcGPT/${BRANCH}/experiments/nano-3"
curl -fsSL "$base/model.py" -o model.py
curl -fsSL "$base/train.py" -o train.py
curl -fsSL "$base/fineweb_cluster/eval_corners.py" -o eval_corners.py

# ---- background log pusher: ${RUN_NAME}/eval.log every 120s ----
push_log() { python - <<'PY' 2>/dev/null
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
if os.path.exists("/tmp/eval.log"):
    try:
        api.upload_file(path_or_fileobj="/tmp/eval.log",
                        path_in_repo=f"{os.environ['RUN_NAME']}/eval.log",
                        repo_id=os.environ["HF_REPO"], repo_type="model")
    except Exception:
        pass
PY
}
( while true; do sleep 120; push_log; done ) &

# ---- download checkpoint + val data ----
echo "=== downloading checkpoint $CKPT_HF ==="
python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ["HF_REPO"], os.environ["CKPT_HF"],
                    repo_type="model", token=os.environ["HF_TOKEN"])
shutil.copy(p, "/workspace/ckpt.pt")
print("got checkpoint ->", os.environ["CKPT_HF"])
PY

echo "=== downloading val data ${DATA_CKEY}/bins.tar.gz -> ${DATA_DIR}/ ==="
python - <<'PY'
import os, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ["HF_REPO"], os.environ["DATA_CKEY"] + "/bins.tar.gz",
                    repo_type="model", token=os.environ["HF_TOKEN"])
out = os.environ["DATA_DIR"]
os.makedirs(out, exist_ok=True)
with tarfile.open(p) as t:
    t.extractall(out)
print("extracted bins ->", out)
PY

# ---- run the eval ----
echo "=== EVAL ($VARIANT, K=$K, eval_iters=$EVAL_ITERS, sample_offdiag=$SAMPLE_OFFDIAG) ==="
python eval_corners.py \
  --data-dir "$DATA_DIR" --ckpt /workspace/ckpt.pt --variant "$VARIANT" \
  --reserve-frac "$RESERVE_FRAC" --k "$K" \
  --eval-iters "$EVAL_ITERS" --sample-offdiag "$SAMPLE_OFFDIAG" \
  --out /workspace/corner_metrics.json || echo "EVAL_FAILED"

# ---- upload result JSON + echo it ----
if [ -f /workspace/corner_metrics.json ]; then
  echo "=== corner_metrics.json ==="
  cat /workspace/corner_metrics.json
  python - <<'PY'
import os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj="/workspace/corner_metrics.json",
    path_in_repo=f"{os.environ['RUN_NAME']}/corner_metrics.json",
    repo_id=os.environ["HF_REPO"], repo_type="model")
print("uploaded ->", os.environ["RUN_NAME"] + "/corner_metrics.json")
PY
else
  echo "no corner_metrics.json produced (EVAL_FAILED?)"
fi

push_log
echo "=== EVAL_DONE ==="
sleep infinity
