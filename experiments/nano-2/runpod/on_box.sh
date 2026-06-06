#!/usr/bin/env bash
# on_box.sh — runs ONE nano-2 training variant on a RunPod pod, then uploads
# the variant's results to HuggingFace Hub and exits.
#
# Expected env vars (set by runpod_fanout.py via the pod's env):
#   VARIANT_NAME  — output subdir name (also used in HF commit message)
#   TRAIN_ARGS    — full argparse string for train.py, e.g.:
#                   "--variant trainable-mn --n-iters 10000 --span 0.5 --mask-lr-ratio 1e-2 --lambda-var 0.1"
#   HF_TOKEN      — HuggingFace token with write access to HF_REPO
#   HF_REPO       — target HF repo (default: iamtrask/abcGPT-nano-2)
#   GIT_SHA       — abcGPT commit to check out (optional, defaults to main)
#
# Each pod produces one variant's run, uploads results/<VARIANT_NAME>/ to HF,
# then exits (RunPod terminates billing for that container).

set -euo pipefail

exec > >(tee -a /tmp/on_box.log) 2>&1

echo "==================================================================="
echo "nano-2 RunPod box startup — $(date)"
echo "==================================================================="

: "${VARIANT_NAME:?need VARIANT_NAME env var}"
: "${TRAIN_ARGS:?need TRAIN_ARGS env var with full argparse string for train.py}"
: "${HF_TOKEN:?need HF_TOKEN env var with write access to target repo}"
HF_REPO="${HF_REPO:-iamtrask/abcGPT-nano-2}"
GIT_SHA="${GIT_SHA:-main}"
REPO_DIR="/workspace/abcGPT"

echo "Variant name: $VARIANT_NAME"
echo "Train args:   $TRAIN_ARGS"
echo "HF repo:      $HF_REPO"
echo "Git ref:      $GIT_SHA"
echo ""

# ------------------------------------------------------------------
# 1. System deps
# ------------------------------------------------------------------
echo "--- system deps ---"
apt-get update -qq
apt-get install -y -qq git curl

# ------------------------------------------------------------------
# 2. Clone abcGPT
# ------------------------------------------------------------------
echo "--- clone abcGPT ---"
if [[ ! -d "$REPO_DIR" ]]; then
  git clone --depth 1 https://github.com/iamtrask/abcGPT.git "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch --depth 1 origin "$GIT_SHA" 2>/dev/null || true
git checkout "$GIT_SHA" 2>/dev/null || git checkout -B detached "$(git rev-parse origin/main)"
echo "checked out: $(git rev-parse --short HEAD)"

# ------------------------------------------------------------------
# 3. Python deps
# ------------------------------------------------------------------
echo "--- python deps ---"
pip install -q --upgrade pip
pip install -q numpy "huggingface_hub>=0.24"
# torch should already be in the pytorch image

# ------------------------------------------------------------------
# 4. Prepare shake+TS data (downloads + tokenizes if not already there)
# ------------------------------------------------------------------
echo "--- prepare shake+TS data ---"
cd "$REPO_DIR"
if [[ ! -f data/shakespeare_tinystories_char/train.bin ]]; then
  python3 data/shakespeare_tinystories_char/prepare.py
else
  echo "  bins already present, skipping prep"
fi

# ------------------------------------------------------------------
# 5. Train
# ------------------------------------------------------------------
echo "--- train: variant=$VARIANT_NAME ---"
cd "$REPO_DIR"
# shellcheck disable=SC2086
python3 experiments/nano-2/train.py \
    --variant-name "$VARIANT_NAME" \
    $TRAIN_ARGS \
    --device cuda

# ------------------------------------------------------------------
# 6. Upload variant's results to HF Hub
# ------------------------------------------------------------------
echo "--- upload to HF ---"
python3 - <<PYEOF
import os, sys
from huggingface_hub import HfApi, login, create_repo

login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
create_repo("$HF_REPO", repo_type="model", exist_ok=True, private=False)

variant_dir = "$REPO_DIR/experiments/nano-2/results/$VARIANT_NAME"
if not os.path.isdir(variant_dir):
    sys.exit(f"error: variant results dir missing: {variant_dir}")

api = HfApi()
api.upload_folder(
    folder_path=variant_dir,
    path_in_repo="$VARIANT_NAME",
    repo_id="$HF_REPO",
    repo_type="model",
    commit_message=f"nano-2: add results for variant '$VARIANT_NAME'",
)
print(f"uploaded {variant_dir} -> {('$HF_REPO')}/$VARIANT_NAME")
PYEOF

echo "==================================================================="
echo "on_box.sh DONE at $(date)"
echo "==================================================================="
