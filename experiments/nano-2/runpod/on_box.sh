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
# 5. Start background log-pusher BEFORE training begins.
#
# Pushes ONLY log.jsonl to HF every 60s while training runs, so the
# orchestrator (and a human) can see live training progress. Pushes only
# the single log file — never summary.json, which the orchestrator uses
# as the "training done" marker (uploaded only by the final step below).
# ------------------------------------------------------------------
LOG_PATH="$REPO_DIR/experiments/nano-2/results/$VARIANT_NAME/log.jsonl"
mkdir -p "$REPO_DIR/experiments/nano-2/results/$VARIANT_NAME"

echo "--- start background log-pusher (60s interval) ---"
(
  # Wait a beat so the path actually exists when we start polling.
  sleep 5
  while true; do
    sleep 55
    if [[ -f "$LOG_PATH" ]]; then
      python3 - <<PYEOF 2>/dev/null || true
import os
from huggingface_hub import HfApi, login, create_repo
try:
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    create_repo("$HF_REPO", repo_type="model", exist_ok=True, private=False)
    api = HfApi()
    api.upload_file(
        path_or_fileobj="$LOG_PATH",
        path_in_repo="$VARIANT_NAME/log.jsonl",
        repo_id="$HF_REPO", repo_type="model",
        commit_message="live: $VARIANT_NAME training in progress",
    )
except Exception as e:
    # quietly swallow upload errors; final upload will sync the truth
    pass
PYEOF
    fi
  done
) &
UPLOADER_PID=$!
echo "background uploader PID: $UPLOADER_PID"

# ------------------------------------------------------------------
# 6. Train (foreground)
# ------------------------------------------------------------------
echo "--- train: variant=$VARIANT_NAME ---"
cd "$REPO_DIR"
# shellcheck disable=SC2086
python3 experiments/nano-2/train.py \
    --variant-name "$VARIANT_NAME" \
    $TRAIN_ARGS \
    --device cuda

# ------------------------------------------------------------------
# 7. Stop the background uploader cleanly before the final upload.
# ------------------------------------------------------------------
echo "--- stop background uploader ---"
kill "$UPLOADER_PID" 2>/dev/null || true
wait "$UPLOADER_PID" 2>/dev/null || true

# ------------------------------------------------------------------
# 8. Final upload (whole folder including summary.json — the marker the
#    orchestrator uses to know training is done and the pod can be killed).
# ------------------------------------------------------------------
echo "--- final upload to HF (includes summary.json done-marker) ---"
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
    commit_message=f"nano-2: final results for variant '$VARIANT_NAME'",
)
print(f"uploaded {variant_dir} -> {('$HF_REPO')}/$VARIANT_NAME")
PYEOF

echo "==================================================================="
echo "on_box.sh DONE at $(date)"
echo "==================================================================="
