#!/usr/bin/env bash
# on_box.sh — runs ONE nano-3 training variant on a RunPod pod, then uploads
# the variant's results to HuggingFace Hub at iamtrask/abcGPT-nano-3 and exits.
#
# Expected env vars (set by runpod_fanout.py via the pod's env):
#   VARIANT_NAME  — output subdir name (also used in HF commit message)
#   TRAIN_ARGS    — full argparse string for nano-3/train.py
#   HF_TOKEN      — HuggingFace token with write access to HF_REPO
#   HF_REPO       — target HF repo (default: iamtrask/abcGPT-nano-3)
#   GIT_SHA       — abcGPT commit to check out (optional, defaults to main)
#
# Modeled on experiments/nano-2/runpod/on_box.sh. Differences:
#   - clones to /workspace/abcGPT
#   - runs data/shake_ts_code_char/prepare.py (not shakespeare_tinystories_char)
#   - runs experiments/nano-3/train.py
#   - uploads to iamtrask/abcGPT-nano-3

set -euo pipefail
exec > >(tee -a /tmp/on_box.log) 2>&1

echo "==================================================================="
echo "nano-3 RunPod box startup — $(date)"
echo "==================================================================="

: "${VARIANT_NAME:?need VARIANT_NAME env var}"
: "${TRAIN_ARGS:?need TRAIN_ARGS env var with full argparse string for train.py}"
: "${HF_TOKEN:?need HF_TOKEN env var with write access to target repo}"
HF_REPO="${HF_REPO:-iamtrask/abcGPT-nano-3}"
GIT_SHA="${GIT_SHA:-main}"
REPO_DIR="/workspace/abcGPT"

echo "Variant name: $VARIANT_NAME"
echo "Train args:   $TRAIN_ARGS"
echo "HF repo:      $HF_REPO"
echo "Git ref:      $GIT_SHA"

# 1. System deps
echo "--- system deps ---"
apt-get update -qq
apt-get install -y -qq git curl

# 2. Clone abcGPT
echo "--- clone abcGPT ---"
if [[ ! -d "$REPO_DIR" ]]; then
  git clone --depth 1 https://github.com/iamtrask/abcGPT.git "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch --depth 1 origin "$GIT_SHA" 2>/dev/null || true
git checkout "$GIT_SHA" 2>/dev/null || git checkout -B detached "$(git rev-parse origin/main)"
echo "checked out: $(git rev-parse --short HEAD)"

# 3. Python deps
echo "--- python deps ---"
pip install -q --upgrade pip
pip install -q numpy "huggingface_hub>=0.24"

# 4. Prepare data — prepare.py self-downloads shake / TinyStories / cpython sources
echo "--- prepare shake+ts+code data ---"
cd "$REPO_DIR"
if [[ ! -f data/shake_ts_code_char/meta.pkl ]]; then
  python3 data/shake_ts_code_char/prepare.py
else
  echo "  bins already present, skipping prep"
fi

# 5. Start background log-pusher (5-minute interval, HF rate limit safe)
LOG_PATH="$REPO_DIR/experiments/nano-3/results/$VARIANT_NAME/log.jsonl"
mkdir -p "$REPO_DIR/experiments/nano-3/results/$VARIANT_NAME"

echo "--- start background log-pusher (5-min interval) ---"
(
  sleep 5
  while true; do
    sleep 295
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
except Exception:
    pass
PYEOF
    fi
  done
) &
UPLOADER_PID=$!
echo "background uploader PID: $UPLOADER_PID"

# 6. Train
echo "--- train: variant=$VARIANT_NAME ---"
cd "$REPO_DIR"
TRAIN_EXIT=0
# shellcheck disable=SC2086
python3 experiments/nano-3/train.py \
    --variant-name "$VARIANT_NAME" \
    $TRAIN_ARGS \
    --device cuda || TRAIN_EXIT=$?
echo "--- train.py exited with code: $TRAIN_EXIT ---"

# 7. Stop background uploader
echo "--- stop background uploader ---"
kill "$UPLOADER_PID" 2>/dev/null || true
wait "$UPLOADER_PID" 2>/dev/null || true

# 8. Final upload (always — push partial logs on crash too)
echo "--- final upload to HF (includes summary.json done-marker on success) ---"
mkdir -p "$REPO_DIR/experiments/nano-3/results/$VARIANT_NAME"
cp /tmp/on_box.log "$REPO_DIR/experiments/nano-3/results/$VARIANT_NAME/on_box.log" 2>/dev/null || true
echo "$TRAIN_EXIT" > "$REPO_DIR/experiments/nano-3/results/$VARIANT_NAME/train_exit_code.txt"

python3 - <<PYEOF || true
import os, sys
from huggingface_hub import HfApi, login, create_repo
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
create_repo("$HF_REPO", repo_type="model", exist_ok=True, private=False)
variant_dir = "$REPO_DIR/experiments/nano-3/results/$VARIANT_NAME"
if not os.path.isdir(variant_dir):
    sys.exit(f"error: variant results dir missing: {variant_dir}")
api = HfApi()
api.upload_folder(
    folder_path=variant_dir,
    path_in_repo="$VARIANT_NAME",
    repo_id="$HF_REPO",
    repo_type="model",
    commit_message=f"nano-3: results for '$VARIANT_NAME' (train_exit=$TRAIN_EXIT)",
)
print(f"uploaded {variant_dir} -> ${HF_REPO}/$VARIANT_NAME")
PYEOF

echo "==================================================================="
echo "on_box.sh DONE at $(date) (train_exit=$TRAIN_EXIT)"
echo "==================================================================="
exit "$TRAIN_EXIT"
