#!/usr/bin/env bash
# on_box.sh — runs on each RunPod pod at startup.
#
# Clones abcGPT, installs deps, trains the assigned source subset, uploads
# results to HuggingFace Hub, then exits (which terminates the pod if it was
# launched with auto-terminate-on-exit).
#
# Expected env vars (set by runpod_fanout.py via the pod's env):
#   SOURCES   — comma-separated source names (e.g. "077_nba-play-by-play,076_retrosheet-baseball")
#               OR the literal string "all" to train all 100 sources
#   HF_TOKEN  — HuggingFace token with write access to iamtrask/abcGPT-nano-1-baselines
#   HF_REPO   — target HF repo (default: iamtrask/abcGPT-nano-1-baselines)
#   GIT_SHA   — abcGPT commit to check out (optional, defaults to main)
#
# Logs everything to /tmp/on_box.log on the pod for debugging.

set -euo pipefail

exec > >(tee -a /tmp/on_box.log) 2>&1

echo "==================================================================="
echo "abcGPT nano-1 RunPod box startup — $(date)"
echo "==================================================================="

: "${SOURCES:?need SOURCES env var (comma-separated names, or 'all')}"
: "${HF_TOKEN:?need HF_TOKEN env var with write access to target repo}"
HF_REPO="${HF_REPO:-iamtrask/abcGPT-nano-1-baselines}"
GIT_SHA="${GIT_SHA:-main}"
REPO_DIR="/workspace/abcGPT"

echo "Sources: $SOURCES"
echo "HF repo: $HF_REPO"
echo "Git ref: $GIT_SHA"
echo ""

# ------------------------------------------------------------------
# 1. System deps (most RunPod PyTorch images already have these)
# ------------------------------------------------------------------
echo "--- system deps ---"
apt-get update -qq
apt-get install -y -qq git rsync curl

# ------------------------------------------------------------------
# 2. Clone abcGPT (corpus is now committed to the repo, no extra rsync needed)
# ------------------------------------------------------------------
echo "--- clone abcGPT ---"
if [[ ! -d "$REPO_DIR" ]]; then
  git clone --depth 1 https://github.com/iamtrask/abcGPT.git "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch --depth 1 origin "$GIT_SHA" || true
git checkout "$GIT_SHA" 2>/dev/null || git checkout -B detached "$(git rev-parse origin/main)"
echo "checked out: $(git rev-parse --short HEAD)"

# ------------------------------------------------------------------
# 3. Python deps (most RunPod PyTorch images already have torch/numpy)
# ------------------------------------------------------------------
echo "--- python deps ---"
pip install -q --upgrade pip
pip install -q "tiktoken==0.7.0" "huggingface_hub>=0.24" tqdm

# ------------------------------------------------------------------
# 4. Train the assigned source subset
# ------------------------------------------------------------------
echo "--- training ---"
cd "$REPO_DIR"
if [[ "$SOURCES" == "all" ]]; then
  bash data/100_simple_voices/baselines/run_all.sh
else
  # run_all.sh accepts space-separated source names as args (or "top5")
  # Convert comma-separated to space-separated.
  SOURCES_SPACE="${SOURCES//,/ }"
  bash data/100_simple_voices/baselines/run_all.sh $SOURCES_SPACE
fi

# ------------------------------------------------------------------
# 5. Upload results to HF Hub
# ------------------------------------------------------------------
echo "--- upload to HF ---"
python3 - <<PYEOF
import os, sys
from huggingface_hub import HfApi, login

login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
api = HfApi()

results_dir = "$REPO_DIR/experiments/nano-1/results"
if not os.path.isdir(results_dir):
    sys.exit(f"error: results dir missing: {results_dir}")

# Ensure repo exists (idempotent — create_repo with exist_ok=True is a no-op if present)
from huggingface_hub import create_repo
create_repo("$HF_REPO", repo_type="model", exist_ok=True, private=False)

# Upload the results dir. HF de-dupes on file hash, so multi-box concurrent
# uploads to the same repo are safe (each commit only adds the files that box
# produced).
sources_tag = os.environ["SOURCES"]
api.upload_folder(
    folder_path=results_dir,
    repo_id="$HF_REPO",
    repo_type="model",
    commit_message=f"nano-1: add results from sources [{sources_tag}]",
)
print(f"uploaded {results_dir} -> {('$HF_REPO')}")
PYEOF

echo "==================================================================="
echo "on_box.sh DONE at $(date) — exiting (pod will auto-terminate)"
echo "==================================================================="
