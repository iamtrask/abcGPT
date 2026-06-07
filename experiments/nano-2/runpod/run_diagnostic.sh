#!/usr/bin/env bash
# Minimal RunPod entrypoint that runs notebooks/precision_diagnostic.py and
# uploads stdout to HF. No training. Used to diff cloud env vs Colab/notebook.
#
# Required env vars:
#   HF_TOKEN          — HuggingFace token with write access to HF_REPO
#   HF_REPO           — defaults to iamtrask/abcGPT-nano-2
#   VARIANT_NAME      — destination filename in HF (e.g. "runpod-3090-default")
#   UPGRADE_TORCH     — "1" to pip-install newer torch before running
#   TORCH_VERSION     — default 2.11.0; only honored if UPGRADE_TORCH=1
set -uo pipefail

exec > >(tee -a /tmp/diag.log) 2>&1

echo "=== diagnostic run start: $(date) ==="
echo "VARIANT_NAME=$VARIANT_NAME  UPGRADE_TORCH=${UPGRADE_TORCH:-0}"

apt-get update -qq && apt-get install -y -qq git curl
pip install -q --upgrade pip
pip install -q numpy "huggingface_hub>=0.24"

git clone --depth 1 https://github.com/iamtrask/abcGPT.git /workspace/abcGPT
cd /workspace/abcGPT

if [[ "${UPGRADE_TORCH:-0}" == "1" ]]; then
  echo "--- upgrading torch ${TORCH_VERSION:-2.11.0} +cu128 ---"
  pip install --upgrade torch=="${TORCH_VERSION:-2.11.0}" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3
fi

echo "--- running diagnostic ---"
python3 notebooks/precision_diagnostic.py > /tmp/diag_output.txt 2>&1 || echo "diagnostic exited non-zero"
cat /tmp/diag_output.txt

echo "--- uploading to HF ---"
HF_REPO="${HF_REPO:-iamtrask/abcGPT-nano-2}"
python3 - <<PYEOF || true
import os
from huggingface_hub import HfApi, login, create_repo
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
create_repo("$HF_REPO", repo_type="model", exist_ok=True, private=False)
api = HfApi()
api.upload_file(
    path_or_fileobj="/tmp/diag_output.txt",
    path_in_repo=f"precision-diagnostic/{os.environ['VARIANT_NAME']}.txt",
    repo_id="$HF_REPO", repo_type="model",
    commit_message=f"diagnostic: {os.environ['VARIANT_NAME']}",
)
print("uploaded.")
PYEOF

echo "=== diagnostic run done: $(date) ==="
