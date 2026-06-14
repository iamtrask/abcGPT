#!/usr/bin/env bash
# Cluster-only pod: download domains.npz, cluster the 4.46M domains into K
# token-balanced buckets, upload clusters_<K>.npz to HF. No training. Used to
# pre-stage cluster files for high-K slider runs.
#   env: K, MICRO, CLUSTERS_HF, HF_TOKEN, HF_REPO, BRANCH
set -uo pipefail
exec > >(tee -a /tmp/cluster.log) 2>&1
echo "=== cluster K=$K micro=$MICRO -> $CLUSTERS_HF  $(date) ==="
: "${HF_TOKEN:?need HF_TOKEN}"
HF_REPO="${HF_REPO:-iamtrask/abcGPT-nano-3}"; BRANCH="${BRANCH:-fineweb-cluster}"
cd /workspace
pip install -q scikit-learn numpy "huggingface_hub>=0.24"
base="https://raw.githubusercontent.com/iamtrask/abcGPT/${BRANCH}/experiments/nano-3"
curl -fsSL "$base/fineweb_cluster/cluster_domains_balanced.py" -o cluster_domains_balanced.py

LOGNAME="clusterlogs/$(echo "$CLUSTERS_HF" | tr '/' '_').log"
push() { python - <<PY 2>/dev/null
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
if os.path.exists("/tmp/cluster.log"):
    try: api.upload_file(path_or_fileobj="/tmp/cluster.log", path_in_repo="$LOGNAME", repo_id="$HF_REPO", repo_type="model")
    except Exception: pass
PY
}
( while true; do sleep 120; push; done ) &

mkdir -p d
python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ["HF_REPO"], "fineweb_cluster/domains.npz", repo_type="model", token=os.environ["HF_TOKEN"])
shutil.copy(p, "d/domains.npz"); print("got domains.npz")
PY

python cluster_domains_balanced.py --in-dir d --out-dir d --k "$K" --micro "$MICRO" || { echo "CLUSTER_FAILED"; push; sleep infinity; }

python - <<PY
import os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj="d/clusters.npz",
    path_in_repo="$CLUSTERS_HF", repo_id="$HF_REPO", repo_type="model")
print("uploaded", "$CLUSTERS_HF")
PY
push
echo "=== CLUSTER_DONE $(date) ==="
sleep infinity
