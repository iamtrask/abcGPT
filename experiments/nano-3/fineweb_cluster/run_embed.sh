#!/usr/bin/env bash
# Pod entrypoint: full FineWeb-10BT domain-embedding + balanced-clustering pass.
# Streams FineWeb at datacenter speed, embeds every domain on the GPU, clusters into
# K token-balanced groups, uploads results (+ log) to HF. No self-terminate — the
# launcher monitors HF for clusters.npz and terminates the pod from outside (so the
# RunPod API key never lands on a community pod).
set -uo pipefail
exec > >(tee -a /tmp/embed.log) 2>&1
echo "=== fineweb-cluster embed box $(date) ==="
: "${HF_TOKEN:?need HF_TOKEN}"
HF_REPO="${HF_REPO:-iamtrask/abcGPT-nano-3}"
BRANCH="${BRANCH:-fineweb-cluster}"
N_DOCS="${N_DOCS:-0}"        # 0 = all ~14.9M
K="${K:-100}"
MICRO="${MICRO:-3000}"
CAP="${CAP:-8}"

cd /workspace
pip install -q --upgrade pip
pip install -q datasets sentence-transformers scikit-learn "huggingface_hub>=0.24" numpy

base="https://raw.githubusercontent.com/iamtrask/abcGPT/${BRANCH}/experiments/nano-3/fineweb_cluster"
curl -fsSL "$base/embed_domains.py" -o embed_domains.py
curl -fsSL "$base/cluster_domains_balanced.py" -o cluster_domains_balanced.py
mkdir -p out

echo "=== STAGE 1: embed domains (N_DOCS=$N_DOCS cap=$CAP) ==="
python embed_domains.py --n-docs "$N_DOCS" --cap "$CAP" --device cuda --batch 512 --out-dir out

echo "=== STAGE 2: balanced cluster (K=$K micro=$MICRO) ==="
python cluster_domains_balanced.py --in-dir out --out-dir out --k "$K" --micro "$MICRO"

echo "=== UPLOAD outputs + log to HF fineweb_cluster/ ==="
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ.get("HF_REPO", "iamtrask/abcGPT-nano-3")
# small/essential first, big domains.npz last
for f, src in [("cluster_labels.json","out/cluster_labels.json"),
               ("samples.jsonl","out/samples.jsonl"),
               ("clusters.npz","out/clusters.npz"),
               ("embed.log","/tmp/embed.log"),
               ("domains.npz","out/domains.npz")]:
    if os.path.exists(src):
        try:
            api.upload_file(path_or_fileobj=src, path_in_repo=f"fineweb_cluster/{f}",
                            repo_id=repo, repo_type="model")
            print("uploaded", f, os.path.getsize(src))
        except Exception as e:
            print("UPLOAD FAILED", f, repr(e))
print("DONE_MARKER")
PY

echo "=== done $(date) — idling (launcher will terminate) ==="
sleep infinity
