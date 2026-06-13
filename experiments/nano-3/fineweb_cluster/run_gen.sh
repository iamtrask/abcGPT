#!/usr/bin/env bash
# Pod entrypoint: generate the 100-cohort (10 models x 10 sources) distillation dataset
# via vLLM, build char bins, upload to HF data_100distill/. All teachers Apache/MIT.
set -uo pipefail
exec > >(tee -a /tmp/gen.log) 2>&1
echo "=== distill dataset gen $(date) ==="
: "${HF_TOKEN:?need HF_TOKEN}"
HF_REPO="${HF_REPO:-iamtrask/abcGPT-nano-3}"
BRANCH="${BRANCH:-fineweb-cluster}"
CHARS="${CHARS:-1000000}"
MAXM="${MAXM:-10}"

cd /workspace
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pip install -q --upgrade pip
pip install -q vllm "huggingface_hub>=0.24" numpy

base="https://raw.githubusercontent.com/iamtrask/abcGPT/${BRANCH}/experiments/nano-3/fineweb_cluster"
curl -fsSL "$base/gen_distill.py" -o gen_distill.py

# background: push the gen log every 10 min so progress is visible
( while true; do sleep 600; python - <<'PY' 2>/dev/null
import os
from huggingface_hub import HfApi
try: HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj="/tmp/gen.log",
        path_in_repo="data_100distill/gen.log", repo_id=os.environ["HF_REPO"], repo_type="model")
except Exception: pass
PY
done ) &

echo "=== generating (chars/cohort=$CHARS, models=$MAXM) ==="
python gen_distill.py --chars-per-cohort "$CHARS" --max-models "$MAXM" --out-dir data_100distill || echo "GEN_FAILED"

python - <<'PY' 2>/dev/null
import os
from huggingface_hub import HfApi
try: HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj="/tmp/gen.log",
        path_in_repo="data_100distill/gen.log", repo_id=os.environ["HF_REPO"], repo_type="model")
except Exception: pass
PY
echo "=== done $(date) DONE_MARKER ==="
sleep infinity
