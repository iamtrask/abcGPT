#!/usr/bin/env bash
# experiments/nano-2/run.sh — local dispatcher for nano-2 sweep work.
#
# Subcommands:
#   smoke                    Validate train.py works locally with --smoke flag (~30s on CPU)
#   sweep [--only=A,B,C]     Run the 13-variant sweep on RunPod (parallel)
#   sweep-dry                Show the sweep plan + cost estimate, no launches
#   fetch                    Pull all results from HF Hub to local
#   status                   Active RunPod pods (nano-2 ones)

set -euo pipefail
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$REPO_ROOT/experiments/nano-2/results"
HF_REPO="iamtrask/abcGPT-nano-2"

usage() {
  cat <<'EOF'
nano-2 sweep dispatcher. Most commands need ~/.runpod/env populated with
RUNPOD_API_KEY and HF_TOKEN (same file used by nano-1).

Usage:
  ./run.sh <command> [args...]

Commands:
  smoke                    Validate train.py locally (smoke config, ~30s)
  sweep                    Run the 13-variant sweep on RunPod (~30 min, ~$4)
  sweep-dry                Show plan, no launches
  sweep --only=A,B,C       Run only specific variants from the default sweep
  fetch                    Pull all variant results from HF Hub
  status                   Show active RunPod pods (nano-2 ones)
  help                     This message
EOF
}

_load_env() {
  if [[ -f "$HOME/.runpod/env" ]]; then
    set -a
    source "$HOME/.runpod/env"
    set +a
  fi
}

_runpod_python() {
  if command -v uv >/dev/null 2>&1; then
    uv run --with runpod --with huggingface_hub python "$@"
  else
    python3 "$@"
  fi
}

cmd_smoke() {
  echo ">>> Running local smoke test (fixed-mn variant) ..."
  cd "$REPO_ROOT"
  if command -v uv >/dev/null 2>&1; then
    uv run --with torch --with numpy python3 experiments/nano-2/train.py \
        --variant fixed-mn --variant-name smoke-fixed --smoke
  else
    python3 experiments/nano-2/train.py --variant fixed-mn --variant-name smoke-fixed --smoke
  fi
}

cmd_sweep() {
  _load_env
  cd "$REPO_ROOT"
  _runpod_python experiments/nano-2/runpod/runpod_fanout.py "$@"
}

cmd_sweep_dry() {
  cd "$REPO_ROOT"
  python3 experiments/nano-2/runpod/runpod_fanout.py --dry-run "$@"
}

cmd_fetch() {
  _load_env
  echo ">>> Pulling all nano-2 results from HF Hub ..."
  cd "$REPO_ROOT"
  if command -v uv >/dev/null 2>&1; then
    uv run --with huggingface_hub python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$HF_REPO', repo_type='model', local_dir='$RESULTS_DIR')
print('pulled to: $RESULTS_DIR')
"
  else
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$HF_REPO', repo_type='model', local_dir='$RESULTS_DIR')
"
  fi
}

cmd_status() {
  _load_env
  _runpod_python -c "
import os, runpod
runpod.api_key = os.environ['RUNPOD_API_KEY']
pods = runpod.get_pods()
nano2 = [p for p in pods if (p.get('name') or '').startswith('nano-2-')]
print(f'nano-2 active pods: {len(nano2)} of {len(pods)} total')
for p in nano2:
    runtime = p.get('runtime') or {}
    uptime = runtime.get('uptimeInSeconds', 0)
    print(f'  {p[\"id\"]}: {p.get(\"name\", \"?\")} uptime={uptime}s')
"
}

case "${1:-help}" in
  smoke)      shift; cmd_smoke "$@" ;;
  sweep)      shift; cmd_sweep "$@" ;;
  sweep-dry)  shift; cmd_sweep_dry "$@" ;;
  fetch)      shift; cmd_fetch "$@" ;;
  status)     shift; cmd_status "$@" ;;
  help|-h|--help) usage ;;
  *) usage; exit 1 ;;
esac
