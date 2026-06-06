#!/usr/bin/env bash
# nano-1 Modal runner. One place, one script. --detach is always on.
#
# Why this exists: the default `modal run <script>` mode requires the local CLI
# to maintain a heartbeat to Modal's control plane. If your laptop loses DNS,
# sleeps, or drops WiFi, Modal kills the in-flight container as a safety
# measure. For runs >5 min that's a real liability. `--detach` decouples the
# container's lifetime from the local CLI — once it's launched it runs to
# completion on Modal regardless of what happens on your laptop.
#
# Every command in this script that invokes Modal uses --detach.

set -euo pipefail

# Try common modal install locations.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP_FILE="$SCRIPT_DIR/modal_nano1.py"
APP_ID_FILE="$SCRIPT_DIR/.last_app_id"
RESULTS_DIR="$REPO_ROOT/experiments/nano-1/results"

usage() {
  cat <<'EOF'
nano-1 Modal runner. Always uses --detach so local network blips can't kill
in-flight containers.

Usage:
  ./run.sh <command> [args...]

Commands:
  preflight              Verify modal CLI + auth + repo state before launching
  smoke                  Train one source (077_nba-play-by-play) — ~12 min pipeline test
  karpathy               Train Karpathy's tinyshakespeare baseline; validates val_loss ≈ 1.4697
  all [--only=A,B,C]     Fan out training across all 100 sources + Karpathy parity check (in parallel). ~2.5 hr, ~$15
                         Add --skip-karpathy if you only want the per-source baselines.
  status                 Show active Modal apps + state of last launched run
  logs [app-id]          Stream logs from a Modal app (defaults to last launched)
  fetch                  Pull all results from the Modal volume into local results/
  aggregate              Run aggregate.py on local results/ to produce summary.csv
  publish [--dry-run]    Upload models to HuggingFace Hub + stage metric files for git commit
                         (Requires `huggingface-cli login` once; models go to HF, metrics go to git)
  commit-metrics         Shortcut for `publish --metrics-only`: stage current metric files for git
                         without re-uploading models. Use this for incremental research-journey commits.

RunPod backend (cheaper alternative to Modal — see runpod/README.md):
  runpod-dry-run         Show RunPod plan + cost estimate, no launches
  runpod-all [args]      Fan out training across RunPod pods. Default: 5 pods, all 100 sources, 4090 spot.
                         Pass --n-boxes N or --sources A,B,C as needed.
  runpod-resume          Identify sources missing from local results/, train just those on RunPod —
                         one pod per source by default (maximum parallelism, same total cost).
                         (Use this after Modal stops at the credit cap.)

Typical workflow:
  ./run.sh preflight       # check setup
  ./run.sh karpathy        # validate pipeline reproduces published number (~12 min)
  ./run.sh all             # commit to full fan-out (~2.5 hr, ~$15)
  ./run.sh status          # check what's running
  ./run.sh fetch           # download results when done
  ./run.sh aggregate       # roll up into summary.csv
  ./run.sh publish         # release: upload models to HF + stage metrics for git commit

The last launched app's ID is saved to .last_app_id so `status` and `logs`
without args query the most recent run.
EOF
}

err() { echo "error: $*" >&2; exit 1; }
log() { echo ">>> $*"; }

require_modal() {
  command -v modal >/dev/null 2>&1 || err "modal CLI not found on PATH. Install: pip install modal (or uv tool install modal)"
}

save_app_id_from_output() {
  # Modal prints "View run at https://modal.com/apps/.../main/ap-XXXXXX"
  # Extract the ap-XXXXXX bit and save it.
  local out_file="$1"
  local app_id
  app_id="$(grep -oE 'ap-[A-Za-z0-9]+' "$out_file" | head -1 || true)"
  if [[ -n "$app_id" ]]; then
    echo "$app_id" > "$APP_ID_FILE"
    log "Saved app ID: $app_id (in .last_app_id)"
    log "Dashboard: https://modal.com/apps/andrew-75645/main/$app_id"
  fi
}

cmd_preflight() {
  require_modal
  log "modal version: $(modal --version 2>&1 | head -1)"
  log "Checking Modal authentication..."
  local profile
  profile="$(modal profile current 2>/dev/null || true)"
  [[ -n "$profile" ]] || err "Modal not authenticated. Run: modal token new"
  log "✅ Modal authenticated as workspace: $profile"
  log "Checking app file exists: $APP_FILE"
  [[ -f "$APP_FILE" ]] || err "Missing $APP_FILE"
  log "✅ App file present"
  log "Checking abcGPT git state..."
  cd "$REPO_ROOT"
  local commit
  commit="$(git rev-parse --short HEAD 2>/dev/null)" || err "Not a git repo at $REPO_ROOT"
  log "abcGPT @ $commit ($(git rev-parse --abbrev-ref HEAD))"
  local pinned
  pinned="$(grep -oE 'ABCGPT_COMMIT = "[a-f0-9]+"' "$APP_FILE" | head -1 | grep -oE '[a-f0-9]{7,}' || true)"
  if [[ -n "$pinned" ]]; then
    log "Image pins abcGPT @ $pinned (in modal_nano1.py)"
    if [[ "$commit" != "$pinned"* && "$pinned" != "$commit"* ]]; then
      log "⚠️  Local HEAD ($commit) differs from pinned image commit ($pinned)."
      log "   That's fine if you've already pushed the changes you want Modal to see,"
      log "   but bump ABCGPT_COMMIT in modal_nano1.py if you changed something the image needs."
    fi
  fi
  log "✅ Preflight passed"
}

run_modal() {
  # All Modal launches go through here so --detach can't be forgotten.
  # cd to repo root so relative paths in modal_nano1.py (e.g., glob for sources)
  # resolve correctly regardless of where the user invoked run.sh from.
  local entrypoint="$1"; shift
  local out_file
  out_file="$(mktemp -t nano1-modal-XXXXXX)"
  log "Launching: modal run --detach $APP_FILE::$entrypoint $*"
  log "(Output also captured to $out_file)"
  cd "$REPO_ROOT"
  modal run --detach "$APP_FILE::$entrypoint" "$@" 2>&1 | tee "$out_file"
  save_app_id_from_output "$out_file"
}

cmd_smoke() {
  require_modal
  run_modal smoke_test
}

cmd_karpathy() {
  require_modal
  run_modal karpathy_baseline
}

cmd_all() {
  require_modal
  run_modal main "$@"
}

cmd_status() {
  require_modal
  log "Active Modal apps:"
  modal app list 2>&1 | head -20
  if [[ -f "$APP_ID_FILE" ]]; then
    local app_id
    app_id="$(cat "$APP_ID_FILE")"
    echo ""
    log "Last launched app: $app_id"
    log "Dashboard: https://modal.com/apps/andrew-75645/main/$app_id"
  fi
}

cmd_logs() {
  require_modal
  local app_id="${1:-}"
  if [[ -z "$app_id" ]]; then
    [[ -f "$APP_ID_FILE" ]] || err "No app ID given and no .last_app_id file. Pass an app ID: ./run.sh logs ap-XXXX"
    app_id="$(cat "$APP_ID_FILE")"
    log "Using last launched app: $app_id"
  fi
  modal app logs "$app_id"
}

cmd_fetch() {
  require_modal
  log "Downloading results from Modal volume to $RESULTS_DIR ..."
  mkdir -p "$RESULTS_DIR"
  # download_results is a local_entrypoint that pulls and aggregates.
  modal run "$APP_FILE::download_results"
  log "Results in $RESULTS_DIR"
}

cmd_aggregate() {
  log "Running aggregate.py on $RESULTS_DIR ..."
  cd "$REPO_ROOT"
  python3 data/100_simple_voices/baselines/aggregate.py
  log "Summary written to $RESULTS_DIR/summary.csv"
}

cmd_publish() {
  log "Publishing models to HuggingFace Hub + staging metrics for git ..."
  cd "$REPO_ROOT"
  python3 experiments/nano-1/publish.py "$@"
}

cmd_commit_metrics() {
  log "Staging metric files for git commit (no HF upload) ..."
  cd "$REPO_ROOT"
  python3 experiments/nano-1/publish.py --metrics-only "$@"
}

# Auto-source RunPod credentials from ~/.runpod/env if present. Keeps tokens
# off the shell history and out of the repo. Mode-600 file recommended.
_load_runpod_env() {
  if [[ -f "$HOME/.runpod/env" ]]; then
    set -a  # auto-export all vars defined in the sourced file
    source "$HOME/.runpod/env"
    set +a
  fi
}

# All RunPod subcommands run via `uv run --with runpod` so the runpod SDK is
# available without polluting the system python. Falls back to plain python3
# for dry-run since that path doesn't import runpod at all.
_runpod_python() {
  if command -v uv >/dev/null 2>&1; then
    uv run --with runpod python "$@"
  else
    python3 "$@"
  fi
}

cmd_runpod_dry_run() {
  log "RunPod dry-run (no launches) ..."
  _load_runpod_env
  cd "$REPO_ROOT"
  # dry-run path doesn't import runpod, so plain python3 is fine and faster
  python3 experiments/nano-1/runpod/runpod_fanout.py --dry-run "$@"
}

cmd_runpod_all() {
  log "Launching RunPod fan-out (max 100 sources, 5 pods by default) ..."
  _load_runpod_env
  cd "$REPO_ROOT"
  _runpod_python experiments/nano-1/runpod/runpod_fanout.py "$@"
}

cmd_runpod_resume() {
  # Identify sources that don't yet have a best_val_loss.txt locally,
  # then launch a RunPod fan-out for just those.
  log "Identifying sources missing from local results/ ..."
  local missing
  missing="$(python3 - <<PYEOF
import glob, os
here = os.path.abspath("$REPO_ROOT")
sources = sorted(os.path.basename(f).replace("source_", "").replace(".txt", "")
                 for f in glob.glob(os.path.join(here, "data/100_simple_voices/sources/source_*.txt")))
results = os.path.join(here, "experiments/nano-1/results")
done = set()
for s in sources:
    if os.path.exists(os.path.join(results, s, "best_val_loss.txt")):
        done.add(s)
missing = [s for s in sources if s not in done]
print(",".join(missing))
PYEOF
)"
  if [[ -z "$missing" ]]; then
    log "Nothing missing — all 100 sources already have local results. Skipping."
    return 0
  fi
  local count
  count="$(echo "$missing" | tr ',' '\n' | wc -l | tr -d ' ')"
  log "Found $count source(s) missing locally."
  log "Launching RunPod fan-out for: ${missing:0:80}${missing:80:1:+...}"
  # Default to ONE BOX PER SOURCE for max parallelism. RunPod bills per-second
  # per-container, so 50 boxes × 25 min costs the same as 5 boxes × 250 min,
  # but finishes in ~25 min wall instead of ~4 hr. Override with --n-boxes if
  # GPU availability or concurrency policy requires fewer.
  _load_runpod_env
  cd "$REPO_ROOT"
  _runpod_python experiments/nano-1/runpod/runpod_fanout.py --sources "$missing" --n-boxes "$count" "$@"
}

case "${1:-help}" in
  preflight)  shift; cmd_preflight "$@" ;;
  smoke)      shift; cmd_smoke "$@" ;;
  karpathy)   shift; cmd_karpathy "$@" ;;
  all)        shift; cmd_all "$@" ;;
  status)     shift; cmd_status "$@" ;;
  logs)       shift; cmd_logs "$@" ;;
  fetch)      shift; cmd_fetch "$@" ;;
  aggregate)  shift; cmd_aggregate "$@" ;;
  publish)    shift; cmd_publish "$@" ;;
  commit-metrics) shift; cmd_commit_metrics "$@" ;;
  runpod-dry-run) shift; cmd_runpod_dry_run "$@" ;;
  runpod-all) shift; cmd_runpod_all "$@" ;;
  runpod-resume) shift; cmd_runpod_resume "$@" ;;
  help|-h|--help) usage ;;
  *) usage; exit 1 ;;
esac
