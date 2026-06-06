#!/usr/bin/env bash
# nano-1: train one baseline char-level model per source in 100_simple_voices.
#
# Each run uses Karpathy's train_shakespeare_char config byte-for-byte
# (./config/train_shakespeare_char.py at the abcGPT root) with only
# `dataset` and `out_dir` overridden via CLI flags. The only thing varying
# between baselines is the input text. This gives directly-comparable
# val_loss numbers across all sources.
#
# Expected runtime per source on a single GPU:
#   T4:      ~12-15 min
#   3090:    ~4-6 min
#   4090:    ~3-5 min
#   A100:    ~3 min
#
# Total cost for all 100:
#   RTX 4090 spot (RunPod, ~$0.30/hr) × ~7 hr = ~$2-3
#   A100 spot   (GCP, ~$1.10/hr) × ~5 hr     = ~$5-6
#
# Prereqs:
#   - run from abcGPT repo root (`cd /path/to/abcGPT && bash data/100_simple_voices/baselines/run_all.sh`)
#   - torch, numpy, tiktoken installed (nanoGPT's requirements)
#   - CUDA-capable GPU available
#   - source files present at data/100_simple_voices/sources/source_<NNN>_<name>.txt
#
# Usage:
#   bash data/100_simple_voices/baselines/run_all.sh             # all 100 sources
#   bash data/100_simple_voices/baselines/run_all.sh top5        # only top 5 (smoke test)
#   bash data/100_simple_voices/baselines/run_all.sh 077_nba-play-by-play 076_retrosheet-baseball
#                                                                # specific sources
#
# Outputs (gitignored):
#   data/100_simple_voices/baselines/data/<src>/{train,val}.bin, meta.pkl
#   data/100_simple_voices/baselines/out/<src>/{ckpt.pt, training.log}
#
# Outputs (committed):
#   experiments/nano-1/results/<src>/{val_loss_curve.csv, best_val_loss.txt, meta.json}
#   experiments/nano-1/results/summary.csv

set -euo pipefail

# Always run from the abcGPT repo root so paths like data/<dataset> resolve
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
echo "running from $REPO_ROOT"

BASELINES_DIR="data/100_simple_voices/baselines"
SOURCES_DIR="data/100_simple_voices/sources"
RESULTS_DIR="experiments/nano-1/results"

# Build the source list:
#   - no args:    all 100 source_*.txt files in SOURCES_DIR
#   - "top5":     hardcoded top 5 by demo_score (smoke test)
#   - other args: treat each arg as a source name (without source_ prefix or .txt suffix)
if [[ $# -eq 0 ]]; then
    SOURCES=()
    for f in "$SOURCES_DIR"/source_*.txt; do
        SOURCES+=("$(basename "$f" .txt | sed 's/^source_//')")
    done
elif [[ "$1" == "top5" ]]; then
    SOURCES=(
        "077_nba-play-by-play"
        "076_retrosheet-baseball"
        "093_pubmed-abs"
        "021_fannie-farmer"
        "097_cia-world-factbook"
    )
else
    SOURCES=("$@")
fi

echo "will train ${#SOURCES[@]} baseline(s)"
GRAND_START=$(date +%s)

for i in "${!SOURCES[@]}"; do
    src="${SOURCES[$i]}"
    echo
    echo "========================================================================"
    echo "=== [$((i+1))/${#SOURCES[@]}] $src"
    echo "========================================================================"

    SOURCE_FILE="$SOURCES_DIR/source_${src}.txt"
    DATA_OUT="$BASELINES_DIR/data/$src"
    TRAIN_OUT="$BASELINES_DIR/out/$src"
    RESULT_OUT="$RESULTS_DIR/$src"

    if [[ ! -f "$SOURCE_FILE" ]]; then
        echo "  WARNING: missing source file $SOURCE_FILE, skipping"
        continue
    fi

    # Idempotency: skip if results already exist
    if [[ -f "$RESULT_OUT/best_val_loss.txt" ]]; then
        echo "  already have results at $RESULT_OUT — skipping (delete to re-run)"
        continue
    fi

    # 1. Char-level prep
    echo "--- prep ---"
    python "$BASELINES_DIR/prepare_char.py" \
        --source "$SOURCE_FILE" \
        --out "$DATA_OUT"

    # 2. Train. nanoGPT's train.py loads config/train_shakespeare_char.py then
    # accepts CLI overrides via configurator.py. We override only dataset+out_dir.
    # `dataset` is interpreted by train.py as data/<dataset>, so we pass a
    # relative path under data/.
    echo "--- train ---"
    mkdir -p "$TRAIN_OUT"
    START_TIME=$(date +%s)
    python train.py config/train_shakespeare_char.py \
        --dataset="100_simple_voices/baselines/data/$src" \
        --out_dir="$TRAIN_OUT" \
        --wandb_log=False \
        --always_save_checkpoint=False \
        2>&1 | tee "$TRAIN_OUT/training.log"
    END_TIME=$(date +%s)
    WALL_S=$((END_TIME - START_TIME))

    # 3. Extract result
    echo "--- extract ---"
    python "$BASELINES_DIR/extract_result.py" \
        --out-dir "$TRAIN_OUT" \
        --data-dir "$DATA_OUT" \
        --result-dir "$RESULT_OUT" \
        --source-name "$src" \
        --wall-clock-s "$WALL_S"
done

# Aggregate
echo
echo "========================================================================"
echo "=== aggregate"
echo "========================================================================"
python "$BASELINES_DIR/aggregate.py"

GRAND_END=$(date +%s)
GRAND_S=$((GRAND_END - GRAND_START))
echo
echo "all baselines complete in $((GRAND_S / 60)) min."
echo "summary: $RESULTS_DIR/summary.csv"
