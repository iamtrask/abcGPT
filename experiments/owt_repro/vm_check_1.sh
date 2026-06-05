#!/usr/bin/env bash
# Runs on the GCP VM. Drives Check 1: run Karpathy's prepare.py twice on the
# same data, verify byte-identical train.bin / val.bin between runs.
#
# Invoked by provision_and_run.sh from your laptop, but can also be run
# manually if you SSH in.
#
# Outputs:
#   /data/check_1_log.txt        — full stdout/stderr of both prep runs
#   /data/check_1_results.txt    — SHA256 hashes + PASS/FAIL verdict
#   /data/run1/{train,val}.bin   — first run's binaries
#   /data/run2/{train,val}.bin   — second run's binaries

set -euo pipefail

LOG=/data/check_1_log.txt
RES=/data/check_1_results.txt

# Ensure /data exists before any log() call writes to it (tee would otherwise
# fail under pipefail because the parent directory doesn't exist yet).
sudo mkdir -p /data
sudo chown -R "$USER" /data 2>/dev/null || true

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- 1. setup
log "=== STAGE 1: SETUP ==="

# Mount the data disk (provisioned with device-name=data-disk)
DEV=$(readlink -f /dev/disk/by-id/google-data-disk)
sudo mkdir -p /data
if ! mountpoint -q /data; then
    if ! sudo blkid "$DEV" >/dev/null 2>&1; then
        log "formatting $DEV"
        sudo mkfs.ext4 -F "$DEV"
    fi
    sudo mount "$DEV" /data
fi
sudo chown -R "$USER" /data
log "data disk: $DEV mounted at /data ($(df -h /data | tail -1 | awk '{print $4}') free)"

# Move HF cache to /data so it survives and fits
export HF_DATASETS_CACHE=/data/hf_cache
mkdir -p "$HF_DATASETS_CACHE"
log "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

# Install system deps
sudo apt-get update -qq
sudo apt-get install -y -qq git curl jq python3.11 python3.11-venv

# Install uv
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
log "uv: $(uv --version)"

# Pinned versions for reproducibility.
# huggingface_hub<1 is required: datasets==2.21.0 declares huggingface_hub<1.0,
# but uv's resolver will happily install hf_hub 1.x unless we pin it. The 1.x
# series rejects bare-name dataset ids ("openwebtext") via a strict URI parser
# (HfUriError: "Repository id must be 'namespace/name'"), which breaks
# nanoGPT's prepare.py.
PIN_DEPS="--with tiktoken==0.7.0 --with datasets==2.21.0 --with huggingface_hub==0.26.5 --with numpy>=1.26,<2 --with tqdm"

# ---------------------------------------------------------------- 2. clone nanoGPT
log "=== STAGE 2: CLONE NANOGPT ==="
cd /data
if [[ ! -d nanoGPT ]]; then
    git clone --depth 1 https://github.com/karpathy/nanoGPT.git
fi
cd nanoGPT
log "nanoGPT HEAD: $(git rev-parse HEAD)"

# Patch prepare.py to use the explicit namespaced dataset id. The original
# script's bare "openwebtext" relied on an HF-side alias that newer
# huggingface_hub versions no longer resolve. The canonical home is
# Skylion007/openwebtext (see snapshot ref logged below).
if grep -q 'load_dataset("openwebtext"' data/openwebtext/prepare.py; then
    sed -i 's|load_dataset("openwebtext"|load_dataset("Skylion007/openwebtext"|' data/openwebtext/prepare.py
    log "patched prepare.py: load_dataset(\"openwebtext\") -> load_dataset(\"Skylion007/openwebtext\")"
fi

# Record the current Skylion007/openwebtext revision (for the audit trail)
REV=$(curl -s https://huggingface.co/api/datasets/Skylion007/openwebtext \
      | jq -r '.sha // .lastModified // "unknown"')
log "Skylion007/openwebtext snapshot reference: $REV"

# ---------------------------------------------------------------- 3. run 1
log "=== STAGE 3: RUN 1 (full prepare.py) ==="
RUN1_START=$(date +%s)
uv run $PIN_DEPS python data/openwebtext/prepare.py 2>&1 | tee -a "$LOG"
RUN1_END=$(date +%s)
log "run 1 took $((RUN1_END - RUN1_START)) seconds"

mkdir -p /data/run1
mv data/openwebtext/train.bin /data/run1/
mv data/openwebtext/val.bin /data/run1/
log "run 1 outputs: $(ls -la /data/run1)"

# ---------------------------------------------------------------- 4. run 2
log "=== STAGE 4: RUN 2 (full prepare.py, should reuse HF cache) ==="
RUN2_START=$(date +%s)
uv run $PIN_DEPS python data/openwebtext/prepare.py 2>&1 | tee -a "$LOG"
RUN2_END=$(date +%s)
log "run 2 took $((RUN2_END - RUN2_START)) seconds"

mkdir -p /data/run2
mv data/openwebtext/train.bin /data/run2/
mv data/openwebtext/val.bin /data/run2/

# ---------------------------------------------------------------- 5. verdict
log "=== STAGE 5: VERIFY BYTE-IDENTITY ==="
: > "$RES"
echo "Check 1 results: $(date -Iseconds)" >> "$RES"
echo "instance: $(hostname), machine type detected via metadata:" >> "$RES"
curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/machine-type 2>/dev/null >> "$RES"
echo >> "$RES"

echo "" >> "$RES"
echo "Dataset reference: Skylion007/openwebtext snapshot ref $REV" >> "$RES"
echo "" >> "$RES"

PASS=true
for f in train.bin val.bin; do
    h1=$(sha256sum /data/run1/$f | awk '{print $1}')
    h2=$(sha256sum /data/run2/$f | awk '{print $1}')
    size=$(stat -c%s /data/run1/$f)
    printf "%-12s  size=%-13s  run1=%s  run2=%s  " "$f" "$size" "${h1:0:16}..." "${h2:0:16}..." | tee -a "$RES"
    if [[ "$h1" == "$h2" ]]; then
        echo "PASS" | tee -a "$RES"
    else
        echo "FAIL" | tee -a "$RES"
        echo "    full run1 sha256: $h1" | tee -a "$RES"
        echo "    full run2 sha256: $h2" | tee -a "$RES"
        PASS=false
    fi
done

echo "" >> "$RES"
echo "Versions:" >> "$RES"
uv run $PIN_DEPS python -c "
import tiktoken, datasets, numpy, sys, importlib.metadata
print('  tiktoken:        ' + (importlib.metadata.version('tiktoken')))
print('  datasets:        ' + datasets.__version__)
print('  huggingface_hub: ' + importlib.metadata.version('huggingface_hub'))
print('  numpy:           ' + numpy.__version__)
print('  python:          ' + sys.version.split()[0])
" 2>/dev/null | tee -a "$RES"

echo "" >> "$RES"
if $PASS; then
    echo "CHECK 1: PASS — canonical OWT prep is byte-deterministic on this VM" >> "$RES"
    echo "Next: Step 2 — build URL→cohort map from Gokaslan raw" >> "$RES"
else
    echo "CHECK 1: FAIL — debug with the byte-divergence diagnostic before proceeding" >> "$RES"
    echo "  python3 -c 'see byte-diff diagnostic in conversation history'" >> "$RES"
fi

echo
cat "$RES"
log "=== DONE ==="
