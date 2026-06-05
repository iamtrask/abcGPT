#!/usr/bin/env bash
# Provision a GCP VM and run Check 1 on it.
#
# Check 1: run Karpathy's data/openwebtext/prepare.py twice on the same VM
# and verify train.bin/val.bin are byte-identical across the two runs.
# If they are, the canonical OWT prep pipeline is deterministic on our
# infrastructure and we can safely add source attribution on top.
#
# This script runs from your laptop. It:
#   1. Provisions a GCP VM (n2-standard-32 spot, 500 GB pd-ssd)
#   2. Copies vm_check_1.sh to the VM
#   3. Runs it remotely
#   4. Pulls back the results file
#   5. Reports pass/fail
# It does NOT tear down the VM automatically — you should delete it manually
# after reviewing results so you can debug if Check 1 fails.
#
# Expected: ~3-5 hours wall-clock, ~$3-5 cost.
#
# Prereqs:
#   - gcloud CLI installed and authenticated
#   - default project set: `gcloud config set project <PROJECT>`
#   - default region/zone set or override below
#
# Usage:
#   bash provision_and_run.sh

set -euo pipefail

INSTANCE="${INSTANCE:-owt-check-1}"
ZONE="${ZONE:-us-central1-a}"
MACHINE="${MACHINE:-n2-standard-32}"
DISK_GB="${DISK_GB:-500}"

PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
if [[ -z "$PROJECT" ]]; then
    echo "ERROR: no default GCP project set. Run: gcloud config set project <PROJECT>"
    exit 1
fi

echo "=== Provisioning $INSTANCE in $ZONE on $MACHINE (project: $PROJECT) ==="

gcloud compute instances create "$INSTANCE" \
    --zone="$ZONE" \
    --machine-type="$MACHINE" \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --boot-disk-size=20GB \
    --create-disk="name=${INSTANCE}-data,size=${DISK_GB}GB,type=pd-ssd,auto-delete=yes,device-name=data-disk" \
    --scopes=cloud-platform

echo "=== Waiting for VM SSH ready ==="
until gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="true" 2>/dev/null; do
    sleep 5
done

echo "=== Copying vm_check_1.sh ==="
gcloud compute scp \
    "$(dirname "$0")/vm_check_1.sh" \
    "${INSTANCE}":~/vm_check_1.sh \
    --zone="$ZONE"

echo "=== Running Check 1 on VM (this takes ~3-5 hours) ==="
echo "    Live log: gcloud compute ssh $INSTANCE --zone=$ZONE --command='tail -f /data/check_1_log.txt'"
gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="bash ~/vm_check_1.sh"

echo "=== Pulling results ==="
gcloud compute scp \
    "${INSTANCE}":/data/check_1_results.txt \
    "$(dirname "$0")/check_1_results.txt" \
    --zone="$ZONE"

echo
echo "=== RESULTS ==="
cat "$(dirname "$0")/check_1_results.txt"
echo
echo "VM still alive at $INSTANCE in $ZONE."
echo "After reviewing results, DELETE THE VM to stop billing:"
echo "  gcloud compute instances delete $INSTANCE --zone=$ZONE --quiet"
