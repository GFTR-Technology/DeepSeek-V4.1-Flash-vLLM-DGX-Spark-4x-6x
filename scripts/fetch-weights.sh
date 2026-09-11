#!/usr/bin/env bash
# Download the checkpoint on the head, then make it visible to every worker.
# Run ON THE HEAD.
#
#   ./scripts/fetch-weights.sh              # download only
#   ./scripts/fetch-weights.sh nfs          # download + export over NFS (default recipe)
#   ./scripts/fetch-weights.sh rsync        # download + copy to every worker (needs 510 GB each)
#
# NFS is what the published recipe uses: 510 GB lives once, on the head's NVMe.
# The cost is 10-18 min of worker load time per boot instead of ~10.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$HERE/lib.sh"
require_cluster

MODE="${1:-download}"
EXPORT_DIR="$(dirname "$WEIGHTS")"

if [ ! -f "$WEIGHTS/config.json" ]; then
  echo "==> downloading $HF_REPO -> $WEIGHTS (about 510 GB, 48 shards)"
  mkdir -p "$WEIGHTS"
  export HF_HUB_ENABLE_HF_TRANSFER=0
  if [ -z "${HF_TOKEN:-}" ] && [ -f ~/.cache/huggingface/token ]; then
    HF_TOKEN="$(cat ~/.cache/huggingface/token)"; export HF_TOKEN
  fi
  hf download "$HF_REPO" --local-dir "$WEIGHTS" --max-workers 8
else
  echo "==> $WEIGHTS already populated, skipping download"
fi

have=$(ls "$WEIGHTS"/model-*-of-*.safetensors 2>/dev/null | wc -l)
want=$(ls "$WEIGHTS"/model-*-of-*.safetensors 2>/dev/null | head -1 \
       | sed -n 's/.*-of-0*\([0-9]\+\)\.safetensors$/\1/p')
echo "    ${have}/${want:-?} shards present"
[ -n "$want" ] && [ "$have" = "$want" ] || { echo "incomplete download"; exit 1; }

case "$MODE" in
  download) ;;
  nfs)
    echo "==> exporting $EXPORT_DIR read-only over NFS"
    subnet="$(echo "$HEAD_IP" | cut -d. -f1-3).0/24"
    line="$EXPORT_DIR ${subnet}(ro,sync,no_subtree_check)"
    grep -qF "$line" /etc/exports 2>/dev/null || echo "$line" | sudo tee -a /etc/exports >/dev/null
    sudo exportfs -ra
    # 8 nfsd threads queue badly when several workers read weights at once.
    echo 32 | sudo tee /proc/fs/nfsd/threads >/dev/null 2>&1 || true
    echo "==> mounting on the workers at $WORKER_WEIGHTS"
    wexport="$(dirname "$WORKER_WEIGHTS")"
    for w in $WORKER_IPS; do
      ssh_to "$w" "sudo mkdir -p $wexport && \
        (mountpoint -q $wexport || sudo mount -t nfs -o ro,vers=3 ${HEAD_IP}:${EXPORT_DIR} $wexport) && \
        ls $WORKER_WEIGHTS/config.json >/dev/null" \
        && echo "    ok $w" || { echo "    FAIL $w"; exit 1; }
      # Two of our workers had this only as a manual mount and lost it after a
      # watchdog reset. Persist it.
      ssh_to "$w" "grep -q '$wexport' /etc/fstab || \
        echo '${HEAD_IP}:${EXPORT_DIR} $wexport nfs ro,vers=3,_netdev 0 0' | sudo tee -a /etc/fstab >/dev/null"
    done
    ;;
  rsync)
    echo "==> rsync $WEIGHTS -> workers (each needs 510 GB free)"
    pids=(); for w in $WORKER_IPS; do
      ssh_to "$w" "mkdir -p $WORKER_WEIGHTS"
      rsync -aH --info=progress2 --partial \
        -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 -b ${HEAD_IP}" \
        "${WEIGHTS}/" "${SSH_USER}@${w}:${WORKER_WEIGHTS}/" \
        > "/tmp/dsv41-rsync-${w}.log" 2>&1 &
      pids+=($!)
    done
    fail=0; i=0
    for w in $WORKER_IPS; do
      if wait "${pids[$i]}"; then echo "    ok $w"
      else echo "    FAIL $w (see /tmp/dsv41-rsync-${w}.log)"; fail=1; fi
      i=$((i+1))
    done
    [ "$fail" = 0 ] || exit 1
    ;;
  *) echo "Usage: $0 {download|nfs|rsync}"; exit 1 ;;
esac
echo "done"
