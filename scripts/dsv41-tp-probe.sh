#!/usr/bin/env bash
# Read-only probe of the vLLM build inside the image, for a target TP.
#
# The dsv41-feat branch moves between releases and this repo bind-mounts seven
# patched files over it, so the padding shim's attachment points cannot be
# verified from a dev box. Run this once before the first TP!=4 boot and read the
# report — it tells you whether the weight iterators the shim wraps still exist,
# whether the mounted engram.py can survive a zero-column rank, and what the plan
# would actually pad.
#
#   ./scripts/dsv41-tp-probe.sh                  # head node, TP = node count
#   DSV41_PROBE_TP=6 ./scripts/dsv41-tp-probe.sh
#   DSV41_PROBE_ALL=1 ./scripts/dsv41-tp-probe.sh   # every node, not just head
#
# Starts a throwaway container with no GPU claim, so it is safe to run while an
# engine is live.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$HERE/lib.sh"
require_cluster

IMAGE="${DSV41_IMAGE:-$IMAGE}"
PATCH_HOST="${DSV41_PATCHES:-$REPO_ROOT/patch}"
TP="${DSV41_PROBE_TP:-$NNODES}"
SITE=/usr/local/lib/python3.12/dist-packages/vllm

if [ ! -f "$PATCH_HOST/dsv41_tp_pad/probe.py" ]; then
  echo "Error: $PATCH_HOST/dsv41_tp_pad/probe.py not found" >&2
  exit 1
fi

# Mount the patched files too: the probe checks the engram.py that actually
# serves, not the one baked into the image.
PATCH_MOUNTS=()
while read -r f rel; do
  [ -z "${f:-}" ] && continue
  case "$f" in \#*) continue ;; esac
  [ -f "$PATCH_HOST/$f" ] && PATCH_MOUNTS+=(-v "$PATCH_HOST/$f:$SITE/$rel:ro")
done < "$PATCH_HOST/mounts.txt"

probe_one() {
  local ip="$1" weights="$2"
  echo
  echo "############ $ip ############"
  ssh_to "$ip" "docker run --rm \
      -v $PATCH_HOST/dsv41_tp_pad:/opt/dsv41:ro \
      -v $weights:/model:ro \
      ${PATCH_MOUNTS[*]} \
      --entrypoint python3 $IMAGE /opt/dsv41/probe.py --tp $TP --model /model" \
    || echo "  probe failed on $ip"
}

if [ "${DSV41_PROBE_ALL:-0}" = 1 ]; then
  for i in "${!NODES[@]}"; do probe_one "${NODES[$i]}" "$(weights_for "$i")"; done
else
  probe_one "$HEAD_IP" "$WEIGHTS"
fi

echo
echo "Local plan for TP=$TP (no container needed):"
python3 "$PATCH_HOST/dsv41_tp_pad/dsv41_tp_pad.py" --tp "$TP" --model "$WEIGHTS"
