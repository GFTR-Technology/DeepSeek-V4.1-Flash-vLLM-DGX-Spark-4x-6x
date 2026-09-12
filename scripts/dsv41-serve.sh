#!/usr/bin/env bash
# Wrapper. Preflight (clock check, cache drop) then dsv41-node-launch.sh.
#
#   ./scripts/dsv41-serve.sh                            # default: 300K lane
#   DSV41_LANE=1m   ./scripts/dsv41-serve.sh            # 1M context (eager)
#   DSV41_LANE=128k ./scripts/dsv41-serve.sh
#   DSV41_ENGRAM_LOCAL=1 ./scripts/dsv41-serve.sh       # node-local Engram rows
#   ./scripts/dsv41-serve.sh stop|status|logs|logs-all|dry-run
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$HERE/lib.sh"
require_cluster

LAUNCH="$HERE/dsv41-node-launch.sh"
IMAGE="${DSV41_IMAGE:-$IMAGE}"
NAME="${DSV41_NAME:-$NAME}"
PORT="${DSV41_PORT:-$PORT}"

case "${1:-start}" in
  stop)
    "$LAUNCH" --stop
    exit 0
    ;;
  status)
    echo "--- container on all nodes ---"
    for ip in "${NODES[@]}"; do
      echo -n "${ip}: "
      ssh_to "$ip" "docker ps --filter name=$NAME --format '{{.Status}}'" 2>/dev/null || echo unreachable
    done
    echo
    echo "--- /v1/models on $HEAD_IP:$PORT ---"
    curl -s --max-time 5 "http://${HEAD_IP}:${PORT}/v1/models" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "(API not ready yet)"
    exit 0
    ;;
  logs)
    docker logs -f "$NAME"
    exit 0
    ;;
  logs-all)
    # The head's traceback is usually a symptom: NCCL reports "remote process
    # exited" while the real error sits in one worker's log. Dump them all.
    n="${2:-40}"
    for ip in "${NODES[@]}"; do
      echo
      echo "############ $ip (last $n lines) ############"
      ssh_to "$ip" "docker logs --tail $n $NAME 2>&1" 2>/dev/null || echo "  unreachable / no container"
    done
    echo
    echo "############ first error on each node ############"
    for ip in "${NODES[@]}"; do
      echo -n "$ip: "
      ssh_to "$ip" "docker logs $NAME 2>&1 | grep -m1 -E 'FATAL|Error|error:|Traceback|SystemExit|Assertion'" 2>/dev/null \
        || echo "(none found)"
    done
    exit 0
    ;;
  dry-run)
    "$LAUNCH" --dry-run
    exit 0
    ;;
  start)
    ;;
  *)
    echo "Usage: $0 {start|stop|status|logs|logs-all [N]|dry-run}"
    exit 1
    ;;
esac

echo "[guard] checkpoint present and complete"
for ip in "${NODES[@]}"; do
  w="$WORKER_WEIGHTS"; [ "$ip" = "$HEAD_IP" ] && w="$WEIGHTS"
  if ! ssh_to "$ip" "test -f $w/config.json"; then
    echo "Error: missing $w/config.json on $ip"
    echo "       Download on the head (scripts/fetch-weights.sh), then export it over NFS"
    exit 1
  fi
  # A worker that lost its NFS mount after a watchdog reset shows an empty dir,
  # not an error. Count the shards rather than trusting config.json alone.
  have=$(ssh_to "$ip" "ls $w/model-*-of-*.safetensors 2>/dev/null | wc -l" || echo 0)
  want=$(ssh_to "$ip" "ls $w/model-*-of-*.safetensors 2>/dev/null | head -1 | sed -n 's/.*-of-0*\([0-9]\+\)\.safetensors$/\1/p'" || echo "")
  if [ -n "$want" ] && [ "$have" != "$want" ]; then
    echo "Error: $ip has $have/$want shards at $w (NFS mount dropped? check /etc/fstab)"
    exit 1
  fi
done

echo "[guard] image $IMAGE present"
for ip in "${NODES[@]}"; do
  if ! ssh_to "$ip" "docker image inspect $IMAGE > /dev/null 2>&1"; then
    echo "Error: image $IMAGE missing on $ip"
    echo "       Build on the head (scripts/build-image.sh), then scripts/copy-image.sh"
    exit 1
  fi
done

echo "[guard] patches + entrypoint present on every node"
PATCH_HOST="${DSV41_PATCHES:-$REPO_ROOT/patch}"
for ip in "${NODES[@]}"; do
  if ! ssh_to "$ip" "test -f $PATCH_HOST/mounts.txt && test -f $PATCH_HOST/engram.py && test -f $REPO_ROOT/scripts/dsv41-container-entrypoint.sh"; then
    echo "Error: clone this repo to the same path on $ip (or set DSV41_PATCHES / DSV41_ENTRYPOINT)"
    echo "       Default path: $REPO_ROOT"
    exit 1
  fi
done

# patch/ is bind-mounted from EACH node's own clone, so a half-synced fleet runs
# different code per rank. That fails as "remote process exited" during NCCL
# init on the head, with the real error buried in one worker's log. Compare the
# checksums here instead.
echo "[guard] patch set identical on every node"
sum_cmd="cat \$(ls $PATCH_HOST/*.py $PATCH_HOST/mounts.txt $PATCH_HOST/dsv41_tp_pad/*.py 2>/dev/null | sort) | md5sum | cut -c1-12"
head_sum="$(ssh_to "$HEAD_IP" "$sum_cmd" 2>/dev/null || bash -c "$sum_cmd")"
mismatch=""
for ip in "${NODES[@]}"; do
  s="$(ssh_to "$ip" "$sum_cmd" 2>/dev/null)"
  [ "$s" = "$head_sum" ] || mismatch="$mismatch $ip($s)"
done
if [ -n "$mismatch" ]; then
  echo "Error: patch set differs from the head ($head_sum) on:$mismatch"
  echo "       Every node bind-mounts its OWN copy, so they must match."
  echo "       Sync the repo to all ${NNODES} nodes and re-run."
  exit 1
fi
echo "        md5 $head_sum on all ${NNODES} nodes"

# A TP that does not divide the attention heads / output groups / MoE
# intermediate needs the padding shim bind-mounted on every rank, not just the
# head. Catch a half-deployed clone here rather than 12 minutes into a load.
TP="$NNODES"
if python3 "$PATCH_HOST/dsv41_tp_pad/dsv41_tp_pad.py" --tp "$TP" --model "$WEIGHTS" --shell 2>/dev/null \
     | grep -q '^DSV41_PAD_ACTIVE=1'; then
  echo "[guard] TP=$TP needs the head/dim padding shim"
  for ip in "${NODES[@]}"; do
    if ! ssh_to "$ip" "test -f $PATCH_HOST/dsv41_tp_pad/dsv41_tp_pad.py && test -f $PATCH_HOST/dsv41_tp_pad/sitecustomize.py && test -f $PATCH_HOST/dsv41_tp_pad/make_overlay.py"; then
      echo "Error: $PATCH_HOST/dsv41_tp_pad/ incomplete on $ip"
      echo "       Pull this repo on every node, then re-run."
      exit 1
    fi
  done
  echo "       run ./scripts/dsv41-tp-probe.sh first if this image is new to you"
fi

# Expert counts shard by COUNT and cannot be padded inertly, so the plan can only
# report them. Refuse here rather than let it assert four minutes into a weight
# load: _init_fused_moe_experts runs after the checkpoint is read.
# `|| true`: under `set -e` a failing command substitution kills the script, and
# an unreadable config must not take the launch down silently.
advice="$(python3 "$PATCH_HOST/dsv41_tp_pad/dsv41_tp_pad.py" --tp "$TP" --model "$WEIGHTS" --shell 2>/dev/null \
          | sed -n "s/^DSV41_PAD_EXPERT_ADVICE=//p" | sed "s/^'//;s/'\$//" || true)"
if [ -n "$advice" ] && [ "${DSV41_FORCE_EXPERTS:-0}" != 1 ]; then
  echo "[guard] $advice"
  echo
  echo "        Not launching: this asserts after the weights are loaded, which"
  echo "        costs about four minutes each attempt. Pick one:"
  echo "          DSV41_SPEC=none ./scripts/dsv41-serve.sh        # drop DSpark, boots today"
  echo "          DSV41_EXTRA='--enable-eplb --eplb-config {\"num_redundant_experts\":N}' \\"
  echo "            ./scripts/dsv41-serve.sh   # the config alone is rejected without --enable-eplb"
  echo "          DSV41_FORCE_EXPERTS=1 ./scripts/dsv41-serve.sh  # try anyway"
  echo "        Check the flag name for your build first:"
  echo "          docker run --rm --entrypoint vllm $IMAGE serve --help | grep -iE 'redundant|eplb'"
  exit 1
fi

# Node-local Engram rows are on by default, and "on" has to mean the copies
# actually exist — mounting a directory nobody built would be a no-op. So build
# what is missing here. engram-local.sh compares each node's recorded row ranges
# against the ranges the current node count implies, so this is a cheap no-op
# once the copies are current: a few ssh round trips, no I/O.
if [ "${DSV41_ENGRAM_LOCAL:-$ENGRAM_LOCAL}" = 1 ] && [ "${DSV41_ENGRAM_DISK:-$ENGRAM_DISK}" = 1 ]; then
  if [ "${DSV41_ENGRAM_AUTOBUILD:-1}" = 1 ]; then
    echo "[preflight] node-local Engram rows"
    echo "            (first run on a fresh fleet writes a ~48 GB sparse copy per"
    echo "             worker and takes a few minutes; after that it is a no-op."
    echo "             DSV41_ENGRAM_AUTOBUILD=0 to skip, DSV41_ENGRAM_LOCAL=0 to"
    echo "             turn the feature off entirely.)"
    if ! "$HERE/engram-local.sh"; then
      echo "            Some ranks have no local copy. They will read their Engram"
      echo "            rows over NFS instead — slower, not wrong. Continuing."
    fi
  else
    missing=""
    for ((r = 1; r < NNODES; r++)); do
      ssh_to "${NODES[$r]}" "test -f ${DSV41_ENGRAM_LOCAL_DIR:-$ENGRAM_LOCAL_DIR}/engram-local.json" \
        || missing="$missing ${NODES[$r]}"
    done
    if [ -n "$missing" ]; then
      echo "[guard] node-local Engram rows missing on:$missing"
      echo "        Autobuild is off (DSV41_ENGRAM_AUTOBUILD=0). Those ranks will"
      echo "        read over NFS. Build them with: ./scripts/engram-local.sh"
    else
      echo "[guard] node-local Engram rows present on every worker"
    fi
  fi
fi

# A GB10 can latch below 1 GHz with no visible cause and nvidia-smi will not show
# it. Every TP step then waits for the slowest rank. Only a cold power cycle
# clears it — a reboot does not. docs/gpu-clock-latch.md
if [ "${DSV41_SKIP_CLOCK_CHECK:-0}" != 1 ]; then
  echo "[preflight] GPU clock-health check (5s burn) ..."
  clock_bad=0
  for ip in "${NODES[@]}"; do
    mhz=$(ssh_to "$ip" "docker run --rm --gpus all --entrypoint python3 $IMAGE -c '
import torch,time,subprocess
a=torch.randn(8192,8192,device=\"cuda\",dtype=torch.bfloat16)
t=time.time()
while time.time()-t<5:(a@a).sum().item()
print(subprocess.run([\"nvidia-smi\",\"--query-gpu=clocks.current.sm\",\"--format=csv,noheader,nounits\"],capture_output=True,text=True).stdout.split()[0])
' 2>/dev/null" 2>/dev/null || true)
    if [ -z "$mhz" ]; then echo "  $ip: clock probe failed (skipped)"
    elif [ "$mhz" -lt 1500 ]; then echo "  $ip: WEDGED — SM ${mhz} MHz"; clock_bad=1
    else echo "  $ip: SM ${mhz} MHz — healthy"; fi
  done
  if [ "$clock_bad" = 1 ]; then
    echo "A GPU is stuck at low clock. Cold power cycle that node, or DSV41_SKIP_CLOCK_CHECK=1."
    exit 1
  fi
fi

"$LAUNCH"

echo
echo "Cold boot is ~12-20 min (10 min on the head, 10-18 over NFS on the workers,"
echo "plus a second pass over all 48 shards for the DSpark draft layers)."
echo "  Poll:    curl -s http://${HEAD_IP}:${PORT}/v1/models"
echo "  Status:  $0 status"
echo "  Logs:    $0 logs        (head)   |   $0 logs-all   (every node)"
echo "  Stop:    $0 stop"
