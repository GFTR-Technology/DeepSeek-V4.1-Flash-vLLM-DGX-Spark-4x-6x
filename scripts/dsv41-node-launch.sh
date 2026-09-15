#!/usr/bin/env bash
# Launch DeepSeek-V4.1-Flash TP=N across the nodes in cluster.env.
# Workers first (headless), then the head. Does not pull weights.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$HERE/lib.sh"
require_cluster

IMAGE="${DSV41_IMAGE:-$IMAGE}"
NAME="${DSV41_NAME:-$NAME}"
PORT="${DSV41_PORT:-$PORT}"
MASTER_PORT="${DSV41_MASTER_PORT:-$MASTER_PORT}"
CACHE_HOST="${DSV41_CACHE:-$CACHE}"
ENTRYPOINT_HOST="${DSV41_ENTRYPOINT:-$REPO_ROOT/scripts/dsv41-container-entrypoint.sh}"
PATCH_HOST="${DSV41_PATCHES:-$REPO_ROOT/patch}"
# cluster.env is the topology and is authoritative; DSV41_* overrides it per run.
ENGRAM_DISK="${DSV41_ENGRAM_DISK:-$ENGRAM_DISK}"
ENGRAM_LOCAL="${DSV41_ENGRAM_LOCAL:-$ENGRAM_LOCAL}"
ENGRAM_LOCAL_DIR="${DSV41_ENGRAM_LOCAL_DIR:-$ENGRAM_LOCAL_DIR}"
SITE=/usr/local/lib/python3.12/dist-packages/vllm

# Lanes. One engine, one max-model-len; restart to switch.
#   300k is the serving recipe (boot 10): CUDA graphs + DSpark k=5, KV pool
#   ~1.07M tokens at gmu 0.80. 1m is the proof boot (boot 7) and needs eager —
#   the indexer prefill buffer grows to ~5.2 GiB at 1M and only eager leaves room.
LANE="${DSV41_LANE:-300k}"
case "$LANE" in
  128k|128K) _LEN=131072;  _SEQS=8; _EAGER=0 ;;
  300k|300K) _LEN=300000;  _SEQS=8; _EAGER=0 ;;
  1m|1M|1000k|1000K) _LEN=1048576; _SEQS=8; _EAGER=1 ;;
  *) printf 'unknown DSV41_LANE=%s (use 128k|300k|1m)\n' "$LANE" >&2; exit 2 ;;
esac
MAX_MODEL_LEN="${DSV41_MAXLEN:-$_LEN}"
MAX_NUM_SEQS="${DSV41_SEQS:-$_SEQS}"
MAX_BATCHED="${DSV41_BATCHED:-8192}"
EAGER="${DSV41_EAGER:-$_EAGER}"
GMU="${DSV41_GMU:-$GMU}"
SERVED_NAME="${DSV41_SERVED_NAME:-deepseek-v4.1-flash}"
CUDAGRAPH_MODE="${DSV41_CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
CG_SIZES="${DSV41_CG_SIZES:-}"
# DSpark speculative decode. k=5 is the recipe: eager decode on this model is
# host-bound at ~200 ms/step, and DSpark + graphs is what makes it serve.
SPEC="${DSV41_SPEC:-dspark}"
SPEC_K="${DSV41_SPEC_K:-5}"
# Adaptive verification forces variable-length FULL decode graphs = padded rows
# on SM120 sparse MLA (FlashInfer #5015, open). Opt-in only.
SPEC_ADAPT="${DSV41_SPEC_ADAPT:-false}"
# Vision + tool calling, both on in the serving config.
TEXT_ONLY="${DSV41_TEXT_ONLY:-0}"
PARSERS="${DSV41_PARSERS:-1}"
THINKING="${DSV41_THINKING:-false}"
MIN_AVAIL_GB="${DSV41_MIN_AVAIL_GB:-$MIN_AVAIL_GB}"
TP="${#NODES[@]}"

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# --- head/dim padding for a TP that does not divide the model ----------------
# The checkpoint's attention heads, output groups and MoE intermediate are sized
# for TP=4. At TP=6 none of them divide and vLLM asserts in divide() before the
# first forward. The shim pads them as the weights stream in; the entrypoint
# hands vLLM a symlink overlay whose config.json carries the padded dims.
#
# Engram is deliberately NOT padded. Its tables are split by hash column and the
# forward already all-gathers then slices back to n_hash_cols, so a rank past the
# last column simply owns nothing. See patch/dsv41_tp_pad/README.md.
PAD_SHIM_DIR="$PATCH_HOST/dsv41_tp_pad"
MODEL_DIR=/model
PAD_ACTIVE=0
if [ "$TP" -gt 1 ]; then
  command -v python3 >/dev/null 2>&1 \
    || die "need python3 on the launching host to compute the padding plan"
  [ -f "$PAD_SHIM_DIR/dsv41_tp_pad.py" ] \
    || die "missing $PAD_SHIM_DIR/dsv41_tp_pad.py"
  # One implementation of the plan, shared by this script and the container.
  # It reads the real dimensions out of the checkpoint's config.json, so it
  # cannot drift from the weights the way hard-coded constants would.
  if PAD_EVAL="$(python3 "$PAD_SHIM_DIR/dsv41_tp_pad.py" --tp "$TP" --model "$WEIGHTS" --spec "$SPEC" --shell 2>&1)"; then
    # stdout is `DSV41_*=...` assignments, but the planner also warns on stderr
    # and both were merged above so a failure keeps its message. Only eval the
    # assignments; a bare warning line would otherwise be run as a command.
    eval "$(printf '%s\n' "$PAD_EVAL" | grep '^DSV41_')"
    printf '%s\n' "$PAD_EVAL" | grep -v '^DSV41_' | sed 's/^/   /' >&2
    PAD_ACTIVE="${DSV41_PAD_ACTIVE:-0}"
    [ "$PAD_ACTIVE" = 1 ] && MODEL_DIR="/model-tp${TP}"
  else
    # No plan. Booting unpadded is only safe if the model actually divides TP —
    # otherwise vLLM rejects it 10 minutes in ("Total number of attention heads
    # (N) must be divisible by tensor parallel size (TP)"). Check the one key
    # that decides it and refuse rather than warn.
    heads="$(python3 -c "
import json,sys
cfg=json.load(open('$WEIGHTS/config.json'))
for k in ('text_config','language_config','llm_config'):
    if isinstance(cfg.get(k),dict) and 'num_attention_heads' in cfg[k]:
        cfg=cfg[k]; break
print(cfg.get('num_attention_heads',''))" 2>/dev/null)"
    printf '\033[33m! could not compute a padding plan for TP=%s:\n%s\033[0m\n' "$TP" "$PAD_EVAL" >&2
    if [ -z "$heads" ]; then
      die "on top of that, num_attention_heads could not be read from
   $WEIGHTS/config.json, so there is no way to tell whether TP=$TP is safe.
   Refusing rather than booting unpadded. Start here:
     python3 $PAD_SHIM_DIR/dsv41_tp_pad.py --tp $TP --model $WEIGHTS"
    fi
    if [ $((heads % TP)) -ne 0 ]; then
      die "num_attention_heads=$heads does not divide TP=$TP and the padding plan
   failed, so this would boot unpadded and vLLM would reject it at init with
   'Total number of attention heads ($heads) must be divisible by tensor
   parallel size ($TP)'. Fix the plan first:
     python3 $PAD_SHIM_DIR/dsv41_tp_pad.py --tp $TP --model $WEIGHTS"
    fi
    printf '  continuing UNPADDED (num_attention_heads=%s divides TP=%s).\n' "$heads" "$TP" >&2
  fi
fi

DRYRUN=0; STOP=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRYRUN=1 ;;
    --stop)    STOP=1 ;;
    *) die "unknown arg: $a (use --dry-run or --stop)" ;;
  esac
done

HEAD="$HEAD_IP"

# run_on <ip> <command...> — the head is this machine, so do not require it to
# have ssh access to itself.
run_on() {
  local ip="$1"; shift
  if [ "$ip" = "$HEAD" ]; then bash -c "$*"; else ssh_to "$ip" "$*"; fi
}

# Stop every rank, HEAD FIRST. This has to happen before ANY rank starts, not as
# part of each rank's own launch: a new worker that meets the still-live old head
# joins that head's rendezvous on the same port and hangs the boot at distributed
# init. That is fix #6 in the README, and it is why this is a separate pass.
stop_all() {
  local ip
  for ip in "${NODES[@]}"; do
    run_on "$ip" "docker rm -f $NAME >/dev/null 2>&1" >/dev/null 2>&1
    printf '   stopped on %s\n' "$ip"
  done
}

if [ "$STOP" = 1 ]; then
  say "stopping '$NAME' on all ${NNODES} nodes (head first)"
  stop_all
  exit 0
fi

# --- the seven patched vLLM files, from patch/mounts.txt ---------------------
# manifest: "<file> <site-relative path>" per line. ENGRAM_DISK=0 drops the
# Engram files, which is the only way to run without the disk-backed tables.
MOUNTS_TXT="$PATCH_HOST/mounts.txt"
[ -f "$MOUNTS_TXT" ] || die "missing $MOUNTS_TXT"
PATCH_MOUNTS=()
while read -r f rel; do
  [ -z "${f:-}" ] && continue
  case "$f" in \#*) continue ;; esac
  if [ "$ENGRAM_DISK" != "1" ]; then
    case "$f" in engram.py|weight_utils.py|model_state.py) continue ;; esac
  fi
  [ -f "$PATCH_HOST/$f" ] || die "missing patch file $PATCH_HOST/$f"
  PATCH_MOUNTS+=(-v "$PATCH_HOST/$f:$SITE/$rel:ro")
done < "$MOUNTS_TXT"

ENVV=(
  -e "VLLM_ENGINE_READY_TIMEOUT_S=3600"
  -e "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800"
  -e "HF_HOME=/cache/huggingface"
  -e "HF_HUB_OFFLINE=1"
  -e "TRANSFORMERS_OFFLINE=1"
  -e "VLLM_CACHE_ROOT=/cache/vllm-$LANE"
  -e "TILELANG_CACHE_DIR=/cache/tilelang"
  -e "TRITON_CACHE_DIR=/cache/triton"
  -e "VLLM_USE_RUST_FRONTEND=${DSV41_RUST_FE:-0}"
  -e "VLLM_HAS_FLASHINFER_CUBIN=1"
  -e "VLLM_USE_FLASHINFER_SAMPLER=0"
  -e "TORCH_CUDA_ARCH_LIST=12.1a"
  -e "FLASHINFER_CUDA_ARCH_LIST=12.1a"
  -e "FLASHINFER_DISABLE_VERSION_CHECK=1"
  # If anything still compiles at runtime it must not be able to take the host
  # down: boot 3 wedged all four nodes with 22 parallel nvcc jobs.
  -e "MAX_JOBS=2"
  -e "FLASHINFER_NVCC_THREADS=1"
  -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  -e "NCCL_NET=IB"
  -e "NCCL_IB_DISABLE=0"
  -e "NCCL_IB_HCA=$NCCL_IB_HCA"
  -e "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
  -e "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
  -e "TP_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
  -e "NCCL_IB_ROCE_VERSION_NUM=2"
  -e "NCCL_IB_ADDR_FAMILY=AF_INET"
  -e "NCCL_CROSS_NIC=1"
  -e "NCCL_IB_MERGE_NICS=0"
  -e "NCCL_IB_SUBNET_AWARE_ROUTING=1"
  -e "NCCL_IB_TC=106"
  -e "NCCL_NET_PLUGIN=none"
  -e "NCCL_NVLS_ENABLE=0"
  -e "NCCL_CUMEM_ENABLE=0"
  -e "NCCL_IGNORE_CPU_AFFINITY=1"
  -e "NCCL_DEBUG=${DSV41_NCCL_DEBUG:-WARN}"
  -e "TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
)

if [ "$ENGRAM_DISK" = "1" ]; then
  ENVV+=(
    -e "DSV41_ENGRAM_DISK=1"
    -e "DSV41_ENGRAM_DISK_THREADS=${DSV41_ENGRAM_THREADS:-32}"
    -e "DSV41_ENGRAM_DISK_CHUNK=${DSV41_ENGRAM_CHUNK:-16}"
  )
else
  ENVV+=(-e "DSV41_ENGRAM_DISK=0")
fi

if [ "$PAD_ACTIVE" = 1 ]; then
  # PYTHONPATH ahead of site-packages so sitecustomize.py is picked up by every
  # rank, including the workers vLLM's mp executor spawns as fresh interpreters.
  ENVV+=(
    -e "PYTHONPATH=/opt/dsv41"
    -e "DSV41_TP_PAD=$TP"
    -e "DSV41_MODEL_DIR=$MODEL_DIR"
    -e "DSV41_MODEL_SRC=/model"
  )
  [ -n "${DSV41_TP_PAD_GROUPS:-}" ] && ENVV+=(-e "DSV41_TP_PAD_GROUPS=$DSV41_TP_PAD_GROUPS")
  [ -n "${DSV41_TP_PAD_DEBUG:-}" ]  && ENVV+=(-e "DSV41_TP_PAD_DEBUG=$DSV41_TP_PAD_DEBUG")
  # The draft's padded expert count. Every rank must plan the same number the
  # host did, or the overlay config and the weight stream disagree. Both default
  # to the same values everywhere, so neither is normally set.
  [ -n "${DSV41_DRAFT_EXPERTS:-}" ] && ENVV+=(-e "DSV41_DRAFT_EXPERTS=$DSV41_DRAFT_EXPERTS")
  [ -n "${DSV41_KERNEL_EXPERT_COUNTS:-}" ] \
    && ENVV+=(-e "DSV41_KERNEL_EXPERT_COUNTS=$DSV41_KERNEL_EXPERT_COUNTS")
fi

# CUDA graph capture sizes. With DSpark k=5 every decode batch is a multiple of
# k+1 target tokens or k draft tokens, so an exact FULL graph exists for each and
# nothing is padded — padded speculative batches can hang SM120 sparse MLA.
GRAPH_ARGS=()
if [ "$EAGER" = "1" ]; then
  GRAPH_ARGS=(--enforce-eager)
else
  # Without the prestage patch the Engram lookup still happens inside the
  # forward, and a host round trip cannot be captured: the run dies with
  # "Engram DISK lookup reached a CUDA-graph capture".
  if [ "$ENGRAM_DISK" = "1" ] && ! grep -q '^model_state.py ' "$MOUNTS_TXT"; then
    die "CUDA graphs (EAGER=0) with ENGRAM_DISK=1 need the Engram prestage patch:
   model_state.py must be listed in $MOUNTS_TXT.
   Either restore it, or run this lane eager with DSV41_EAGER=1."
  fi
  if [ -z "$CG_SIZES" ]; then
    if [ "$SPEC" = "dspark" ]; then
      CG_SIZES=$( { seq "$SPEC_K" "$SPEC_K" $((SPEC_K * MAX_NUM_SEQS)); \
                    seq $((SPEC_K + 1)) $((SPEC_K + 1)) $(((SPEC_K + 1) * MAX_NUM_SEQS)); } \
                  | sort -n -u | paste -sd, - )
    else
      CG_SIZES=$(seq 1 "$MAX_NUM_SEQS" | paste -sd, -)
    fi
  fi
  GRAPH_ARGS=(--compilation-config "{\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"cudagraph_capture_sizes\":[$CG_SIZES]}")
  # Explicit on every node: eager_break_during_capture binds at model import.
  ENVV+=(-e "VLLM_USE_BREAKABLE_CUDAGRAPH=1")
fi

SERVE=(
  vllm serve "$MODEL_DIR"
  --served-model-name "$SERVED_NAME" --host 0.0.0.0 --port "$PORT"
  --tensor-parallel-size "$TP" --pipeline-parallel-size 1
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_BATCHED"
  --gpu-memory-utilization "$GMU"
  # Required. vLLM would otherwise pick 64, the smallest size the patched main
  # backend lists, and the V4 indexer backend refuses it at KV init.
  --block-size 128
  --engram-config '{"cpu_offload": false}'
  --default-chat-template-kwargs "{\"thinking\": $THINKING}"
  --distributed-executor-backend mp
)
# The published day-0 images need the tokenizer mode spelled out. On the overlay
# images it auto-resolves to deepseek_v41 from model_type and their CLI choices
# list does not even carry the literal, so passing it there would fail.
case "$IMAGE" in
  *deepseekv41-flash-0909*) SERVE+=(--tokenizer-mode deepseek_v41) ;;
esac
[ "$SPEC" = "dspark" ] && SERVE+=(--speculative-config \
  "{\"method\":\"dspark\",\"num_speculative_tokens\":$SPEC_K,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":$SPEC_ADAPT}")
[ "$TEXT_ONLY" = "1" ] && SERVE+=(--language-model-only) \
                       || SERVE+=(--limit-mm-per-prompt '{"image":4}' --mm-processor-cache-gb 1)
[ "$PARSERS" = "1" ] && SERVE+=(--tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41)
SERVE+=("${GRAPH_ARGS[@]}")
# shellcheck disable=SC2206
[ -n "${DSV41_EXTRA:-}" ] && SERVE+=(${DSV41_EXTRA})

docker_run_cmd() {
  local rank="$1" headless="$2" host_ip="$3" weights="$4"
  local base=(docker run -d --name "$NAME"
    --gpus all --network host --ipc host --shm-size 32g
    --memory 112g --memory-swap 112g --oom-score-adj 500
    --cap-add IPC_LOCK --ulimit memlock=-1:-1
    # NCCL opens a socket per peer per channel, and dual-rail doubles it. The
    # daemon default (often 1024 soft) survives TP=4 and dies at TP=6 with
    # "Call to socket failed: Too many open files" during ncclCommInitRank.
    --ulimit "nofile=${DSV41_NOFILE:-65536}:${DSV41_NOFILE:-65536}"
    --device /dev/infiniband:/dev/infiniband
    -v "$weights:/model:ro"
    -v "$CACHE_HOST:/cache"
    -v "$ENTRYPOINT_HOST:/dsv41-container-entrypoint.sh:ro"
    "${PATCH_MOUNTS[@]}")
  # Node-local Engram rows, where this node holds a copy. engram.py checks the
  # copy's recorded row range against the rank's rows and falls back to NFS.
  if [ "$ENGRAM_DISK" = "1" ] && [ "$ENGRAM_LOCAL" = "1" ]; then
    base+=(-v "$ENGRAM_LOCAL_DIR:/engram-local:ro" -e "DSV41_ENGRAM_DIR=/engram-local")
  fi
  [ "$PAD_ACTIVE" = 1 ] && base+=(-v "$PAD_SHIM_DIR:/opt/dsv41:ro")
  local cmd=("${base[@]}" "${ENVV[@]}"
    -e "VLLM_HOST_IP=$host_ip" -e "NODE_RANK=$rank" -e "MASTER_ADDR=$HEAD"
    --entrypoint /dsv41-container-entrypoint.sh
    "$IMAGE" "${SERVE[@]}"
    --nnodes "$NNODES" --node-rank "$rank" --master-addr "$HEAD" --master-port "$MASTER_PORT")
  [ "$headless" = 1 ] && cmd+=(--headless)
  local out="" t
  for t in "${cmd[@]}"; do out+=" $(printf '%q' "$t")"; done
  printf '%s' "${out# }"
}

# Every boot drops the page cache and refuses to start below MIN_AVAIL_GB: the
# weight load needs ~100 GiB of the 121.7 GiB the OS can see, and a leftover
# page cache is what turns that into an OOM kill 10 minutes in.
prelude() {
  printf 'docker rm -f %s 2>/dev/null; mkdir -p %s; ' "$NAME" "$CACHE_HOST"
  # The bind mount is unconditional so the command is identical on every node;
  # docker would otherwise create this path as root. The entrypoint decides
  # per node whether the copy is real (engram-local.json) and drops it if not.
  [ "$ENGRAM_DISK" = "1" ] && [ "$ENGRAM_LOCAL" = "1" ] \
    && printf 'mkdir -p %s; ' "$ENGRAM_LOCAL_DIR"
  printf 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true; '
  printf 'gb=$(awk "/^MemAvailable:/{print \\$2}" /proc/meminfo); gb=$((${gb:-0}/1048576)); '
  printf 'if [ $gb -lt %s ]; then echo "MemAvailable ${gb} GiB < %s GiB, refusing to boot" >&2; exit 4; fi; ' \
    "$MIN_AVAIL_GB" "$MIN_AVAIL_GB"
}

say "DeepSeek-V4.1-Flash launch: ${NNODES} nodes TP=$TP head=$HEAD:$PORT image=$IMAGE"
say "weights=$WEIGHTS served-as=$SERVED_NAME lane=$LANE ctx=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS gmu=$GMU eager=$EAGER spec=$SPEC"
printf '   engram  disk=%s local=%s\n' "$ENGRAM_DISK" "$ENGRAM_LOCAL"
printf '   patches %s (%d file(s))\n' "$PATCH_HOST" "$((${#PATCH_MOUNTS[@]} / 2))"
if [ "$PAD_ACTIVE" = 1 ]; then
  say "TP=$TP does not divide the model — padding: ${DSV41_PAD_SUMMARY}"
  printf '   groups  %s\n' "${DSV41_PAD_GROUPS}"
  printf '   serving %s (config overlay, symlinks to /model)\n' "$MODEL_DIR"
  printf '   lane seq counts were measured at TP=4; re-bench\n'
  # Experts shard by count and cannot be padded inertly — the plan can only warn.
  [ -n "${DSV41_PAD_EXPERT_ADVICE:-}" ] \
    && printf '\033[33m   ! %s\033[0m\n' "$DSV41_PAD_EXPERT_ADVICE"
fi
[ "$DRYRUN" = 1 ] && echo "   (dry-run — nothing will be executed)"

if [ "$DRYRUN" = 1 ]; then
  echo
  echo "# first, every rank is stopped HEAD FIRST (see stop_all)"
  for ip in "${NODES[@]}"; do printf '#   %s: docker rm -f %s\n' "$ip" "$NAME"; done
else
  say "stopping any previous '$NAME', head first"
  stop_all
fi

for ((rank=1; rank<NNODES; rank++)); do
  w="${NODES[$rank]}"
  run="$(docker_run_cmd "$rank" 1 "$w" "$(weights_for "$rank")")"
  shell="$(prelude)$run"
  if [ "$DRYRUN" = 1 ]; then
    printf '\n# worker %s (rank %d, headless)\nssh %s@%s %q\n' "$w" "$rank" "$SSH_USER" "$w" "$shell"
  else
    printf '   worker %s rank=%d (headless)\n' "$w" "$rank"
    ssh_to "$w" "$shell" || die "worker launch failed on $w"
  fi
done

run="$(docker_run_cmd 0 0 "$HEAD" "$(weights_for 0)")"
shell="$(prelude)$run"
if [ "$DRYRUN" = 1 ]; then
  printf '\n# head %s (rank 0)\n%s\n' "$HEAD" "$shell"
  exit 0
fi
printf '   head %s rank=0\n' "$HEAD"
bash -c "$shell" || die "head launch failed"

say "launched"
echo "   poll:  curl -s http://${HEAD}:$PORT/v1/models"
echo "   logs:  docker logs -f $NAME   (on the head node)"
echo "   stop:  $0 --stop"
