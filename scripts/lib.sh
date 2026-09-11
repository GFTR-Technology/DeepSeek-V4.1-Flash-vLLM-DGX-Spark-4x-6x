#!/usr/bin/env bash
# Shared loader. Source this from every script.
# Looks for cluster.env next to this file, then $DSV41_ENV, then CWD.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_repo="$(cd "$_here/.." && pwd)"

load_cluster_env() {
  local f
  for f in \
    "${DSV41_ENV:-}" \
    "$_here/cluster.env" \
    "$_repo/cluster.env" \
    "$PWD/cluster.env"
  do
    [ -n "$f" ] && [ -f "$f" ] && {
      set -a
      # shellcheck source=/dev/null
      . "$f"
      set +a
      DSV41_ENV_FILE="$f"
      return 0
    }
  done
  echo "Missing cluster.env." >&2
  echo "  cp scripts/cluster.env.example scripts/cluster.env" >&2
  echo "  edit SSH_USER, HEAD_IP, WORKER_IPS, NCCL_* to match your fabric" >&2
  echo "  see README_CN.md" >&2
  return 1
}

require_cluster() {
  load_cluster_env || exit 2
  : "${SSH_USER:?set SSH_USER in cluster.env}"
  : "${HEAD_IP:?set HEAD_IP in cluster.env}"
  : "${WORKER_IPS:?set WORKER_IPS in cluster.env}"
  # The head keeps the checkpoint on local NVMe; the workers usually see the same
  # bytes over NFS at a different path. Set WORKER_WEIGHTS only if it differs.
  : "${WEIGHTS:=/var/tmp/models/DeepSeek-V4.1-Flash}"
  : "${WORKER_WEIGHTS:=$WEIGHTS}"
  : "${CACHE:=/var/tmp/dsv41-vllm-cache}"
  : "${IMAGE:=vllm-dsv41:overlay5}"
  : "${NAME:=vllm_dsv41}"
  : "${PORT:=8000}"
  : "${MASTER_PORT:=29541}"
  : "${HF_REPO:=deepseek-ai/DeepSeek-V4.1-Flash}"
  : "${HF_BASE_IMAGE:=vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c}"
  : "${NCCL_IB_HCA:=rocep1s0f0,roceP2p1s0f0}"
  : "${NCCL_SOCKET_IFNAME:=enp1s0f0np0,enP2p1s0f0np0}"
  : "${GLOO_SOCKET_IFNAME:=enp1s0f0np0}"
  : "${ENGRAM_DISK:=1}"
  # On by default: reading Engram rows over NFS costs 5.9-7.8 ms per step against
  # 2.8 ms locally, and every step waits for the slowest rank. A node without a
  # copy is not an error — the entrypoint drops the mount and that rank reads the
  # model dir as before. dsv41-serve.sh builds the copies automatically
  # (scripts/engram-local.sh); DSV41_ENGRAM_AUTOBUILD=0 turns that off.
  : "${ENGRAM_LOCAL:=1}"
  : "${ENGRAM_LOCAL_DIR:=/var/tmp/engram-local/DeepSeek-V4.1-Flash}"
  # NODES: head first
  # shellcheck disable=SC2206
  NODES=($HEAD_IP $WORKER_IPS)
  NNODES="${#NODES[@]}"
  if [ "$NNODES" -lt 2 ]; then
    echo "Need HEAD_IP plus at least one worker in WORKER_IPS." >&2
    exit 2
  fi
  REPO_ROOT="$_repo"
  SCRIPT_DIR="$_here"
  # weights_for <rank> — rank 0 reads local NVMe, workers read the NFS mount
  weights_for() { [ "$1" = 0 ] && printf '%s' "$WEIGHTS" || printf '%s' "$WORKER_WEIGHTS"; }
  # ssh_to <host> <command...>
  # The shift matters: without it the host lands in the remote argv too and
  # every remote command runs as `<ip> docker ...` -> command not found.
  ssh_to() {
    local host="$1"; shift
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
      -o ConnectTimeout=8 "$SSH_USER@$host" "$@"
  }
}
