#!/usr/bin/env bash
# IN-CONTAINER ENTRYPOINT — pick a RoCEv2 IPv4 GID, then exec vllm.
# Bind-mounted read-only from the host. The image ENTRYPOINT is ["vllm","serve"];
# we replace it with this wrapper and pass `vllm serve ...` as args.
set -eu

# Dual-rail: NCCL_IB_HCA carries both devices and both are used for data. The GID
# index only has to be resolved once, on rail 0 — the RoCEv2 IPv4 GID sits at the
# same index on both ports of a Spark. Probing beats hard-coding 3: the index
# moves with the number of IPv4/IPv6 addresses configured on the netdev.
HCA="${NCCL_IB_HCA%%,*}"
if [ -n "${HCA:-}" ] && [ -d "/sys/class/infiniband/$HCA/ports/1" ]; then
  for i in $(seq 0 15); do
    t=$(cat "/sys/class/infiniband/$HCA/ports/1/gid_attrs/types/$i" 2>/dev/null || true)
    g=$(cat "/sys/class/infiniband/$HCA/ports/1/gids/$i" 2>/dev/null || true)
    case "$t" in
      *"RoCE v2"*)
        case "$g" in
          *"0000:0000:0000:0000:0000:ffff:"*)
            export NCCL_IB_GID_INDEX=$i
            echo "[dsv41-entrypoint] HCA=$HCA NCCL_IB_GID_INDEX=$i gid=$g"
            break
            ;;
        esac
        ;;
    esac
  done
fi
if [ -z "${NCCL_IB_GID_INDEX:-}" ]; then
  echo "[dsv41-entrypoint] WARNING: no RoCEv2 IPv4 GID for HCA=$HCA; NCCL will auto-select" >&2
fi

# Node-local Engram rows are on by default, but only this node knows whether it
# actually holds a copy. Without engram-local.json the shim would warn once per
# Engram layer and fall back anyway, so drop the variable here instead and say so
# once. Build the copies with scripts/engram-local.sh.
if [ -n "${DSV41_ENGRAM_DIR:-}" ]; then
  if [ -f "${DSV41_ENGRAM_DIR}/engram-local.json" ]; then
    echo "[dsv41-entrypoint] node-local Engram rows: ${DSV41_ENGRAM_DIR}"
  else
    echo "[dsv41-entrypoint] no engram-local.json in ${DSV41_ENGRAM_DIR}; this rank"
    echo "[dsv41-entrypoint] reads its Engram rows from the model dir (slower over NFS)."
    echo "[dsv41-entrypoint] dsv41-serve.sh normally builds this; run"
    echo "[dsv41-entrypoint] scripts/engram-local.sh on the head to make one."
    unset DSV41_ENGRAM_DIR
  fi
fi

# TP that does not divide the head counts (TP=6): hand vLLM a symlink overlay of
# /model whose config.json carries the padded dims. The weights themselves are
# padded as they stream in, by the shim on PYTHONPATH.
if [ -n "${DSV41_TP_PAD:-}" ] && [ "${DSV41_TP_PAD}" != 1 ] && [ -n "${DSV41_MODEL_DIR:-}" ]; then
  echo "[dsv41-entrypoint] building TP=${DSV41_TP_PAD} config overlay at ${DSV41_MODEL_DIR}"
  python3 /opt/dsv41/make_overlay.py \
    --src "${DSV41_MODEL_SRC:-/model}" \
    --dst "${DSV41_MODEL_DIR}" \
    --tp "${DSV41_TP_PAD}"
fi

exec "$@"
