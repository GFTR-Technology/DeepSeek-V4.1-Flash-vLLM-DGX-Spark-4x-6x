#!/bin/bash
# overlay5 = overlay4 + sparse_mla_sm120 rebuilt under the launcher's exact runtime env (the overlay3 prewarm was a
# FLASHINFER_JIT_VERBOSE=1 debug build with --threads 2 -> stale at runtime). Then verify both kernels load from cache.
LOG=/tmp/build-overlay5b.log; exec > "$LOG" 2>&1
av() { echo $(( $(grep MemAvailable /proc/meminfo | awk '{print $2}') / 1048576 )); }
echo "start $(date -u +%FT%TZ) host=$(hostname) avail=$(av)GiB"
D=$(mktemp -d /tmp/ov5.XXXX); cp /tmp/prewarm5.py /tmp/verify5.py "$D/"
cat > "$D/Dockerfile" <<'DF'
FROM vllm-dsv41:overlay4
COPY prewarm5.py /tmp/prewarm5.py
RUN FLASHINFER_CUDA_ARCH_LIST=12.1a TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_HAS_FLASHINFER_CUBIN=1 FLASHINFER_NVCC_THREADS=1 MAX_JOBS=4 timeout 3000 python3 /tmp/prewarm5.py && ls /root/.cache/flashinfer/0.7.0rc1/121a/cached_ops/
DF
( m=999; while [ -f "$D/Dockerfile" ]; do a=$(av); [ "$a" -lt "$m" ] && m=$a && echo "$m" > /tmp/ov5b-minavail; sleep 2; done ) &
SP=$!; T0=$(date +%s)
DOCKER_BUILDKIT=1 docker build --progress=plain -t vllm-dsv41:overlay5 "$D"; rc=$?
rm -f "$D/Dockerfile"; sleep 3; kill $SP 2>/dev/null
echo "nvcc: $(grep -c 'cuda/bin/nvcc' $LOG) lines, debug-flag lines: $(grep 'cuda/bin/nvcc' $LOG | grep -cE ' -G( |$)|--device-debug|lineinfo')"
echo "build rc=$rc secs=$(( $(date +%s) - T0 )) minAvail=$(cat /tmp/ov5b-minavail 2>/dev/null)GiB"
[ "$rc" = 0 ] && docker run --rm --gpus all -v "$D/verify5.py:/v.py:ro" \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_HAS_FLASHINFER_CUBIN=1 -e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 \
  --entrypoint bash vllm-dsv41:overlay5 -c 'timeout 600 python3 /v.py 2>&1 | grep -E "VERIFY|Building JIT" | cut -c1-160'
echo "done $(date -u +%H:%M:%SZ)"
