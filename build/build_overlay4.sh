#!/bin/bash
# overlay4 = overlay3 + FlashInfer mxfp8_gemm_cutlass_sm120 JIT module prebuilt for sm_121a (the module whose
# runtime compile wedged all four nodes in boot 3). Samples min MemAvailable during the compile.
LOG=/tmp/build-overlay4.log; exec > "$LOG" 2>&1
av() { echo $(( $(grep MemAvailable /proc/meminfo | awk '{print $2}') / 1048576 )); }
echo "start $(date -u +%FT%TZ) host=$(hostname) avail=$(av)GiB"
D=$(mktemp -d /tmp/ov4.XXXX)
cat > "$D/Dockerfile" <<'DF'
FROM vllm-dsv41:overlay3
ARG MAXJ=4
RUN FLASHINFER_CUDA_ARCH_LIST=12.1a MAX_JOBS=${MAXJ} FLASHINFER_NVCC_THREADS=1 timeout 3000 python3 -c "from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8 as gen; spec = gen(); b = getattr(spec, 'build', None); (b(verbose=True) if b else spec.build_and_load()); print('MXFP8-SM120-BUILT')" && ls -la /root/.cache/flashinfer/0.7.0rc1/121a/cached_ops/ /root/.cache/flashinfer/0.7.0rc1/121a/cached_ops/mxfp8_gemm_cutlass_sm120/
DF
( m=999; while [ -f "$D/Dockerfile" ]; do a=$(av); [ "$a" -lt "$m" ] && m=$a && echo "$m" > /tmp/ov4-minavail; sleep 2; done ) &
SP=$!
T0=$(date +%s)
DOCKER_BUILDKIT=1 docker build --progress=plain -t vllm-dsv41:overlay4 --build-arg MAXJ=4 "$D"; rc=$?
rm -f "$D/Dockerfile"; sleep 3; kill $SP 2>/dev/null
echo "build rc=$rc secs=$(( $(date +%s) - T0 )) minAvail=$(cat /tmp/ov4-minavail 2>/dev/null)GiB end=$(date -u +%H:%M:%SZ)"
docker image inspect vllm-dsv41:overlay4 --format 'overlay4 id={{.Id}}' 2>&1 | cut -c1-40
