#!/bin/bash
# overlay3 = overlay1 + FlashInfer v0.7.0rc1 (07869c61ba581e6d6b8ad8d142f4a6c89b707cc1) with pinned submodules, built clean (BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1), stale 0.6.18 jit-cache/cubin removed, sparse_mla_sm120 JIT pre-warmed for sm_121a.
set -e
D=/var/tmp/v41overlay3; rm -rf $D; mkdir -p $D; cd $D
cat > Dockerfile <<'DF'
FROM vllm-dsv41:overlay1
ARG FI_SHA CUTLASS_SHA CCCL_SHA SPDLOG_SHA
RUN pip uninstall -y -q flashinfer-jit-cache flashinfer-cubin flashinfer-python || true
RUN mkdir -p /opt/fi-src && curl -sL -m 900 "https://codeload.github.com/flashinfer-ai/flashinfer/tar.gz/${FI_SHA}" | tar xz -C /opt/fi-src --strip-components=1 \
 && mkdir -p /opt/fi-src/3rdparty/cutlass /opt/fi-src/3rdparty/cccl /opt/fi-src/3rdparty/spdlog \
 && curl -sL -m 900 "https://codeload.github.com/NVIDIA/cutlass/tar.gz/${CUTLASS_SHA}" | tar xz -C /opt/fi-src/3rdparty/cutlass --strip-components=1 \
 && curl -sL -m 900 "https://codeload.github.com/NVIDIA/cccl/tar.gz/${CCCL_SHA}" | tar xz -C /opt/fi-src/3rdparty/cccl --strip-components=1 \
 && curl -sL -m 900 "https://codeload.github.com/gabime/spdlog/tar.gz/${SPDLOG_SHA}" | tar xz -C /opt/fi-src/3rdparty/spdlog --strip-components=1 \
 && ls /opt/fi-src/3rdparty/cutlass/include/cutlass/cutlass.h /opt/fi-src/3rdparty/cccl/README.md /opt/fi-src/3rdparty/spdlog/include/spdlog/spdlog.h >/dev/null \
 && cd /opt/fi-src && BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1 pip install --no-deps --no-build-isolation -q . \
 && pip list 2>/dev/null | grep -iE "^flashinfer|nvidia-nccl" && rm -rf /opt/fi-src/build
ENV VLLM_HAS_FLASHINFER_CUBIN=1
RUN python3 -c "import flashinfer; from flashinfer.mla import supported_sparse_mla_sm120_configs as f; c=f()['dsv4']; assert c.supports_decode(num_heads=16, topk=1152); print('overlay3 python ok', flashinfer.__version__)"
# pre-warm the SM120 sparse-MLA JIT module for sm_121a (no GPU needed for nvcc); non-fatal
RUN FLASHINFER_CUDA_ARCH_LIST=12.1a MAX_JOBS=16 FLASHINFER_NVCC_THREADS=2 FLASHINFER_JIT_VERBOSE=1 timeout 3000 python3 -c "from flashinfer.mla._sparse_mla_sm120 import get_sparse_mla_sm120_module; get_sparse_mla_sm120_module(); print('sparse_mla_sm120 JIT prewarmed')" 2>&1 | tail -5 || echo "PREWARM SKIPPED (will JIT at first use)"
DF
docker build -q --build-arg FI_SHA=07869c61ba581e6d6b8ad8d142f4a6c89b707cc1 --build-arg CUTLASS_SHA=b46b16d003484063bca4ed365e44095c4c6ed633 --build-arg CCCL_SHA=16bd510c9b712e82b0ab6cbb630d8e29ba1f7116 --build-arg SPDLOG_SHA=c3aed4b68373955e1cc94307683d44dca1515d2b -t vllm-dsv41:overlay3 . >/dev/null 2>build.err && echo "$(hostname): $(docker images vllm-dsv41:overlay3 --format '{{.Repository}}:{{.Tag}} {{.Size}}')" || { echo "$(hostname): BUILD FAILED"; tail -20 build.err; }
