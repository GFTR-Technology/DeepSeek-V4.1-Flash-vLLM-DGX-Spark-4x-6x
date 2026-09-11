#!/bin/bash
# v41build2.sh: configure+build _C_stable_libtorch inside the existing v41build container (git + cutlass already present)
echo "=== $(date) start2 ==="
docker exec v41build bash -c '
set -o pipefail
export TORCH_CUDA_ARCH_LIST=12.1a
cd /src
mkdir -p /src/build/_deps
if [ ! -f /src/build/_deps/cutlass-src/include/cutlass/cutlass.h ]; then
  rm -rf /src/build/_deps/cutlass-src /src/build/_deps/cutlass-subbuild /src/build/_deps/cutlass-build
  if [ -f /tmp/cutlass-speedtest/include/cutlass/cutlass.h ]; then mv /tmp/cutlass-speedtest /src/build/_deps/cutlass-src
  else mkdir -p /src/build/_deps/cutlass-src && curl -sL -m 600 https://codeload.github.com/NVIDIA/cutlass/tar.gz/refs/tags/v4.7.1 | tar xz -C /src/build/_deps/cutlass-src --strip-components=1; fi
fi
ls /src/build/_deps/cutlass-src/include/cutlass/cutlass.h >/dev/null && echo "cutlass ok" || { echo "CUTLASS MISSING"; echo "CONFIGURE FAILED"; exit 2; }
cp -n CMakeLists.txt CMakeLists.txt.orig
cp CMakeLists.txt.orig CMakeLists.txt
sed -i -E "s|^(\s*)include\(cmake/external_projects/|\1# STABLE-ONLY BUILD (kai): include(cmake/external_projects/|" CMakeLists.txt
grep -c "STABLE-ONLY BUILD" CMakeLists.txt
rm -rf /src/build/CMakeCache.txt /src/build/CMakeFiles
PYPATH=$(python3 -c "import sys;print(\":\".join(p for p in sys.path if p))")
TORCH_PREFIX=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)")
NVRTC=$(ls /usr/local/cuda/lib64/libnvrtc.so /usr/local/cuda/lib64/libnvrtc.so.* /usr/local/lib/python3.12/dist-packages/nvidia/*/lib/libnvrtc.so* /usr/lib/aarch64-linux-gnu/libnvrtc.so* 2>/dev/null | head -1)
echo "nvrtc=$NVRTC"
echo "=== $(date) configure ==="
cmake -S /src -B /src/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=cuda \
  -DVLLM_PYTHON_EXECUTABLE=$(which python3) -DVLLM_PYTHON_PATH="$PYPATH" \
  -DFETCHCONTENT_BASE_DIR=/src/build/_deps -DFETCHCONTENT_SOURCE_DIR_CUTLASS=/src/build/_deps/cutlass-src \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DNVCC_THREADS=2 -DCUDA_nvrtc_LIBRARY="$NVRTC" 2>&1 | tail -25
rc=$?; echo "configure rc=$rc"; [ $rc -ne 0 ] && { echo "CONFIGURE FAILED"; exit 2; }
echo "=== $(date) build _C_stable_libtorch ==="
cmake --build /src/build --target _C_stable_libtorch -j 20 2>&1 | grep -E --line-buffered "^\[[0-9]+/[0-9]+\]|error|Error|FAILED|Linking" | awk "NR%20==1 || /error|Error|FAILED|Linking/ {print; fflush()}"
rc=${PIPESTATUS[0]}; echo "build rc=$rc"
find /src/build -maxdepth 2 -name "_C_stable_libtorch*.so" -exec ls -la {} \;
echo "=== $(date) done rc=$rc ==="
[ $rc -eq 0 ] && echo "BUILD OK" || echo "BUILD FAILED"
'
