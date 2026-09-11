#!/usr/bin/env bash
# Build the overlay image chain on this node. Run ON EVERY NODE (images are
# node-local), or build on the head and use scripts/copy-image.sh.
# Final tag is $IMAGE from cluster.env (default vllm-dsv41:overlay5).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$HERE/lib.sh"
require_cluster

BUILD_DIR="$REPO_ROOT/build"
cd "$BUILD_DIR"

echo "base $HF_BASE_IMAGE must already be pulled"
docker image inspect "$HF_BASE_IMAGE" >/dev/null

# overlay1: the dsv41-feat branch plus _C_stable_libtorch rebuilt for sm_121a.
# The branch's kernel changes all live in that one extension.
echo "===== overlay1: branch + sm121 stable extension ====="
docker build -f Dockerfile.overlay -t vllm-dsv41:overlay1 "$REPO_ROOT"
bash "$BUILD_DIR/build_stable_ext.sh"

# overlay3: FlashInfer 0.7.0rc1. 0.6.18's SM120 sparse-MLA decode has no kernel
# for V4.1's topk of 1152.
echo "===== overlay3: FlashInfer 0.7.0rc1 ====="
bash "$BUILD_DIR/build_overlay3.sh"

# overlay4: prebuild mxfp8_gemm_cutlass_sm120. Its runtime compile (7 CUTLASS
# files, 22 parallel jobs) exhausted host memory on all four nodes in boot 3.
echo "===== overlay4: prebuilt mxfp8 GEMM ====="
bash "$BUILD_DIR/build_overlay4.sh"

# overlay5: rebuild sparse_mla_sm120 under the exact runtime environment.
echo "===== overlay5: prebuilt sparse MLA (serving image) ====="
bash "$BUILD_DIR/build_overlay5.sh"

echo "===== verify: nothing may compile at runtime ====="
python3 "$BUILD_DIR/verify5.py"

docker tag vllm-dsv41:overlay5 "$IMAGE" 2>/dev/null || true
echo "tagged $IMAGE"
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -E 'vllm-dsv41' || true
