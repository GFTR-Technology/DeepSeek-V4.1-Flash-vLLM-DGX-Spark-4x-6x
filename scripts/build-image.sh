#!/usr/bin/env bash
# Build the overlay image chain on this node. Images are node-local: build on the
# head and use scripts/copy-image.sh, or run this everywhere.
#
#   ./scripts/build-image.sh                 # full chain, overlay1 -> overlay5
#   ./scripts/build-image.sh --from overlay3 # resume after a failure
#   ./scripts/build-image.sh --from-published   # skip overlay1, see the caveat below
#   ./scripts/build-image.sh --check          # only report what is missing
#
# THIS IS THE LEAST AUTOMATED PART OF THE REPO. overlay3/4/5 are self-contained
# (every SHA is pinned inside build/build_overlay{3,4,5}.sh), but overlay1 needs
# two things this repo cannot ship:
#
#   1. build/vllm/ — the Python tree of vllm-project/vllm branch `dsv41-feat`,
#      which Dockerfile.overlay copies over the base image's site-packages.
#   2. a running container named `v41build` with the same checkout at /src, in
#      which build/build_stable_ext.sh rebuilds _C_stable_libtorch for sm_121a.
#      The branch's kernel changes all live in that one extension, which the
#      stock ARM64 wheels do not carry for SM 12.1.
#
# The preflight below tells you exactly which of those is missing and how to get
# it, rather than letting docker fail on a missing COPY source.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$HERE/lib.sh"
require_cluster

BUILD_DIR="$REPO_ROOT/build"
VLLM_TREE="${DSV41_VLLM_TREE:-$BUILD_DIR/vllm}"
BUILD_CTR="${DSV41_BUILD_CONTAINER:-v41build}"
PUBLISHED="${DSV41_PUBLISHED_IMAGE:-vllm/vllm-openai:deepseekv41-flash-0909-arm64}"
# The exact merge-base of branch dsv41-feat; Dockerfile.overlay FROMs it.
BASE_IMAGE="$HF_BASE_IMAGE"

START=overlay1; CHECK_ONLY=0; FROM_PUBLISHED=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from)           START="$2"; shift ;;
    --from=*)         START="${1#--from=}" ;;
    --from-published) FROM_PUBLISHED=1; START=overlay3 ;;
    --check)          CHECK_ONLY=1 ;;
    -h|--help)        sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

have_image() { docker image inspect "$1" >/dev/null 2>&1; }
stage_num() {
  case "$1" in
    overlay1) echo 1 ;; overlay3) echo 3 ;; overlay4) echo 4 ;; overlay5) echo 5 ;;
    *) echo "unknown stage: $1 (use overlay1|overlay3|overlay4|overlay5)" >&2; exit 2 ;;
  esac
}
START_NUM="$(stage_num "$START")"
stage_ge() { [ "$(stage_num "$1")" -ge "$START_NUM" ]; }

# ---- preflight --------------------------------------------------------------
miss=0
echo "== preflight"
if have_image "$BASE_IMAGE"; then
  echo "   base image        $BASE_IMAGE"
else
  echo "   base image        MISSING: $BASE_IMAGE"
  echo "                     docker pull $BASE_IMAGE"
  [ "$FROM_PUBLISHED" = 1 ] || miss=1
fi

if [ "$FROM_PUBLISHED" = 1 ]; then
  if have_image "$PUBLISHED"; then
    echo "   published image   $PUBLISHED"
  else
    echo "   published image   MISSING: $PUBLISHED"
    echo "                     docker pull $PUBLISHED"
    miss=1
  fi
elif stage_ge overlay1; then
  if [ -d "$VLLM_TREE" ] && [ -f "$VLLM_TREE/__init__.py" ]; then
    echo "   vllm tree         $VLLM_TREE"
    if ls "$VLLM_TREE"/_C_stable_libtorch*.so >/dev/null 2>&1; then
      echo "   sm121 extension   $(cd "$VLLM_TREE" && ls _C_stable_libtorch*.so)"
    else
      echo "   sm121 extension   MISSING from $VLLM_TREE"
      echo "                     build it, then copy it in:"
      echo "                       bash $BUILD_DIR/build_stable_ext.sh"
      echo "                       docker cp $BUILD_CTR:/src/build/_C_stable_libtorch.<abi>.so $VLLM_TREE/"
      miss=1
    fi
  else
    echo "   vllm tree         MISSING: $VLLM_TREE"
    echo "                     Dockerfile.overlay copies it over site-packages. Get it with:"
    echo "                       git clone https://github.com/vllm-project/vllm /src/vllm-dsv41"
    echo "                       git -C /src/vllm-dsv41 checkout dsv41-feat"
    echo "                       cp -a /src/vllm-dsv41/vllm $VLLM_TREE"
    echo "                     or point DSV41_VLLM_TREE at an existing checkout's vllm/ dir."
    miss=1
  fi
  if docker ps --format '{{.Names}}' | grep -qx "$BUILD_CTR"; then
    echo "   build container   $BUILD_CTR running"
  else
    echo "   build container   NOT running: $BUILD_CTR"
    echo "                     build_stable_ext.sh does 'docker exec $BUILD_CTR' with the"
    echo "                     checkout at /src. Start one, e.g.:"
    echo "                       docker run -d --name $BUILD_CTR --gpus all \\"
    echo "                         -v /src/vllm-dsv41:/src -w /src --entrypoint sleep \\"
    echo "                         $BASE_IMAGE infinity"
    echo "                     (only needed to produce the .so above; skip if you already have it)"
  fi
fi

if [ "$miss" != 0 ]; then
  echo
  echo "== missing prerequisites above. Two ways forward:"
  echo "   1. supply them and re-run ./scripts/build-image.sh"
  echo "   2. ./scripts/build-image.sh --from-published"
  echo "      tags $PUBLISHED"
  echo "      as vllm-dsv41:overlay1 and starts at overlay3. That image carries the"
  echo "      dsv41-feat branch but NOT _C_stable_libtorch rebuilt for sm_121a, which"
  echo "      is the whole point of overlay1 on GB10. Untested here — if the engine"
  echo "      dies in a kernel at load, you need the real overlay1."
  exit 1
fi
[ "$CHECK_ONLY" = 1 ] && { echo "== check only, nothing built"; exit 0; }

# ---- build ------------------------------------------------------------------
if [ "$FROM_PUBLISHED" = 1 ]; then
  echo "== tagging $PUBLISHED as vllm-dsv41:overlay1 (no sm121 extension rebuild)"
  docker tag "$PUBLISHED" vllm-dsv41:overlay1
fi

if [ "$START" = overlay1 ]; then
  echo "===== overlay1: dsv41-feat tree + sm121 stable extension ====="
  # The COPY source is build/vllm, so the build context is build/.
  docker build -f "$BUILD_DIR/Dockerfile.overlay" -t vllm-dsv41:overlay1 "$BUILD_DIR"
fi

if stage_ge overlay3; then
  echo "===== overlay3: FlashInfer 0.7.0rc1 ====="
  # 0.6.18's SM120 sparse-MLA decode has no kernel for V4.1's topk of 1152.
  bash "$BUILD_DIR/build_overlay3.sh"
  have_image vllm-dsv41:overlay3 || { echo "overlay3 did not produce an image"; exit 1; }
fi

if stage_ge overlay4; then
  echo "===== overlay4: prebuilt mxfp8 GEMM (logs to /tmp/build-overlay4.log) ====="
  # Its runtime compile (7 CUTLASS files, 22 parallel jobs) exhausted host memory
  # on all four nodes at once in boot 3.
  bash "$BUILD_DIR/build_overlay4.sh" || true
  have_image vllm-dsv41:overlay4 || { echo "overlay4 failed; see /tmp/build-overlay4.log"; exit 1; }
fi

if stage_ge overlay5; then
  echo "===== overlay5: prebuilt sparse MLA — the serving image (logs to /tmp/build-overlay5b.log) ====="
  # build_overlay5.sh reads these two from /tmp; they live in build/.
  cp "$BUILD_DIR/prewarm5.py" "$BUILD_DIR/verify5.py" /tmp/
  bash "$BUILD_DIR/build_overlay5.sh" || true
  have_image vllm-dsv41:overlay5 || { echo "overlay5 failed; see /tmp/build-overlay5b.log"; exit 1; }
fi

docker tag vllm-dsv41:overlay5 "$IMAGE" 2>/dev/null || true
echo "== tagged $IMAGE"
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -E 'vllm-dsv41' || true
echo
echo "Next: ./scripts/copy-image.sh   # docker save + rsync + load on every worker"
