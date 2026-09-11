#!/bin/bash
# Boot 8 = FINAL 300K serving config, every speed fix at once: DSpark k=5 + CUDA graphs (FULL_AND_PIECEWISE; the
#          launcher derives exact DSpark capture sizes so decode batches are never padded, FlashInfer #5015) +
#          Engram rows staged before the forward with parallel reads (engram.py + nvidia/model_state.py) +
#          gmu 0.78 (graph memory headroom) + TileLang/Triton caches on the host. Adaptive verification off.
export EXP_NAME=boot8 IMAGE=vllm-dsv41:overlay5 PATCH_DIR=$HOME/patches/dsv41-boot3 GMU=0.78 MAXLEN=300000 SEQS=8 EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE SPEC=dspark SPEC_K=5 ENGRAM_DISK=1 TEXT_ONLY=1 THINKING=false PARSERS=0 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton"
export VLLM_EXTRA='--block-size 128'
echo "$EXP_NAME go $(date -u +%FT%TZ) gmu=$GMU maxlen=$MAXLEN spec=$SPEC eager=$EAGER cg=$CUDAGRAPH_MODE"
bash $HOME/boot_dsv41.sh
echo "boot_dsv41 exit=$? $(date -u +%FT%TZ)"
