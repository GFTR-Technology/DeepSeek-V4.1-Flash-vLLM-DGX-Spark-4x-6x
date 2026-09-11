#!/bin/bash
# Boot 9 = boot 8 (DSpark k=5 + CUDA graphs + Engram staged before the forward + SM12x top-k fix, 300K) with
#          tools on (deepseek_v41 tool + reasoning parsers, auto tool choice), vision on (encoder loaded,
#          up to 4 images per prompt, 1 GiB processor cache) and gmu 0.80.
export EXP_NAME=boot9 IMAGE=vllm-dsv41:overlay5 PATCH_DIR=$HOME/patches/dsv41-boot3 GMU=0.80 MAXLEN=300000 SEQS=8 EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE SPEC=dspark SPEC_K=5 ENGRAM_DISK=1 TEXT_ONLY=0 THINKING=false PARSERS=1 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton"
export VLLM_EXTRA='--block-size 128 --limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1'
echo "$EXP_NAME go $(date -u +%FT%TZ) gmu=$GMU maxlen=$MAXLEN spec=$SPEC eager=$EAGER cg=$CUDAGRAPH_MODE text_only=$TEXT_ONLY parsers=$PARSERS"
bash $HOME/boot_dsv41.sh
echo "boot_dsv41 exit=$? $(date -u +%FT%TZ)"
