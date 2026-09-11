#!/bin/bash
# Boot 10 = boot 9 (DSpark k=5 + CUDA graphs + Engram staged before the forward + SM12x top-k fix, 300K,
#           tools on, vision on, gmu 0.80) with ONE change: node-local Engram rows on the three workers
#           (tools/engram_local.py copies each rank's rows to /var/tmp/engram-local; engram.py reads them
#           there when the copied range covers the rank, else NFS as before). Reddie already reads locally.
export EXP_NAME=boot10 IMAGE=vllm-dsv41:overlay5 PATCH_NAME=dsv41-boot10 ENGRAM_LOCAL=1 GMU=0.80 MAXLEN=300000 SEQS=8 EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE SPEC=dspark SPEC_K=5 ENGRAM_DISK=1 TEXT_ONLY=0 THINKING=false PARSERS=1 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton"
export VLLM_EXTRA='--block-size 128 --limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1'
echo "$EXP_NAME go $(date -u +%FT%TZ) gmu=$GMU maxlen=$MAXLEN spec=$SPEC eager=$EAGER cg=$CUDAGRAPH_MODE text_only=$TEXT_ONLY parsers=$PARSERS patch=$PATCH_NAME engram_local=$ENGRAM_LOCAL"
bash $HOME/boot_dsv41.sh
echo "boot_dsv41 exit=$? $(date -u +%FT%TZ)"
