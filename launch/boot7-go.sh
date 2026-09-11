#!/bin/bash
# Boot 7 = boot 6 + DSpark k=5 at 1048576 max context: the 1M proof boot (vLLM refuses at startup if the
#          DSpark KV pool cannot hold one 1M-token request). Adaptive verification off under eager.
export EXP_NAME=boot7 IMAGE=vllm-dsv41:overlay5 PATCH_DIR=$HOME/patches/dsv41-boot3 GMU=${GMU:-0.80} MAXLEN=1048576 SEQS=8 EAGER=1 SPEC=dspark SPEC_K=5 ENGRAM_DISK=1 TEXT_ONLY=1 THINKING=false PARSERS=0 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0"
export VLLM_EXTRA='--block-size 128'
echo "$EXP_NAME go $(date -u +%FT%TZ) gmu=$GMU maxlen=$MAXLEN spec=$SPEC"
bash $HOME/boot_dsv41.sh
echo "boot_dsv41 exit=$? $(date -u +%FT%TZ)"
