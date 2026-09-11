#!/bin/bash
# Boot 5 = boot 4 + --block-size 128 (boot 4 died at KV init: vLLM picked min([128,64]) = 64; the V4 indexer
#          backend takes only 128 on SM12x) + Engram rank-offset fix (/root/swap_engram.sh) + 300000 max context.
#          Text-only, eager, no DSpark, thinking off.
export EXP_NAME=boot5 IMAGE=vllm-dsv41:overlay5 PATCH_DIR=$HOME/patches/dsv41-boot3 GMU=${GMU:-0.80} MAXLEN=300000 SEQS=8 EAGER=1 SPEC=none ENGRAM_DISK=1 TEXT_ONLY=1 THINKING=false PARSERS=0 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0"
export VLLM_EXTRA='--block-size 128'
echo "$EXP_NAME go $(date -u +%FT%TZ) gmu=$GMU maxlen=$MAXLEN spec=$SPEC"
bash $HOME/boot_dsv41.sh
echo "boot_dsv41 exit=$? $(date -u +%FT%TZ)"
