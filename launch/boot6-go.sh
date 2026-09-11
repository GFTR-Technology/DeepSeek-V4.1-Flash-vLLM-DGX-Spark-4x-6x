#!/bin/bash
# Boot 6 = boot 5 + SM12x indexer pages of 64 states (64 tokens at ratio 1, 128 at ratio 2) in attention.py
#          (boot 5 died in decode warmup: DeepGEMM paged MQA logits needs block_kv 32 or 64; ratio-1 indexer had 128).
export EXP_NAME=boot6 IMAGE=vllm-dsv41:overlay5 PATCH_DIR=$HOME/patches/dsv41-boot3 GMU=${GMU:-0.80} MAXLEN=300000 SEQS=8 EAGER=1 SPEC=none ENGRAM_DISK=1 TEXT_ONLY=1 THINKING=false PARSERS=0 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0"
export VLLM_EXTRA='--block-size 128'
echo "$EXP_NAME go $(date -u +%FT%TZ) gmu=$GMU maxlen=$MAXLEN spec=$SPEC"
bash $HOME/boot_dsv41.sh
echo "boot_dsv41 exit=$? $(date -u +%FT%TZ)"
