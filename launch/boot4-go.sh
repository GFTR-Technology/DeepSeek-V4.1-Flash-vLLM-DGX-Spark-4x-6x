#!/bin/bash
# Boot 4 = boot 3 + overlay5 (FlashInfer mxfp8_gemm + sparse_mla prebuilt under the runtime env; its runtime compile wedged all 4 nodes in boot 3)
#          + MAX_JOBS=2 cap so any residual runtime JIT cannot exhaust unified memory. Text-only, eager, no DSpark, thinking off.
export EXP_NAME=boot4 IMAGE=vllm-dsv41:overlay5 PATCH_DIR=$HOME/patches/dsv41-boot3 GMU=${GMU:-0.80} MAXLEN=131072 SEQS=8 EAGER=1 SPEC=none ENGRAM_DISK=1 TEXT_ONLY=1 THINKING=false PARSERS=0 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1"
echo "boot4 go $(date -u +%FT%TZ) gmu=$GMU"
bash $HOME/boot_dsv41.sh
echo "boot_dsv41 exit=$? $(date -u +%FT%TZ)"
