# Boot 4: "No common block size for 64" (2026-09-10 05:44 ET)

## What happened

Boot 4 (image `vllm-dsv41:overlay5`, text-only, eager, no speculation, gmu 0.80, 128K, 8 seqs, Engram on disk):
- Loaded 78.79 GiB per rank and ran the profiling forward with **no runtime FlashInfer JIT**, which confirms the boot 3 wedge fix.
- Sized the KV cache at 2,026,695 tokens (11.65 GiB, 15.46x at 131,072 tokens).
- One second later it died in `initialize_kv_cache`:

```
vllm/v1/worker/gpu/attn_utils.py:130  init_attn_backend
vllm/v1/worker/utils.py:492           prepare_kernel_block_sizes
vllm/v1/worker/utils.py:394           select_common_block_size
ValueError: No common block size for 64.
```

It was a clean exit: no wedge, and all four nodes stayed healthy. Full head log: `results/boot4-head.log`.

## Root cause

1. Kai's SM12x patch has the main backend `DeepseekV4FlashInferMLASparseBackend` advertise kernel block sizes `[128, 64]`, meaning "128 first so the cache-config default stays 128".
2. vLLM's default `get_preferred_block_size(16)` does not look at list order. When 16 is unsupported it returns `min(supported)`, which is **64**. The boot log shows it: `Setting kv cache block size to 64 for FLASHINFER_MLA_SPARSE_DSV41 backend.`
3. `DeepseekV4IndexerCache.get_kv_cache_spec` builds the indexer cache at `cache_config.block_size`, now 64.
4. That group's backend, `DeepseekV4IndexerBackend`, supports only `[128]` on anything that isn't Hopper (`indexer.py`: `[64 if family(90) else 128]`).
5. 64 is not supported, and 128 does not divide 64, so the check raises.

Boot 1 never hit this because the main backend then advertised only `[128]`, which made the default block 128.

## Offline reproduction

`tools/repro_block_size.py` ran in a GPU container on Reddie, with the same three patch files bind-mounted as in the boot and no model loaded:

```
capability DeviceCapability(major=12, minor=1) family120 True
DEFAULT_BLOCK_SIZE 16 | main supported [128, 64] -> preferred 64
indexer [128] | swa [64] | compressor [MultipleOf(...)]
  group block  64 backend indexer    -> ValueError: No common block size for 64.
  group block  64 backend main       -> kernel block 64
  group block  64 backend swa        -> kernel block 64
  group block  64 backend compressor -> kernel block 64
  group block 128 backend indexer    -> kernel block 128
  group block 128 backend main       -> kernel block 128
  group block 128 backend swa        -> kernel block 64
  group block 128 backend compressor -> kernel block 128
```

## Proposed fix (boot 5, one change): `--block-size 128`

**What it does:**
- It pins the default block size to 128, which is what Kai's patch intended. The indexer cache then gets 128.
- Kai's per-layer overrides still apply:
  - kv-source ratio-1 layers get 64 through `get_compressed_block_size`.
  - Ratio-2 layers get 128.
  - The SWA cache gets 64 through `get_swa_block_size`.

**Why the flag is not overridden** (checked in the image):
- A user-set `--block-size` skips Phase 1 (`platforms/interface.py`).
- Phase 2 (hybrid Mamba alignment) and Phase 3 (mixed KV dtypes) apply only to those model types, and both only ever raise the block size.
- `v1/engine/core.py:345` resets the scheduler's global block size after the per-group KV specs already exist. `prepare_kernel_block_sizes` reads the per-group spec.

**Not yet exercised:** everything after KV init.
- The next step is the eager decode warmup. Boot 1 died there on FlashInfer 0.6.18 (no SM120 sparse-MLA decode kernel for `page_block_size=32`).
- Boots 3 and 4 run FlashInfer 0.7.0rc1 with a 64-token SWA page, but neither reached that step.
