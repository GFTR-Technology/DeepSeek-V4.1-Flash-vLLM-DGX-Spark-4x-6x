# Boot 5: DeepGEMM `block_kv == 32 or block_kv == 64` (2026-09-10 06:27 ET)

## What happened

Boot 5 = boot 4 + `--block-size 128` + the Engram rank-offset fix + 300,000 max context. Text-only, eager, no speculation.

**What worked:**
- Load: 78.79 GiB per rank.
- Engram rows: each rank logged its own contiguous row range, and together the ranges cover the full table.
- Profiling.
- KV init: **2,346,690 tokens** (10.9 GiB, 7.82x at 300,000 tokens). Boot 4 got 2,026,695 at 128K.
- The block-size check that killed boot 4.
- DeepGEMM's JIT kernel warmup (89.6 s).

**Then it died.** During the FlashInfer SM120 sparse-MLA decode autotune, the first real mixed prefill/decode step raised:

```
vllm/v1/attention/backends/mla/indexer.py:1268  build  ->  get_paged_mqa_logits_metadata(seq_lens, self.kv_cache_spec.num_states, ...)
RuntimeError: Assertion error (deepgemm-src/csrc/apis/attention.hpp:262): block_kv == 32 or block_kv == 64
```

It was a clean exit. Head log: `results/boot5-head.log`.

## Root cause

- The Lightning Indexer's decode path uses DeepGEMM's paged MQA logits, which takes **32 or 64 states per block**.
- The indexer's own K-cache ops are written for 64-entry blocks (`common/ops/cache_utils.py`: "K Cache block layout (block_size=64 tokens)").
- The indexer metadata builder passes the cache spec's `num_states` (`block_size // tokens_per_state`) straight to DeepGEMM, so kernel-level block splitting cannot help.
- `DeepseekV4IndexerCache` built its spec at `cache_config.block_size`, now 128. That gave:
  - ratio-1 layers (kv source 20): 128 states per block, which **fails**
  - ratio-2 layers (kv sources 2, 8, 14): 64 states, which passes
- The upstream `DeepseekV4IndexerBackend` advertises `[128]` tokens off Hopper. That fits DSV4's compress ratio of 4 (128 tokens is 32 states), not V4.1's ratio 1.

## Fix (in `patch/sm12x-pages/attention.py`, 51-line diff: `indexer-64state.diff`)

This is Kai's SM12x rule for the main compressed KV cache (64 states per page), applied to the indexer cache:

- On SM12x, `DeepseekV4IndexerCache.get_kv_cache_spec` uses `64 * compress_ratio` tokens per page: 64 at ratio 1, 128 at ratio 2, so 64 states in both.
- On SM12x, `get_attn_backend` returns `DeepseekV4IndexerSM12xBackend`, a subclass that accepts `[128, 64]`, so vLLM uses those page sizes directly.
- Other GPUs keep the upstream behavior.

**Offline check** (`patch/sm12x-pages/test_indexer64.py`, GB10, patched files mounted as in a boot):

```
ratio 1: spec block 64 tokens, num_states 64, backend DeepseekV4IndexerSM12xBackend, kernel block 64 -> OK
ratio 2: spec block 128 tokens, num_states 64, backend DeepseekV4IndexerSM12xBackend, kernel block 128 -> OK
unchanged: main 64 -> 64, main 128 -> 128, swa 64 -> 64
ALL OK
```
