# SM12x indexer decode top-k: measured on GB10 (2026-09-10)

`top_k_per_row_decode` matches `torch.topk` exactly (same index sets) at row widths 600 to 300,000, including
DSpark's 6 verification rows per request. Numbers from `test_topk_sm12x.py` and `test_topk_rows.py`.

`persistent_topk` only fails when a row needs more thread blocks than GB10's 48 SMs. Boot 7's 1M-wide logits
buffer needed `ctas_per_group=90`, and the FilteredTopK fallback wants 128 KB of shared memory per block, which
GB10 doesn't have (99 KB). At 300K-wide buffers it runs.

Timing (`bench_topk.py`, CUDA events, mean of 50 calls, topk 512, 300,000-wide buffer):

| rows | context | persistent_topk | top_k_per_row_decode |
|---|---|---|---|
| 6 | 4,096 | 96.1 us | 59.8 us |
| 6 | 65,536 | 248.9 us | 119.9 us |
| 6 | 300,000 | 241.6 us | 140.2 us |
| 48 | 4,096 | 239.6 us | 66.1 us |
| 48 | 65,536 | 810.6 us | 337.7 us |
| 48 | 300,000 | 977.8 us | 512.8 us |

On GB10, `top_k_per_row_decode` is the faster kernel in every measured case, so the patch routes SM12x to it
unconditionally.
