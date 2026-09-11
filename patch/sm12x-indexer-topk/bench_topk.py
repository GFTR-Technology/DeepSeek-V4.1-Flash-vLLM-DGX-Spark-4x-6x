import torch
import vllm._custom_ops as ops
import vllm.model_executor.layers.sparse_attn_indexer as sai
dev = "cuda"; ws = torch.zeros(sai.RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=dev)
def timeit(fn, n=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n
for rows in (6, 48):
    for width, ctx in ((300000, 4096), (300000, 65536), (300000, 300000)):
        logits = torch.randn(rows, width, device=dev)
        seq = torch.full((rows, 1), ctx, dtype=torch.int32, device=dev)
        out = torch.full((rows, 512), -1, dtype=torch.int32, device=dev)
        tp = timeit(lambda: torch.ops._C.persistent_topk(logits, seq, out, ws, 512, width))
        tr = timeit(lambda: ops.top_k_per_row_decode(logits, 1, seq, out, rows, logits.stride(0), logits.stride(1), 512))
        print(f"rows {rows:2d} buffer {width} ctx {ctx:6d}: persistent_topk {tp*1000:7.1f} us | top_k_per_row_decode {tr*1000:7.1f} us")
