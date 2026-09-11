import torch
import vllm._custom_ops as ops
import vllm.model_executor.layers.sparse_attn_indexer as sai
dev = "cuda"
ws = torch.zeros(sai.RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=dev)
for rows in (2, 6, 12, 48):
    for width in (32768, 300000):
        logits = torch.randn(rows, width, device=dev)
        seq = torch.full((rows, 1), width, dtype=torch.int32, device=dev)
        out = torch.full((rows, 512), -1, dtype=torch.int32, device=dev)
        try:
            torch.ops._C.persistent_topk(logits, seq, out, ws, 512, width); torch.cuda.synchronize(); r = "ok"
        except RuntimeError as e:
            r = "FAILS total_ctas=" + str(e).split("total_ctas=")[-1].split(" (")[0]
        out2 = torch.full((rows, 512), -1, dtype=torch.int32, device=dev)
        ops.top_k_per_row_decode(logits, 1, seq, out2, rows, logits.stride(0), logits.stride(1), 512); torch.cuda.synchronize()
        ref = torch.topk(logits, 512, dim=1).indices
        ok = all(set(out2[i][out2[i] >= 0].tolist()) == set(ref[i].tolist()) for i in range(rows))
        print(f"rows {rows:3d} width {width:6d}: persistent_topk {r:<40} | top_k_per_row_decode {'MATCH' if ok else 'MISMATCH'}")
