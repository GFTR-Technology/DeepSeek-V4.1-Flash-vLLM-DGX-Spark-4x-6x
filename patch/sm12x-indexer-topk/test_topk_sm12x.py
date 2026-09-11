"""GB10 check for the indexer decode top-k: persistent_topk vs top_k_per_row_decode vs torch.topk.

Run in the image with the GPU visible and the patched sparse_attn_indexer.py mounted:
  docker run --rm --gpus all --network none -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -v <patched>:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py:ro \
    -v <this file>:/t.py:ro --entrypoint python3 vllm-dsv41:overlay5 /t.py
"""
import inspect
import time

import torch

import vllm._custom_ops as ops
import vllm.model_executor.layers.sparse_attn_indexer as sai
from vllm.platforms import current_platform

dev = "cuda"
props = torch.cuda.get_device_properties(0)
print("capability", torch.cuda.get_device_capability(), "SMs", props.multi_processor_count,
      "| family120", current_platform.is_device_capability_family(120))
print("patched dispatch present:", "not current_platform.is_device_capability_family(120)" in inspect.getsource(sai.sparse_attn_indexer))

TOPK = 512
ws = torch.zeros(sai.RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=dev)


def persistent(logits, seq_lens):
    out = torch.full((logits.shape[0], TOPK), -1, dtype=torch.int32, device=dev)
    torch.ops._C.persistent_topk(logits, seq_lens, out, ws, TOPK, logits.shape[1])
    torch.cuda.synchronize()
    return out


def per_row(logits, seq_lens, next_n):
    out = torch.full((logits.shape[0], TOPK), -1, dtype=torch.int32, device=dev)
    ops.top_k_per_row_decode(logits, next_n, seq_lens, out, logits.shape[0], logits.stride(0), logits.stride(1), TOPK)
    torch.cuda.synchronize()
    return out


all_ok = True
for width in (600, 4096, 16384, 32768, 150000, 300000):
    B = 2
    logits = torch.randn(B, width, device=dev, dtype=torch.float32)
    seq_lens = torch.full((B, 1), width, dtype=torch.int32, device=dev)
    try:
        persistent(logits, seq_lens)
        p = "ok"
    except RuntimeError as e:
        p = "FAILS (" + str(e).split("persistent_topk")[-1][:60].strip() + ")"
    per_row(logits, seq_lens, 1)  # warm
    t0 = time.time()
    out = per_row(logits, seq_lens, 1)
    dt = (time.time() - t0) * 1000
    ref = torch.topk(logits, min(TOPK, width), dim=1).indices
    ok = all(set(out[b][out[b] >= 0].tolist()) == set(ref[b].tolist()) for b in range(B))
    all_ok &= ok
    print(f"width {width:6d}: persistent_topk {p:<70} | top_k_per_row_decode {'MATCH' if ok else 'MISMATCH'} ({dt:.2f} ms, B={B})")

# DSpark verification rows: next_n = k + 1 = 6 rows per request, flattened
for width in (4096, 32768, 300000):
    B, nn_ = 2, 6
    logits = torch.randn(B * nn_, width, device=dev, dtype=torch.float32)
    seq_lens = torch.full((B, nn_), width, dtype=torch.int32, device=dev)
    try:
        out = per_row(logits, seq_lens, nn_)
        valid = bool(((out >= -1) & (out < width)).all()) and bool((out >= 0).sum(1).min() > 0)
        all_ok &= valid
        print(f"next_n=6 width {width:6d}: top_k_per_row_decode ran, indices in range: {valid}")
    except RuntimeError as e:
        all_ok = False
        print(f"next_n=6 width {width:6d}: top_k_per_row_decode FAILED: {str(e)[:120]}")
print("ALL OK" if all_ok else "FAILED")
