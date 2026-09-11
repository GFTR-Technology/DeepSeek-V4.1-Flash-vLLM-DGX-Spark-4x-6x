"""Offline check of the SM12x indexer page patch (run in the image with the patched files mounted, GPU visible)."""
import types

import torch

import vllm.models.deepseek_v4_1.attention as A
from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
    DeepseekSparseSWAFlashInferSM120Backend as SWA,
    DeepseekV4FlashInferMLASparseBackend as MAIN,
)
from vllm.v1.worker.utils import select_common_block_size

print("sm12x detected:", A._dsv41_indexer_sm12x())
IB = A.DeepseekV4IndexerSM12xBackend
print("indexer SM12x backend sizes:", IB.get_supported_kernel_block_sizes(), "| name:", IB.get_name())
vcfg = types.SimpleNamespace(cache_config=types.SimpleNamespace(cache_dtype="fp8_ds_mla"))
ok = True
for ratio in (1, 2):
    c = A.DeepseekV4IndexerCache.__new__(A.DeepseekV4IndexerCache)
    torch.nn.Module.__init__(c)
    c.cache_config = types.SimpleNamespace(block_size=128)
    c.head_dim, c.dtype, c.compress_ratio = 132, torch.uint8, ratio
    spec = c.get_kv_cache_spec(vcfg)
    backend = c.get_attn_backend()
    kb = select_common_block_size(spec.block_size, [backend])
    good = spec.num_states == 64 and kb == spec.block_size and backend is IB
    ok &= good
    print(f"ratio {ratio}: spec block {spec.block_size} tokens, num_states {spec.num_states}, "
          f"backend {backend.__name__}, kernel block {kb} -> {'OK' if good else 'BAD'}")
for bs, name, b in ((64, "main", MAIN), (128, "main", MAIN), (64, "swa", SWA)):
    print(f"unchanged: group block {bs} backend {name} -> kernel block {select_common_block_size(bs, [b])}")
print("ALL OK" if ok else "FAILED")
