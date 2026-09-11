from vllm.config.cache import CacheConfig
from vllm.platforms import current_platform
from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLASparseBackend as MB, DeepseekSparseSWAFlashInferSM120Backend as SB)
from vllm.v1.attention.backends.mla.indexer import DeepseekV4IndexerBackend as IB
from vllm.models.deepseek_v4_1.compressor import CompressorBackend as CB
from vllm.v1.worker.utils import select_common_block_size
D = CacheConfig.DEFAULT_BLOCK_SIZE
print("capability", current_platform.get_device_capability(), "family120", current_platform.is_device_capability_family(120))
print("DEFAULT_BLOCK_SIZE", D, "| main supported", MB.get_supported_kernel_block_sizes(), "-> preferred", MB.get_preferred_block_size(D))
print("indexer", IB.get_supported_kernel_block_sizes(), "| swa", SB.get_supported_kernel_block_sizes(), "| compressor", CB.get_supported_kernel_block_sizes())
for bs in (64, 128):
    for name, b in (("indexer", IB), ("main", MB), ("swa", SB), ("compressor", CB)):
        try:
            r = select_common_block_size(bs, [b]); print(f"  group block {bs:3d} backend {name:10s} -> kernel block {r}")
        except ValueError as e:
            print(f"  group block {bs:3d} backend {name:10s} -> ValueError: {e}")
