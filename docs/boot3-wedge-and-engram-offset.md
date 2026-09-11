# Boot 3 wedge and the Engram rank-offset bug (2026-09-10)

Two findings from the boot 3 post-mortem. Measured and reproduced items are marked as such. Estimates are labeled.

## 1. Boot 3 wedge: host memory exhaustion during the profiling run

**What happened (from the journals, container logs and sar on all four nodes, UTC):**

| Time | Event |
|---|---|
| 08:52:00 | Reddie finishes loading (232 s). Workers finish at 08:53:45-51, after 340-343 s over NFS. |
| 08:53:52-59 | The profiling run starts on all ranks. |
| 08:54:00-01 | FlashInfer starts a runtime JIT build of `mxfp8_gemm_cutlass_sm120` on all 4 ranks at once. |
| 08:54:43-53 | systemd-logind D-Bus timeouts on all 4 nodes (PID 1 unresponsive for 25 s or more). |
| 08:55-08:57 | Journals end. pstore holds SBSA Generic Watchdog panics. |
| 08:58-09:01 | All four nodes boot again. No OOM-kill line anywhere, which points to reclaim thrash rather than a clean OOM kill. |

**Cause (moderate-high confidence):**
- `mxfp8_gemm_cutlass_sm120` is 7 heavy CUTLASS translation units.
- With `MAX_JOBS` unset, ninja runs 22 jobs, so all 7 compile at once. That is an estimated several GiB per job, on nodes that had 10-15 GiB available and under 1 GiB truly free.
- The JIT cache lives inside the container, and the launcher's `docker rm -f` discards it, so every boot would recompile.

**Fix used from boot 4 on:**
- Image `vllm-dsv41:overlay5` ships `mxfp8_gemm_cutlass_sm120` and `sparse_mla_sm120` prebuilt under the exact runtime environment. The FlashInfer version check, arch list and NVCC thread count all match, so the cache hits.
- Runtime `MAX_JOBS=2` and `FLASHINFER_NVCC_THREADS=1`, in case anything else JITs.
- `VLLM_USE_FLASHINFER_SAMPLER=0` removes the FlashInfer sampler JIT from the first request.
- Checked: under the runtime environment, loading the mxfp8 GEMM module takes 1.5 s and sparse_mla 0.0 s, with no compile.

**Guards not active on the fleet (recommendations, not applied; they change system settings on the boxes):**
- `dgx-anti-oom` is disabled on all four nodes. Its regex on the workers (`^comfy|^vllm_nemotron`) does not match `vllm_dsv41`.
- `vm.min_free_kbytes` is 44 MiB, so background reclaim has almost no room. Proposed: `vm.min_free_kbytes=1048576` and `vm.watermark_scale_factor=200`.
- The container's `--memory 112g` very likely does not bound host usage, because GPU allocations on GB10 are not charged to the cgroup (read from the driver source, about 85% confidence). To confirm, read `memory.current` after "Model loading took".
- Reddie runs 8 nfsd threads while serving three workers.
- The NFS mounts on Asusi and Bluey are not in fstab. After the reset they had to be remounted by hand.

## 2. Engram DISK mode: ranks 1-3 read rank 0's rows (correctness bug)

**Where:** `vllm/models/deepseek_v4_1/common/engram.py`, `DiskEngramTable` plus `ParallelEngramEmbedding._disk_lookup`. This is the Engram-on-disk patch (`DSV41_ENGRAM_DISK=1`).

**Why it is wrong:**
- The checkpoint stores each Engram layer's **full** table, all TP ranks' hash heads, as one tensor.
- The in-memory path narrows to this rank's rows: `loaded_weight.narrow(0, engram_vocab_start, part_rows)`.
- The disk path computes rank-local row ids (`rel = rows - vocab_start_idx`) but bases every read at the start of the full tensor. So rank k>0 reads rows `[0, part_rows)`, which belong to rank 0.

**Evidence:**
- Boot 4 logs: every rank reports `Engram DISK mode: layer 14 ... (off=672)`, the start of the full tensor. The scale tensor starts at byte 98,308,271,264, which is 4 × 96.0M rows × 256 B, so the file holds all four ranks' rows.
- Harness (`patch/engram-offset-fix/engram_offset_harness.py`): runs the real `DiskEngramTable` and `_disk_lookup` code against a synthetic 4-rank table laid out like the checkpoint:

```
deployed engram.py (md5 0ab3330a):  rank 0 MATCH, ranks 1-3 WRONG (reads rank-0 region)
fixed engram.py    (md5 dc4b2104):  ranks 0-3 MATCH
```

**Impact:**
- Every boot with the disk patch (boots 3 and 4) serves Engram embeddings for the wrong hash buckets on 3 of 4 ranks.
- The output still looks like text because the rows are real embedding rows, just the wrong ones. That is why smoke tests can pass.
- Throughput is unaffected; the same number of rows is read.

**Fix** (`patch/engram-offset-fix/engram-offset-fix.diff`, 61 lines): `DiskEngramTable` takes `row_start` and `num_rows`, and shifts the weight and scale base offsets by `row_start` rows. The call site passes `vocab_start_idx` and `part_num_embeddings`, mirroring the in-memory loader. Nothing else changes.

**Rollout:** the fix gets its own boot (boot 5, text-only, no speculation), following one change per boot. `launch/swap_engram.sh` installs it on all four nodes and keeps `engram.py.bak-pre-offsetfix`. It refuses to run while a container is up.

**Side effect to watch:** after the fix, the four ranks read four different quarters of each table. Page cache for Engram rows on Reddie (the NFS server) can therefore grow to about 4× what it was under the bug. It is reclaimable, but on GB10 unified memory it competes with the GPU pool.
