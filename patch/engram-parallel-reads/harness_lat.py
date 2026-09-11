# Harness: exercise the REAL DiskEngramTable + ParallelEngramEmbedding._disk_lookup
# source from an engram.py (patched or proposed) against a synthetic safetensors
# shard laid out like the DSV4.1 checkpoint (one full-table tensor per layer,
# TP ranks own contiguous head ranges). No vllm import needed: the two code
# objects are extracted with ast and exec'd with small shims.
# usage: python3 engram_offset_harness.py <engram.py>
import time
LAT = 0.001
import ast, inspect, json, os, struct, sys, tempfile, types
from concurrent.futures import ThreadPoolExecutor
import torch

FADV_CALLS = []


class OSShim:
    """os with preadv/posix_fadvise shims (macOS py3.9 lacks them)."""
    POSIX_FADV_RANDOM = 1
    POSIX_FADV_DONTNEED = 4

    def __getattr__(self, k):
        return getattr(os, k)

    @staticmethod
    def preadv(fd, bufs, off, *flags):
        total = 0
        for b in bufs:
            time.sleep(LAT); data = os.pread(fd, len(b), off + total)
            b[: len(data)] = data
            total += len(data)
            if len(data) < len(b):
                break
        return total

    @staticmethod
    def posix_fadvise(fd, off, ln, adv):
        FADV_CALLS.append((fd, off, ln, adv))


src_path = sys.argv[1]
tree = ast.parse(open(src_path).read())
ns = {"_kai_os": OSShim(), "_kai_json": json, "_kai_struct": struct,
      "_KaiPool": ThreadPoolExecutor, "torch": torch,
      "logger": types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)}
import __future__
_FLAGS = __future__.annotations.compiler_flag  # PEP 604 hints on local py3.9
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == "DiskEngramTable":
        exec(compile(ast.Module(body=[node], type_ignores=[]), src_path, "exec", flags=_FLAGS), ns)
    if isinstance(node, ast.ClassDef) and node.name == "ParallelEngramEmbedding":
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "_disk_lookup":
                exec(compile(ast.Module(body=[item], type_ignores=[]), src_path, "exec", flags=_FLAGS), ns)
DiskEngramTable, disk_lookup = ns["DiskEngramTable"], ns["_disk_lookup"]

# ---- synthetic checkpoint: 24 hash cols (3 n-gram orders x 8 heads), TP4 -> 6/rank
S, NCOLS, TP, dim, sb = 1000, 24, 4, 256, 8
L = NCOLS // TP
N = S * NCOLS  # full table rows
torch.manual_seed(0)
w = torch.randint(0, 256, (N, dim), dtype=torch.uint8)
s = torch.randint(118, 136, (N, sb), dtype=torch.uint8)
d = tempfile.mkdtemp()
hdr = {"layers.1.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [N, dim], "data_offsets": [0, N * dim]},
       "layers.1.engram.embed.scale": {"dtype": "F8_E8M0", "shape": [N, sb], "data_offsets": [N * dim, N * dim + N * sb]},
       "__metadata__": {"format": "pt"}}
hb = json.dumps(hdr).encode()
hb += b" " * ((8 - len(hb) % 8) % 8)
with open(d + "/model-00047-of-00048.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(hb))); f.write(hb); f.write(w.numpy().tobytes()); f.write(s.numpy().tobytes())
json.dump({"weight_map": {k: "model-00047-of-00048.safetensors" for k in hdr if k != "__metadata__"}},
          open(d + "/model.safetensors.index.json", "w"))


def ref_rows(idx):
    vals = w[idx].view(torch.float8_e4m3fn).to(torch.float32).view(-1, sb, 32)
    scale = (s[idx].to(torch.int32) << 23).view(torch.float32)
    return (vals * scale[:, :, None]).reshape(-1, dim).to(torch.bfloat16)


sig = inspect.signature(DiskEngramTable.__init__)
T = int(__import__("os").environ.get("T_TOKENS", "1"))
all_ok = True
for rank in range(TP):
    vs, ve = rank * L * S, (rank + 1) * L * S
    kw = {}
    if "row_start" in sig.parameters:
        kw["row_start"] = vs
    if "num_rows" in sig.parameters:
        kw["num_rows"] = ve - vs
    table = DiskEngramTable(d, 1, dim, 32, **kw)
    table.pool = ThreadPoolExecutor(max_workers=4)
    fake = types.SimpleNamespace(part_n_hash_cols=L, head_start=rank * L, n_hash_cols=NCOLS,
                                 vocab_start_idx=vs, vocab_end_idx=ve, dim=dim, disk=table)
    # global row ids, as produced by _hash_ids_kernel: head h -> [h*S, (h+1)*S)
    ids = torch.stack([torch.randint(h * S, (h + 1) * S, (T,)) for h in range(NCOLS)], dim=1)
    out = torch.empty(T, L, dim, dtype=torch.bfloat16)
    FADV_CALLS.clear()
    t0 = time.time(); disk_lookup(fake, ids, out); dt = time.time() - t0
    expect = ref_rows(ids[:, rank * L:(rank + 1) * L].reshape(-1)).view(T, L, dim)
    ok = torch.allclose(out.float(), expect.float(), equal_nan=True)
    all_ok &= ok
    print(f"rank {rank}: vocab_start={vs:6d} rows {'MATCH' if ok else 'WRONG (reads rank-0 region)'}"
          f" | lookup {dt*1000:.1f} ms for T={T} ({T*L} rows)")
print("ALL RANKS CORRECT" if all_ok else "BUG: ranks with vocab_start>0 read the wrong rows")
