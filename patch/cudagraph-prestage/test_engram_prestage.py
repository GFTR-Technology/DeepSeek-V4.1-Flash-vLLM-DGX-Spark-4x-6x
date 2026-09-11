#!/usr/bin/env python3
"""Offline (CPU-only, no vLLM import) test of the Engram prestage patch.

Extracts the REAL code from the old (boot-6, md5 dc4b2104) and patched engram.py
with `ast` and checks, on a fake 2-layer safetensors checkpoint:
  1. new gather_dequant_many == old DiskEngramTable.gather_dequant, bit-exact
     (random rows, duplicates, unowned rows, sizes 1..49152)
  2. new _disk_lookup == old _disk_lookup, bit-exact, incl. padded heads (TP=5)
  3. stager CPU half (both layers in one batch via disk_rel_owned +
     gather_dequant_many) == old per-layer _disk_lookup
  4. read scheduling: row coverage for chunk boundaries, single-row path
  5. timing (informational): old serial vs new batched, page-cached file
usage: python3 test_engram_prestage.py OLD_engram.py NEW_engram.py
"""
import ast
import json
import logging
import os
import struct
import sys
import tempfile
import time
import types
from concurrent.futures import ThreadPoolExecutor

import torch

OLD, NEW = sys.argv[1], sys.argv[2]
KEEP_FUNCS = {"_kai_pool", "_kai_pread_rows", "_kai_parallel_read", "gather_dequant_many"}
KEEP_VARS = {"_KAI_THREADS", "_KAI_CHUNK", "_KAI_POOL"}
METHODS = {"disk_rel_owned", "_disk_lookup"}


def load(path):
    tree = ast.parse(open(path).read())
    body, methods = [], {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "DiskEngramTable":
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in KEEP_FUNCS:
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = [t.id for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
                     if isinstance(t, ast.Name)]
            if set(names) & KEEP_VARS:
                body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "ParallelEngramEmbedding":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in METHODS:
                    methods[item.name] = item
    mod = ast.Module(body=body + list(methods.values()), type_ignores=[])
    ns = {"torch": torch, "_kai_os": os, "_kai_json": json, "_kai_struct": struct,
          "_KaiPool": ThreadPoolExecutor, "logger": logging.getLogger("t")}
    exec(compile(ast.fix_missing_locations(mod), path, "exec",
                 flags=__import__("__future__").annotations.compiler_flag), ns)
    return ns


old, new = load(OLD), load(NEW)
assert "gather_dequant_many" in new and "disk_rel_owned" in new, "patched symbols missing"

# ---- fake checkpoint: layers 1 and 14, fp8 rows + ue8m0 scales -------------------
d = tempfile.mkdtemp()
R, dim, sb = 200000, 256, 8
g = torch.Generator().manual_seed(0)
hdr, blobs, off = {}, [], 0
for L in (1, 14):
    w = torch.randint(0, 256, (R, dim), dtype=torch.uint8, generator=g)
    s = torch.randint(118, 136, (R, sb), dtype=torch.uint8, generator=g)
    for name, t, dt in ((f"layers.{L}.engram.embed.weight", w, "F8_E4M3"),
                        (f"layers.{L}.engram.embed.scale", s, "F8_E8M0")):
        b = t.numpy().tobytes()
        hdr[name] = {"dtype": dt, "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
hb = json.dumps(hdr).encode()
hb += b" " * ((8 - len(hb) % 8) % 8)
with open(d + "/model-00001-of-00001.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(hb)))
    f.write(hb)
    for b in blobs:
        f.write(b)
json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in hdr}},
          open(d + "/model.safetensors.index.json", "w"))

ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# row_start mimics a TP rank's slice (offset-fix path)
T_old = {L: old["DiskEngramTable"](d, L, dim, 32, row_start=1000, num_rows=R - 1000) for L in (1, 14)}
T_new = {L: new["DiskEngramTable"](d, L, dim, 32, row_start=1000, num_rows=R - 1000) for L in (1, 14)}

print("1. gather_dequant_many vs old gather_dequant (bit-exact)")
for n in (1, 2, 6, 24, 33, 144, 4096, 49152):
    rel = torch.randint(0, R - 1000, (n,), dtype=torch.int64, generator=g)
    if n > 4:
        rel[: n // 4] = rel[0]  # duplicates -> dedupe path
    owned = torch.ones(n, dtype=torch.bool)
    owned[::7] = False
    a = T_old[1].gather_dequant(rel, owned)
    b1, b14 = new["gather_dequant_many"]([(T_new[1], rel, owned), (T_new[14], rel, owned)])
    c14 = T_old[14].gather_dequant(rel, owned)
    check(f"n={n:5d} layer1 {tuple(b1.shape)} equal", torch.equal(a.view(torch.int16), b1.view(torch.int16)))
    check(f"n={n:5d} layer14 equal", torch.equal(c14.view(torch.int16), b14.view(torch.int16)))
    check(f"n={n:5d} single-table wrapper equal",
          torch.equal(a.view(torch.int16), T_new[1].gather_dequant(rel, owned).view(torch.int16)))

print("2. _disk_lookup old vs new (bit-exact), TP=4 (6 heads) and TP=5 (padded heads)")
n_hash_cols = 24
for tp, rank in ((4, 1), (5, 4)):
    L_ = -(-n_hash_cols // tp)
    head_start = rank * L_
    def emb(ns, table):
        e = types.SimpleNamespace(part_n_hash_cols=L_, head_start=head_start, n_hash_cols=n_hash_cols,
                                  vocab_start_idx=1000, vocab_end_idx=R, dim=dim, disk=table)
        for m in METHODS & set(ns):
            setattr(e, m, types.MethodType(ns[m], e))
        return e
    eo, en = emb(old, T_old[1]), emb(new, T_new[1])
    T = 37
    idx = torch.randint(0, R + 500, (T, n_hash_cols), dtype=torch.int64, generator=g)  # some foreign ids
    out_o = torch.full((T + 3, L_, dim), 7.0, dtype=torch.bfloat16)
    out_n = out_o.clone()
    eo._disk_lookup(idx, out_o)
    en._disk_lookup(idx, out_n)
    check(f"tp={tp} rank={rank} heads [{head_start},{min(head_start + L_, n_hash_cols)}) equal",
          torch.equal(out_o.view(torch.int16), out_n.view(torch.int16)))

print("3. stager CPU half (both layers, one batch) vs old per-layer lookup")
L_ = 6
T = 6 * 8  # C8 DSpark verify batch
hashes = torch.randint(1000, R, (T, 2, n_hash_cols), dtype=torch.int32, generator=g)
e1 = emb(new, T_new[1]); e1.part_n_hash_cols = 6; e1.head_start = 6
e14 = emb(new, T_new[14]); e14.part_n_hash_cols = 6; e14.head_start = 6
host = hashes[:, :, 6:12].contiguous()  # what the D2H copies (local heads only)
reqs = []
for li, e in ((0, e1), (1, e14)):
    rel, owned = e.disk_rel_owned(host[:, li, :].to(torch.int64))
    reqs.append((e.disk, rel, owned))
staged = [r.view(T, 6, dim) for r in new["gather_dequant_many"](reqs)]
for li, tab in ((0, T_old[1]), (1, T_old[14])):
    eo = emb(old, tab); eo.part_n_hash_cols = 6; eo.head_start = 6
    ref = torch.zeros((T, 6, dim), dtype=torch.bfloat16)
    eo._disk_lookup(hashes[:, li, :], ref)
    check(f"layer_hash_index={li} staged rows == in-forward rows",
          torch.equal(ref.view(torch.int16), staged[li].view(torch.int16)))

print("4. read scheduling coverage")
buf_rows = 1000
for total, threads in ((1, 32), (24, 32), (33, 32), (144, 32), (5000, 32)):
    new["_KAI_THREADS"] = threads
    seen = []
    orig = new["_kai_pread_rows"]
    def spy(fd, base, rel, lo, hi, rb, buf, _o=orig):
        seen.append((id(rel), lo, hi))
        _o(fd, base, rel, lo, hi, rb, buf)
    new["_kai_pread_rows"] = spy
    rel = torch.randint(0, R - 1000, (total,), generator=g).tolist()
    w = torch.empty((total, dim), dtype=torch.uint8)
    s = torch.empty((total, sb), dtype=torch.uint8)
    new["_kai_parallel_read"](T_new[1].read_jobs(rel, w, s))
    new["_kai_pread_rows"] = orig
    cover = sorted((lo, hi) for _, lo, hi in seen)
    rows_read = sum(hi - lo for lo, hi in cover)
    check(f"total={total:5d}: {len(seen)} tasks, rows read {rows_read} == 2*{total}", rows_read == 2 * total)

print("5. timing (informational; file is page-cached, so this is Python overhead only)")
new["_KAI_THREADS"] = 32
for n in (24, 144, 12288):
    rel = torch.randint(0, R - 1000, (n,), dtype=torch.int64, generator=g)
    owned = torch.ones(n, dtype=torch.bool)
    t0 = time.perf_counter(); T_old[1].gather_dequant(rel, owned); t_old = time.perf_counter() - t0
    t0 = time.perf_counter(); new["gather_dequant_many"]([(T_new[1], rel, owned)]); t_new = time.perf_counter() - t0
    print(f"  rows={n:6d}: old {t_old * 1e3:8.2f} ms   new {t_new * 1e3:8.2f} ms")

print("RESULT:", "ALL PASS" if ok else "FAILURES")
sys.exit(0 if ok else 1)
