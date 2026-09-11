#!/usr/bin/env python3
"""4-Spark collective check: do the cross-node all-reduces flip between fast and slow on their own?
Same image, same NCCL env and rank order as vLLM, through vLLM's own PyNccl wrapper when it loads.
1) per-op latency at 8 KB-1 MB.
2) a decode-shaped step: 88 all-reduces of 60 KB (6 tokens x 5120 x bf16), each after a ~0.6 ms GPU
   sleep standing in for layer compute; eager and CUDA-graphed (like vLLM), plus back-to-back.
Every loop has a fixed count so all ranks stay in lockstep."""
import os
import socket
import time

import torch
import torch.distributed as dist

rank, ws = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
H = os.environ.get("NODE", socket.gethostname())
torch.cuda.set_device(0)
dist.init_process_group("gloo", rank=rank, world_size=ws)
try:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    comm = PyNcclCommunicator(dist.group.WORLD, device=torch.device("cuda:0"))

    def ar(x):
        return comm.all_reduce(x)

    mode = "pynccl (vLLM's NCCL wrapper)"
except Exception as e:  # noqa: BLE001
    ng = dist.new_group(backend="nccl")

    def ar(x):
        dist.all_reduce(x, group=ng)
        return x

    mode = f"torch nccl (pynccl failed: {type(e).__name__}: {str(e)[:120]})"


def ev():
    return torch.cuda.Event(enable_timing=True)


def q(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * p))]


def summ(name, v, times=None, base=None):
    lo = q(v, 0.1)
    slow = sum(1 for x in v if x > 1.25 * lo) / len(v)
    line = (f"{H:7} r{rank} {name:36} n={len(v):5d} p10={lo:8.2f} p50={q(v, .5):8.2f} p90={q(v, .9):8.2f} "
            f"max={max(v):8.2f} slow(>1.25x p10)={100 * slow:3.0f}%")
    if base is not None:
        line += f" | collective cost/step p50={q(v, .5) - base:.2f} ms"
    if times:
        b = {}
        for t, x in zip(times, v):
            b.setdefault(int(t), []).append(x)
        line += " | 1s medians: " + " ".join(f"{q(bb, .5):.1f}" for _, bb in sorted(b.items()))
    print(line, flush=True)


if rank == 0:
    print(f"mode: {mode}", flush=True)
w = torch.zeros(30720, dtype=torch.bfloat16, device="cuda")
for _ in range(50):
    ar(w)
torch.cuda.synchronize()

for nb in (8192, 32768, 61440, 131072, 262144, 1048576):
    t = torch.zeros(nb // 2, dtype=torch.bfloat16, device="cuda")
    lat = []
    for _ in range(300):
        s, e = ev(), ev()
        s.record()
        ar(t)
        e.record()
        e.synchronize()
        lat.append(s.elapsed_time(e) * 1000)
    summ(f"per-op us, {nb // 1024}KB", lat)

GAP = int(os.environ.get("GAP_CYC", "1300000"))  # ~0.6 ms at 2.19 GHz
t = torch.zeros(30720, dtype=torch.bfloat16, device="cuda")


def step(gap, coll=True):
    for _ in range(88):
        if gap:
            torch.cuda._sleep(gap)
        if coll:
            ar(t)


def time_it(fn, n):
    v, ts, t0 = [], [], time.time()
    for _ in range(n):
        s, e = ev(), ev()
        s.record()
        fn()
        e.record()
        e.synchronize()
        v.append(s.elapsed_time(e))
        ts.append(time.time() - t0)
    return v, ts


v, ts = time_it(lambda: step(GAP, coll=False), 40)
base = q(v, 0.5)
summ("eager step ms, gaps only", v)
v, ts = time_it(lambda: step(GAP), 150)
summ("eager step ms, 88 AR + gaps", v, ts, base)


def capture(gap):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            step(gap)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step(gap)
    torch.cuda.synchronize()
    return g


try:
    g1 = capture(GAP)
    v, ts = time_it(g1.replay, 300)
    summ("graph step ms, 88 AR + gaps", v, ts, base)
    g2 = capture(0)
    v, ts = time_it(g2.replay, 2000)
    summ("graph step ms, 88 AR back-to-back", v, ts)
except Exception as e:  # noqa: BLE001
    print(f"{H} r{rank} graph capture failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
dist.barrier()
print(f"{H} r{rank} done", flush=True)
