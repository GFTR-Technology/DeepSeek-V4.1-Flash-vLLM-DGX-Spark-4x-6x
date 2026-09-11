#!/usr/bin/env python3
"""Single-GPU check: does this GB10 flip between a fast and a slow memory mode on its own?
Phase 1 (20 s): continuous 256 MB copies and a decode-shaped GEMV (6 x 5120 @ 5120 x 16384 bf16).
Phase 2 (20 s): the same GEMV in ~70 ms bursts with 30 ms idle, like a decode step.
Prints p10/p50/p90 GB/s, share of slow samples, and 2 s medians over time."""
import os
import socket
import time

import torch

H = os.environ.get("NODE", socket.gethostname())
torch.cuda.set_device(0)
x = torch.empty(128 * 2**20, dtype=torch.bfloat16, device="cuda")
y = torch.empty_like(x)
W = torch.randn(5120, 16384, dtype=torch.bfloat16, device="cuda")
a = torch.randn(6, 5120, dtype=torch.bfloat16, device="cuda")


def timed(fn, n):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / n  # ms per call


def q(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * p))]


for _ in range(20):
    y.copy_(x)
    a @ W
torch.cuda.synchronize()
R = {"copy GB/s": [], "gemv GB/s": [], "gemv duty GB/s": []}
t0 = time.time()
while time.time() - t0 < 20:
    t = time.time() - t0
    R["copy GB/s"].append((t, 4 * x.numel() / timed(lambda: y.copy_(x), 4) / 1e6))
    R["gemv GB/s"].append((t, 2 * W.numel() / timed(lambda: a @ W, 8) / 1e6))
t1 = time.time()
while time.time() - t1 < 20:
    t = time.time() - t0
    R["gemv duty GB/s"].append((t, 2 * W.numel() / timed(lambda: a @ W, 90) / 1e6))
    time.sleep(0.03)
for k, v in R.items():
    vals = [b for _, b in v]
    hi = q(vals, 0.9)
    slow = sum(1 for b in vals if b < 0.8 * hi) / len(vals)
    buckets = {}
    for t, b in v:
        buckets.setdefault(int(t // 2), []).append(b)
    series = " ".join(f"{q(bb, 0.5):.0f}" for _, bb in sorted(buckets.items()))
    print(f"{H:7} {k:15} n={len(vals):5d} p10={q(vals, .1):4.0f} p50={q(vals, .5):4.0f} p90={hi:4.0f} "
          f"min={min(vals):4.0f} slow(<80% of p90)={100 * slow:3.0f}% | 2s medians: {series}", flush=True)
