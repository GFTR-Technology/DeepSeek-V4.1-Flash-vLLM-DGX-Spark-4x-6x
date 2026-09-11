#!/usr/bin/env python3
"""Passive speed-lever sampler for a live vLLM node. Reads /proc only; sends no requests.

Every 0.5 s it reads each vLLM thread's CPU ticks and the core it last ran on
(X925 big: 5-9,15-19; A725 little: 0-4,10-14). Every 10 s it writes one line:
NFS READ stats for /mnt/reddie-models (ops, MB, avg RTT, avg exec), nfsd READ ops and
NVMe MB read (server side), page cache GB, and the 4 busiest threads (ticks, % on little cores).
Usage: lever_sampler.py SECONDS OUTFILE
"""
import os
import sys
import time

DUR, OUT = float(sys.argv[1]), sys.argv[2]
BIG = set(range(5, 10)) | set(range(15, 20))
MNT = "/mnt/reddie-models"


def vllm_pids():
    pids = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/cmdline", "rb") as f:
                c = f.read(300)
        except OSError:
            continue
        if b"VLLM" in c or b"vllm" in c:
            pids.append(p)
    return pids


def thread_stats(pids):
    out = {}
    for p in pids:
        try:
            tids = os.listdir(f"/proc/{p}/task")
        except OSError:
            continue
        for t in tids:
            try:
                with open(f"/proc/{p}/task/{t}/stat") as f:
                    s = f.read()
            except OSError:
                continue
            r = s.rindex(")")
            fl = s[r + 2:].split()
            out[t] = (p, s[s.index("(") + 1:r], int(fl[11]) + int(fl[12]), int(fl[36]))
    return out


def nfs_read():
    try:
        with open("/proc/self/mountstats") as f:
            txt = f.read()
    except OSError:
        return None
    i = txt.find(f"mounted on {MNT} ")
    if i < 0:
        return None
    end = txt.find("\ndevice ", i)
    for line in txt[i:end if end > 0 else len(txt)].splitlines():
        s = line.strip()
        if s.startswith("READ:"):
            v = s.split()[1:]
            return int(v[0]), int(v[4]), int(v[6]), int(v[7])  # ops, bytes recv, rtt ms, exec ms (cumulative)
    return None


def cached_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("Cached:"):
                return int(line.split()[1]) / 1048576
    return 0.0


def nfsd_reads():
    try:
        with open("/proc/net/rpc/nfsd") as f:
            for line in f:
                if line.startswith("proc3 "):
                    return int(line.split()[8])
    except OSError:
        pass
    return None


def nvme_sectors_read():
    tot = 0
    with open("/proc/diskstats") as f:
        for line in f:
            v = line.split()
            if v[2].startswith("nvme") and "p" not in v[2][5:]:
                tot += int(v[5])
    return tot


t0 = time.time()
pids, scan_t = vllm_pids(), t0
prev = thread_stats(pids)
win, tot = {}, {}
nfs0, nfsd0, sec0 = nfs_read(), nfsd_reads(), nvme_sectors_read()
next_w = t0 + 10
with open(OUT, "w", buffering=1) as log:
    log.write(f"# start {time.strftime('%H:%M:%S', time.gmtime(t0))} UTC, {len(pids)} vllm pids, {len(prev)} threads\n")
    while time.time() - t0 < DUR:
        time.sleep(0.5)
        now = time.time()
        if now - scan_t > 30:
            pids, scan_t = vllm_pids(), now
        cur = thread_stats(pids)
        for t, (p, name, ticks, cpu) in cur.items():
            if t not in prev:
                continue
            d = ticks - prev[t][2]
            if d <= 0:
                continue
            little = cpu not in BIG
            w = win.setdefault(t, [name, 0, 0])
            w[1] += d
            w[2] += d if little else 0
            a = tot.setdefault(t, [p, name, 0, 0, 0, None])
            a[2] += d
            a[3] += d if little else 0
            if a[5] is not None and a[5] != little:
                a[4] += 1
            a[5] = little
        prev = cur
        if now >= next_w:
            next_w += 10
            top = sorted(win.items(), key=lambda x: -x[1][1])[:4]
            thr = " ".join(f"{n}[{t}]:{tk}t/{100 * lt // max(tk, 1)}%L" for t, (n, tk, lt) in top if tk >= 5)
            n1, nfsd1, sec1 = nfs_read(), nfsd_reads(), nvme_sectors_read()
            nf = ""
            if n1 and nfs0:
                dops = n1[0] - nfs0[0]
                nf = (f"nfsREAD {dops} ops {(n1[1] - nfs0[1]) / 1e6:.1f}MB "
                      f"rtt {(n1[2] - nfs0[2]) / max(dops, 1):.2f}ms exec {(n1[3] - nfs0[3]) / max(dops, 1):.2f}ms ")
            sd = f"nfsd {nfsd1 - nfsd0} " if nfsd1 is not None and nfsd0 is not None else ""
            log.write(f"{time.strftime('%H:%M:%S', time.gmtime(now))} {nf}{sd}disk {(sec1 - sec0) * 512 / 1e6:.1f}MB "
                      f"cache {cached_gb():.1f}G | {thr}\n")
            nfs0, nfsd0, sec0 = n1, nfsd1, sec1
            win = {}
    log.write("# summary: busiest threads over the run (ticks, % on little cores, big<->little switches seen)\n")
    for t, (p, name, tk, lt, sw, _) in sorted(tot.items(), key=lambda x: -x[1][2])[:15]:
        log.write(f"# {p}/{t} {name:<16} {tk:>7}t {100 * lt / max(tk, 1):5.1f}%L sw {sw}\n")
