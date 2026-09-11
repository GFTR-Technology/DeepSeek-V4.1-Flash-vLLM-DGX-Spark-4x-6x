#!/usr/bin/env python3
"""Summarize flip-<NODE>.txt probes run at the same time on several Sparks.
Per node: 1 char per second (F fast state, s slow state, - copy block, which cannot tell), fast/slow
seconds, median SM clock and power in each state. Then, on shared epoch seconds, how often ALL nodes
were fast at once (what a TP4 step needs). usage: flipsum.py DIR"""
import collections
import glob
import os
import sys


def med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else float("nan")


D = sys.argv[1]
state, starts = {}, []
for f in sorted(x for x in glob.glob(os.path.join(D, "flip-*.txt")) if not x.endswith("flip-summary.txt")):
    node = os.path.basename(f)[5:-4]
    S, N = [], []
    for line in open(f):
        p = line.split()
        if p[:1] == ["S"] and len(p) >= 4:
            S.append((float(p[1]), p[2], float(p[3])))
        elif p[:1] == ["N"] and len(p) >= 3:
            try:
                sm, pw = (float(x) for x in " ".join(p[2:]).split(","))
                N.append((float(p[1]), sm, pw))
            except ValueError:
                pass
    if not S:
        print(f"{node:7} no samples ({open(f).read()[-300:].strip()!r})")
        continue
    per = collections.defaultdict(list)
    for t, k, v in S:
        per[int(t)].append((k, v))
    st = {}
    for sec, kv in per.items():
        g = [v for k, v in kv if k.startswith("gemv")]
        m = [v for k, v in kv if k == "mm_duty"]
        st[sec] = ("F" if med(g) > 150 else "s") if g else (("F" if med(m) > 70 else "s") if m else "-")
    state[node] = st
    t0, t1 = min(st), max(st)
    tl = "".join(st.get(s, " ") for s in range(t0, t1 + 1))
    starts.append(t0)
    smp = collections.defaultdict(list)
    for t, sm, pw in N:
        smp[int(t)].append((sm, pw))
    by = {c: [x for s in st if st[s] == c for x in smp.get(s, [])] for c in "Fs"}
    clk = " ".join(f"{c}: {med([x[0] for x in by[c]]):.0f} MHz {med([x[1] for x in by[c]]):.1f} W" for c in "Fs" if by[c])
    print(f"{node:7} {tl}  fast {tl.count('F')}s slow {tl.count('s')}s | {clk}")
if len(state) > 1:
    common = set.intersection(*(set(s for s, c in st.items() if c != "-") for st in state.values()))
    allf = sum(1 for s in common if all(st[s] == "F" for st in state.values()))
    anys = sum(1 for s in common if any(st[s] == "s" for st in state.values()))
    print(f"shared seconds {len(common)}: all {len(state)} fast {allf}s, at least one slow {anys}s "
          f"(start skew {max(starts) - min(starts)}s)")
print("(F fast state, s slow state, - copy block)")
