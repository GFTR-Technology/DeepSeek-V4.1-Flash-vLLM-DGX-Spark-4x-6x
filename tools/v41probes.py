#!/usr/bin/env python3
"""Greedy probe set. Run twice against a lane; keep only prompts that reproduce
byte-identically, so the cross-lane comparison has a real noise floor.
Usage: probes.py <base_url> <model> <outfile>"""
import json, sys, time, urllib.request, hashlib

URL, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

PROBES = [
    ("count",   "Count from 1 to 60, comma separated, nothing else.", 200),
    ("math",    "What is 17 * 23 + 145? Reply with only the number.", 32),
    ("fact",    "Name the capital of Australia. One word only.", 16),
    ("logic",   "A bat and ball cost $1.10. The bat costs $1.00 more than the ball. "
                "How much is the ball? Answer with just the amount.", 64),
    ("list",    "List exactly the first 8 prime numbers, comma separated, nothing else.", 64),
    ("json",    'Return only this JSON with no prose: {"status":"ok","count":3}', 48),
    ("prose",   "In exactly two sentences, explain what a cache is.", 128),
    ("seq",     "Continue this sequence with the next 5 numbers only: 2, 4, 8, 16", 48),
]

def ask(prompt, max_tokens):
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0,
                       "stream": False}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=300))
    dt = time.time() - t0
    m = r["choices"][0]["message"]
    return {"text": m.get("content") or "", "reasoning": m.get("reasoning") or "",
            "finish": r["choices"][0]["finish_reason"],
            "tokens": r["usage"]["completion_tokens"], "secs": round(dt, 3)}

res = {}
for name, prompt, mt in PROBES:
    a = ask(prompt, mt); b = ask(prompt, mt)
    stable = a["text"] == b["text"]
    res[name] = {"prompt": prompt, "max_tokens": mt, "stable": stable, **a,
                 "sha": hashlib.sha256(a["text"].encode()).hexdigest()[:12]}
    print(f"{name:8} stable={str(stable):5} {a['tokens']:4d}tok "
          f"{a['tokens']/a['secs']:6.1f}tok/s  {a['text'][:52]!r}")

json.dump(res, open(OUT, "w"), indent=1)
n = sum(1 for v in res.values() if v["stable"])
print(f"\n{n}/{len(res)} probes reproduce byte-identically -> usable as reference")
print(f"written: {OUT}")
