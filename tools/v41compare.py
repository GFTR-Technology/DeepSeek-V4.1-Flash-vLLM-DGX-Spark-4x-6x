#!/usr/bin/env python3
"""Compare a lane against a captured reference probe set.
Only probes marked stable in the reference count toward the verdict --
an unstable probe cannot distinguish a real regression from lane noise.
Usage: compare.py <ref.json> <base_url> <model>"""
import json, sys, time, urllib.request

ref = json.load(open(sys.argv[1]))
URL, MODEL = sys.argv[2], sys.argv[3]

def ask(prompt, mt):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": mt, "temperature": 0, "stream": False}).encode()
    r = json.load(urllib.request.urlopen(
        urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=300))
    m = r["choices"][0]["message"]
    return (m.get("content") or ""), r["usage"]["completion_tokens"], r["choices"][0]["finish_reason"]

match = graded = 0
print(f"{'probe':8} {'verdict':9} {'ref':<34} {'new':<34}")
print("-" * 92)
for name, v in ref.items():
    got, ntok, fin = ask(v["prompt"], v["max_tokens"])
    same = got == v["text"]
    if v["stable"]:
        graded += 1; match += same
        verdict = "MATCH" if same else "DIFFER"
    else:
        verdict = "unstable"
    print(f"{name:8} {verdict:9} {v['text'][:32]!r:<34} {got[:32]!r:<34}")

print("-" * 92)
print(f"{match}/{graded} stable probes byte-identical to the boot-3 reference")
if match < graded:
    print("NOT equivalent -- the missing SwiGLU clamp changes output. Treat any")
    print("speed number from this lane as coming from a different model.")
