#!/usr/bin/env python3
"""Smoke + throughput test for DeepSeek-V4.1-Flash TP4 on :8000. Single stream, temperature 0, thinking off."""
import json, time, urllib.request, sys
BASE="http://127.0.0.1:8000/v1"
def post(path, body, timeout=600):
    req=urllib.request.Request(BASE+path, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    t=time.time(); r=json.load(urllib.request.urlopen(req, timeout=timeout)); return r, time.time()-t
models=[m["id"] for m in json.load(urllib.request.urlopen(BASE+"/models", timeout=20))["data"]]
print("served:", models)
PROMPTS={
 "count":  ("Count from 1 to 100, separated by spaces. Output only the numbers.", 400),
 "prose":  ("Write a 350-word essay on why local inference hardware matters for small studios. No headings.", 450),
 "code":   ("Write a Python function that parses an ISO-8601 duration string like PT1H30M into seconds, with a docstring and 5 unit tests using pytest.", 450),
}
def chat(model, prompt, max_tokens):
    body={"model":model,"messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0,
          "chat_template_kwargs":{"thinking":False}}
    r,dt=post("/chat/completions", body)
    u=r.get("usage",{}); ct=u.get("completion_tokens",0); txt=r["choices"][0]["message"].get("content") or ""
    return ct, dt, txt
# 1) both names answer with a real completion
for m in models:
    ct,dt,txt=chat(m,"Say hello in one short sentence and tell me your model family.",60)
    print(f"[{m}] {ct} tok in {dt:.2f}s :: {txt.strip()[:160]!r}")
# 2) throughput, two runs each on the primary name, keep the best (second run is warm)
m=models[0]; res={}
for k,(p,mt) in PROMPTS.items():
    best=0; sample=""
    for run in range(2):
        ct,dt,txt=chat(m,p,mt); tps=ct/dt if dt else 0
        if tps>best: best,sample=tps,txt
        print(f"  {k} run{run+1}: {ct} tok / {dt:.2f}s = {tps:.1f} tok/s")
    res[k]=round(best,1)
    if k=="count":
        nums=[int(x) for x in sample.replace(",", " ").split() if x.isdigit()]
        ok = nums[:100]==list(range(1,101))
        print(f"  count correctness: {'OK 1..100' if ok else 'MISMATCH'} ({len(nums)} numbers)")
print("RESULT", json.dumps(res))

# --- corruption check (vLLM #54150 symptom on the old ModelOpt export: garbage/non-language tokens mid-generation)
import re
def junk_ratio(s):
    toks=s.split(); 
    if not toks: return 1.0
    bad=sum(1 for t in toks if re.search(r"[^\x09\x0a\x0d\x20-\x7e‘’“”–—…éèêàüöäñ]", t))
    return bad/len(toks)
m=models[0]
ct,dt,txt=chat(m,"Write a 600-word explainer on how RoCE differs from InfiniBand for a small GPU cluster. Plain prose, no lists.",800)
rep=len(re.findall(r"\b(\w+ \w+ \w+) \1\b", txt))
print(f"corruption check: {ct} tok, junk-token ratio {junk_ratio(txt):.3%}, repeated-trigram runs {rep}, tail: {txt.strip()[-160:]!r}")
