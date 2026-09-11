#!/bin/bash
# boot 7 (DSpark k=5, max context 1,048,576, eager) proof: pool, serving, one answer, DSpark speed, 32K needle
O=/var/tmp/boot-results/boot7; mkdir -p $O
{
echo "=== boot7 1M proof $(date -u +%FT%TZ) ==="
docker logs vllm_dsv41 2>&1 | grep -E "Available KV cache memory|GPU KV cache size|Maximum concurrency|Application startup complete" | sed -E 's/^.*\] //' | sort -u
curl -s -m 20 http://127.0.0.1:8000/v1/models | python3 -c "import json,sys;d=json.load(sys.stdin)['data'][0];print('serving',d['id'],'max_model_len',d.get('max_model_len'))"
python3 - <<'PY'
import json, time, urllib.request
U = "http://127.0.0.1:8000/v1/chat/completions"
def chat(p, mt):
    body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": p}], "max_tokens": mt,
            "temperature": 0, "chat_template_kwargs": {"thinking": False}}
    t = time.time(); r = json.load(urllib.request.urlopen(urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    return r, time.time() - t
r, dt = chat("In one sentence: what is 17 times 23, and why?", 60)
print("ANSWER:", repr(r["choices"][0]["message"]["content"]), "| %d tok in %.1fs" % (r["usage"]["completion_tokens"], dt))
for i in range(2):
    r, dt = chat("Count from 1 to 100, separated by spaces. Output only the numbers.", 256)
    ct = r["usage"]["completion_tokens"]
    print("DSpark eager count run%d: %d tok / %.1fs = %.1f tok/s" % (i + 1, ct, dt, ct / dt))
PY
echo "--- needle 32K ---"; python3 /root/v41needle.py --targets 32768 --out $O/needle-32k.json
echo "--- spec decode stats from the log ---"; docker logs vllm_dsv41 2>&1 | grep -iE "accept|spec" | grep -viE "speculative_config|SpeculativeConfig" | sed -E 's/^.*\] //' | tail -4 | cut -c1-200
echo "=== boot7 proof done $(date -u +%FT%TZ) ==="
} 2>&1 | tee $O/proof.txt
