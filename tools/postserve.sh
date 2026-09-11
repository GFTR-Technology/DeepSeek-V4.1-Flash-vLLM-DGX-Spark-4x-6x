#!/bin/bash
# Post-serve checks for V4.1 boots (root on Reddie).  usage: postserve.sh <label>
L=${1:?label}; B=http://127.0.0.1:8000/v1; M=deepseek-v4.1-flash; O=/var/tmp/boot-results/$L; mkdir -p "$O"
REF_LABEL=${REF_LABEL:-boot6}; REF=/var/tmp/ref-v41-$REF_LABEL.json
echo "=== $L post-serve $(date -u +%FT%TZ) ==="
curl -s -m 20 $B/models | python3 -c "import json,sys;d=json.load(sys.stdin)['data'][0];print('serving',d['id'],'max_model_len',d.get('max_model_len'))"
for i in 1 2 3; do curl -s -m 600 $B/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"warm $i: say ok\"}],\"max_tokens\":8}" >/dev/null; done
echo "--- smoke (Kai) ---"; python3 /home/tonyspark2/dsv41_smoke.py 2>&1 | tail -16 | tee "$O/smoke.txt"
if [ "$L" = "$REF_LABEL" ] || [ ! -f "$REF" ]; then echo "--- capture greedy reference ($REF) ---"; python3 /root/v41probes.py $B/chat/completions $M "$REF" 2>&1 | tail -3; fi
for R in /var/tmp/ref-v41-*.json; do r=$(basename "$R" .json); r=${r#ref-v41-}; [ -f "$R" ] && [ "$r" != "$L" ] || continue
  echo "--- quality vs $r reference ---"; python3 /root/v41compare.py "$R" $B/chat/completions $M 2>&1 | tail -4 | tee "$O/quality-vs-$r.txt"; done
if [ "${GATE:-1}" = 1 ]; then echo "--- garble gate ---"; python3 /root/v41gate.py $B $M 2>&1 | tail -3 | tee "$O/gate.txt"; fi
if [ "${VISION:-0}" = 1 ]; then echo "--- vision request (solid red 64x64 PNG) ---"
python3 - <<'PY'
import base64, json, struct, zlib, urllib.request
w = h = 64
raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
body = {"model": "deepseek-v4.1-flash", "max_tokens": 20, "temperature": 0, "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What single color fills this image? Answer with one word."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]}]}
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
print("VISION answer:", repr(r["choices"][0]["message"].get("content")), "| prompt_tokens:", r["usage"]["prompt_tokens"])
PY
fi
if [ "${BENCH:-1}" = 1 ]; then echo "--- fixed-prompt bench v1: C1-C6 + cold prefill ---"
python3 /root/v41bench.py --base $B --model $M --label "$L" --out "$O" --notes "${BENCH_NOTES:-}" 2>&1 | tee "$O/bench.txt" | tail -40; fi
echo "=== $L post-serve done $(date -u +%FT%TZ) ==="
