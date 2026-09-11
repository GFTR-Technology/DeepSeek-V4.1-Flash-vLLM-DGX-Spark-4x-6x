#!/bin/bash
# clockcheck.sh (root on Reddie): sample SM clock / power / util on all 4 GPUs every 0.5 s while one
# count-to-100 generation runs, then print per-node min / median / max.
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10"
D=/tmp/clockcheck; rm -rf $D; mkdir -p $D
Q="nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu --format=csv,noheader,nounits -lms 500"
( timeout 14 $Q > $D/REDDIE.csv 2>&1 ) &
for p in SPARK4:tonyspark4@192.168.192.4 ASUSI:tonyspark3@192.168.192.3 BLUEY:tonyspark1@192.168.192.1; do
  ( $J ${p#*:} "timeout 14 $Q" > $D/${p%%:*}.csv 2>&1 ) &
done
sleep 1
python3 - <<'PY' &
import json, time, urllib.request
body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "Count from 1 to 300, separated by spaces. Output only the numbers."}], "max_tokens": 900, "temperature": 0}
t = time.time(); r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
ct = r["usage"]["completion_tokens"]; dt = time.time() - t
print(f"load request: {ct} tok in {dt:.1f}s = {ct/dt:.1f} tok/s", flush=True)
PY
wait
for n in REDDIE SPARK4 ASUSI BLUEY; do
  python3 - "$n" "$D/$n.csv" <<'PY'
import statistics as st, sys
n, f = sys.argv[1], sys.argv[2]
rows = [l.split(",") for l in open(f) if l.count(",") == 2]
c = [float(r[0]) for r in rows]; p = [float(r[1]) for r in rows]; u = [float(r[2]) for r in rows]
if c:
    print(f"{n:7s} SM clock MHz min {min(c):5.0f} median {st.median(c):5.0f} max {max(c):5.0f} | power W median {st.median(p):5.1f} max {max(p):5.1f} | util median {st.median(u):3.0f}% | n={len(c)}")
else:
    print(n, "no samples:", open(f).read()[:200])
PY
done
