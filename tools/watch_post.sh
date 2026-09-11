#!/bin/bash
# watch_post8.sh (Mac): stream boot 8 post-serve progress; exits on RUN_POST DONE, head down, or a hang.
L=boot8; n=0; seen=""; lastwarn=""; miss=0
while true; do
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=20 root@100.113.138.96 "bash /root/postpoll.sh $L $n 2>/dev/null; grep -E 'ceiling_count|prefill +[0-9]|wrote ' /var/tmp/boot-results/$L/bench.txt 2>/dev/null | sed 's/^/@@B /'; echo \"@@HANG \$(bash /root/hangcheck.sh)\"; bash /root/v41poll.sh 2>/dev/null | sed -n 1p | grep -oE 'minAvail~[0-9]+GiB' | sed 's/^/@@MEM /'" 2>/dev/null) || { miss=$((miss+1)); [ $miss -ge 5 ] && echo "WATCH: Reddie unreachable 5x"; sleep 60; continue; }
  miss=0
  printf '%s\n' "$out" | grep -v '^@@'
  printf '%s\n' "$out" | sed -n 's/^@@B //p' | while IFS= read -r b; do case "$seen" in *"$b"*) ;; *) echo "BENCH $b" | cut -c1-150;; esac; done
  seen="$seen
$(printf '%s\n' "$out" | sed -n 's/^@@B //p')"
  n=$(printf '%s\n' "$out" | sed -n 's/^@@LINES //p'); n=${n:-0}
  h=$(printf '%s\n' "$out" | sed -n 's/^@@HEAD //p')
  hang=$(printf '%s\n' "$out" | sed -n 's/^@@HANG //p')
  m=$(printf '%s\n' "$out" | sed -n 's/^@@MEM minAvail~\([0-9]*\)GiB/\1/p')
  if [ -n "$m" ] && [ "$m" -lt 5 ] && [ "$lastwarn" != "$m" ]; then echo "MEM WARNING: minAvail ~${m} GiB"; lastwarn=$m; fi
  case "$hang" in HANG*) echo "POST $L RESULT: $hang"; exit 0;; esac
  if printf '%s\n' "$out" | grep -q 'RUN_POST DONE'; then echo "POST $L RESULT: DONE"; exit 0; fi
  if [ -n "$h" ] && [ "$h" != "running" ]; then echo "POST $L RESULT: HEAD $h (engine died during post-serve)"; exit 0; fi
  sleep 60
done
