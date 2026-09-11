#!/bin/bash
# usage: run_post.sh <label> [needle_targets]   (root on Reddie; env REF_LABEL, BENCH_NOTES, VISION pass through to postserve.sh)
L=${1:?label}; NT=${2:-}; O=/var/tmp/boot-results/$L; mkdir -p "$O"
{
  bash /root/postserve.sh "$L"
  if [ -n "$NT" ]; then echo "--- needle test: $NT ---"; python3 /root/v41needle.py --targets "$NT" --out "$O/needle.json" 2>&1; fi
  echo "=== RUN_POST DONE $L $(date -u +%FT%TZ) ==="
} > "$O/post.log" 2>&1
