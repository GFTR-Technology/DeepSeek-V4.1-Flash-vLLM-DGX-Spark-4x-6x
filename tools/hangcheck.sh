#!/bin/bash
# hangcheck.sh: prints HANG if vLLM's last 9 stat lines (90 s) all show Running>=1 with 0.0 generation throughput
L=$(docker logs --since 3m vllm_dsv41 2>&1 | grep "Avg generation throughput" | tail -9)
n=$(printf '%s\n' "$L" | grep -c .)
z=$(printf '%s\n' "$L" | grep -E "Avg generation throughput: 0\.0 tokens/s, Running: [1-9]" | grep -c .)
last=$(printf '%s\n' "$L" | tail -1 | grep -oE "generation throughput: [0-9.]+ tokens/s, Running: [0-9]+ reqs" )
if [ "$n" -ge 9 ] && [ "$z" -eq "$n" ]; then echo "HANG: 90 s of 0.0 tok/s with requests running"; else echo "ok ($last)"; fi
