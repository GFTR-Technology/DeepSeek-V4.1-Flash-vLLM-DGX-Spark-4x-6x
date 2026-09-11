#!/bin/bash
# flip10.sh (any node, GPU free): gpuflip.py probe with nvidia-smi clock/power sampled every 200 ms.
# The sampler stops itself (timeout), so nothing is pkill'ed by pattern.
cd /tmp/lvr
( timeout 115 nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits -lms 200 </dev/null \
  | while read -r l; do echo "N $(date +%s.%N) $l"; done > /tmp/lvr/smi10.txt ) &
docker run --rm --gpus all --network none --memory 8g -v /tmp/lvr:/w --entrypoint timeout vllm-dsv41:overlay5 150 \
  python3 /w/gpuflip.py > /tmp/lvr/flip10.txt 2>&1
wait
cat /tmp/lvr/flip10.txt /tmp/lvr/smi10.txt
