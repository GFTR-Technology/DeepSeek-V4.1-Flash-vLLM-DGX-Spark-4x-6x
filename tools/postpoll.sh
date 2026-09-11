#!/bin/bash
# usage: postpoll.sh <label> <lines_seen>  prints new key lines of post.log, then @@LINES <n> and @@HEAD <state>
f=/var/tmp/boot-results/$1/post.log; n=${2:-0}
t=$(wc -l < "$f" 2>/dev/null || echo 0)
[ "$t" -gt "$n" ] && sed -n "$((n+1)),${t}p" "$f" | grep -E '^(---|===)|ceiling_count|prefill +[0-9]|"target"|calibration|VISION|Traceback|Error|FAIL|PASS|mismatch|match|serving ' | cut -c1-220
echo "@@LINES $t"; echo "@@HEAD $(docker ps -a --filter name=vllm_dsv41 --format '{{.State}}')"
