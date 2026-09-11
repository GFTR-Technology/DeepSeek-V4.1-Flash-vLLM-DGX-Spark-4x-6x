#!/usr/bin/env bash
# Build each worker's node-local copy of its own Engram rows. Run ON THE HEAD.
#
#   ./scripts/engram-local.sh              # build what is missing or stale
#   ./scripts/engram-local.sh status       # just report what each node holds
#   ./scripts/engram-local.sh --force      # rebuild even if the copy looks current
#
# Why: a worker reading its Engram rows over NFS costs 5.9-7.8 ms per step
# against 2.8 ms locally, and every step waits for the slowest rank. Boot 10's
# counting case went 60.8 -> 84.9 tok/s on this one change. Rank 0 already reads
# local NVMe and is skipped.
#
# The copy is sparse: only this rank's rows exist, at the original byte offsets,
# so it costs ~48 GB per worker at TP=4 rather than 510. This script compares the
# ranges each node recorded against the ranges the current node count implies and
# rebuilds only where they actually differ — a TP change does not always move
# them (the split is ceil(hash_columns / TP) columns per rank, so with 4 columns
# rank 1 owns the same rows at TP=4 and TP=6).
#
# Safe to run while an engine is live: reads are rate limited (DSV41_ENGRAM_MBPS,
# default 600) and dropped from the page cache as they go.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$HERE/lib.sh"
require_cluster

MBPS="${DSV41_ENGRAM_MBPS:-600}"
DST="$ENGRAM_LOCAL_DIR"
RANGES="$REPO_ROOT/tools/engram_ranges.py"
COPIER="$REPO_ROOT/tools/engram_local.py"
TP="$NNODES"

MODE=build; FORCE=0
for a in "$@"; do
  case "$a" in
    status)  MODE=status ;;
    --force) FORCE=1 ;;
    *) echo "Usage: $0 [status] [--force]" >&2; exit 2 ;;
  esac
done

for f in "$RANGES" "$COPIER"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done

PLAN="$(python3 "$RANGES" "$WEIGHTS" "$TP" --json)" \
  || { echo "could not compute Engram ranges for TP=$TP from $WEIGHTS" >&2; exit 1; }

# want_for <rank> <field> — field is specs | layers | gib | rows
want_for() {
  RANK="$1" FIELD="$2" python3 -c '
import json, os, sys
plan = json.loads(sys.stdin.read())["ranks"][os.environ["RANK"]]
f = os.environ["FIELD"]
print(" ".join(plan["specs"]) if f == "specs"
      else json.dumps(plan["layers"], sort_keys=True) if f == "layers"
      else plan[f])' <<< "$PLAN"
}

echo "== node-local Engram rows: TP=$TP, ${NNODES} node(s), dst $DST"
built=0; skipped=0; failed=0
for ((rank = 0; rank < NNODES; rank++)); do
  ip="${NODES[$rank]}"
  if [ "$rank" = 0 ]; then
    printf '   rank %-2s %-16s head, already reads local NVMe — skipped\n' "$rank" "$ip"
    continue
  fi
  want_layers="$(want_for "$rank" layers)"
  want_specs="$(want_for "$rank" specs)"
  want_gib="$(want_for "$rank" gib)"
  want_rows="$(want_for "$rank" rows)"

  # A TP that does not divide the hash-column count leaves the last rank(s) past
  # the end: they own no Engram rows at all, so there is nothing to copy. The
  # forward all-gathers and slices their (zero) output away.
  if [ "$want_rows" = 0 ]; then
    printf '   rank %-2s %-16s owns no Engram rows at TP=%s — nothing to copy\n' "$rank" "$ip" "$TP"
    continue
  fi

  have="$(ssh_to "$ip" "cat $DST/engram-local.json 2>/dev/null" 2>/dev/null)"
  state=missing
  if [ -n "$have" ]; then
    if HAVE="$have" WANT="$want_layers" python3 -c '
import json, os, sys
try:
    have = json.loads(os.environ["HAVE"]).get("layers", {})
except ValueError:
    sys.exit(1)
have = {str(k): list(v) for k, v in have.items()}
want = {str(k): list(v) for k, v in json.loads(os.environ["WANT"]).items()}
sys.exit(0 if have == want else 1)'; then
      state=current
    else
      state=stale
    fi
  fi

  if [ "$MODE" = status ]; then
    printf '   rank %-2s %-16s %s\n' "$rank" "$ip" "$state"
    continue
  fi
  if [ "$state" = current ] && [ "$FORCE" != 1 ]; then
    printf '   rank %-2s %-16s current — skipped (--force to rebuild)\n' "$rank" "$ip"
    skipped=$((skipped + 1))
    continue
  fi
  [ "$state" = stale ] && printf '   rank %-2s %-16s STALE (ranges moved, probably a TP change) — rebuilding\n' "$rank" "$ip"

  # Free space, before spending an hour finding out there is none.
  free_gb="$(ssh_to "$ip" "df -BG --output=avail $(dirname "$DST") 2>/dev/null | tail -1 | tr -dc 0-9" 2>/dev/null)"
  if [ -n "$free_gb" ] && [ "$free_gb" -lt "${want_gib%.*}" ]; then
    printf '   rank %-2s %-16s SKIP: %s GB free, needs about %s GB\n' "$rank" "$ip" "$free_gb" "$want_gib"
    failed=$((failed + 1))
    continue
  fi

  printf '   rank %-2s %-16s building ~%s GiB from %s (%s MB/s)\n' \
    "$rank" "$ip" "$want_gib" "$WORKER_WEIGHTS" "$MBPS"
  scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    "$COPIER" "$SSH_USER@$ip:/tmp/engram_local.py" \
    || { printf '   rank %-2s %-16s FAILED to copy the tool\n' "$rank" "$ip"; failed=$((failed+1)); continue; }
  # shellcheck disable=SC2029
  if ssh_to "$ip" "mkdir -p $DST && python3 /tmp/engram_local.py $WORKER_WEIGHTS $DST $want_specs --mbps=$MBPS"; then
    printf '   rank %-2s %-16s done\n' "$rank" "$ip"
    built=$((built + 1))
  else
    printf '   rank %-2s %-16s FAILED (the copier verifies rows and exits non-zero on a mismatch)\n' "$rank" "$ip"
    failed=$((failed + 1))
  fi
done

[ "$MODE" = status ] && exit 0
echo "== built $built, skipped $skipped, failed $failed"
if [ "$failed" != 0 ]; then
  echo "   Nodes without a copy still serve: the entrypoint drops the mount and"
  echo "   that rank reads its rows from the model dir, just more slowly."
  exit 1
fi
echo "   ENGRAM_LOCAL is on by default; ./scripts/dsv41-serve.sh will mount these."
