#!/usr/bin/env bash
# Dual-rail RoCE check, from the head, across every node in cluster.env.
#
#   ./scripts/roce-check.sh
#
# Per node: is each rail up, which IB device it maps to, its IPv4, link speed and
# the RoCEv2 IPv4 GID index the container entrypoint will pick. Then it pings
# every worker on EVERY rail, from the head's address on that same rail — which
# is what catches the usual dual-rail failure: rail 1 has a link but no route, so
# NCCL silently falls back to one rail and you only notice it as half the
# bandwidth.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$HERE/lib.sh"
require_cluster

probe='
i=0
for dev in $(echo IFS_LIST | tr "," " "); do
  st=$(cat /sys/class/net/$dev/operstate 2>/dev/null || echo absent)
  hca=$(ls /sys/class/net/$dev/device/infiniband 2>/dev/null | head -1)
  ip4=$(ip -o -4 addr show dev $dev 2>/dev/null | awk "{print \$4; exit}")
  spd=$(cat /sys/class/net/$dev/speed 2>/dev/null)
  gidx=""
  if [ -n "$hca" ]; then
    for g in $(seq 0 15); do
      t=$(cat /sys/class/infiniband/$hca/ports/1/gid_attrs/types/$g 2>/dev/null)
      v=$(cat /sys/class/infiniband/$hca/ports/1/gids/$g 2>/dev/null)
      case "$t" in *"RoCE v2"*) case "$v" in *0000:0000:0000:0000:0000:ffff:*) gidx=$g; break;; esac;; esac
    done
  fi
  printf "  rail%s %-16s %-7s %-14s %-19s %-8s gid_index=%s\n" \
    "$i" "$dev" "$st" "${hca:--}" "${ip4:--}" "${spd:--}Mb" "${gidx:--}"
  i=$((i+1))
done'
probe="${probe//IFS_LIST/$NCCL_SOCKET_IFNAME}"

declare -A RAIL_IP
echo "=== rails (NCCL_IB_HCA=$NCCL_IB_HCA)"
for ip in "${NODES[@]}"; do
  echo "$ip"
  out="$(ssh_to "$ip" "$probe" 2>&1)"
  echo "$out"
  while read -r rail dev _ _ cidr _; do
    case "$rail" in rail[0-9]*) ;; *) continue ;; esac
    [ "$cidr" = "-" ] && cidr=""
    RAIL_IP["$ip,${rail#rail}"]="${cidr%%/*}"
  done <<< "$out"
done

echo "=== per-rail reachability (head -> each worker, source pinned to the rail)"
nrails="$(echo "$NCCL_SOCKET_IFNAME" | tr ',' '\n' | grep -c .)"
bad=0
for r in $(seq 0 $((nrails - 1))); do
  src="${RAIL_IP["$HEAD_IP,$r"]:-}"
  if [ -z "$src" ]; then echo "  rail$r: head has no address, SKIPPED"; bad=1; continue; fi
  for ip in "${NODES[@]}"; do
    [ "$ip" = "$HEAD_IP" ] && continue
    dst="${RAIL_IP["$ip,$r"]:-}"
    if [ -z "$dst" ]; then printf '  rail%-2s %-16s NO ADDRESS on this rail\n' "$r" "$ip"; bad=1; continue; fi
    if ssh_to "$HEAD_IP" "ping -c 2 -W 2 -I $src $dst" >/dev/null 2>&1 \
       || ping -c 2 -W 2 -I "$src" "$dst" >/dev/null 2>&1; then
      printf '  rail%-2s %-16s %-16s <- %-16s OK\n' "$r" "$ip" "$dst" "$src"
    else
      printf '  rail%-2s %-16s %-16s <- %-16s UNREACHABLE\n' "$r" "$ip" "$dst" "$src"; bad=1
    fi
  done
done

[ "$bad" = 0 ] && echo "=== all $nrails rail(s) reachable on every node" \
               || echo "=== PROBLEMS above: the fleet will run degraded or fail to init"
exit "$bad"
