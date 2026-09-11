#!/bin/bash
# Pre-launch for boot 8 (run by launch.sh after all containers are stopped): install the final patch set into
# ~/patches/dsv41-boot3 on all 4 nodes. Backups: engram.py.bak-pre-prestage, mounts.txt.bak-pre-prestage.
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10"
cat > /tmp/swapfinal-remote.sh <<'R'
cd ~/patches/dsv41-boot3 || exit 1
[ "$(md5sum < engram.py.prestage | cut -c1-32)" = 0ae8f1a5cbdcee027ba2392640b7fed1 ] || { echo "BAD engram md5"; exit 1; }
[ "$(md5sum < model_state.py.prestage | cut -c1-32)" = 0a14bee67f103f1d616c6044134c0bab ] || { echo "BAD model_state md5"; exit 1; }
docker ps --format '{{.Names}}' | grep -q '^vllm_dsv41$' && { echo "REFUSING: vllm_dsv41 running"; exit 1; }
[ -f engram.py.bak-pre-prestage ] || cp -p engram.py engram.py.bak-pre-prestage
[ -f mounts.txt.bak-pre-prestage ] || cp -p mounts.txt mounts.txt.bak-pre-prestage
cp engram.py.prestage engram.py && cp model_state.py.prestage model_state.py && cp mounts.txt.prestage mounts.txt && echo "installed engram $(md5sum < engram.py | cut -c1-8) model_state $(md5sum < model_state.py | cut -c1-8) mounts $(wc -l < mounts.txt) lines"
R
ok=0
r=$(sudo -u tonyspark2 bash /tmp/swapfinal-remote.sh 2>&1); echo "REDDIE: $r"; echo "$r" | grep -q "^installed engram 0ae8f1a5" && ok=$((ok+1))
for p in SPARK4:tonyspark4@192.168.192.4 ASUSI:tonyspark3@192.168.192.3 BLUEY:tonyspark1@192.168.192.1; do
  r=$($J ${p#*:} 'bash -s' < /tmp/swapfinal-remote.sh 2>&1); echo "${p%%:*}: $r"; echo "$r" | grep -q "^installed engram 0ae8f1a5" && ok=$((ok+1))
done
[ $ok -eq 4 ] || { echo "PRELAUNCH FAILED: $ok/4 nodes installed"; exit 1; }
echo "PRELAUNCH OK: 4/4"
