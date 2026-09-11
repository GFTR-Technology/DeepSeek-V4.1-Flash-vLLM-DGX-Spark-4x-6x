#!/bin/bash
# Install the Engram rank-offset fix (md5 dc4b2104...) into the mounted patch dir on all 4 nodes.
# Keeps the original as engram.py.bak-pre-offsetfix. Run only while no vllm_dsv41 container is running.
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10"
cat > /tmp/swap-remote.sh <<'R'
cd ~/patches/dsv41-boot3 || exit 1
[ "$(md5sum < engram.py.offsetfix | cut -c1-32)" = dc4b2104616ef213b427db10f3def089 ] || { echo "BAD md5 on staged fix"; exit 1; }
docker ps --format '{{.Names}}' | grep -q '^vllm_dsv41$' && { echo "REFUSING: vllm_dsv41 still running"; exit 1; }
[ -f engram.py.bak-pre-offsetfix ] || cp -p engram.py engram.py.bak-pre-offsetfix
cp engram.py.offsetfix engram.py && echo "installed: $(md5sum engram.py | cut -c1-8) (backup $(md5sum engram.py.bak-pre-offsetfix | cut -c1-8))"
R
echo "REDDIE: $(sudo -u tonyspark2 bash /tmp/swap-remote.sh 2>&1)"
for p in SPARK4:tonyspark4@192.168.192.4 ASUSI:tonyspark3@192.168.192.3 BLUEY:tonyspark1@192.168.192.1; do echo "${p%%:*}: $($J ${p#*:} 'bash -s' < /tmp/swap-remote.sh 2>&1)"; done
