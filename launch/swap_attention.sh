#!/bin/bash
# Install the SM12x indexer-page attention.py (md5 da9ef196...) into the mounted patch dir on all 4 nodes.
# Keeps the previous file as attention.py.bak-pre-indexer64. Refuses while vllm_dsv41 is running.
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10"
cat > /tmp/swapatt-remote.sh <<'R'
cd ~/patches/dsv41-boot3 || exit 1
[ "$(md5sum < /tmp/attention.py.indexer64 | cut -c1-32)" = da9ef19608848b6686c17300108610af ] || { echo "BAD md5 on staged file"; exit 1; }
docker ps --format '{{.Names}}' | grep -q '^vllm_dsv41$' && { echo "REFUSING: vllm_dsv41 running"; exit 1; }
[ -f attention.py.bak-pre-indexer64 ] || cp -p attention.py attention.py.bak-pre-indexer64
cp /tmp/attention.py.indexer64 attention.py && echo "installed $(md5sum < attention.py | cut -c1-8) (backup $(md5sum < attention.py.bak-pre-indexer64 | cut -c1-8))"
R
cp /root/indexerfix/attention.py /tmp/attention.py.indexer64 && chmod 644 /tmp/attention.py.indexer64
echo "REDDIE: $(sudo -u tonyspark2 bash /tmp/swapatt-remote.sh 2>&1)"
for p in SPARK4:tonyspark4@192.168.192.4 ASUSI:tonyspark3@192.168.192.3 BLUEY:tonyspark1@192.168.192.1; do
  $J ${p#*:} 'cat > /tmp/attention.py.indexer64' < /tmp/attention.py.indexer64
  echo "${p%%:*}: $($J ${p#*:} 'bash -s' < /tmp/swapatt-remote.sh 2>&1)"
done
