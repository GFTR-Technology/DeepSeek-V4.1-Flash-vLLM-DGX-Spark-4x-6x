#!/bin/bash
# recover.sh (root on Reddie) after Tony's power-cycle: boot times, GPU clocks idle + under a 15 s fp16 burn,
# NFS export/mounts (remount where the reboot dropped a non-fstab mount), re-stage /tmp/boot9-go.sh, MemAvailable.
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
NODES="SPARK4:tonyspark4@192.168.192.4 ASUSI:tonyspark3@192.168.192.3 BLUEY:tonyspark1@192.168.192.1"
echo "===== boot times / idle GPU / MemAvailable"
Q='echo "booted $(uptime -s) | idle $(nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader) | MemAvail $(( $(grep MemAvailable /proc/meminfo | tr -s " " | cut -d" " -f2) / 1048576 ))G | vllm: $(docker ps -a --filter name=vllm_dsv41 --format {{.State}})"'
echo "REDDIE: $(bash -c "$Q")"
for p in $NODES; do echo "${p%%:*}: $($J ${p#*:} "$Q" 2>&1)"; done
echo "===== NFS server on Reddie"
echo "nfs-server: $(systemctl is-active nfs-server 2>&1) | exports: $(exportfs -v 2>/dev/null | tr -s ' \t' ' ' | head -2 | tr '\n' ' ')"
echo "===== worker mounts"
for p in $NODES; do
  h=${p#*:}; k=${p%%:*}
  if $J $h 'timeout 10 test -f /mnt/reddie-models/DeepSeek-V4.1-Flash/config.json' 2>/dev/null; then echo "$k: mount OK"
  else
    echo "$k: mount missing or stale -> remount"
    $J $h 'sudo -n umount -l /mnt/reddie-models 2>/dev/null; sudo -n mount -t nfs -o ro,vers=3 192.168.192.2:/var/tmp/models /mnt/reddie-models && timeout 10 test -f /mnt/reddie-models/DeepSeek-V4.1-Flash/config.json && echo "   remounted OK" || echo "   REMOUNT FAILED"'
  fi
done
echo "===== re-stage /tmp/boot9-go.sh on Asusi"
$J tonyspark3@192.168.192.3 'cat > /tmp/boot9-go.sh' < /root/final/boot9-go.sh && echo "asusi /tmp/boot9-go.sh $($J tonyspark3@192.168.192.3 'md5sum < /tmp/boot9-go.sh | cut -c1-8')"
echo "===== 15 s fp16 burn on all four (healthy: ~2000+ MHz and 80 W+ under load; latched: ~715 MHz / ~18 W)"
cat > /tmp/burn.sh <<'R'
( docker run --rm --gpus all --network none --entrypoint python3 vllm-dsv41:overlay5 -c "
import torch, time
a = torch.randn(4096, 4096, dtype=torch.float16, device='cuda'); b = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
for _ in range(10): c = a @ b
torch.cuda.synchronize(); t0 = time.time(); n = 0
while time.time() - t0 < 15:
    c = a @ b; n += 1
torch.cuda.synchronize(); print(f'fp16 {2*4096**3*n/(time.time()-t0)/1e12:.1f} TFLOPS')
" 2>&1 | grep TFLOPS ) &
sleep 12; s=$(nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader | tr '\n' ' '); wait
echo "under load: $s"
R
echo "REDDIE: $(bash /tmp/burn.sh 2>&1 | tr '\n' ' ')" &
for p in $NODES; do (echo "${p%%:*}: $($J ${p#*:} 'bash -s' < /tmp/burn.sh 2>&1 | tr '\n' ' ')") & done
wait
