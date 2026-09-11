#!/bin/bash
# prelaunch-10.sh (root on Reddie). launch.sh runs it after every vllm_dsv41 container is stopped.
# With the GPUs free: the hidden-state flip probe on all four Sparks at once, then the 4-Spark
# collective check. Informational only: it always exits 0, so boot 10 goes ahead either way.
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
NODES="1:SPARK4:tonyspark4@192.168.192.4 2:ASUSI:tonyspark3@192.168.192.3 3:BLUEY:tonyspark1@192.168.192.1"
IMG=vllm-dsv41:overlay5
O=/var/tmp/boot-results/boot10/prelaunch; mkdir -p $O /tmp/lvr
cp /root/lvr/gpuflip.py /root/lvr/nccl_lat.py /root/lvr/flip10.sh /tmp/lvr/
for p in $NODES; do tar cf - -C /root/lvr gpuflip.py nccl_lat.py flip10.sh | $J ${p##*:} 'mkdir -p /tmp/lvr && tar xf - -C /tmp/lvr'; done
gib() { awk '/^MemAvailable:/{print int($2/1048576)}'; }
for i in $(seq 1 24); do
  ok=1
  [ "$(gib < /proc/meminfo)" -ge 90 ] || ok=0
  for p in $NODES; do w=$($J -n ${p##*:} 'cat /proc/meminfo' | gib); [ "${w:-0}" -ge 90 ] || ok=0; done
  [ $ok = 1 ] && break; sleep 5
done
echo "=== prelaunch-10 $(date -u +%T): memory back=$ok; flip probe on all four at once"
( bash /tmp/lvr/flip10.sh > $O/flip-REDDIE.txt 2>&1 ) &
for p in $NODES; do n=$(echo $p | cut -d: -f2); ( $J ${p##*:} 'bash /tmp/lvr/flip10.sh' > $O/flip-$n.txt 2>&1 ) & done
wait
python3 /root/lvr/flipsum.py $O | tee $O/flip-summary.txt
echo "=== collective check across four Sparks $(date -u +%T)"
NENV="-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_ROCE_VERSION_NUM=2 \
 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=192.168.192.0/24 -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
 -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
 -e MASTER_ADDR=192.168.192.2 -e MASTER_PORT=29610 -e WORLD_SIZE=4"
nr() { echo "docker run --rm --gpus all --network host --ipc host --memory 8g --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
 --device /dev/infiniband:/dev/infiniband $NENV -e RANK=$1 -e NODE=$2 $3 -v /tmp/lvr:/w --entrypoint timeout $IMG 240 python3 /w/nccl_lat.py"; }
( $(nr 0 REDDIE "-e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET") > $O/nccl-REDDIE.txt 2>&1 ) &
for p in $NODES; do r=${p%%:*}; n=$(echo $p | cut -d: -f2); ( $J ${p##*:} "$(nr $r $n "-e NCCL_DEBUG=WARN")" > $O/nccl-$n.txt 2>&1 ) & done
wait
{ grep -hE "^mode:|NCCL INFO (Channel 00/|Using network|NET/IB : Using)" $O/nccl-REDDIE.txt | head -4
  for n in REDDIE SPARK4 ASUSI BLUEY; do grep -hE " r[0-3] |Error|Traceback" $O/nccl-$n.txt | grep -v "NCCL INFO" | tail -12; done; } | tee $O/nccl-summary.txt
echo "=== prelaunch-10 done $(date -u +%T)"
exit 0
