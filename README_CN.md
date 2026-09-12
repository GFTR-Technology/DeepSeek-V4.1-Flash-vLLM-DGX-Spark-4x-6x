# DeepSeek-V4.1-Flash 多机 DGX Spark vLLM 部署（中文）

一个文件驱动整套流程：`scripts/cluster.env`。

```bash
cp scripts/cluster.env.example scripts/cluster.env   # IP、用户、网卡，就这一个文件
./scripts/dsv41-serve.sh                             # 默认 300K 档
```

不要把 IP 或用户名写进启动脚本里。`cluster.env` 已被 gitignore。

> 老路径 `launch/dsv41-tp4.sh` / `launch/boot_dsv41.sh` / `launch/bootN-go.sh` 原样保留，仍然可用；`patch/` 下七个 bind-mount 文件也没有被这次重构改动（`weight_utils.py` md5 仍是 `7e1027f1`）。

---

## 目录

1. [需要什么](#1-需要什么)
2. [填 cluster.env](#2-填-clusterenv)
3. [每个节点克隆本仓库](#3-每个节点克隆本仓库)
4. [下载权重并共享](#4-下载权重并共享)
5. [构建镜像](#5-构建镜像)
6. [双通道 RoCE 自检](#6-双通道-roce-自检)
7. [启动](#7-启动)
8. [档位（lane）](#8-档位lane)
9. [TP=6：不整除时自动补齐](#9-tp6不整除时自动补齐)
10. [节点本地 Engram 行（**默认启用**）](#10-节点本地-engram-行默认启用)
11. [验证与压测](#11-验证与压测)
12. [环境变量速查](#12-环境变量速查)
13. [故障排查](#13-故障排查)
14. [脚本清单](#14-脚本清单)

---

## 1. 需要什么

- N 台 NVIDIA DGX Spark（GB10，SM121，ARM64，每台约 128 GB 统一内存，OS 可见约 121.7 GiB），每台 1 张 GPU
- 每台都装好 docker + NVIDIA runtime
- head 能免密 ssh 到所有 worker，同一个用户，密钥认证
- 之间有 RoCE / InfiniBand 网络（200G QSFP-DD 口，不是 10G 管理口）
- Hugging Face 上的权重（约 510 GB，48 个分片）和基础镜像
- **本仓库克隆到每个节点的同一路径**（启动时从这个路径 bind-mount `patch/` 和容器入口脚本）

**为什么至少 4 台：** 路由专家（296 GB MXFP4）四等分放得下；两张 Engram n-gram 表（203 GB FP8）放不下——原版 vLLM 把它们放在主机内存，而 GB10 上主机内存就是显存池。用本仓库的 Engram-on-disk 补丁后每个 rank 只需 81.58 GiB，表本身留在磁盘上。TP2 无论如何都放不下。

---

## 2. 填 cluster.env

| 变量 | 填什么 |
|---|---|
| `SSH_USER` | 能用 docker、能在节点间 ssh 的 Linux 用户 |
| `HEAD_IP` | rank 0 的 fabric IP |
| `WORKER_IPS` | 空格分隔的 worker fabric IP。**TP = 1 + worker 数量** |
| `NCCL_IB_HCA` | InfiniBand 设备，逗号分隔。Spark 双通道默认值已在示例里 |
| `NCCL_SOCKET_IFNAME` | 对应的 RoCE 网卡 |
| `GLOO_SOCKET_IFNAME` | 给 gloo 用的单个网卡（通常是通道 0） |
| `WEIGHTS` | head 上权重的本地路径 |
| `WORKER_WEIGHTS` | worker 上看到同一份权重的路径（NFS 挂载点）。路径相同就删掉这行 |
| `CACHE` | 编译缓存目录，默认 `/var/tmp/dsv41-vllm-cache` 即可 |
| `IMAGE` | `build-image.sh` 之后的本地 tag，默认 `vllm-dsv41:overlay5` |
| `ENGRAM_DISK` / `ENGRAM_LOCAL` | Engram 放磁盘（必须 1）/ 挂载节点本地行副本（**默认 1**，见[第 10 节](#10-节点本地-engram-行默认启用)） |

在 Spark 上怎么找 fabric IP 和 HCA：

```bash
ip -4 addr show | grep -E 'enp1s0f0np0|enP2p1s0f0np0'
ls /sys/class/infiniband
ibstat 2>/dev/null | head
```

不要拿 1/10G 管理口跑 NCCL。rank 0 必须能 ssh 到每个 worker：

```bash
for ip in $WORKER_IPS; do ssh "$SSH_USER@$ip" hostname; done
```

可选：`export DSV41_ENV=/abs/path/to/cluster.env`（文件不在 scripts/ 旁边时）。

---

## 3. 每个节点克隆本仓库

```bash
git clone <this-repo>
# 四台/六台都放同一路径，例如 ~/DeepSeek-V4.1-Flash-vLLM-DGX-Spark
```

worker 上不需要 `cluster.env`（始终从 head 启动就行），但**必须**有 `patch/` 和 `scripts/dsv41-container-entrypoint.sh`，因为启动时是从这个路径 bind-mount 进容器的。

---

## 4. 下载权重并共享

在 **head** 上：

```bash
./scripts/fetch-weights.sh nfs      # 下载 + 导出 NFS + 在每个 worker 上挂好并写进 fstab
```

三种模式：

| 模式 | 做什么 |
|---|---|
| `download` | 只下载到 `$WEIGHTS`（默认） |
| `nfs` | 下载 + 从 head 导出只读 NFS + 每个 worker 挂到 `$WORKER_WEIGHTS` + 写 `/etc/fstab`。**这是本仓库的推荐做法**：510 GB 只存一份 |
| `rsync` | 下载 + 复制到每个 worker（每台各需 510 GB） |

**权重放哪很重要。** 导出目录的每一级父目录都必须对 others 可穿越（`o+x`）。容器以 root 运行，NFS 默认的 `root_squash` 把它映射成 `nobody`，而 `/root` 是 0700——挂载会成功，但每次读都失败。脚本会在导出前逐级检查并拒绝继续，给出三个选择（挪到 `/var/tmp/models`、`chmod o+x`、或用 `no_root_squash` 导出）。推荐第一个，也就是本仓库默认的 `WEIGHTS=/var/tmp/models/DeepSeek-V4.1-Flash`。

**导出 ACL 按节点实际源地址生成。** 脚本不再猜 `HEAD_IP` 的 /24——集群跨网段或节点多网卡时那就是错的。它逐个问 worker `ip route get $HEAD_IP`，拿到它真正用来连 head 的源地址，把这些地址（以及 `cluster.env` 里写的地址）都写进 `/etc/exports`。特殊拓扑可以在 `cluster.env` 里用 `NFS_EXPORT_CLIENTS="10.10.0.0/16"` 直接指定。同一导出路径的旧条目会被重写而不是追加，避免堆积后第一条生效。

`nfs` 模式需要 head 上有 NFS **服务端**（`exportfs` 来自 `nfs-kernel-server`），worker 上有**客户端**（`mount.nfs` 来自 `nfs-common`）。DGX Spark 出厂镜像通常只带客户端，所以脚本会自己检测并安装缺的那个；不想让它装就 `DSV41_INSTALL_NFS=0 ./scripts/fetch-weights.sh nfs`，它会告诉你该跑哪条命令：

```bash
sudo apt-get install -y nfs-kernel-server   # head
sudo apt-get install -y nfs-common          # 每个 worker
```

> 一定要写进 `/etc/fstab`。我们有两台 worker 只做了手工挂载，watchdog 复位后就丢了挂载点，表现是加载到一半失败。
>
> 权重已经下好的话重跑 `./scripts/fetch-weights.sh nfs` 是安全的：它检测到 `config.json` 就跳过下载，只做导出和挂载。
>
> NFS 的代价：head 读权重约 10 分钟，worker 走 NFS 要 10-18 分钟；DSpark 还要再扫一遍全部 48 个分片取 draft 层；head 会等最慢的 worker。head 的 nfsd 线程数会被脚本从 8 提到 32。

---

## 5. 构建镜像

镜像是**节点本地**的：要么每台都构建，要么在 head 构建后分发。

```bash
./scripts/build-image.sh     # head 上构建整条 overlay 链，最终 tag 为 $IMAGE
./scripts/copy-image.sh      # docker save + rsync + docker load 到每个 worker
```

overlay 链解决的问题：

| 镜像 | 解决什么 |
|---|---|
| `overlay1` | vLLM `dsv41-feat` 分支的 kernel 改动全在 `_C_stable_libtorch` 里，为 sm_121a 重编 |
| `overlay3` | FlashInfer 0.6.18 的 SM120 sparse-MLA decode 不支持 V4.1 的 topk=1152，换 0.7.0rc1 |
| `overlay4` | `mxfp8_gemm_cutlass_sm120` 运行时编译（7 个 CUTLASS 文件、22 并发）曾把四台机器的主机内存同时耗尽 |
| **`overlay5`** | 预编 `sparse_mla_sm120`，`verify5.py` 校验运行时零 JIT。**服务用的就是它** |

---

## 6. 双通道 RoCE 自检

```bash
./scripts/roce-check.sh
```

逐节点输出每条 rail 的 up/down、IB 设备、IPv4、链路速率、以及容器入口会选的 RoCEv2 IPv4 GID index；然后**从 head 在每条 rail 上的地址**去 ping 每个 worker 的**同一条 rail**。

最容易踩的坑就在这里：rail 1 链路是 up 的但没有路由，NCCL 会静默退回单通道，表现只是带宽减半。这个检查专门抓它。

GID index 不再写死成 3 —— `scripts/dsv41-container-entrypoint.sh` 在容器里遍历 `/sys/class/infiniband/$HCA/ports/1/gid_attrs/`，挑第一个 RoCE v2 + IPv4-mapped 的 GID。网卡上配了几个 IPv4/IPv6 地址会让这个 index 漂移，写死是错的。

---

## 7. 启动

```bash
./scripts/dsv41-serve.sh                # 启动（默认 300K 档）
./scripts/dsv41-serve.sh status         # 每个节点的容器状态 + /v1/models
./scripts/dsv41-serve.sh logs           # head 上 docker logs -f
./scripts/dsv41-serve.sh stop           # 停掉所有节点
./scripts/dsv41-serve.sh dry-run        # 打印将要执行的每一条 ssh/docker 命令，什么都不做
```

`dsv41-serve.sh` 先做守卫和预检，再调 `dsv41-node-launch.sh`：

1. **守卫**：每个节点上权重齐不齐（数分片数量，专抓丢掉的 NFS 挂载）、镜像在不在、`patch/` 和入口脚本在不在；TP 需要补齐时再查一遍补齐 shim 在不在——**在 12 分钟的加载之前**发现半个克隆，而不是之后。还会点名哪些节点缺 Engram 本地副本（只提示，不阻断）。
2. **预检**：5 秒 fp16 burn 测每台的 SM 时钟。GB10 会毫无征兆地闩在 1 GHz 以下，`nvidia-smi` 看不出来，重启也清不掉，**只有拔电源冷启动**才行，而每个 TP step 都要等最慢的那台。低于 1500 MHz 直接拒绝启动（`DSV41_SKIP_CLOCK_CHECK=1` 可跳过）。
3. **启动**：worker 先起（`--headless`），head 最后。每个节点起容器前会 `drop_caches` 并检查 `MemAvailable ≥ 100 GiB`——加载权重要吃掉 121.7 GiB 里的约 100 GiB，残留的页缓存就是 10 分钟后被 OOM kill 的原因。

停机时 head 优先：新 worker 碰上还活着的旧 head，会加入旧 head 的汇合点然后一直挂住。

---

## 8. 档位（lane）

一个引擎一个 `max-model-len`，切换要重启。

```bash
DSV41_LANE=300k ./scripts/dsv41-serve.sh   # 默认
DSV41_LANE=128k ./scripts/dsv41-serve.sh
DSV41_LANE=1m   ./scripts/dsv41-serve.sh
```

| 档位 | 上下文 | 序列槽 | eager | 说明 |
|---|---|---|---|---|
| `128k` | 131,072 | 8 | 否 | CUDA graphs + DSpark k=5 |
| `300k` | 300,000 | 8 | 否 | **服务配置**（boot 10）。KV 池约 1,070,168 tokens（300K 的 3.57 倍），gmu 0.80 |
| `1m` | 1,048,576 | 8 | **是** | 1M 验证配置（boot 7），KV 池 1,078,380。indexer prefill buffer 在 1M 时涨到约 5.2 GiB，只有 eager 才腾得出来 |

CUDA graphs 是吞吐的关键：eager decode 在这个模型上是 host-bound 的，约 200 ms/step，GPU 基本闲着。DSpark k=5 下每个 decode batch 都是 k+1 个目标 token 或 k 个 draft token 的整数倍，所以每个 batch 都有精确的 FULL graph、没有补零行——补零的投机 batch 会挂死 SM120 sparse MLA（FlashInfer #5015，未修）。捕获尺寸由脚本按 `SPEC_K` 和 `MAX_NUM_SEQS` 自动算出来。

---

## 9. TP=6：不整除时自动补齐

**TP 就是节点数。** 四台整除这个 checkpoint，六台不整除，vLLM 会在第一次 forward 之前就在 `divide()` 里断言失败。做法不是退回小 TP，而是把维度**向上补齐**，补出来的部分填 0 —— 0 在算术上是惰性的。

```
6 台 → TP=6
  o_groups               64 ->   66
  num_attention_heads    64 ->   66      ← 一头一组时就是 64→66
  moe_intermediate_size 2048 -> 2304
```

两半，必须对同一个故事：

| 半边 | 在哪 | 做什么 |
|---|---|---|
| 按补齐尺寸**建模型** | `make_overlay.py`，由容器入口调用 | 生成 `/model-tp6`：**symlink** 48 个真实分片（零拷贝、不占额外磁盘）+ 一个重写过的 `config.json` |
| 把**权重补齐**到这个尺寸 | `sitecustomize.py` + `dsv41_tp_pad.py`，挂在 `PYTHONPATH` | 包住 vLLM 的 `*_weights_iterator`，张量在被任何 rank 切分之前就补好 |

用 `sitecustomize.py` 是因为 vLLM 的 mp executor 把每个 worker 起成全新的解释器，而 `site` 在每个解释器里都会 import `sitecustomize` —— 这是唯一能覆盖所有 rank 的入口。它保持极轻：不 import torch、不 import vllm，只注册一个 post-import 钩子。

**维度从 checkpoint 里读，不写死。** `config.json` 里有 `num_attention_heads` / `o_groups` / `head_dim` / `o_lora_rank` / `moe_intermediate_size` / `intermediate_size` 和 `quantization_config.weight_block_size`，`load_dims()` 读它们再校验。写死常量在权重卡被重新上传的那天就变成了谎言。

**两条必须遵守的约束：**

1. **头和组一起补，以组为准。** `wo_a` 是按 `o_groups` 做的批量矩阵乘（`is_bmm=True`），每组恰好吃 `heads/groups` 个头。这个比值是权重的结构，改了等于重新切分 checkpoint。所以按**整组**补，每组带上它那一整套虚拟头：`groups' = round_up(groups, tp)`，`heads' = groups' * heads_per_group`。`groups'` 是 tp 的倍数，`heads'` 自动也是。每组输入宽度始终不变，所以 `wo_a` 的第二维不需要任何规则。
2. **量化轴必须保持整块。** 块量化权重 `[N,K]` 带 `[N/block,K/block]` 的 scale，N 补到非 block 整数倍时 scale 行数就成了小数——根本补不了。所以 FFN 宽度补到 `tp * block` 的倍数：2048 在 TP=6、block=128 时补到 **2304**（不是 2052）。头数/组数不需要这一步：它们进张量时乘了 `head_dim` / `o_lora_rank`，本身已经是 block 的倍数。

**Engram 不补齐。** 它按哈希列切，而 forward 本来就是「all-gather 之后切回 `n_hash_cols`」，所以除不尽时最后几个 rank 单纯什么都不持有、它们的 0 会被切掉。补齐反而是错的：行是从 `engram_vocab_size` 生成的素数，虚拟列在磁盘上没有行。只有两处原先假设这不会发生（stager 的缓冲区宽度会变成负数、空 rank 的去重读会越过表尾），已在 `patch/engram.py` 修掉，diff 见 `patch/dsv41_tp_pad/engram-uneven-tp.diff`。

**第一次跑 TP≠4 之前：**

```bash
./scripts/dsv41-tp-probe.sh          # 只读探针：这个镜像里 shim 要挂的接缝还在吗
python3 tests/test_tp_pad.py         # 计划 + 配置 overlay
python3 tests/test_tp_pad.py --torch # + 形状与数值（在 Spark 上跑）
```

> **状态说明。** 计划、overlay、补齐函数都有测试覆盖，但**整条路径没有在真实 checkpoint 上跑过**——本仓库从未在 TP≠4 下启动过。第一次务必用贪心参考对拍：补齐后的贪心输出必须与 TP=4 **完全一致**（`tools/postserve.sh`），因为每一片补齐都应当是算术惰性的。不一致就是补错了，不是"差不多"。另外补齐是有代价的：2048→2304 是 +12.5% 的专家 FFN 宽度，显存和每次矩阵乘都要付。

---

## 10. 节点本地 Engram 行（**默认启用**）

worker 通过 NFS 读 Engram 行每步要 5.9-7.8 ms（head 本地只要 2.8 ms），而每一步都要等最慢的 rank。把每个 rank 自己的行复制到本地 NVMe 后，boot 10 的计数用例从 60.8 提到 84.9 tok/s。所以 `ENGRAM_LOCAL=1` 现在是**默认值**。

**不需要单独手动执行。** `./scripts/dsv41-serve.sh` 启动时会自己调 `scripts/engram-local.sh`：第一次在新集群上会给每个 worker 建约 48 GB 的稀疏副本（几分钟），之后每次启动都是空操作（几个 ssh 往返、不产生 I/O）。

```bash
./scripts/dsv41-serve.sh                        # 需要时自动建副本，然后启动
DSV41_ENGRAM_AUTOBUILD=0 ./scripts/dsv41-serve.sh   # 跳过自动建，缺副本就走 NFS
DSV41_ENGRAM_LOCAL=0     ./scripts/dsv41-serve.sh   # 整个特性关掉

./scripts/engram-local.sh            # 也可以单独跑（预先建好，或改了 TP 之后）
./scripts/engram-local.sh status     # 只看每个节点现在有什么
./scripts/engram-local.sh --force    # 强制重建
```

它做的事：用 `tools/engram_ranges.py` 从 `config.json` 直接算出当前节点数下每个 rank 的行区间（不用先启动一次看日志），检查每个 worker 的剩余磁盘，把 `tools/engram_local.py` 送过去执行。副本是**稀疏文件**：只有本 rank 的行是实体数据，落在原始字节偏移上，TP=4 时每个 worker 约 48 GB 而不是 510 GB。复制带限速（`DSV41_ENGRAM_MBPS`，默认 600 MB/s）并随读随丢页缓存，**可以在服务运行时跑**；结束时随机抽 2000 行逐字节回验，不一致就以非 0 退出。rank 0 本来就读本地 NVMe，会被跳过。

**幂等且只在需要时重建。** 脚本比较的是每个节点记录的行区间和当前节点数推出来的区间，只有**真的变了**才重建。TP 变化不一定会移动区间——切分是每 rank `ceil(哈希列数 / TP)` 列，所以哈希列数是 4 时，rank 1 在 TP=4 和 TP=6 下拥有的是同一批行，不需要重建 48 GB。

**建不成也不会挡住启动。** 磁盘不够、ssh 不通之类的失败只打印一行说明然后继续；容器入口（`dsv41-container-entrypoint.sh`）在每个节点上单独检查 `engram-local.json` 在不在，不在就把 `DSV41_ENGRAM_DIR` 取消掉，那个 rank 照常从模型目录读，只是慢一些——**是性能问题，不是正确性问题**。

**哈希列除不尽时，靠后的 rank 本来就没有行**（例如 4 列 3 节点时的 rank 2），脚本会直接跳过它们，不会白建一份空副本。

手工做同样的事（脚本内部就是这两步）：

```bash
python3 tools/engram_ranges.py "$WEIGHTS" 4        # 所有 rank 的区间和占用
python3 tools/engram_ranges.py "$WEIGHTS" 4 1      # 只要 rank 1 的参数，可直接粘贴
1:96000564:192001740 14:96003054:192007016
python3 tools/engram_ranges.py "$WEIGHTS" 4 --json # 给脚本用的机器可读格式
```

---

## 11. 验证与压测

```bash
bash tools/postserve.sh boot11        # 冒烟 + 贪心参考对比 + 乱码门 + C1-C6 基准
python3 tools/vision_tools_demo.py    # 3 个图像用例 + 4 个工具调用用例，端到端
python3 bench/v41needle.py            # 长上下文 needle
bash tools/hangcheck.sh               # 抓 #5015 式挂死（请求 90 秒 0 tok/s）
python3 tools/nccl_lat.py             # 全节点 all-reduce 时延
```

接口是 OpenAI 兼容的，模型名 `deepseek-v4.1-flash`，视觉和工具调用默认开着（每请求最多 4 张图），思考默认关闭、按需逐请求打开：

```bash
curl -s http://$HEAD_IP:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash",
  "messages": [{"role": "user", "content": "9.11 和 9.9 哪个大？"}],
  "chat_template_kwargs": {"thinking": true}}'
```

---

## 12. 环境变量速查

`cluster.env` 管拓扑且**优先于环境变量**；单次运行的覆盖统一走 `DSV41_` 前缀。

| 变量 | 默认 | 说明 |
|---|---|---|
| `DSV41_LANE` | `300k` | `128k` / `300k` / `1m` |
| `DSV41_MAXLEN` / `DSV41_SEQS` / `DSV41_BATCHED` | 随档位 | 单独覆盖上下文/并发槽/batched tokens |
| `DSV41_GMU` | `0.80` | `--gpu-memory-utilization` |
| `DSV41_EAGER` | 随档位 | `1` 关掉 CUDA graphs |
| `DSV41_SPEC` / `DSV41_SPEC_K` | `dspark` / `5` | 投机解码；`DSV41_SPEC=none` 关掉 |
| `DSV41_SPEC_ADAPT` | `false` | 自适应验证。开了会强制变长 FULL 图并补零行，是 #5015 的触发条件 |
| `DSV41_TEXT_ONLY` / `DSV41_PARSERS` | `0` / `1` | 视觉编码器 / 工具与推理解析器 |
| `DSV41_THINKING` | `false` | 默认思考开关 |
| `DSV41_ENGRAM_DISK` / `DSV41_ENGRAM_LOCAL` | `1` / `1` | 覆盖 Engram 相关开关。`DSV41_ENGRAM_LOCAL=0` 关掉本地行副本 |
| `DSV41_ENGRAM_AUTOBUILD` | `1` | 启动时自动建缺失的 Engram 本地副本；`0` 则只提示不建 |
| `DSV41_ENGRAM_MBPS` | `600` | `engram-local.sh` 复制副本的限速（MB/s） |
| `DSV41_EXTRA` | — | 追加给 `vllm serve` 的参数 |
| `DSV41_IMAGE` / `DSV41_NAME` / `DSV41_PORT` | 随 cluster.env | 临时换镜像/容器名/端口 |
| `DSV41_SKIP_CLOCK_CHECK` | `0` | 跳过 GPU 时钟预检 |
| `DSV41_TP_PAD_GROUPS` | `attn,dense,moe` | 只补其中某些组 |
| `DSV41_TP_PAD_DEBUG` | — | 打印每一个被补齐的张量 |
| `DSV41_INSTALL_NFS` | `1` | `fetch-weights.sh nfs` 自动安装缺失的 NFS 包；`0` 则只提示 |
| `NFS_EXPORT_CLIENTS` | 自动探测 | 覆盖导出 ACL，例如 `10.10.0.0/16` 或空格分隔的地址列表（写在 `cluster.env` 里） |
| `DSV41_NFS_OPTS` | `ro,sync,no_subtree_check` | 导出选项，需要时可加 `no_root_squash` |
| `DSV41_ALLOW_PRIVATE_EXPORT` | `0` | 跳过导出路径可穿越性检查 |
| `DSV41_ENV` | — | `cluster.env` 的绝对路径 |

**两个不能动的参数：**
- `--block-size 128`：不写的话 vLLM 会选 64，V4 indexer 后端在 KV 初始化时直接拒绝（[详情](docs/boot4-block-size.md)）。补丁里每层 64-state 的页设置仍然生效。
- `MAX_JOBS=2` + `FLASHINFER_NVCC_THREADS=1`：万一还有东西在运行时编译，也不会把主机拖死。

---

## 13. 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| `Missing cluster.env.` | `cp scripts/cluster.env.example scripts/cluster.env` 后填 IP |
| `exportfs：找不到命令` / `exportfs: command not found` | head 上没装 NFS 服务端。`sudo apt-get install -y nfs-kernel-server`，或直接重跑 `./scripts/fetch-weights.sh nfs`（新版会自己装；权重已下好会跳过下载） |
| worker 挂载报 `wrong fs type` | worker 上没装 `nfs-common`，同样由 `fetch-weights.sh nfs` 自动处理 |
| `mount.nfs: access denied by server` | 导出 ACL 不含 worker 实际用的源地址（跨网段/多网卡时最常见）。worker 上 `ip route get <head>` 看 `src`，head 上 `sudo exportfs -v` 看导出给了谁；重跑 `./scripts/fetch-weights.sh nfs` 会按源地址自动重建 ACL |
| 挂载成功但读文件 `Permission denied` | 导出路径某级父目录不是 `o+x`（`/root` 是 0700），`root_squash` 下读不了。把权重挪到 `/var/tmp/models` 并改 `cluster.env` 的 `WEIGHTS` |
| worker 挂载超时 | 从 worker 上 `showmount -e <head>` 看导出；确认 head 放行 2049/tcp |
| `Error: $ip has 47/48 shards` | NFS 挂载掉了（watchdog 复位后最常见），或没下全。检查 `/etc/fstab` |
| `Error: image ... missing on $ip` | 镜像是节点本地的，每台都要有。`./scripts/copy-image.sh` |
| `Error: clone this repo to the same path on $ip` | worker 上缺 `patch/` 或入口脚本；或路径不一致，用 `DSV41_PATCHES` / `DSV41_ENTRYPOINT` 指定 |
| `$ip: WEDGED — SM xxx MHz` | GPU 时钟闩锁，**拔电源冷启动 30-60 秒**，重启没用（[详情](docs/gpu-clock-latch.md)） |
| `MemAvailable ... refusing to boot` | 有残留进程占着统一内存；先 `./scripts/dsv41-serve.sh stop`，再重来 |
| 启动卡在 distributed init | 上一轮的 head 还活着。`./scripts/dsv41-serve.sh stop` 全清后重来 |
| 吞吐只有一半，NCCL 日志只出现一个 HCA | 双通道退化。跑 `./scripts/roce-check.sh`，通常是 rail 1 有链路无路由 |
| `[dsv41-entrypoint] WARNING: no RoCEv2 IPv4 GID` | 该 HCA 上没有 IPv4-mapped 的 RoCE v2 GID；确认网卡配了 IPv4 地址 |
| `Error: $PATCH_HOST/dsv41_tp_pad/ incomplete` | TP 需要补齐但某个节点上仓库没拉全 |
| 单流速度忽快忽慢（差约 1.5 倍） | GB10 的 GPU 慢状态，`tools/gpuflip.py` 可复现，见 [issue #1](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/issues/1) |
| 请求挂住、tok/s 为 0 | FlashInfer #5015。确认 `DSV41_SPEC_ADAPT=false`；`tools/hangcheck.sh` 可以抓 |
| 换了 TP 之后 worker 步进变慢 | 本地 Engram 行区间过期了。跑 `./scripts/engram-local.sh`，它会自己判断哪些节点真的需要重建 |
| `[guard] node-local Engram rows missing on: ...` | 那些 rank 会走 NFS 读 Engram（慢，不是错）。`./scripts/engram-local.sh` 建副本 |
| `[dsv41-entrypoint] no engram-local.json in ...` | 同上，容器已自动回退到从模型目录读 |

---

## 14. 脚本清单

| 文件 | 作用 |
|---|---|
| `scripts/cluster.env.example` | 拓扑模板。复制成 `cluster.env` 后编辑，这是唯一要改的文件 |
| `scripts/lib.sh` | 共享加载器：找 `cluster.env`、拼 `NODES` 数组（head 在前）、`ssh_to`、`weights_for` |
| `scripts/dsv41-serve.sh` | **主入口**。`start\|stop\|status\|logs\|dry-run` + 守卫 + 预检 |
| `scripts/dsv41-node-launch.sh` | 组装并下发每个 rank 的 `docker run`，worker 先起、head 最后 |
| `scripts/dsv41-container-entrypoint.sh` | 容器内入口：挑 RoCEv2 GID index，需要时建 TP 配置 overlay |
| `scripts/dsv41-tp-probe.sh` | TP≠4 之前的只读探针 |
| `scripts/roce-check.sh` | 双通道体检 + 逐 rail 连通性 |
| `scripts/build-image.sh` / `copy-image.sh` | 构建 overlay 链 / 分发镜像 |
| `scripts/fetch-weights.sh` | 下载权重，`nfs` 或 `rsync` 共享给 worker |
| `patch/dsv41_tp_pad/` | TP 补齐：计划 + `sitecustomize` shim + 配置 overlay + 探针（[说明](patch/dsv41_tp_pad/README.md)） |
| `tests/test_tp_pad.py` | 补齐的离线测试 |
| `scripts/engram-local.sh` | 给每个 worker 建节点本地 Engram 行副本（`status` / `--force`） |
| `tools/engram_ranges.py` | 从 `config.json` 直接算各 rank 的 Engram 行区间（`--json` 供脚本用） |
| `launch/`（旧） | 原来的 `dsv41-tp4.sh <rank>` / `boot_dsv41.sh` / `bootN-go.sh`，保留可用 |

其余目录（`patch/`、`build/`、`bench/`、`docs/`、`results/`）见 [README.md](README.md) 的仓库结构表。
