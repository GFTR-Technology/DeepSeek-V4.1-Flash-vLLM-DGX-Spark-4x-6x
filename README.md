# DeepSeek-V4.1-Flash on NVIDIA DGX Sparks (vLLM, TP4, DSpark, CUDA graphs)

中文完整安装流程见 [README_CN.md](README_CN.md).

**Status (2026-09-10): serving (boot 10).**
- The model dropped at about 2 AM ET, and this stack started serving it at 9:17 AM ET the same day.
- **One stream, decode tok/s** (after the first token): **code 73.8**, tables 55.4, JSON 52.1, math 50.9, reasoning 37.8, narrative 24.9, prose 24.4.
  - Counting ran at **92.2 tok/s** back to back. It was 62.2 in the bench run, which hit a GPU slow-state phase ([issue #1](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/issues/1)).
- **Concurrency:** six streams give **131.9 tok/s** aggregate across the 8 categories (boot 9: 98.0).
  - Peak aggregates: counting **259.9** (C5), **code 225.5** (C6), tables 215.1 (C5), math 182.7 (C6).
- **Prefill:** 902-1,539 tok/s cold. A 93K-token prompt takes 78 s. TTFT on short prompts is 0.3-0.5 s.
- **Context:** **300K** max context with a **1,070,168-token** KV pool (3.57x at 300K).
- **Vision and tool calling are on:** up to 4 images per request, tool calls including parallel calls and a full round trip. 7/7 end-to-end checks pass ([below](#vision-and-tool-calling-on-in-the-serving-config)).
  - 1M max context was proven on a separate boot, with a 1,078,380-token DSpark KV pool.
- Nothing on this page is a projection. Every number here was measured at **TP=4**.

## Quick start

One file drives the whole recipe: `scripts/cluster.env`. Everything runs from the
head; the workers only need this repo at the same path, the image, and the weights.

```bash
cp scripts/cluster.env.example scripts/cluster.env   # IPs, user, fabric. One file.
./scripts/fetch-weights.sh nfs                       # 510 GB once, exported to the workers

# the engine image — try the published one first, see "The engine image" below
docker pull vllm/vllm-openai:deepseekv41-flash-0909-arm64
# cluster.env:  IMAGE=vllm/vllm-openai:deepseekv41-flash-0909-arm64

./scripts/dsv41-serve.sh                             # serve. 300K lane.
```

```bash
./scripts/dsv41-serve.sh status|logs|stop|dry-run
DSV41_LANE=1m   ./scripts/dsv41-serve.sh    # 1M context (eager)
DSV41_LANE=128k ./scripts/dsv41-serve.sh
```

| Lane | Context | Seq slots | Notes |
|---|---|---|---|
| `128k` | 131,072 | 8 | CUDA graphs + DSpark k=5 |
| `300k` | 300,000 | 8 | **Default.** The boot-10 serving config: KV pool 1,070,168 tokens, gmu 0.80 |
| `1m` | 1,048,576 | 8 | The boot-7 proof: KV pool 1,078,380. Eager — the indexer prefill buffer grows to ~5.2 GiB at 1M |

On the first run `serve` also builds each worker's node-local copy of its Engram
rows (~48 GB sparse per worker, a few minutes; after that it is a no-op). That is
worth 60.8 -> 84.9 tok/s on the counting case. `DSV41_ENGRAM_AUTOBUILD=0` skips
it, `./scripts/engram-local.sh` does it on its own, `DSV41_ENGRAM_LOCAL=0` turns
the feature off.

`dry-run` prints the exact `ssh` and `docker run` for every rank and touches
nothing. The launcher stops the head first (a worker that meets a still-live old
head joins its rendezvous and hangs), drops the page cache, refuses to boot below
100 GiB of `MemAvailable`, and starts the workers headless before the head.

**Node count is TP.** Four Sparks is the published, measured recipe. Six or eight
also launch — see [Any TP](#any-tp) below.

## The engine image

This repo was written on day 0, against vLLM's `dsv41-feat` branch, and every
number on this page was measured on the image it builds, `vllm-dsv41:overlay5`.
Since then the branch was **merged into mainline and deleted**, which changes what
the build path costs you. Two options:

| | published day-0 image | `vllm-dsv41:overlay5` |
|---|---|---|
| how | one `docker pull` | `./scripts/build-image.sh` |
| sm_121a kernels for GB10 | **unknown — untested here** | yes, that is the point of it |
| measured numbers on this page | no | yes |
| cost if it does not work | one pull | — |

Try the published one first; failing costs a pull.

```bash
# the tags are DATED — there is no bare `deepseekv41-flash` tag. List them:
curl -s 'https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page_size=100&name=deepseekv41' \
  | python3 -c "import json,sys;[print(t['name']) for t in json.load(sys.stdin)['results']]"

docker pull vllm/vllm-openai:deepseekv41-flash-0909-arm64    # DGX Spark is ARM64
# cluster.env:  IMAGE=vllm/vllm-openai:deepseekv41-flash-0909-arm64
```

`dsv41-node-launch.sh` adds `--tokenizer-mode deepseek_v41` by itself for any
`*deepseekv41-flash-0909*` image; the overlay images resolve it from `model_type`
and reject the flag, so you do not have to think about it.

If the engine dies in a kernel at load, that answers the unknown above and you
need `overlay5`. `./scripts/build-image.sh --check` prints what is missing and how
to get it; `--from overlay3` resumes after a failure. The parts that bite:

- **overlay1 needs `build/vllm/`**, the branch's Python tree. The branch is gone,
  so fetch the PR ref instead:
  `git fetch origin pull/56214/head:dsv41-feat` (then `pull/56201/head` as a
  fallback). `git ls-remote --heads https://github.com/vllm-project/vllm | grep -i dsv41`
  confirms it really is gone.
- **overlay1's base is pinned in `build/Dockerfile.overlay`**, not in
  `cluster.env`. `HF_BASE_IMAGE` does not override it; the script reads the
  `FROM` line and says so if the two disagree.
- **`_C_stable_libtorch` is rebuilt inside a container**, `v41build`, whose
  `/src` is the **full** vllm checkout (`CMakeLists.txt`, `cmake/`, `csrc/`) —
  not the `vllm/` subdirectory. The branch's kernel changes all live in that one
  extension, and the stock ARM64 wheels do not carry it for SM 12.1.
- **`vllm/vllm-openai:*` are runtime images**: no cmake, no ninja, possibly no
  nvcc. Check with
  `docker exec v41build sh -c 'which cmake ninja g++ nvcc'`, add the first two
  with `pip install cmake ninja`. No nvcc means that base cannot compile CUDA at
  all — use a `-devel` CUDA base with a matching torch, or take the published
  image.
- **`--from-published`** tags a published image as `vllm-dsv41:overlay1` and
  starts at overlay3, if you want FlashInfer 0.7.0rc1 and the prebuilt SM120
  kernels on top of it without doing overlay1 yourself.

## Vision and tool calling (on in the serving config)

Both are live on boot 10 and tested end to end with `tools/vision_tools_demo.py`; the output is in [`results/boot10/vision-tools.txt`](results/boot10/vision-tools.txt). The test images are generated in the script, so each expected answer is known exactly.

| test | result |
|---|---|
| V1: one image, three color stripes, name them left to right | PASS: "red, green, blue" (0.6 s) |
| V2: two images in one message | PASS: first=red, second=blue (1.2 s) |
| V3: 2x2 grid, color of the top-right square | PASS: "Green" (0.6 s) |
| T1: tool call with arguments | PASS: `get_weather {"city": "Paris", "unit": "c"}` (1.4 s) |
| T2: full round trip (the tool result goes back, the model answers from it) | PASS: "18°C with light rain" (2.2 s) |
| T3: parallel calls in one turn | PASS: `get_weather` for Tokyo and Berlin (2.9 s) |
| T4: `tool_choice` forcing a named function | PASS: `get_time {"city": "Sydney"}` (1.6 s) |

Both are **on by default** in `scripts/dsv41-node-launch.sh`:
- the vision encoder loads (no `--language-model-only`). It adds about 0.22 GiB per rank. `DSV41_TEXT_ONLY=1` turns it off.
- `--limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1`: up to 4 images per request.
- `--tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41`. `DSV41_PARSERS=0` turns them off.
- Thinking is off by default; turn it on per request with `"chat_template_kwargs": {"thinking": true}`.
- The older `launch/dsv41-tp4.sh` path defaults the other way (`TEXT_ONLY=1`, `PARSERS=0`) and needs `launch/boot10-go.sh` to switch both on.

Example requests (OpenAI-compatible API; the served model name is `deepseek-v4.1-flash`):

```bash
# image (base64 data URL; up to 4 images per message)
curl -s http://<head>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash", "max_tokens": 200,
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}}]}]}'
```

```bash
# tool call
curl -s http://<head>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash", "tool_choice": "auto",
  "messages": [{"role": "user", "content": "What is the weather in Paris in celsius?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
    "required": ["city"]}}}]}'
```

## Benchmark (boot 10, the serving config)

4x DGX Spark TP4, DSpark k=5, FULL_AND_PIECEWISE CUDA graphs, Engram rows node-local on every rank, tools and vision on, 300K context, gmu 0.80.

How it was measured:
- Fixed prompt set `bench/prompts-v1.json`, identical on every boot: 8 categories plus a counting ceiling.
- Streaming, temperature 0, thinking off, after a warmup. Short prompts (about 30-120 tokens), 150-256 token budgets.
- At concurrency C, C streams are released together. There is one batch per cell.
- Token counts come from the server's `usage` block, never from stream chunks (DSpark packs several tokens per chunk).
- **Decode** = tokens after the first / time after the first token, per stream. **Aggregate** = all streams' tokens / batch wall time. **TTFT** = first token delta.
- Raw output and more tables: [`results/boot10/`](results/boot10/) ([report](results/boot10/report.md), made with `tools/bench_report.py`).
- One caveat: the GB10 GPU slow state ([issue #1](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/issues/1)) can move a single cell by up to about 1.5x. Counting at C1 is the clearest case (62.2 here, 92.2 back to back).

**Throughput by concurrency** (mean of the 8 categories; the counting ceiling is excluded):

| C | aggregate tok/s | per-stream decode tok/s | mean TTFT (s) | boot 9 aggregate |
|---|---|---|---|---|
| C1 | 37.95 | 43.12 | 0.441 | 33.88 |
| C2 | 64.30 | 37.43 | 0.441 | 58.99 |
| C3 | 78.70 | 30.62 | 0.793 | 67.56 |
| C4 | 85.72 | 24.66 | 0.579 | 85.50 |
| C5 | 114.20 | 27.01 | 0.585 | 91.96 |
| C6 | **131.86** | 25.35 | 0.502 | 97.99 |

**Decode: per-stream tok/s after the first token**

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| **code** | **73.8** | 45.0 | 33.8 | 34.3 | 44.1 | 41.5 |
| JSON | 52.1 | 30.4 | 20.7 | 22.7 | 26.8 | 27.5 |
| math | 50.9 | 59.3 | 54.4 | 31.4 | 28.9 | 34.3 |
| reasoning | 37.8 | 42.4 | 36.2 | 26.8 | 20.8 | 24.6 |
| tables (format) | 55.4 | 59.5 | 41.5 | 35.5 | 53.1 | 35.3 |
| summary | 25.7 | 22.4 | 22.4 | 13.7 | 15.9 | 16.1 |
| prose | 24.4 | 23.6 | 19.4 | 18.4 | 14.8 | 13.8 |
| narrative | 24.9 | 16.9 | 16.5 | 14.7 | 11.8 | 9.8 |
| counting (ceiling) | 62.2 | 58.9 | 43.0 | 58.3 | 57.2 | 33.4 |

**Aggregate throughput: tok/s across all streams** (wall time, TTFT included)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| **code** | 66.5 | 81.1 | 92.4 | 123.8 | 196.7 | **225.5** |
| JSON | 43.7 | 49.2 | 51.9 | 76.3 | 94.7 | 130.4 |
| math | 45.6 | 105.5 | 148.4 | 110.2 | 127.9 | 182.7 |
| reasoning | 35.0 | 75.3 | 96.5 | 97.0 | 94.1 | 133.6 |
| tables (format) | 44.0 | 94.3 | 101.7 | 112.0 | 215.1 | 182.1 |
| summary | 21.9 | 37.6 | 38.2 | 43.1 | 64.0 | 73.5 |
| prose | 22.9 | 41.5 | 55.2 | 69.9 | 67.9 | 74.8 |
| narrative | 23.8 | 30.0 | 45.2 | 53.4 | 53.1 | 52.4 |
| counting (ceiling) | 57.3 | 109.0 | 118.6 | 208.8 | **259.9** | 184.9 |

**TTFT: mean time to first token (s)**

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| code | 0.31 | 0.44 | 0.49 | 0.51 | 0.40 | 0.42 |
| JSON | 0.33 | 0.50 | 0.55 | 0.57 | 0.45 | 0.42 |
| math | 0.47 | 0.35 | 0.38 | 0.46 | 0.62 | 0.42 |
| reasoning | 0.44 | 0.35 | 0.42 | 0.55 | 0.57 | 0.41 |
| tables (format) | 0.51 | 0.40 | 0.57 | 0.59 | 0.47 | 0.47 |
| summary (~300-token passage) | 0.79 | 0.77 | 3.18 | 1.28 | 1.33 | 1.02 |
| prose | 0.38 | 0.39 | 0.31 | 0.32 | 0.36 | 0.48 |
| narrative | 0.29 | 0.32 | 0.44 | 0.35 | 0.49 | 0.37 |
| counting (ceiling) | 0.34 | 0.35 | 0.46 | 0.40 | 0.35 | 0.53 |

**Prefill: cold, unique prompt, 1-token reply** (TTFT here is the whole prefill)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2K | 2,950 | 3.27 | 902 |
| 8K | 11,592 | 11.29 | 1,026 |
| 32K | 46,810 | 30.43 | **1,539** |
| 64K | 93,335 | 78.17 | 1,194 |

**Decode per stream, boot 9 → boot 10** (same prompts; boot 10 adds node-local Engram rows):

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| code | 52.4 → **73.8** | 60.4 → 45.0 | 33.4 → 33.8 | 31.3 → 34.3 | 28.5 → 44.1 | 26.3 → 41.5 |
| JSON | 38.6 → 52.1 | 37.1 → 30.4 | 20.0 → 20.7 | 20.8 → 22.7 | 18.8 → 26.8 | 16.2 → 27.5 |
| math | 46.9 → 50.9 | 46.2 → 59.3 | 46.2 → 54.4 | 49.1 → 31.4 | 27.6 → 28.9 | 25.8 → 34.3 |
| reasoning | 39.0 → 37.8 | 27.2 → 42.4 | 29.1 → 36.2 | 20.9 → 26.8 | 25.4 → 20.8 | 17.1 → 24.6 |
| tables (format) | 71.4 → 55.4 | 50.2 → 59.5 | 38.9 → 41.5 | 35.4 → 35.5 | 36.3 → 53.1 | 30.8 → 35.3 |
| summary | 24.7 → 25.7 | 17.4 → 22.4 | 13.6 → 22.4 | 11.5 → 13.7 | 11.4 → 15.9 | 10.9 → 16.1 |
| prose | 23.3 → 24.4 | 16.1 → 23.6 | 16.1 → 19.4 | 17.6 → 18.4 | 12.0 → 14.8 | 14.0 → 13.8 |
| narrative | 17.4 → 24.9 | 18.7 → 16.9 | 11.2 → 16.5 | 11.6 → 14.7 | 10.2 → 11.8 | 9.8 → 9.8 |
| counting (ceiling) | 77.2 → 62.2 | 56.0 → 58.9 | 39.3 → 43.0 | 40.2 → 58.3 | 36.5 → 57.2 | 38.1 → 33.4 |

**Step time, fast vs slow GPU state** (streamed; every speculative step is timed from the stream; [`results/boot10/idletest.txt`](results/boot10/idletest.txt)):

| request | ms/step | tokens/step | decode tok/s |
|---|---|---|---|
| count 1-100, first request after about 13 min idle | 94 (the whole request) | 5.85 | 61.7 |
| count, back to back | 63 | 5.85 | **92.2** |
| code, back to back | 70.5 | 4.97 | 70.2 |
| count, after 45 s idle | 63 | 5.85 | 92.1 |
| code, back to back | 67 | 5.30 | **77.3** |

**DSpark acceptance** (vLLM SpecDecoding metrics across the boot 10 bench, 35 ten-second windows): mean acceptance length **3.57** tokens per step, range 1.88-5.92.
- Counting, code and tables sit near the maximum of 6.
- Prose and narrative stay near 2, which is why per-stream speed spans 10-92 tok/s by content.

**What each fix bought (counting prompt, one stream, measured):**

| stack | tok/s |
|---|---|
| eager, no speculation (boot 6) | 5.1 |
| eager + DSpark k=5 (boot 7) | 19.5-22.1 |
| DSpark + CUDA graphs + Engram rows staged before the forward (boot 8) | 41.5 (count-to-100 check) |
| + GPU clock latch cleared on two nodes (boot 9) | 60.8 (count-to-100 check); 77.2 bench ceiling |
| **+ node-local Engram rows, GPU clocks locked (boot 10)** | **84.9** (count-to-100 check, same method); **92.2** decode back to back |

**Memory (boot 10):**

| | value |
|---|---|
| Weights per rank, with the DSpark draft layers and the vision encoder | 81.58 GiB |
| CUDA graphs | 1.99 GiB target + 0.55 GiB draft |
| KV | 5.15 GiB = 1,070,168 tokens (3.57x at 300K) |
| Node-local Engram rows per worker | 48 GB on disk (sparse copy of the rank's rows) |

**Earlier boots:**
- Boot 9: C1 per-stream 39.2, code 52.4 ([`results/boot9/`](results/boot9/)).
- Boot 8, before the GPU clock fix: C1 per-stream 24.2, code 32.9 ([`results/boot8/`](results/boot8/)).
- **1M max context (boot 7):** served at `--max-model-len 1048576` with DSpark: KV 1,078,380 tokens (5.21 GiB, 1.03x at 1M), eager, gmu 0.80.

## The model and the problem

`deepseek-ai/DeepSeek-V4.1-Flash`:
- 552B-backbone MoE (769B counting the Engram tables), 16B active decode / 8B prefill, 1M context.
- MXFP4 experts, MXFP8 dense, 510 GB on disk.

It does not fit four GB10s as shipped. The 296 GB of experts split four ways is fine. The problem is the two Engram n-gram tables (203 GB FP8): vLLM keeps them in host memory, and on a DGX Spark host memory *is* the GPU pool.

This repo:
- keeps the Engram tables on disk;
- fixes what that and the GB10's SM 12.1 break in day-0 vLLM;
- gets CUDA graphs working around a host-side lookup.

**Full recipe: [docs/RECIPE.md](docs/RECIPE.md)**

## What had to be fixed

In boot order. Details in `docs/`.

1. **Engram on disk** (`patch/engram.py`, `patch/weight_utils.py`): the two 101 GB tables stay in the safetensors files, and rows are read on demand. Without this it does not fit.
2. **Runtime JIT wedge** (boot 3). A FlashInfer MXFP8 GEMM compiled at runtime with 22 parallel jobs and exhausted host memory on all four nodes, and the watchdogs reset them. Fix: kernels prebuilt in the image (`build/build_overlay5.sh`) plus `MAX_JOBS=2`. [docs](docs/boot3-wedge-and-engram-offset.md)
3. **Engram rank offset** (found in audit). The disk reader ignored each rank's row offset, so ranks 1-3 read rank 0's rows, silently. [docs](docs/boot3-wedge-and-engram-offset.md)
4. **`No common block size for 64`** (boot 4). vLLM picked the smallest listed block size, and the V4 indexer backend refused it. Fix: `--block-size 128`. [docs](docs/boot4-block-size.md)
5. **DeepGEMM `block_kv == 32 or 64`** (boot 5). The ratio-1 indexer cache had 128 states per block. Fix: SM12x indexer pages of 64 states. [docs](docs/boot5-indexer-pages.md)
6. **Relaunch race.** A new worker joined the still-live old head's rendezvous on the same port. Fix: stop every node, head first, before launching. `scripts/dsv41-serve.sh stop` and the start path both do it.
7. **`persistent_topk` on GB10** (boot 7, long context). It oversubscribes the 48 SMs, and its fallback needs 128 KB of shared memory per block (GB10 has 99 KB). Fix: `top_k_per_row_decode`, which is also 1.6-3.6x faster there. [results](patch/sm12x-indexer-topk/RESULTS.md)
8. **Eager mode was the throughput ceiling.** About 200 ms per step, host-bound; the GPUs sat near idle. Fix: the Engram lookup moves out of the forward into `prepare_inputs` (`patch/model_state.py`), with all rows read in parallel. The whole decode step is then captured as a CUDA graph, with exact capture sizes so DSpark batches are never padded (FlashInfer #5015).
9. **GPU clock latch** (2 of 4 Sparks). Reddie and Asusi sat at 630-950 MHz with no visible cause. Every TP step waited for them. Fix: unplug the adapter for 30-60 s; a reboot does not clear it. Result: count 41.5 to 60.8 tok/s, code 32.9 to 57.1. [docs](docs/gpu-clock-latch.md)
10. **Engram rows over NFS** (boot 10).
    - Problem: the workers read their Engram rows from the head's NFS export. That took 5.9-7.8 ms per step against 2.8 ms on the head, and every step waited for the slowest rank.
    - Fix: `tools/engram_local.py` copies each worker's rows to local NVMe (47 GiB per worker, a sparse copy at the original offsets, verified row by row).
    - `patch/engram.py` reads the copy only when its recorded row range covers the rank, and falls back to NFS otherwise. It is **on by default** and `scripts/dsv41-serve.sh` builds the copies itself on the first boot.
    - The same boot fixed `boot_dsv41.sh`, which was not forwarding the patch folder to each node (now `PATCH_NAME`).

## Any TP

**TP is the node count**, and four Sparks is the only size this repo has measured.
Six is the interesting case: the checkpoint's attention heads, output groups and
MoE intermediate are all sized for TP=4, and vLLM asserts inside `divide()` — and
in `deepseek_v4_1/attention.py`'s own `assert self.n_heads % tp_size == 0` —
before the first forward.

The launcher's answer is to pad those dimensions UP rather than fall back to a
smaller TP. The padded slices are zeros, which are arithmetically inert:

```
6 nodes -> TP=6
  o_groups               64 ->   66
  num_attention_heads    64 ->   66      # one output group per head
  moe_intermediate_size 2048 -> 2304     # lcm(6, 128): stays whole FP8 blocks
```

Two halves, wired in automatically when the node count needs them:

| half | where | what |
|---|---|---|
| build the model padded | `make_overlay.py`, run by the container entrypoint | a `/model-tp6` directory that **symlinks** the 48 real shards and carries one rewritten `config.json` |
| pad the weights to match | `sitecustomize.py` on `PYTHONPATH` | wraps vLLM's `*_weights_iterator`, so each tensor is padded before anything narrows it for its rank |

Heads and groups move together and groups lead — `wo_a` is a batched matmul over
`o_groups`, so padding adds *whole groups* with their full set of dummy heads and
the per-group input width never changes. Engram is not padded at all: its tables
split by hash column and the forward already all-gathers then slices back to
`n_hash_cols`, so a rank past the last column simply owns nothing.

Details and the derivation: [`patch/dsv41_tp_pad/README.md`](patch/dsv41_tp_pad/README.md).
Before a first TP≠4 boot:

```bash
./scripts/dsv41-tp-probe.sh              # does this image still have the seams?
python3 tests/test_tp_pad.py --torch     # plan, overlay, shapes, numerics
```

> **Unverified.** The plan, the overlay and the padding are covered by those
> tests, but this repo has never *booted* at TP≠4 and none of the numbers on this
> page were measured there. Gate a padded boot on the greedy-reference check
> (`tools/postserve.sh`) against a TP=4 capture: the output must be identical,
> because every padded slice is inert. It is also not free — 2048 → 2304 is +12.5%
> of expert FFN width, in memory and in every matmul.

## Boot log

| boot | change | outcome |
|---|---|---|
| 1 | overlay1, eager, text-only, Engram on disk | KV 1,989,514. Died in decode warmup: FlashInfer 0.6.18 has no SM120 sparse-MLA decode kernel for `page_block_size=32`. |
| 2 | overlay2 + SWA override | Died at KV init: `No common block size for 32`. |
| 3 | overlay3 (FlashInfer 0.7.0rc1) + SM12x page patches | Wedged all 4 nodes (runtime JIT compile exhausted host memory). |
| 4 | overlay5 (kernels prebuilt) | No runtime JIT; KV 2,026,695. Died at KV init: `No common block size for 64`. |
| 5 | + `--block-size 128`, Engram offset fix, 300K | KV 2,346,690. Died in decode warmup: DeepGEMM `block_kv == 32 or block_kv == 64`. |
| 6 | + indexer pages of 64 states | **Served.** Eager, no speculation: 5.1 tok/s (count). Smoke clean, greedy reference 8/8, garble gate 30/30. |
| 7 | + DSpark k=5 at 1M max context | **Served at 1M.** KV 1,078,380. DSpark eager 19.5-22.1 tok/s. A 32K request then killed it in `persistent_topk` (fix 7). |
| 8 | + CUDA graphs, Engram staged before the forward, top-k fix, gmu 0.78, 300K | Served. Count check 41.5 tok/s, code 32.9 (two GPUs clock-latched, found later). |
| 9 | + tools, vision, gmu 0.80; Reddie and Asusi power-cycled to clear a GPU clock latch | Served. KV 1,032,963 (3.44x at 300K). Count-to-100 60.8 tok/s, code 57.1, count-to-300 68.2. Tool call and image OK. |
| 10 | + node-local Engram rows on the 3 workers (one change); GPU clocks locked by the owner before launch | **Serving.** KV 1,070,168 (3.57x at 300K). Count-to-100 84.9 tok/s, code 65.3 (end to end). Bench: C1 code 73.8, C6 aggregate 131.9. Tool call and image OK. |

## Known limits and next steps

- **GB10 GPU slow state** ([issue #1](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/issues/1); help wanted).
  - The GPU switches between a fast and a slow state that `nvidia-smi` does not show. A decode-shaped GEMV runs at 70 vs 230 GB/s.
  - In serving that is 63 vs 94 ms per step. A long idle reliably leaves it slow; it also flips during work, less often.
  - It reproduces with a plain PyTorch script (`tools/gpuflip.py`), so it is not the engine. Network, thermals, CPU placement, ASPM and KV caching are ruled out.
- **Vision and tools.**
  - Both on: up to 4 images per request, with the `deepseek_v41` tool and reasoning parsers.
  - Thinking is off by default; a request can turn it on with `"chat_template_kwargs": {"thinking": true}`.
  - FlashInfer #4973 (vision on SM120) did not reproduce in the 3 image tests, one of which sends two images in one message. Heavier image traffic and large photos are untested.
- **Adaptive verification is off.** It pads speculative batches, and padded batches can hang SM120 sparse MLA (FlashInfer #5015, open).
- **Step time in the fast state:** 63 ms on counting and 67-71 ms on code.
  - Measured parts: 88 all-reduces per step cost about 5 ms (`tools/nccl_lat.py`, stable across the four Sparks). The Engram staging still runs before each forward (2-3 ms with local rows).
  - The rest is the MoE and attention kernels, the next lever.
- **1M long context after the top-k fix.** The top-k fix is validated on GB10 at row widths up to 300,000. A 1M-token request has not been re-run on the fixed stack.
- **TP other than 4 has never been booted.** The padding shim that makes TP=6 or TP=8 possible is tested offline (`tests/test_tp_pad.py`) and probed against the image (`scripts/dsv41-tp-probe.sh`), but nothing on this page was measured at those sizes, and the lane seq counts in particular were sized at TP=4. MLA keeps its KV latent replicated on every rank, so a wider TP buys KV room only through the weights it frees — re-bench before trusting the lanes there. [details](patch/dsv41_tp_pad/README.md)
- **Host hardening** (system settings, not applied here): see [docs/RECIPE.md](docs/RECIPE.md) step 7.

## Repo layout

| path | what |
|---|---|
| `scripts/` | **The launch path.** `cluster.env` (the one file you edit) + `lib.sh`, `dsv41-serve.sh` (`start\|stop\|status\|logs\|dry-run`), `dsv41-node-launch.sh` (per-rank `docker run`, workers first), `dsv41-container-entrypoint.sh` (RoCEv2 GID pick, TP config overlay), `engram-local.sh` (node-local Engram rows, run automatically by `serve`), `roce-check.sh`, `dsv41-tp-probe.sh`, `fetch-weights.sh`, `build-image.sh`, `copy-image.sh`. |
| `patch/` | The exact files bind-mounted over vLLM (md5s in `patch/README.md`), plus one folder per fix with its diff and test. `dsv41_tp_pad/` is the TP padding shim (not bind-mounted over vLLM — it rides on `PYTHONPATH`). |
| `build/` | Image chain: overlay1 (the dsv41-feat tree + `_C_stable_libtorch` for sm121), overlay3 (FlashInfer 0.7.0rc1), overlay4/5 (prebuilt kernels). overlay1 needs a `vllm/` tree the repo cannot ship; `scripts/build-image.sh --check` explains how to get it now that the branch is gone. |
| `launch/` | The original path, still working: `dsv41-tp4.sh <rank>`, `boot_dsv41.sh` (worker-first fan-out), one `bootN-go.sh` per boot. `boot10-go.sh` is the config every number here was measured on. |
| `tools/` | Pre-launch steps, boot poll, post-serve checks, bench report, `engram_local.py` / `engram_ranges.py` (node-local Engram rows and their per-rank ranges), `gpuflip.py` / `flipsum.py` (GPU slow-state probe), `nccl_lat.py` (all-reduce check), `idletest.py` (per-step timing after idle vs back to back), `vision_tools_demo.py` (vision and tool-calling checks). |
| `tests/` | `test_tp_pad.py`: the padding plan, the config overlay, and the padded shapes/numerics. |
| `bench/` | Fixed prompt set v1, C1-C6 bench, long-context needle test. |
| `docs/` | Recipe and one post-mortem per failure. |
| `results/` | Head logs of every boot, proofs, bench output, pre-launch GPU and network checks. |

**Fleet (the measured one):**
- Reddie is the head: model on local NVMe, exported over NFS.
- Asusi, Bluey and Spark4 are workers, each with a local copy of its Engram rows.
- ConnectX-7 RoCE fabric, 192.168.192.0/24. Dual rail (`rocep1s0f0` + `roceP2p1s0f0`) is the launcher's default; `./scripts/roce-check.sh` pings every worker on each rail from the head's address on that same rail, which is what catches a rail that has a link but no route and silently halves the bandwidth.

The topology is not baked into any script — `scripts/cluster.env` holds it, and it
is gitignored.

**Sister repos:** [DeepSeek-V4-Flash-Vision-Exp (vLLM, 2x/4x Spark)](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark) · [DeepSeek-V4-Flash-Vision (SGLang, 2x Spark)](https://github.com/tonyd2wild/DeepSeek-V4-Flash-Vision-SGLang-DGX-Spark)

**Credits:**
- The vLLM team, for the day-0 `dsv41-feat` branch (since merged into mainline and deleted; the day-0 images live on as dated `deepseekv41-flash-0909*` tags).
- Kai, for the first Engram-on-disk patch and the SM12x page-size patches.
- The Engram-on-disk idea follows our own PLE-on-disk patch for Qwen3.8-Flash-Next.
- Prior art acknowledged at the idea level: vLLM PR #54129 (`VLLM_PLE_MMAP`). No code was copied.
