## boot8 (2026-09-10T13:19:24Z)

boot8 FINAL: 4x DGX Spark TP4, vllm-dsv41:overlay5 (vLLM dsv41-feat + FlashInfer 0.7.0rc1), DSpark k=5 (adaptive verification off), CUDA graphs FULL_AND_PIECEWISE (capture sizes 5..48), Engram rows staged before the forward, SM12x indexer top-k fix, 300K ctx, gmu 0.78, block 128, text-only, thinking off

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 20.26 | 24.22 | 1.004 |
| C2 | 34.03 | 21.07 | 1.78 |
| C3 | 53.19 | 21.75 | 0.933 |
| C4 | 64.13 | 18.93 | 0.926 |
| C5 | 64.61 | 15.53 | 1.204 |
| C6 | 82.01 | 16.78 | 1.18 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 32.92 | 27.44 | 35.27 | 31.0 | 19.86 | 27.03 |
| json | 23.72 | 21.08 | 21.25 | 18.94 | 11.78 | 15.71 |
| narrative | 12.4 | 10.37 | 8.3 | 7.2 | 7.68 | 7.58 |
| prose | 15.23 | 14.08 | 12.05 | 9.92 | 9.23 | 8.6 |
| math | 30.6 | 27.52 | 28.29 | 23.44 | 24.87 | 20.2 |
| reasoning | 22.81 | 16.19 | 17.51 | 14.27 | 15.78 | 17.33 |
| summary | 17.0 | 14.09 | 14.17 | 12.14 | 10.62 | 9.31 |
| format | 39.09 | 37.75 | 37.19 | 34.55 | 24.41 | 28.48 |
| ceiling_count | 40.46 | 41.3 | 44.52 | 33.1 | 27.64 | 34.56 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
