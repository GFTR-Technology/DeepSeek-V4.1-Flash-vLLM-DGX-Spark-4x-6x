## boot9 (2026-09-10T14:28:19Z)

boot9: 4x DGX Spark TP4 (all 4 GPUs unlatched after power-cycling Reddie and Asusi), vllm-dsv41:overlay5, DSpark k=5, CUDA graphs FULL_AND_PIECEWISE, Engram rows staged before the forward, SM12x top-k fix, tools on (deepseek_v41 parsers), vision on, 300K ctx, gmu 0.80, block 128, thinking off

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 33.88 | 39.2 | 0.533 |
| C2 | 58.99 | 34.17 | 0.704 |
| C3 | 67.56 | 26.07 | 0.61 |
| C4 | 85.5 | 24.75 | 0.592 |
| C5 | 91.96 | 21.25 | 0.692 |
| C6 | 97.99 | 18.87 | 0.736 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 52.37 | 60.44 | 33.39 | 31.25 | 28.53 | 26.25 |
| json | 38.59 | 37.06 | 19.99 | 20.81 | 18.76 | 16.19 |
| narrative | 17.41 | 18.72 | 11.22 | 11.56 | 10.16 | 9.85 |
| prose | 23.26 | 16.13 | 16.08 | 17.57 | 11.96 | 14.02 |
| math | 46.89 | 46.24 | 46.24 | 49.08 | 27.55 | 25.78 |
| reasoning | 38.98 | 27.17 | 29.08 | 20.87 | 25.39 | 17.13 |
| summary | 24.73 | 17.41 | 13.61 | 11.49 | 11.38 | 10.87 |
| format | 71.41 | 50.21 | 38.94 | 35.36 | 36.29 | 30.85 |
| ceiling_count | 77.17 | 55.96 | 39.3 | 40.21 | 36.53 | 38.1 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
