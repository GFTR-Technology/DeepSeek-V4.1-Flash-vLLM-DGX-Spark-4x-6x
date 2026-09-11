## boot10 (2026-09-10T16:09:56Z)

boot10: boot 9 config (DSpark k=5, CUDA graphs FULL_AND_PIECEWISE, Engram staged before the forward, SM12x top-k fix, tools on, vision on, 300K ctx, gmu 0.80, block 128, thinking off) + ONE change: node-local Engram rows on the 3 workers. GPU clocks locked by Tony before launch. KV pool 1,070,168 tokens.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 37.95 | 43.12 | 0.441 |
| C2 | 64.3 | 37.43 | 0.441 |
| C3 | 78.7 | 30.62 | 0.793 |
| C4 | 85.72 | 24.66 | 0.579 |
| C5 | 114.2 | 27.01 | 0.585 |
| C6 | 131.86 | 25.35 | 0.502 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 73.78 | 44.98 | 33.76 | 34.26 | 44.1 | 41.47 |
| json | 52.1 | 30.4 | 20.66 | 22.69 | 26.77 | 27.5 |
| narrative | 24.93 | 16.88 | 16.53 | 14.66 | 11.75 | 9.79 |
| prose | 24.37 | 23.63 | 19.44 | 18.37 | 14.8 | 13.76 |
| math | 50.86 | 59.29 | 54.44 | 31.38 | 28.88 | 34.26 |
| reasoning | 37.79 | 42.35 | 36.19 | 26.78 | 20.82 | 24.55 |
| summary | 25.72 | 22.43 | 22.42 | 13.69 | 15.88 | 16.09 |
| format | 55.41 | 59.48 | 41.53 | 35.46 | 53.1 | 35.35 |
| ceiling_count | 62.19 | 58.94 | 42.99 | 58.27 | 57.23 | 33.41 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 3.27 | 902.2 |
| 8000 | 11592 | 11.293 | 1026.5 |
| 32000 | 46810 | 30.426 | 1538.5 |
| 64000 | 93335 | 78.173 | 1194.0 |
