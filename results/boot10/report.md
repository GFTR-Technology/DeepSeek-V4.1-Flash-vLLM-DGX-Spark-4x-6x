## boot10 (2026-09-10T16:09:56Z to 2026-09-10T16:17:42Z)

boot10: boot 9 config (DSpark k=5, CUDA graphs FULL_AND_PIECEWISE, Engram staged before the forward, SM12x top-k fix, tools on, vision on, 300K ctx, gmu 0.80, block 128, thinking off) + ONE change: node-local Engram rows on the 3 workers. GPU clocks locked by Tony before launch. KV pool 1,070,168 tokens.

Prompt set `v1`, temperature 0, thinking off, streaming; one batch per cell (C streams released together). Short prompts (about 30-120 tokens), 150-256 token budgets.

### Decode: per-stream tok/s after the first token

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 73.8 | 45.0 | 33.8 | 34.3 | 44.1 | 41.5 |
| json | 52.1 | 30.4 | 20.7 | 22.7 | 26.8 | 27.5 |
| math | 50.9 | 59.3 | 54.4 | 31.4 | 28.9 | 34.3 |
| reasoning | 37.8 | 42.4 | 36.2 | 26.8 | 20.8 | 24.6 |
| format | 55.4 | 59.5 | 41.5 | 35.5 | 53.1 | 35.3 |
| summary | 25.7 | 22.4 | 22.4 | 13.7 | 15.9 | 16.1 |
| prose | 24.4 | 23.6 | 19.4 | 18.4 | 14.8 | 13.8 |
| narrative | 24.9 | 16.9 | 16.5 | 14.7 | 11.8 | 9.8 |
| counting (ceiling) | 62.2 | 58.9 | 43.0 | 58.3 | 57.2 | 33.4 |

### Aggregate throughput: tok/s across all streams (wall time, TTFT included)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 66.5 | 81.1 | 92.4 | 123.8 | 196.7 | 225.5 |
| json | 43.7 | 49.2 | 51.9 | 76.3 | 94.7 | 130.4 |
| math | 45.6 | 105.5 | 148.4 | 110.2 | 127.9 | 182.7 |
| reasoning | 35.0 | 75.3 | 96.5 | 97.0 | 94.1 | 133.6 |
| format | 44.0 | 94.3 | 101.7 | 112.0 | 215.1 | 182.1 |
| summary | 21.9 | 37.6 | 38.2 | 43.1 | 64.0 | 73.5 |
| prose | 22.9 | 41.5 | 55.2 | 69.9 | 67.9 | 74.8 |
| narrative | 23.8 | 30.0 | 45.2 | 53.4 | 53.1 | 52.4 |
| counting (ceiling) | 57.3 | 109.0 | 118.6 | 208.8 | 259.9 | 184.9 |

### TTFT: mean time to first token (s)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 0.31 | 0.44 | 0.49 | 0.51 | 0.40 | 0.42 |
| json | 0.33 | 0.50 | 0.55 | 0.57 | 0.45 | 0.42 |
| math | 0.47 | 0.35 | 0.38 | 0.46 | 0.62 | 0.42 |
| reasoning | 0.44 | 0.35 | 0.42 | 0.55 | 0.57 | 0.41 |
| format | 0.51 | 0.40 | 0.57 | 0.59 | 0.47 | 0.47 |
| summary | 0.79 | 0.77 | 3.18 | 1.28 | 1.33 | 1.02 |
| prose | 0.38 | 0.39 | 0.31 | 0.32 | 0.36 | 0.48 |
| narrative | 0.29 | 0.32 | 0.44 | 0.35 | 0.49 | 0.37 |
| counting (ceiling) | 0.34 | 0.35 | 0.46 | 0.40 | 0.35 | 0.53 |

### End-to-end per stream: tok/s over the whole request, TTFT included

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 66.5 | 41.1 | 31.3 | 31.6 | 40.7 | 38.3 |
| json | 43.7 | 26.1 | 18.4 | 19.9 | 23.6 | 24.5 |
| math | 45.6 | 53.9 | 49.5 | 29.4 | 26.6 | 32.1 |
| reasoning | 35.0 | 39.6 | 33.8 | 25.1 | 19.7 | 23.5 |
| format | 44.1 | 48.6 | 33.9 | 29.7 | 43.1 | 30.7 |
| summary | 22.0 | 19.2 | 13.7 | 11.9 | 13.5 | 13.9 |
| prose | 22.9 | 22.1 | 18.7 | 17.7 | 14.3 | 13.2 |
| narrative | 23.8 | 16.4 | 15.8 | 14.2 | 11.4 | 9.6 |
| counting (ceiling) | 57.3 | 54.5 | 39.9 | 53.3 | 53.0 | 31.2 |

### Decode per stream vs boot9

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 52.4 → 73.8 | 60.4 → 45.0 | 33.4 → 33.8 | 31.3 → 34.3 | 28.5 → 44.1 | 26.3 → 41.5 |
| json | 38.6 → 52.1 | 37.1 → 30.4 | 20.0 → 20.7 | 20.8 → 22.7 | 18.8 → 26.8 | 16.2 → 27.5 |
| math | 46.9 → 50.9 | 46.2 → 59.3 | 46.2 → 54.4 | 49.1 → 31.4 | 27.6 → 28.9 | 25.8 → 34.3 |
| reasoning | 39.0 → 37.8 | 27.2 → 42.4 | 29.1 → 36.2 | 20.9 → 26.8 | 25.4 → 20.8 | 17.1 → 24.6 |
| format | 71.4 → 55.4 | 50.2 → 59.5 | 38.9 → 41.5 | 35.4 → 35.5 | 36.3 → 53.1 | 30.8 → 35.3 |
| summary | 24.7 → 25.7 | 17.4 → 22.4 | 13.6 → 22.4 | 11.5 → 13.7 | 11.4 → 15.9 | 10.9 → 16.1 |
| prose | 23.3 → 24.4 | 16.1 → 23.6 | 16.1 → 19.4 | 17.6 → 18.4 | 12.0 → 14.8 | 14.0 → 13.8 |
| narrative | 17.4 → 24.9 | 18.7 → 16.9 | 11.2 → 16.5 | 11.6 → 14.7 | 10.2 → 11.8 | 9.8 → 9.8 |
| counting (ceiling) | 77.2 → 62.2 | 56.0 → 58.9 | 39.3 → 43.0 | 40.2 → 58.3 | 36.5 → 57.2 | 38.1 → 33.4 |

### Cold prefill (unique prompt, 1-token reply; TTFT = whole prefill)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2K | 2,950 | 3.27 | 902 |
| 8K | 11,592 | 11.29 | 1,026 |
| 32K | 46,810 | 30.43 | 1,538 |
| 64K | 93,335 | 78.17 | 1,194 |
