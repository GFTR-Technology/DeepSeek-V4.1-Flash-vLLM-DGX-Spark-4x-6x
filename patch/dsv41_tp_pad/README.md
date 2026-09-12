# `dsv41_tp_pad/`: TP alignment by padding

**Problem.** TP is the node count. Four Sparks divide this checkpoint; six do
not, and vLLM asserts inside `divide()` — and in
`deepseek_v4_1/attention.py`'s own `assert self.n_heads % tp_size == 0` — before
the first forward.

**What this does.** Pads the dimensions UP so TP divides them, instead of
dropping the fleet to a smaller TP. With one output group per head, 64 attention
heads become 66 at TP=6; the two extra heads are zero and contribute exactly
nothing.

Two halves, and they have to tell vLLM the same story:

| half | where | what |
|---|---|---|
| build the model padded | `make_overlay.py`, run by the container entrypoint | writes a `config.json` with the padded dims into `/model-tp6`, a directory that **symlinks** the 48 real shards (zero copy, no extra disk) |
| pad the weights to match | `sitecustomize.py` + `dsv41_tp_pad.py` on `PYTHONPATH` | wraps vLLM's `*_weights_iterator` generators, so each tensor is padded before anything narrows it for this rank |

`sitecustomize.py` is how the shim reaches **every** rank: vLLM's `mp` executor
starts each worker as a fresh interpreter, and `site` imports `sitecustomize` in
all of them. It stays cheap — no torch, no vllm — and only registers a
post-import hook.

The launcher never re-derives any of this in bash. It runs
`dsv41_tp_pad.py --tp N --model $WEIGHTS --shell` and `eval`s the result, so the
shell and the container share one implementation of the plan.

## Dimensions come from the checkpoint

Unlike the GLM shim this one does not hard-code `64`/`2048`. DeepSeek-V4.1-Flash
ships the TP-sensitive extents in `config.json` and `load_dims()` reads them:
`num_attention_heads`, `o_groups`, `head_dim`, `o_lora_rank`,
`moe_intermediate_size`, `intermediate_size`, and the quantization block from
`quantization_config.weight_block_size`. Hard-coded constants would become a lie
the first time the card is re-uploaded. Everything read is then validated, and
`make_overlay.py` re-checks the config against the plan before rewriting it.

## Why zero padding is exact

| group | dummy slice | why the output does not move |
|---|---|---|
| `attn` | `wo_a` rows for a dummy group | that group's `o_lora` latent is 0, and `wo_b`'s columns for it are 0 |
| `attn` | `wq_b` rows, `attn_sink` for a dummy head | its q is 0, so it attends uniformly to something finite — which is then multiplied by its group's zero `wo_a` rows. Its sink is `-inf`, so it adds nothing to the softmax either. |
| `moe` / `dense` | `gate_proj`/`up_proj` rows, `down_proj` columns | `act(0) * 0 = 0` |
| `vocab` | extra embedding rows | no weight rule at all — see below |

## Two constraints the plan has to respect

**1. Heads and groups move together, and groups lead.** `wo_a` is a batched
matmul over `o_groups` (`attention.py`: `is_bmm = True`,
`bmm_batch_size = n_local_groups`), each group consuming exactly
`num_attention_heads / o_groups` heads — its per-group input width is
`num_heads*head_dim/o_groups`. That ratio is structural; changing it would re-cut
the checkpoint. So padding adds *whole groups*, each carrying its full set of
dummy heads:

```
heads_per_group = heads / groups          # must already be exact
groups'         = round_up(groups, tp)
heads'          = groups' * heads_per_group
```

`groups'` is a multiple of tp, so `heads'` is too, for free. The per-group input
width never moves, which is why `wo_a`'s second dimension has no rule at all.

**2. A quantized axis must stay a whole number of blocks.** A block-quantized
weight `[N, K]` carries a scale of `[N/block, K/block]`. Pad `N` off a block
boundary and the scale has a fractional number of rows — the checkpoint cannot be
padded at all. So FFN widths round to a multiple of `tp * block`:
`moe_intermediate_size` 2048 at TP=6 with 128-wide blocks goes to **2304**, not
2052. Head and group counts need no such rounding: they enter the tensors
multiplied by `head_dim` / `o_lora_rank`, which are already multiples of the
block. `_validate()` asserts the per-rank shard is still a whole number of blocks
and refuses the plan otherwise.

**A padded scale cannot be filled with zero.** MX scales are stored as E8M0, a
bare exponent: the format has no zero and no -inf, and `torch.full(..., 0.0,
dtype=float8_e8m0fnu)` raises *"value cannot be converted to type
c10::Float8_e8m0fnu without overflow"*. Padded E8M0 rows are filled with **1.0**
instead. That is inert: the scale multiplies the data row it belongs to, and that
row was padded with zero.

**A scale is recognised by a declared block ratio, nothing looser.** A rule sized
for a weight also has to cover that weight's blockwise scale, which is the same
extent divided by the quantization block. Accepting *any* small divisor is too
loose: the 32-head sparse indexer's `wq_b` is `[4096, 1280]` against the main
attention's 32768 — a ratio of 8 — and it is `ReplicatedLinear`, never sharded.
Padding it produced *"Tried to load weights of size [6144, 1280] to a parameter of
size [4096, 1280]"*. Only ratios listed in the checkpoint's
`quantization_config.weight_block_size` are accepted now, so a ratio of 8 (or of
256, for the indexer's own scale) is skipped.

Rules key on the **module** (`.wq_b`), not the parameter, so `.wq_b.weight` and
`.wq_b.weight_scale_inv` are padded consistently; each scale's block is inferred
from the ratio of its size to the weight's. A per-tensor `input_scale` (size 1)
is never grown.

## The vocab is rounded, not padded

vLLM shards the embedding *after* rounding the vocab up with `pad_vocab_size`
(default unit 64). This checkpoint's 129280 is a whole number of 64s but not of
6, so `divide(129280, 6)` asserts inside `VocabParallelEmbedding`. The `vocab`
group makes that rounding land on `lcm(64, tp)` instead — 192 at TP=6, giving
129408, which is a whole number of both 6 and 64.

No weight rule accompanies it, and none is wanted: vLLM already builds the
embedding at the padded size and its loader fills only the real rows, leaving the
rest zero. The logits processor slices back to `org_vocab_size`, so the extra 128
rows can never be sampled. The hook therefore only rewrites `pad_vocab_size` in
`vllm.model_executor.layers.vocab_parallel_embedding`, and `make_overlay.py`
deliberately leaves `vocab_size` in the config alone — the checkpoint's embedding
rows have not moved.

`lcm(64, tp)` is 64 whenever tp divides 64, so at TP=4 or TP=8 this group is a
no-op. That is why it never surfaced until TP=6.

## Engram is not padded

The Engram tables are split by hash column, and
`ParallelEngramEmbedding.forward` already all-gathers every rank's columns and
slices back to `n_hash_cols`. Under a TP that does not divide the column count
the last rank(s) simply own nothing, and their zeros are sliced away. Padding
would be wrong anyway: the rows are primes generated from `engram_vocab_size`, so
a synthetic column has no rows on disk.

Two places in `patch/engram.py` assumed that could not happen, and both are fixed
there (`engram-uneven-tp.diff`):

- `EngramDiskStager.__init__` sized a buffer `head_end - head_start`, which goes
  **negative** once `head_start` passes `n_hash_cols`. Clamped to zero width.
- `gather_dequant_many` always issued one de-duplicated read, and a rank that
  owns no rows has its `row_start` at the very end of the table, so that read
  runs off it. It now returns zeros without reading when nothing is owned.

Node-local Engram copies stay valid data across a TP change, but they cover a
different rank's range — regenerate with `tools/engram_ranges.py`.

## Files

| file | what |
|---|---|
| `dsv41_tp_pad.py` | dims, plan, rules, the padding, the loader hooks, and the `--shell` CLI |
| `sitecustomize.py` | the `PYTHONPATH` shim that reaches every rank |
| `make_overlay.py` | the symlink overlay with the padded `config.json` |
| `probe.py` | read-only report on whether this image still has the seams |
| `engram-uneven-tp.diff` | the two `patch/engram.py` clamps |

## Before the first TP≠4 boot

```bash
./scripts/dsv41-tp-probe.sh        # does this image still have the seams?
python3 tests/test_tp_pad.py       # plan + overlay
python3 tests/test_tp_pad.py --torch   # + shapes and numerics (on a Spark)
```

## Status

The plan, the overlay and the padding function are covered by the tests above.
**The full path has not been run against the real checkpoint** — this repo has
never booted at TP≠4. Before trusting a padded boot, run the greedy-reference
gate against a TP=4 capture (`tools/postserve.sh`): the padded model must produce
identical greedy output, because every padded slice is arithmetically inert. If
it does not, the padding is wrong somewhere, not "close enough".

Padding also costs: `moe_intermediate_size` 2048 → 2304 is +12.5% of expert FFN
width, in memory and in every matmul.
