# `dsv41_tp_pad/`: TP alignment by padding

**Problem.** TP is the node count. Four Sparks divide this checkpoint; six do
not, and vLLM asserts inside `divide()` — and in
`deepseek_v4_1/attention.py`'s own `assert self.n_heads % tp_size == 0` — before
the first forward.

**What this does.** Pads the dimensions UP so TP divides them, instead of
dropping the fleet to a smaller TP. At TP=6 this checkpoint's 8 output groups
become 12 and its 64 attention heads become 96; the extra groups and heads are
zero and contribute exactly nothing.

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
padded at all. So FFN widths round to a multiple of `tp * block`: a 2048-wide
`moe_intermediate_size` at TP=6 with 128-wide blocks would go to **2304**, not
2052. (This checkpoint's is already 2304 with 32-wide blocks, so it does not move
at TP=6 — but a re-upload could change either number, which is why the rule is
computed rather than written down.) Head and group counts need no such rounding:
they enter the tensors multiplied by `head_dim` / `o_lora_rank`, which are
already multiples of the block. `_validate()` asserts the per-rank shard is still
a whole number of blocks and refuses the plan otherwise.

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

## Expert counts: the draft is padded, the backbone is not

Experts shard by **count**, so the count has to divide TP. This checkpoint has
two of them (`tools/inspect_experts.py`):

| where | blocks | experts | 384 or 128 mod 6 |
|---|---|---|---|
| backbone `layers.0..39.ffn` | 40 | 384 | 0 — already fine |
| DSpark draft `mtp.0..2.ffn` | 3 | 128 | 2 — does not divide |

vLLM's own remedy, `num_redundant_experts`, cannot fix this: it is a single
global number added to every MoE, and no `r` satisfies `384 + r ≡ 0` and
`128 + r ≡ 0 (mod 6)` at once. That is the dead end the cluster kept hitting as
`n_physical_experts=388 must be divisible by tp_size=6`.

So the plan pads the **draft's count instead**, 128 → 192 at TP=6 (why not 132
— see below), and leaves the backbone alone. Two things make a dead expert
harmless there:

- Each block carries a `gate.bias`, one entry per expert, added to the routing
  score. A dead expert's entry is set to `NEG_BIG` (−1e4, not −inf: `0 * -inf`
  is a NaN, and real logits sit within about ±20), which keeps it out of top-k.
- Even if one were selected, this is a *speculative draft* — every token it
  proposes is verified against the real model before it is emitted. The cost
  would be acceptance rate, i.e. speed, never correctness.

Neither argument holds for the backbone, so a backbone count that does not
divide TP is still refused with the `num_redundant_experts` advice.

Unlike every other rule here, this one has to **create** tensors: ids 128..191
do not exist in the checkpoint, and vLLM's fused-MoE loader fills its expert
slots one id at a time, so a slot nobody loads keeps whatever `torch.empty` left
in it. `Padder.extra` emits explicit zero experts (1.0 for e8m0 scales, which
cannot represent 0) keyed on expert 0 of each draft block.

The router's own per-expert tensors are found by **shape, not by name**
(`Padder._expert_indexed`): inside a draft block a dim-0 extent equal to the
expert count *is* the expert axis. Guessing the module name was wrong on the
first cluster boot — this build calls it `mtp.N.ffn.gate.bias_vl`, not
`gate.bias` — and a false positive cannot pass silently, because the parameter
it loads into would then have the wrong size and assert. A 2-D candidate must
also be `[experts, hidden]`: `index_head_dim` is 128, the same as the draft's
expert count, so an indexer weight would otherwise qualify.

### The count also has to be one the routing kernel was compiled for

`topk_softplus_sqrt_kernels.cu` templates on the expert count and rejects
anything else:

```
RuntimeError: topkGatingSoftplusSqrtKernelLauncher, ...:841,
Unsupported expert number: 132
```

It dispatches on `1 2 4 8 16 32 64 128 192 256 320 384 448 512 576` — powers of
two to 128, then multiples of 64. So at TP=6 the draft's 128 cannot go to 132.

The target is therefore rounded up **twice**: first to a multiple of `tp`, then
on to the next count the kernel dispatches on that is *also* a multiple of `tp`
— 128 → 132 → **192**. Both steps are automatic; no environment variable is
needed for TP=6. A count that is legal but does not divide `tp` is skipped over
(160 at TP=6), and if nothing in the set is reachable the plan **refuses** rather
than emitting a number that dies at the first forward, pointing at
`DSV41_SPEC=none`.

That set is not read from the checkpoint — it belongs to the image — so it lives
in `KERNEL_EXPERT_COUNTS` and both halves are overridable:

| variable | when |
|---|---|
| `DSV41_KERNEL_EXPERT_COUNTS=128,192,384` | a build with a different dispatch set. `tools/kernel_expert_counts.sh` reads the real one out of the kernel source or the image |
| `DSV41_DRAFT_EXPERTS=384` | pick the target outright. A value off the dispatch set only warns — the operator may know something the table does not |

The launcher passes either into every container when set, because a rank that
plans a different count than the host would disagree with the overlay config.

Cost at 192: +64 dead experts per block over 3 `mtp` blocks. With
`hidden_size` 5120, `moe_intermediate_size` 2304 and `expert_dtype: fp4` that is
~3.2 GiB of zeros, ~0.53 GiB per rank at TP=6. Compute is unaffected — a dead
expert never enters top-k, so it is never assigned a token.

There is no `n_group`/`topk_group` in this config (`topk_method: noaux_tc` runs
ungrouped), so growing the count does not move any group boundary and the real
experts still compete exactly as they did at 128.

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
TP=6 has booted on the real cluster with `DSV41_SPEC=none` (no draft), so the
attention, vocab and overlay halves are real; the expert half has not been
through a full boot yet.

**Greedy output has not been compared.** Every padded slice is arithmetically
inert by construction, but that is an argument, not a measurement. Before
trusting a padded boot, run the greedy-reference gate against a TP=4 capture
(`tools/postserve.sh`): `temperature=0`, same prompts, output must be identical.
If it is not, the padding is wrong somewhere, not "close enough".

Padding also costs. At TP=6 on this checkpoint: `o_groups` 8 → 12 and heads
64 → 96 is +50% of attention projection width, and the draft's 128 → 192 experts
is ~0.53 GiB of zeros per rank. `moe_intermediate_size` is 2304, which already
divides 6, so the FFN width does not move.
