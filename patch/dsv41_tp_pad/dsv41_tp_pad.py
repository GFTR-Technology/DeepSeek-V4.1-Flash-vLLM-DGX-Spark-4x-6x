"""TP padding for DeepSeek-V4.1-Flash: pad the model up to the node count.

TP is the node count. Four Sparks divide this checkpoint; six do not, and vLLM
asserts inside ``divide()`` before the first forward. Rather than dropping to a
smaller TP, the dimensions are padded UP and the padded slices are filled with
zeros, which are arithmetically inert.

Two halves, and they have to agree:

* ``make_overlay.py`` writes a ``config.json`` carrying the padded dims into a
  symlink overlay of the model directory, so vLLM BUILDS every layer padded.
* this module hooks the weight loader and pads each checkpoint tensor as it
  streams in, so the weights MATCH what was built.

Unlike the GLM shim this one does not hard-code the checkpoint's dimensions:
DeepSeek-V4.1-Flash ships them in ``config.json`` and they are read from there,
then validated. Hard-coded constants would be a lie the first time the card is
re-uploaded.

Why zero padding is exact, per group:

``attn``   ``wo_a`` is a batched matmul over ``o_groups``, each group consuming
           exactly ``num_attention_heads / o_groups`` heads. That ratio is
           structural, so padding adds WHOLE GROUPS, each carrying its full set
           of dummy heads::

               groups' = round_up(o_groups, tp)
               heads'  = groups' * (heads / groups)

           ``groups'`` is a multiple of tp, so ``heads'`` is too, for free. With
           one group per head that is 64 heads -> 66 at TP=6. A dummy head's
           ``wq_b`` rows are zero so its q is zero; whatever it attends to is
           multiplied by its group's zero ``wo_a`` rows. Its ``attn_sink`` is
           ``-inf``. A dummy group's ``wo_a`` rows are zero, and ``wo_b``'s
           columns for it are zero.
``moe``    ``gate_proj``/``up_proj`` rows and ``down_proj`` columns are zero, so
           ``act(0)*0 = 0``. Rounded to ``tp * block`` so every rank still owns
           whole blockwise-FP8 blocks.
``dense``  same, for the dense FFN of the first layers.

Engram is deliberately absent from the groups. Its tables are split by hash
column and ``ParallelEngramEmbedding.forward`` already all-gathers every rank's
columns and slices back to ``n_hash_cols``, so a rank past the last column simply
owns nothing and its zeros are dropped. ``patch/engram.py`` carries the two
clamps that stop such a rank from building a negative-width buffer or reading off
the end of the table. Padding it would be wrong: the rows are primes generated
from ``engram_vocab_size`` and a synthetic column has no rows on disk.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable, Iterable

LOG_PREFIX = "[dsv41-tp-pad]"

# Blockwise FP8/MXFP8 scale granularity. A block-quantized weight [N, K] carries
# a scale of [N/block, K/block]; pad N off a block boundary and that scale has a
# fractional number of rows, which cannot be expressed at all.
DEFAULT_BLOCK = 128

ALL_GROUPS = ("attn", "moe", "dense", "vocab", "experts")
DEFAULT_GROUPS = ("attn", "moe", "dense", "vocab", "experts")

# vLLM already rounds the vocab up to a multiple of this before sharding it
# (DEFAULT_VOCAB_PADDING_SIZE). The real value is read from the module at
# patch time; this is only the plan-side default.
VOCAB_PAD_BASE = 64


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


# ---------------------------------------------------------------------------
# Dimensions, read from the checkpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dims:
    """The TP-sensitive extents, as the checkpoint ships them.

    Only ``heads`` is required. Everything else is optional: a checkpoint that
    does not carry ``o_groups`` simply has its heads padded directly, and rules
    whose sizes we cannot compute are not emitted. Refusing to plan because one
    optional key is absent is how you end up booting unpadded.
    """

    heads: int
    groups: "int | None" = None
    head_dim: "int | None" = None
    o_lora_rank: "int | None" = None
    moe_intermediate: "int | None" = None
    intermediate: "int | None" = None
    vocab_size: "int | None" = None
    #: every {key: value} in the config that looks like an expert count,
    #: including any nested draft/dspark section. Experts shard by COUNT.
    expert_counts: tuple = ()
    block: int = DEFAULT_BLOCK
    #: every distinct quantization block the checkpoint declares. A tensor
    #: may only be treated as a scale of another if their sizes differ by
    #: exactly one of these.
    blocks: tuple = (DEFAULT_BLOCK,)

    @property
    def heads_per_group(self) -> int:
        return self.heads // self.groups if self.groups else 1


def _flatten(raw: dict) -> dict:
    flat = {}
    for key in ("text_config", "language_config", "llm_config"):
        sub = raw.get(key)
        if isinstance(sub, dict):
            flat.update(sub)
    flat.update({k: v for k, v in raw.items() if not isinstance(v, dict)})
    return flat


def _quant_blocks(raw: dict) -> tuple:
    """Every distinct block size the checkpoint's quantization declares."""
    q = raw.get("quantization_config")
    sizes = []
    if isinstance(q, dict):
        for key in ("weight_block_size", "block_size", "group_size"):
            v = q.get(key)
            if isinstance(v, int) and v > 1:
                sizes.append(v)
            elif isinstance(v, (list, tuple)):
                sizes += [x for x in v if isinstance(x, int) and x > 1]
    return tuple(sorted(set(sizes))) if sizes else (DEFAULT_BLOCK,)


#: an expert count that belongs to the speculative draft model, not the
#: backbone. With speculative decoding off the draft is never built, so its
#: count cannot assert and must not block the boot.
_DRAFT_EXPERT_KEY = re.compile(r"dspark|draft|mtp|eagle|specul", re.I)

_EXPERT_KEY = re.compile(r"(^|_)(n_routed_experts|num_experts|n_experts"
                         r"|n_physical_experts|num_local_experts)$")


def _draft_expert_pads(dims: "Dims", tp: int) -> tuple:
    """Expert counts we may grow by adding dead experts — the draft's only.

    An expert count is sharded by count, so it has to divide tp, and vLLM's own
    remedy (``num_redundant_experts``) is a single global number: it cannot fix a
    backbone at 384 and a draft at 128 at once, because 384 already divides 6 and
    128 needs +4. Growing the *draft* to 132 fixes it without touching either.

    Padding an expert count is only safe where a picked dummy cannot change the
    answer, and the draft is exactly that place:

    * The checkpoint has a per-block ``gate.bias`` (one entry per expert, added
      to the routing score), so a dummy expert can be pushed out of top-k
      outright rather than merely scoring zero.
    * Even if a dummy were somehow selected, its weights are zero, and this is a
      *speculative draft*: every token it proposes is verified against the real
      model before it is emitted. The cost would be acceptance rate, i.e. speed.

    Neither argument transfers to the backbone, so backbone counts are never
    padded here — they fall through to ``expert_advice``.
    """
    pads = []
    for key, count in dims.expert_counts:
        if count % tp == 0 or not _DRAFT_EXPERT_KEY.search(key):
            continue
        pads.append((key, count, _round_up(count, tp)))
    return tuple(pads)


def _expert_counts(raw: dict, _path: str = "") -> tuple:
    """Walk the whole config for expert counts, nested sections included.

    The DSpark draft layers carry their own count, and it need not match the
    backbone's: the backbone built fine at TP=6 while the draft asserted with
    n_physical_experts=128. Reading one top-level key would miss that.
    """
    found = []
    if isinstance(raw, dict):
        for k, v in raw.items():
            here = f"{_path}.{k}" if _path else k
            if isinstance(v, int) and v > 1 and _EXPERT_KEY.search(k):
                found.append((here, v))
            elif isinstance(v, dict):
                found.extend(_expert_counts(v, here))
    return tuple(found)


def load_dims(model_dir: str) -> Dims:
    path = os.path.join(model_dir, "config.json")
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    cfg = _flatten(raw)

    def opt(key: str):
        v = cfg.get(key)
        return v if isinstance(v, int) and v > 0 else None

    heads = opt("num_attention_heads")
    if heads is None:
        raise SystemExit(
            f"{LOG_PREFIX} {path}: no positive int num_attention_heads. "
            "Cannot plan TP padding without it."
        )
    groups = opt("o_groups")
    if groups is not None and heads % groups:
        _log(f"WARNING num_attention_heads={heads} is not a whole number of heads "
             f"per o_groups={groups}; ignoring o_groups and padding heads directly")
        groups = None
    return Dims(
        heads=heads,
        groups=groups,
        head_dim=opt("head_dim"),
        o_lora_rank=opt("o_lora_rank"),
        moe_intermediate=opt("moe_intermediate_size"),
        intermediate=opt("intermediate_size"),
        vocab_size=opt("vocab_size"),
        expert_counts=_expert_counts(raw),
        # The coarsest block drives the lcm rounding; the whole set drives
        # which size ratios may be read as "this is a scale of that".
        block=max(_quant_blocks(raw)),
        blocks=_quant_blocks(raw),
    )


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PadPlan:
    """Target extents for one tensor parallel size."""

    tp: int
    groups: frozenset
    dims: Dims
    heads: int
    o_groups: int
    moe_intermediate: int
    intermediate: int
    vocab_pad_to: int = VOCAB_PAD_BASE
    #: ``(config path, old count, padded count)`` per expert count this plan
    #: grows. Only ever the speculative draft's — see ``_draft_expert_pads``.
    expert_pads: tuple = ()

    def expert_pad_for(self, count: int) -> "tuple | None":
        """The pad entry whose *old* count is ``count``, if any."""
        for key, old, new in self.expert_pads:
            if old == count:
                return (key, old, new)
        return None

    @property
    def vocab_padded(self) -> "int | None":
        """What vLLM will round the vocab up to under this plan."""
        if not self.dims.vocab_size:
            return None
        return _round_up(self.dims.vocab_size, self.vocab_pad_to)

    def expert_advice(self, spec: str = "dspark") -> "str | None":
        """Expert counts that do not divide tp and that this plan cannot fix.

        Experts shard by count. Where a dummy expert can be made harmless the
        plan pads the count itself (``_draft_expert_pads``) and nothing is
        reported here. What is left is the backbone, where a dummy is NOT safe:
        even with a ``gate.bias`` to push it out of top-k, a wrong routing
        decision would change the answer with nothing downstream to catch it.
        vLLM's own remedy for those is ``num_redundant_experts`` — replicas of
        real experts — which is what its assertion message points at. This is
        only advice; nothing here can fix it.

        ``spec`` is the speculative method in force. With it off, draft-only
        counts are dropped -- otherwise this would block the very command it
        recommends.
        """
        bad = [(k, v) for k, v in self.dims.expert_counts
               if v % self.tp and not self.expert_pad_for(v)]
        if spec in ("", "none", "off", "0"):
            bad = [(k, v) for k, v in bad if not _DRAFT_EXPERT_KEY.search(k)]
        if not bad:
            return None
        # num_redundant_experts is GLOBAL: the same r is added to every MoE. A
        # single r works only if every count needs the same residue.
        relevant = [(k, v) for k, v in self.dims.expert_counts
                    if not (spec in ("", "none", "off", "0")
                            and _DRAFT_EXPERT_KEY.search(k))
                    and not self.expert_pad_for(v)]
        bits = ", ".join(f"{k}={v} (needs +{(-v) % self.tp})" for k, v in bad)
        needs = {(-v) % self.tp for _, v in relevant}
        head = (f"expert counts that do not divide tp={self.tp}: {bits}. Experts "
                f"shard by COUNT, and outside the speculative draft a dummy "
                f"expert is not safe to add, so this cannot be padded. "
                f"vLLM will assert in _init_fused_moe_experts. ")
        if len(needs) == 1:
            need = needs.pop()
            return (head + f"Enable EPLB with matching replicas: '--enable-eplb "
                    f'--eplb-config {{"num_redundant_experts":{need}}}\' (the '
                    f"config alone is rejected: 'num_redundant_experts is set but "
                    f"EPLB is not enabled'). Or drop the component that carries "
                    f"the odd count -- DSV41_SPEC=none if it is only the DSpark "
                    f"draft.")
        # No common r: the counts disagree on the residue they need.
        vals = sorted({v for _, v in relevant})
        feasible = [t for t in range(2, 4 * self.tp + 1)
                    if len({(-v) % t for v in vals}) == 1]
        return (head + f"num_redundant_experts CANNOT fix this: it is global, and "
                f"these counts {vals} need different residues mod {self.tp} "
                f"({sorted(needs)}). For one r to exist, tp must divide every "
                f"pairwise difference ({', '.join(str(vals[i+1]-vals[i]) for i in range(len(vals)-1))}), "
                f"which tp={self.tp} does not. TP values that would work: "
                f"{feasible}. So at tp={self.tp} you must drop the component with "
                f"the odd count -- DSV41_SPEC=none for the DSpark draft.")

    @property
    def active(self) -> bool:
        d = self.dims
        return (
            self.heads != d.heads
            or (self.o_groups or 0) != (d.groups or 0)
            or self.moe_intermediate != d.moe_intermediate
            or self.intermediate != d.intermediate
            or (self.vocab_padded is not None
                and self.vocab_padded != _round_up(d.vocab_size, VOCAB_PAD_BASE))
            or bool(self.expert_pads)
        )

    def summary(self) -> str:
        d = self.dims
        bits = []
        if self.o_groups and d.groups and self.o_groups != d.groups:
            bits.append(f"o_groups {d.groups}->{self.o_groups}")
        if self.heads != d.heads:
            bits.append(f"attention heads {d.heads}->{self.heads}")
        if self.intermediate and d.intermediate and self.intermediate != d.intermediate:
            bits.append(f"dense intermediate {d.intermediate}->{self.intermediate}")
        if self.moe_intermediate and d.moe_intermediate and self.moe_intermediate != d.moe_intermediate:
            bits.append(
                f"moe intermediate {d.moe_intermediate}->{self.moe_intermediate}"
            )
        vp = self.vocab_padded
        if vp is not None and vp != _round_up(d.vocab_size, VOCAB_PAD_BASE):
            bits.append(f"vocab {_round_up(d.vocab_size, VOCAB_PAD_BASE)}->{vp}")
        for key, old, new in self.expert_pads:
            bits.append(f"draft experts {old}->{new} ({key.rsplit('.', 1)[-1]})")
        return ", ".join(bits) if bits else "nothing to pad"


def plan_for_tp(dims: Dims, tp: int, groups: Iterable[str] | None = None) -> PadPlan:
    """Compute the padded extents for ``tp`` ranks."""
    if tp < 1:
        raise ValueError(f"tp must be >= 1, got {tp}")
    selected = frozenset(DEFAULT_GROUPS if groups is None else groups)
    unknown = selected - frozenset(ALL_GROUPS)
    if unknown:
        raise ValueError(f"unknown pad groups: {sorted(unknown)}")

    if "attn" in selected:
        if dims.groups:
            # wo_a is a bmm over o_groups: pad whole groups, each carrying its
            # full set of dummy heads, so heads/groups never moves.
            o_groups = _round_up(dims.groups, tp)
            heads = o_groups * dims.heads_per_group
        else:
            # no grouped output projection in this config: pad heads directly
            o_groups = None
            heads = _round_up(dims.heads, tp)
    else:
        o_groups, heads = dims.groups, dims.heads

    plan = PadPlan(
        tp=tp,
        groups=selected,
        dims=dims,
        heads=heads,
        o_groups=o_groups,
        moe_intermediate=(
            _round_up(dims.moe_intermediate, tp * dims.block)
            if "moe" in selected and dims.moe_intermediate and dims.moe_intermediate % tp
            else dims.moe_intermediate
        ),
        intermediate=(
            _round_up(dims.intermediate, tp * dims.block)
            if "dense" in selected and dims.intermediate and dims.intermediate % tp
            else dims.intermediate
        ),
        # The embedding is sharded AFTER vLLM rounds the vocab up, so that
        # rounding has to land on a multiple of tp as well.
        vocab_pad_to=(math.lcm(VOCAB_PAD_BASE, tp) if "vocab" in selected
                      else VOCAB_PAD_BASE),
        expert_pads=(_draft_expert_pads(dims, tp) if "experts" in selected else ()),
    )
    _validate(plan)
    return plan


def _validate(plan: PadPlan) -> None:
    """Check the extents this plan is responsible for.

    Only enabled groups are checked. Switching a group off is an explicit claim
    that the dimension is not TP-sharded in this build, so demanding
    divisibility there would be wrong.
    """
    tp, d = plan.tp, plan.dims
    checks = [
        ("attn", "attention heads", plan.heads),
        ("attn", "o_groups", plan.o_groups),
        ("attn", "wq_b out", plan.heads * d.head_dim if d.head_dim else None),
        ("attn", "wo_a out",
         plan.o_groups * d.o_lora_rank if plan.o_groups and d.o_lora_rank else None),
        ("moe", "moe intermediate", plan.moe_intermediate),
        ("dense", "dense intermediate", plan.intermediate),
        ("vocab", "padded vocab", plan.vocab_padded),
    ]
    for group, what, size in checks:
        if group in plan.groups and size and size % tp:
            raise AssertionError(f"{what}={size} is not divisible by tp={tp}")

    if "attn" in plan.groups and plan.o_groups and plan.heads % plan.o_groups:
        raise AssertionError(
            f"padded heads={plan.heads} is not a whole number of heads per "
            f"padded o_groups={plan.o_groups}; wo_a's bmm would be re-cut"
        )

    # If the head count grows, the q projection MUST grow with it, or the overlay
    # advertises more heads than the weights carry and the load fails on a shape
    # mismatch. That rule needs head_dim, so refuse rather than emit a config the
    # weights cannot satisfy.
    if "attn" in plan.groups and plan.heads != d.heads and not d.head_dim:
        raise AssertionError(
            f"attention heads would be padded {d.heads}->{plan.heads}, but "
            "config.json has no head_dim, so the q projection cannot be padded "
            "to match. Add head_dim to the config, or drop the attn group "
            "(DSV41_TP_PAD_GROUPS=moe,dense) and pick a TP that divides "
            f"{d.heads} heads."
        )

    # Blockwise-quantized shards must stay whole blocks on every rank.
    for group, what, size in (
        ("attn", "wq_b out", plan.heads * d.head_dim if d.head_dim else None),
        ("attn", "wo_a out",
         plan.o_groups * d.o_lora_rank if plan.o_groups and d.o_lora_rank else None),
        ("moe", "moe intermediate", plan.moe_intermediate),
        ("dense", "dense intermediate", plan.intermediate),
    ):
        if group not in plan.groups or not size:
            continue
        per_rank = size // tp
        if per_rank % d.block:
            raise AssertionError(
                f"{what} per rank = {per_rank} is not a multiple of {d.block}"
            )


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

ZERO = "zero"
NEG_INF = "neg_inf"
#: a router logit low enough that top-k can never reach it, but finite. -inf in a
#: routing score is one `0 * -inf` away from a NaN that would spread through the
#: whole MoE output; real logits live within about +-20.
NEG_BIG = "neg_big"
NEG_BIG_VALUE = -1.0e4


@dataclass(frozen=True)
class Rule:
    group: str
    what: str
    pattern: "re.Pattern"
    dim: int
    old: int
    new: int
    mode: str = ZERO


def _rx(tail: str) -> "re.Pattern":
    """Match a MODULE suffix, whatever parameter hangs off it.

    Keying on the module rather than the parameter is what keeps a weight and its
    blockwise scale consistent: `.wq_b` covers `.wq_b.weight` and
    `.wq_b.weight_scale_inv` together.
    """
    return re.compile(
        r"(^|\.)" + re.escape(tail.lstrip(".")) +
        r"(\.(weight|weight_scale_inv|weight_scale|weight_packed|scales|scale|qweight|bias))?$"
    )


#: the module path of a speculative draft block: `mtp.0.ffn`, `draft.1.mlp`, ...
_DRAFT_PREFIX = r"(^|\.)(mtp|dspark|draft|eagle)\.?\d*\."
_DRAFT_PREFIX_RX = re.compile(_DRAFT_PREFIX)
#: its router. Split by parameter because the two halves take different fill:
#: a dead expert's logit row is zero, its bias is what actually excludes it.
_DRAFT_GATE_W = re.compile(_DRAFT_PREFIX + r".*\b(gate|router)\.(weight|qweight)$")
_DRAFT_GATE_B = re.compile(
    _DRAFT_PREFIX + r".*\b(gate|router)\.(bias|e_score_correction_bias)$")
#: `<prefix>.experts.<id>.<rest>` inside a draft block
_DRAFT_EXPERT_TENSOR = re.compile(_DRAFT_PREFIX + r".*\.experts\.(\d+)\.")


def build_rules(plan: PadPlan) -> list[Rule]:
    d = plan.dims
    rules: list[Rule] = []

    def rule(group, what, tail, dim, old, new, mode=ZERO):
        if old == new:
            return
        if new < old:
            raise AssertionError(f"{what}: rule would shrink {old} -> {new}")
        rules.append(Rule(group, what, _rx(tail), dim, old, new, mode))

    if "attn" in plan.groups:
        if d.head_dim:
            rule("attn", "wq_b out", ".wq_b", 0,
                 d.heads * d.head_dim, plan.heads * d.head_dim)
        rule("attn", "attn sink", ".attn_sink", 0, d.heads, plan.heads, NEG_INF)
        if d.groups and plan.o_groups and d.o_lora_rank:
            rule("attn", "wo_a out", ".wo_a", 0,
                 d.groups * d.o_lora_rank, plan.o_groups * d.o_lora_rank)
            rule("attn", "wo_b in", ".wo_b", 1,
                 d.groups * d.o_lora_rank, plan.o_groups * d.o_lora_rank)
        # wo_a's per-group input is heads*head_dim/groups. Heads and groups grow
        # by the same factor, so it does not move and needs no rule.
    for tail in (".gate_proj", ".up_proj", ".w1", ".w3"):
        if "moe" in plan.groups:
            rule("moe", f"moe {tail} out", tail, 0, d.moe_intermediate, plan.moe_intermediate)
        if "dense" in plan.groups:
            rule("dense", f"dense {tail} out", tail, 0, d.intermediate, plan.intermediate)
    for tail in (".down_proj", ".w2"):
        if "moe" in plan.groups:
            rule("moe", f"moe {tail} in", tail, 1, d.moe_intermediate, plan.moe_intermediate)
        if "dense" in plan.groups:
            rule("dense", f"dense {tail} in", tail, 1, d.intermediate, plan.intermediate)
    # The draft's router grows a row per dead expert. The weight row is zero (it
    # contributes nothing to the logit); the bias entry is NEG_BIG, which is what
    # keeps the dead expert out of top-k. Both patterns are draft-scoped, so the
    # backbone's 384-wide router is never touched.
    for _key, old, new in plan.expert_pads:
        rules.append(Rule("experts", "draft router logits", _DRAFT_GATE_W,
                          0, old, new, ZERO))
        rules.append(Rule("experts", "draft router bias", _DRAFT_GATE_B,
                          0, old, new, NEG_BIG))
    return rules


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------



def _pad_fill(rule: Rule, dtype, torch):
    """The value to fill a padded slice with, adjusted for the tensor's dtype.

    E8M0 (the MX scale format) stores a bare exponent: it can represent neither
    0 nor -inf, and ``torch.full(..., 0.0, dtype=float8_e8m0fnu)`` raises
    "value cannot be converted to type c10::Float8_e8m0fnu without overflow".
    A padded *scale* row is multiplied by a padded *data* row, which is zero, so
    the scale itself is arithmetically irrelevant — use 1.0, the one value that
    is always representable and never introduces an inf or NaN.
    """
    fill = float("-inf") if rule.mode is NEG_INF else 0.0
    if rule.mode is NEG_BIG:
        fill = NEG_BIG_VALUE
    for name in ("float8_e8m0fnu", "float8_e8m0"):
        dt = getattr(torch, name, None)
        if dt is not None and dtype == dt:
            return 1.0
    return fill


def pad_tensor(tensor, rule: Rule, blocks=(DEFAULT_BLOCK,)):
    """Zero-pad ``tensor`` along ``rule.dim`` if it is the size the rule expects.

    A blockwise scale tensor is the weight's extent divided by the quantization
    block, so scales need no rules of their own. The divisor must be one the
    checkpoint actually declares: accepting *any* small divisor matches unrelated
    modules whose sizes happen to divide. The 32-head indexer's ``wq_b`` is
    [4096, 1280] against the main attention's 32768 — a ratio of 8 — and it is
    ReplicatedLinear, so padding it produced "Tried to load weights of size
    [6144, 1280] to a parameter of size [4096, 1280]".

    Returns the tensor unchanged when the rule does not apply.
    """
    import torch

    dim = rule.dim
    if dim >= tensor.dim():
        return tensor
    have, src, dst = tensor.shape[dim], rule.old, rule.new
    if have == src:
        block = 1
    elif (
        1 < have < src
        and src % have == 0
        and (src // have) in blocks
        and dst % (src // have) == 0
    ):
        block = src // have
    else:
        # a different module, already padded, or a per-tensor scale (never grows)
        return tensor
    want = dst // block
    if want == have:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = want - have
    fill = _pad_fill(rule, tensor.dtype, torch)
    try:
        tail = torch.full(shape, fill, dtype=tensor.dtype, device=tensor.device)
    except RuntimeError as exc:
        # A dtype that cannot hold `fill` at all. 1.0 is representable in every
        # float format vLLM uses here; the data rows it pairs with are zero.
        if fill == 1.0:
            raise RuntimeError(
                f"{LOG_PREFIX} cannot build padding for dtype {tensor.dtype}: {exc}"
            ) from exc
        tail = torch.full(shape, 1.0, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, tail], dim=dim)


class Padder:
    """Applies the plan to a stream of (name, tensor) pairs."""

    def __init__(self, plan: PadPlan):
        self.plan = plan
        self.rules = build_rules(plan)
        self.blocks = tuple(plan.dims.blocks)
        self.count = 0
        self.synthesized = 0
        self.by_rule: dict[str, int] = {}

    def __call__(self, name: str, tensor):
        for rule in self.rules:
            if not rule.pattern.search(name):
                continue
            padded = pad_tensor(tensor, rule, self.blocks)
            if padded is tensor:
                if _debug():
                    _log(f"skip {name} dim{rule.dim} size {tensor.shape[rule.dim]} "
                         f"(rule {rule.what} expects {rule.old} or it / {self.blocks})")
                continue
            self.count += 1
            self.by_rule[rule.what] = self.by_rule.get(rule.what, 0) + 1
            if _debug() or self.count <= 8:
                _log(f"{name} dim{rule.dim} {tensor.shape[rule.dim]} -> "
                     f"{padded.shape[rule.dim]} [{rule.what}]")
            tensor = padded
        return self._expert_indexed(name, tensor)

    def _expert_indexed(self, name: str, tensor):
        """Catch any draft tensor indexed by expert that the rules missed.

        The named rules above have to guess what the router module is called
        (`gate`, `router`, ...). Getting that wrong is not hypothetical: the
        first cluster boot with a padded draft died on
        "Attempted to load weight (torch.Size([128])) into parameter
        (torch.Size([132]))" -- vLLM had built the parameter at the padded count
        from the overlay config, but the checkpoint tensor came through
        unpadded because no pattern matched its name.

        So key on the shape instead of the name: inside a draft block, a dim-0
        extent that is exactly the draft's expert count IS the expert axis.
        Nothing else in these blocks is 128 wide -- hidden is 7168, head_dim
        512, moe_intermediate 2304 -- and a false positive cannot pass silently,
        because the parameter it is loaded into would then have the wrong size
        and assert exactly as above.

        `.experts.<id>.` tensors are excluded: there the expert is in the *name*
        and dim 0 is an intermediate width.
        """
        pads = self.plan.expert_pads
        if not pads or not tensor.dim():
            return tensor
        if ".experts." in name or not _DRAFT_PREFIX_RX.search(name):
            return tensor
        for _key, old, new in pads:
            if tensor.shape[0] != old:
                continue
            # 1-D is a per-expert score or bias: it has to be pushed out of
            # top-k, not zeroed -- a zero bias competes with real logits.
            # 2-D is the router's logit rows, where zero contributes nothing.
            mode = NEG_BIG if tensor.dim() == 1 else ZERO
            rule = Rule("experts", "draft expert axis", _DRAFT_PREFIX_RX,
                        0, old, new, mode)
            padded = pad_tensor(tensor, rule, self.blocks)
            if padded is not tensor:
                self.count += 1
                self.by_rule["draft expert axis"] = \
                    self.by_rule.get("draft expert axis", 0) + 1
                _log(f"{name} dim0 {old} -> {new} [draft expert axis, "
                     f"{'excluded from top-k' if mode is NEG_BIG else 'zeroed'}]")
                return padded
        if _debug():
            _log(f"draft tensor left alone: {name} {tuple(tensor.shape)}")
        return tensor

    def report(self) -> str:
        if not self.count:
            return "padded nothing (no tensor matched a rule)"
        parts = ", ".join(f"{k} x{v}" for k, v in sorted(self.by_rule.items()))
        out = f"padded {self.count} tensor(s): {parts}"
        if self.synthesized:
            out += f"; synthesized {self.synthesized} dead draft-expert tensor(s)"
        return out

    def extra(self, name: str, tensor):
        """Dead experts to emit alongside ``name``, as (name, tensor) pairs.

        A padded expert *count* is different from every other rule here: the
        tensors for ids >= the real count do not exist in the checkpoint at all,
        and vLLM's fused-MoE loader fills its expert slots one id at a time. A
        slot nobody loads keeps whatever ``torch.empty`` left there, so the
        draft would route through uninitialized memory. Emitting explicit zero
        experts is what makes the padding inert rather than merely legal.

        Keyed on expert 0 so each block emits its dummies exactly once, and
        shaped from the tensor as it leaves ``__call__`` — already widened by the
        moe rules, if those applied.
        """
        if not self.plan.expert_pads:
            return ()
        match = _DRAFT_EXPERT_TENSOR.search(name)
        if match is None or match.group(3) != "0":
            return ()
        import torch

        out = []
        head, tail = name.split(".experts.0.", 1)
        for _key, old, new in self.plan.expert_pads:
            for eid in range(old, new):
                # e8m0 holds a bare exponent and cannot represent 0; the data it
                # scales is zero either way. Same reasoning as _pad_fill.
                try:
                    dead = torch.zeros_like(tensor)
                except (RuntimeError, TypeError):
                    dead = torch.full_like(tensor, 1.0)
                out.append((f"{head}.experts.{eid}.{tail}", dead))
        self.synthesized += len(out)
        if _debug() or self.synthesized <= 8:
            _log(f"{name} -> +{len(out)} dead expert tensor(s) "
                 f"(ids {self.plan.expert_pads[0][1]}..{self.plan.expert_pads[0][2] - 1})")
        return out


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def _log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", file=sys.stderr, flush=True)


def _debug() -> bool:
    return os.environ.get("DSV41_TP_PAD_DEBUG", "").strip() not in ("", "0")


_PLAN: PadPlan | None = None
_PADDER: Padder | None = None
_PATCHED: set = set()


def _env_plan() -> PadPlan | None:
    """The plan this container was launched with, from the environment."""
    global _PLAN
    if _PLAN is not None:
        return _PLAN
    raw_tp = os.environ.get("DSV41_TP_PAD", "").strip()
    if raw_tp in ("", "0", "1"):
        return None
    src = os.environ.get("DSV41_MODEL_SRC", "/model")
    groups_env = os.environ.get("DSV41_TP_PAD_GROUPS", "").strip()
    groups = [g.strip() for g in groups_env.split(",") if g.strip()] or None
    _PLAN = plan_for_tp(load_dims(src), int(raw_tp), groups)
    return _PLAN


def _padder() -> Padder | None:
    global _PADDER
    if _PADDER is None:
        plan = _env_plan()
        if plan is None or not plan.active:
            return None
        _PADDER = Padder(plan)
        _log(f"TP={plan.tp}: {plan.summary()}")
    return _PADDER


def patch_weight_loader(module) -> bool:
    """Wrap the weight iterators so every tensor is padded before vLLM shards it.

    The seam is the generator that yields (name, tensor): whatever narrows the
    tensor for this rank runs after it, on the padded extent.
    """
    padder = _padder()
    if padder is None:
        return False
    key = getattr(module, "__name__", str(module))
    if key in _PATCHED:
        return True
    wrapped = 0
    for attr in dir(module):
        if not attr.endswith("_weights_iterator"):
            continue
        original = getattr(module, attr, None)
        if not callable(original):
            continue

        def make(original=original):
            def gen(*args, **kwargs):
                for name, tensor in original(*args, **kwargs):
                    padded = padder(name, tensor)
                    yield name, padded
                    # Dead experts for a padded expert count have no entry in the
                    # checkpoint, so they can only enter the stream here.
                    for extra_name, extra_tensor in padder.extra(name, padded):
                        yield extra_name, extra_tensor
            return gen

        setattr(module, attr, make())
        wrapped += 1
    if wrapped:
        _PATCHED.add(key)
        _log(f"wrapped {wrapped} weight iterator(s) in {key}")
    return bool(wrapped)


def patch_vocab_padding(module) -> bool:
    """Round the vocab up to a multiple of tp as well as of the default unit.

    vLLM shards the embedding over tp AFTER padding the vocab with
    ``pad_vocab_size`` (default unit 64). 129280 is a whole number of 64s but not
    of 6, so ``divide(129280, 6)`` asserts. Rounding to ``lcm(64, tp)`` instead
    satisfies both.

    No weight rule goes with this: vLLM builds the embedding at the padded size
    and its loader fills only the real rows, and the logits processor slices back
    to ``org_vocab_size``, so the extra rows can never be sampled.
    """
    plan = _env_plan()
    if plan is None or "vocab" not in plan.groups:
        return False
    key = getattr(module, "__name__", str(module))
    if key in _PATCHED:
        return True
    original = getattr(module, "pad_vocab_size", None)
    if not callable(original):
        return False
    base = getattr(module, "DEFAULT_VOCAB_PADDING_SIZE", VOCAB_PAD_BASE)
    tp = plan.tp

    def pad_vocab_size(vocab_size, pad_to=base, **kwargs):
        want = math.lcm(pad_to or base, tp)
        return -(-vocab_size // want) * want

    module.pad_vocab_size = pad_vocab_size
    module.DEFAULT_VOCAB_PADDING_SIZE = math.lcm(base, tp)
    _PATCHED.add(key)
    _log(f"vocab padding unit {base} -> {math.lcm(base, tp)} so the padded vocab "
         f"divides tp={tp}")
    return True


HOOKS: dict[str, Callable] = {
    "vllm.model_executor.layers.vocab_parallel_embedding": patch_vocab_padding,
    "vllm.model_executor.model_loader.weight_utils": patch_weight_loader,
    "vllm.model_executor.model_loader.default_loader": patch_weight_loader,
    # older layouts kept the loaders in one module
    "vllm.model_executor.model_loader.loader": patch_weight_loader,
}


def install() -> bool:
    """Patch whatever vLLM modules are already imported. Idempotent."""
    ok = False
    for name, fn in HOOKS.items():
        module = sys.modules.get(name)
        if module is not None:
            ok = fn(module) or ok
    return ok


# ---------------------------------------------------------------------------
# CLI — so the launch scripts and the container share one plan implementation
# instead of re-deriving 66 / 2304 in bash.
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="DeepSeek-V4.1-Flash TP padding plan")
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--model", default=os.environ.get("DSV41_MODEL_SRC", "/model"),
                        help="model directory holding config.json")
    parser.add_argument("--groups", default="")
    parser.add_argument("--spec", default="dspark",
                        help="speculative method in force; 'none' drops "
                             "draft-only expert counts from the advice")
    parser.add_argument("--shell", action="store_true",
                        help="emit eval-able shell assignments instead of prose")
    args = parser.parse_args(argv)

    groups = [g.strip() for g in args.groups.split(",") if g.strip()] or None
    plan = plan_for_tp(load_dims(args.model), args.tp, groups)

    if args.shell:
        import shlex

        # A dimension the config does not carry becomes an empty string, not the
        # literal "None": these are eval'd by the launcher.
        def sh(v):
            return "" if v is None else str(v)

        print(f"DSV41_PAD_ACTIVE={1 if plan.active else 0}")
        print(f"DSV41_PAD_SUMMARY={shlex.quote(plan.summary())}")
        print(f"DSV41_PAD_GROUPS={shlex.quote(','.join(sorted(plan.groups)))}")
        print(f"DSV41_PAD_HEADS={sh(plan.heads)}")
        print(f"DSV41_PAD_O_GROUPS={sh(plan.o_groups)}")
        print(f"DSV41_PAD_MOE_INTERMEDIATE={sh(plan.moe_intermediate)}")
        print(f"DSV41_PAD_INTERMEDIATE={sh(plan.intermediate)}")
        print(f"DSV41_PAD_VOCAB={sh(plan.vocab_padded)}")
        import shlex as _sh
        print(f"DSV41_PAD_EXPERT_ADVICE="
              f"{_sh.quote(plan.expert_advice(args.spec) or '')}")
        return 0

    d = plan.dims
    print(f"TP={plan.tp} active={plan.active} groups={','.join(sorted(plan.groups))}")
    print(f"  checkpoint: heads={d.heads} o_groups={d.groups} "
          f"head_dim={d.head_dim} o_lora_rank={d.o_lora_rank} "
          f"moe_intermediate={d.moe_intermediate} intermediate={d.intermediate} "
          f"vocab={d.vocab_size} experts={list(d.expert_counts)} "
          f"quant blocks={list(d.blocks)}")
    missing = [k for k, v in (("o_groups", d.groups), ("head_dim", d.head_dim),
                              ("vocab_size", d.vocab_size),
                              ("o_lora_rank", d.o_lora_rank),
                              ("moe_intermediate_size", d.moe_intermediate),
                              ("intermediate_size", d.intermediate)) if v is None]
    if missing:
        print(f"  absent from config.json (no rules emitted for them): {', '.join(missing)}")
    print(f"  {plan.summary()}")
    for rule in build_rules(plan):
        print(f"  {rule.group:6s} {rule.what:22s} dim{rule.dim} "
              f"{rule.old} -> {rule.new}  [{rule.mode}]")
    advice = plan.expert_advice(args.spec)
    if advice:
        print(f"\n  WARNING {advice}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
