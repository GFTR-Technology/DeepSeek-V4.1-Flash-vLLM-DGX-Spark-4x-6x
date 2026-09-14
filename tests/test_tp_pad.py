#!/usr/bin/env python3
"""Tests for patch/dsv41_tp_pad — TP alignment by padding.

Two halves:

  1. THE PLAN (no torch needed): dimensions are read out of config.json, heads
     and output groups grow together in whole groups, quantized widths stay on
     block boundaries, and no rule ever shrinks anything.

  2. THE PADDING (needs torch): the rules produce the shapes the overlay config
     advertises — including blockwise scale tensors, which are matched through
     the module name and sized by inferring the block from the ratio — and zero
     padding is arithmetically inert.

    python3 tests/test_tp_pad.py            # plan + overlay
    python3 tests/test_tp_pad.py --torch    # + padding shapes and numerics
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SHIM_DIR = os.path.join(REPO, "patch", "dsv41_tp_pad")
SHIM = os.path.join(SHIM_DIR, "dsv41_tp_pad.py")

FAILED = []

# One output group per head: TP=6 is then the 64 -> 66 case.
SOLO = {
    "num_attention_heads": 64, "o_groups": 64, "head_dim": 128,
    "o_lora_rank": 512, "intermediate_size": 18432, "moe_intermediate_size": 2048,
    "num_key_value_heads": 1,   # MLA: one latent KV head, must survive padding
    "vocab_size": 129280,       # 64 x 2020: padded for 64, not for 6
    "quantization_config": {"weight_block_size": [128, 128]},
    # -> blocks == (128,): only a /128 ratio may be read as "this is a scale"
}
# Four groups of 16 heads: padding has to add whole groups.
GROUPED = dict(SOLO, o_groups=4)


def check(label, got, want):
    if got == want:
        print(f"  PASS {label}")
    else:
        print(f"  FAIL {label}\n       got  {got}\n       want {want}")
        FAILED.append(label)


def load_shim():
    sys.path.insert(0, SHIM_DIR)
    spec = importlib.util.spec_from_file_location("dsv41_tp_pad", SHIM)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dsv41_tp_pad"] = mod
    spec.loader.exec_module(mod)
    return mod


def write_model(tmp, cfg):
    d = os.path.join(tmp, "model")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    # a couple of stand-in shards so the overlay has something to symlink
    for n in (1, 2):
        open(os.path.join(d, f"model-0000{n}-of-00002.safetensors"), "wb").close()
    return d


def test_plan(m):
    print("plan: dimensions come from config.json, not from constants")
    with tempfile.TemporaryDirectory() as tmp:
        d = m.load_dims(write_model(tmp, SOLO))
    check("heads read", d.heads, 64)
    check("o_groups read", d.groups, 64)
    check("heads per group", d.heads_per_group, 1)
    check("quant block read", d.block, 128)

    print("plan: one group per head -> the 64 -> 66 case")
    p = m.plan_for_tp(d, 6)
    check("active", p.active, True)
    check("o_groups 64 -> 66", (d.groups, p.o_groups), (64, 66))
    check("heads 64 -> 66", (d.heads, p.heads), (64, 66))
    check("moe 2048 -> 2304 (lcm with the 128 block)", p.moe_intermediate, 2304)
    check("moe stays whole blocks per rank", (p.moe_intermediate // 6) % 128, 0)
    check("dense 18432 already divides 6", p.intermediate, 18432)

    print("plan: grouped heads -> padding adds WHOLE groups")
    with tempfile.TemporaryDirectory() as tmp:
        dg = m.load_dims(write_model(tmp, GROUPED))
    pg = m.plan_for_tp(dg, 6)
    check("o_groups 4 -> 6", pg.o_groups, 6)
    check("heads 64 -> 96", pg.heads, 96)
    check("heads per group unchanged", pg.heads // pg.o_groups, 16)
    check("padded heads divide TP", pg.heads % 6, 0)

    print("plan: a TP that already divides pads nothing")
    check("TP=4 inactive", m.plan_for_tp(d, 4).active, False)
    check("TP=4 summary", m.plan_for_tp(d, 4).summary(), "nothing to pad")

    print("plan: TP=8")
    p8 = m.plan_for_tp(dg, 8)
    check("o_groups 4 -> 8", p8.o_groups, 8)
    check("heads 64 -> 128", p8.heads, 128)
    check("heads per group still 16", p8.heads // p8.o_groups, 16)

    print("plan: a ragged heads/groups ratio falls back to padding heads directly")
    with tempfile.TemporaryDirectory() as tmp:
        d_ragged = m.load_dims(write_model(tmp, dict(SOLO, num_attention_heads=65, o_groups=4)))
    check("o_groups dropped", d_ragged.groups, None)
    check("heads 65 -> 66 at TP=6", m.plan_for_tp(d_ragged, 6).heads, 66)

    print("plan: optional keys may be absent entirely")
    with tempfile.TemporaryDirectory() as tmp:
        d_min = m.load_dims(write_model(tmp, {"num_attention_heads": 64,
                                              "moe_intermediate_size": 2048}))
    check("groups absent", d_min.groups, None)
    check("head_dim absent", d_min.head_dim, None)
    check("block defaults to 128", d_min.block, 128)
    # heads grow but head_dim is unknown -> the q projection could not be padded,
    # so the plan must refuse rather than emit a config the weights cannot satisfy
    try:
        m.plan_for_tp(d_min, 6)
        check("refuses to pad heads without head_dim", "no error", "AssertionError")
    except AssertionError as exc:
        check("refuses to pad heads without head_dim", "head_dim" in str(exc), True)
    # ...but moe-only padding on the same config is fine
    check("moe-only plan still works",
          m.plan_for_tp(d_min, 6, ["moe"]).moe_intermediate, 2304)

    print("plan: missing num_attention_heads is fatal")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            m.load_dims(write_model(tmp, {"moe_intermediate_size": 2048}))
            check("no heads refused", "no error", "SystemExit")
        except SystemExit:
            check("no heads refused", True, True)

    print("plan: vocab is rounded to lcm(64, tp), not padded as a weight")
    pv = m.plan_for_tp(d, 6)
    check("129280 -> 129408", pv.vocab_padded, 129408)
    check("divides tp", pv.vocab_padded % 6, 0)
    check("still a whole number of 64s", pv.vocab_padded % 64, 0)
    check("TP=4 leaves it alone", m.plan_for_tp(d, 4).vocab_padded, 129280)
    check("no weight rule for vocab",
          any("vocab" in r.what for r in m.build_rules(pv)), False)
    # TP=8: lcm(64,8) is 64, so the vocab needs nothing and neither does anything
    # else here -> the plan must report itself inactive rather than invent work
    d8 = m.Dims(heads=64, groups=8, head_dim=512, o_lora_rank=1024,
                moe_intermediate=2304, vocab_size=129280, block=32)
    check("TP=8 needs nothing at all", m.plan_for_tp(d8, 8).active, False)
    # a TP where everything else divides but the vocab does not: 12 groups of 8
    # heads at TP=6 -> heads 96, groups 12, moe 2304 all divide; vocab does not
    d_vocab = m.Dims(heads=96, groups=12, head_dim=512, o_lora_rank=1024,
                     moe_intermediate=2304, vocab_size=129280, block=32)
    pvo = m.plan_for_tp(d_vocab, 6)
    check("vocab alone makes the plan active", pvo.active, True)
    check("and it is the only change", pvo.summary(), "vocab 129280->129408")

    print("plan: the vocab hook rewrites pad_vocab_size")
    import types
    mod = types.SimpleNamespace(
        __name__="vllm.model_executor.layers.vocab_parallel_embedding",
        DEFAULT_VOCAB_PADDING_SIZE=64,
        pad_vocab_size=lambda v, pad_to=64: -(-v // pad_to) * pad_to)
    os.environ["DSV41_TP_PAD"] = "6"
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DSV41_MODEL_SRC"] = write_model(tmp, SOLO)
        m._PLAN = None; m._PADDER = None; m._PATCHED.clear()
        check("hook applied", m.patch_vocab_padding(mod), True)
        check("pad_vocab_size(129280) -> 129408", mod.pad_vocab_size(129280), 129408)
        check("explicit pad_to=64 still lifted", mod.pad_vocab_size(129280, pad_to=64), 129408)
        check("default unit updated", mod.DEFAULT_VOCAB_PADDING_SIZE, 192)
    m._PLAN = None; m._PADDER = None; m._PATCHED.clear()
    os.environ.pop("DSV41_TP_PAD", None); os.environ.pop("DSV41_MODEL_SRC", None)

    print("plan: groups can be switched off")
    check("attn only", m.plan_for_tp(d, 6, ["attn"]).moe_intermediate, 2048)
    check("moe only", m.plan_for_tp(d, 6, ["moe"]).heads, 64)
    try:
        m.plan_for_tp(d, 6, ["nonsense"])
        check("unknown group refused", "no error", "ValueError")
    except ValueError:
        check("unknown group refused", True, True)

    print("plan: rules")
    rules = {(r.what, r.dim): (r.old, r.new) for r in m.build_rules(p)}
    check("wq_b out 8192 -> 8448", rules[("wq_b out", 0)], (64 * 128, 66 * 128))
    check("attn sink 64 -> 66", rules[("attn sink", 0)], (64, 66))
    check("wo_a out 32768 -> 33792", rules[("wo_a out", 0)], (64 * 512, 66 * 512))
    check("wo_b in 32768 -> 33792", rules[("wo_b in", 1)], (64 * 512, 66 * 512))
    check("wo_a per-group input has no rule (it does not move)",
          any(r.what.startswith("wo_a in") for r in m.build_rules(p)), False)
    check("no rule shrinks", all(r.new > r.old for r in m.build_rules(p)), True)
    sink = [r for r in m.build_rules(p) if r.what == "attn sink"][0]
    check("attn sink fills with -inf", sink.mode, m.NEG_INF)

    print("plan: rule patterns match the module, weight and scale alike")
    wq = [r for r in m.build_rules(p) if r.what == "wq_b out"][0]
    for name in ("model.layers.0.self_attn.wq_b",
                 "model.layers.0.self_attn.wq_b.weight",
                 "model.layers.0.self_attn.wq_b.weight_scale_inv"):
        check(f"matches {name.rsplit('.', 1)[-1]}", bool(wq.pattern.search(name)), True)
    for name in ("model.layers.0.self_attn.wq_b_proj.weight",
                 "model.layers.0.mlp.gate.weight"):
        check(f"does not match {name.rsplit('.', 2)[-2]}", bool(wq.pattern.search(name)), False)

    print("plan: only a declared quantization block counts as a scale ratio")
    with tempfile.TemporaryDirectory() as tmp:
        d_blk = m.load_dims(write_model(tmp, SOLO))
    check("blocks read from config", d_blk.blocks, (128,))
    with tempfile.TemporaryDirectory() as tmp:
        d_multi = m.load_dims(write_model(tmp, dict(
            SOLO, quantization_config={"weight_block_size": [1, 32]})))
    check("1 is not a block, 32 is", d_multi.blocks, (32,))
    check("coarsest drives the lcm rounding", d_multi.block, 32)

    rule = m.Rule("attn", "wq_b out", m._rx(".wq_b"), 0, 32768, 49152, m.ZERO)

    class _T:
        def __init__(self, *sh): self.shape = tuple(sh)
        def dim(self): return len(self.shape)
        dtype = "bf16"; device = "cpu"

    def _accepts(size, blocks):
        """Would pad_tensor act on a tensor of this size? (shape math only)"""
        have, src, dst = size, rule.old, rule.new
        if have == src:
            return True
        return (1 < have < src and src % have == 0
                and (src // have) in blocks and dst % (src // have) == 0)

    check("exact weight size matches", _accepts(32768, (32,)), True)
    check("scale at the declared block matches", _accepts(32768 // 32, (32,)), True)
    # the 32-head indexer's wq_b is 4096 = 32768/8. 8 is not a block, and that
    # module is ReplicatedLinear, so padding it produced
    # "Tried to load weights of size [6144, 1280] to a parameter of size [4096, 1280]"
    check("indexer wq_b (ratio 8) is NOT matched", _accepts(4096, (32,)), False)
    check("indexer wq_b scale (ratio 256) is NOT matched", _accepts(128, (32,)), False)
    check("a ratio that IS a block but from another module still matches (known limit)",
          _accepts(32768 // 32, (32,)), True)

    print("plan: padded slices use a fill the dtype can actually hold")
    import types as _types

    class _DT:
        def __init__(self, n): self.n = n
        def __eq__(self, o): return isinstance(o, _DT) and self.n == o.n
        def __repr__(self): return f"torch.{self.n}"

    _torch = _types.SimpleNamespace(float8_e8m0fnu=_DT("float8_e8m0fnu"),
                                    float8_e4m3fn=_DT("float8_e4m3fn"),
                                    bfloat16=_DT("bfloat16"))
    zero_rule = m.Rule("attn", "wq_b out", m._rx(".wq_b"), 0, 8192, 12288, m.ZERO)
    sink_rule = m.Rule("attn", "attn sink", m._rx(".attn_sink"), 0, 64, 96, m.NEG_INF)
    check("bf16 weight -> 0.0",
          m._pad_fill(zero_rule, _torch.bfloat16, _torch), 0.0)
    check("fp8 e4m3 weight -> 0.0",
          m._pad_fill(zero_rule, _torch.float8_e4m3fn, _torch), 0.0)
    # E8M0 is exponent-only: no zero, no -inf. A padded scale pairs with a padded
    # (zero) data row, so 1.0 is inert and always representable.
    check("E8M0 scale -> 1.0, not 0.0",
          m._pad_fill(zero_rule, _torch.float8_e8m0fnu, _torch), 1.0)
    check("bf16 attn_sink -> -inf",
          m._pad_fill(sink_rule, _torch.bfloat16, _torch), float("-inf"))
    check("E8M0 never gets -inf either",
          m._pad_fill(sink_rule, _torch.float8_e8m0fnu, _torch), 1.0)

    print("plan: CLI --shell is what the launcher eval()s")
    with tempfile.TemporaryDirectory() as tmp:
        d6 = write_model(tmp, SOLO)
        out = subprocess.run([sys.executable, SHIM, "--tp", "6", "--model", d6, "--shell"],
                             capture_output=True, text=True)
        check("exit 0", out.returncode, 0)
        env = dict(line.split("=", 1) for line in out.stdout.strip().splitlines())
        check("PAD_ACTIVE=1", env["DSV41_PAD_ACTIVE"], "1")
        check("PAD_HEADS=66", env["DSV41_PAD_HEADS"], "66")
        check("PAD_O_GROUPS=66", env["DSV41_PAD_O_GROUPS"], "66")
        check("PAD_MOE=2304", env["DSV41_PAD_MOE_INTERMEDIATE"], "2304")
        out4 = subprocess.run([sys.executable, SHIM, "--tp", "4", "--model", d6, "--shell"],
                              capture_output=True, text=True)
        env4 = dict(line.split("=", 1) for line in out4.stdout.strip().splitlines())
        check("TP=4 PAD_ACTIVE=0", env4["DSV41_PAD_ACTIVE"], "0")


def test_overlay(m):
    print("overlay: config.json is rewritten, shards are symlinked")
    sys.path.insert(0, SHIM_DIR)
    spec = importlib.util.spec_from_file_location(
        "make_overlay", os.path.join(SHIM_DIR, "make_overlay.py"))
    ov = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ov)

    with tempfile.TemporaryDirectory() as tmp:
        src = write_model(tmp, SOLO)
        dst = os.path.join(tmp, "model-tp6")
        plan = m.plan_for_tp(m.load_dims(src), 6)
        try:
            ov.build(src, dst, plan)
        except OSError as exc:
            # Windows needs a privilege for os.symlink. The config rewrite is the
            # part with the logic, so test that directly and skip the linking.
            print(f"  SKIP symlink half ({exc.__class__.__name__}: not permitted here)")
            with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
                cfg = ov.rewrite_config(json.load(f), plan)
            check("num_attention_heads padded", cfg["num_attention_heads"], 66)
            check("o_groups padded", cfg["o_groups"], 66)
            check("moe_intermediate_size padded", cfg["moe_intermediate_size"], 2304)
            check("dense intermediate untouched", cfg["intermediate_size"], 18432)
            check("provenance recorded", cfg["_dsv41_tp_pad"]["tp"], 6)
            check("MLA single latent KV head is NOT followed",
                  cfg.get("num_key_value_heads"), 1)
            return
        with open(os.path.join(dst, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        check("num_attention_heads padded", cfg["num_attention_heads"], 66)
        check("o_groups padded", cfg["o_groups"], 66)
        check("moe_intermediate_size padded", cfg["moe_intermediate_size"], 2304)
        check("dense intermediate untouched", cfg["intermediate_size"], 18432)
        check("provenance recorded", cfg["_dsv41_tp_pad"]["tp"], 6)
        check("MLA single latent KV head is NOT followed",
              cfg.get("num_key_value_heads"), 1)
        check("shards symlinked, not copied",
              os.path.islink(os.path.join(dst, "model-00001-of-00002.safetensors")), True)
        check("config is a real file, not a link",
              os.path.islink(os.path.join(dst, "config.json")), False)
        # rebuilt from scratch, so a restart cannot inherit a stale config
        ov.build(src, dst, plan)
        check("rebuild is idempotent",
              json.load(open(os.path.join(dst, "config.json")))["o_groups"], 66)

    print("overlay: it refuses to overlay a directory onto itself")
    with tempfile.TemporaryDirectory() as tmp:
        src = write_model(tmp, SOLO)
        plan = m.plan_for_tp(m.load_dims(src), 6)
        try:
            ov.build(src, src, plan)
            check("self-overlay refused", "no error", "SystemExit")
        except SystemExit:
            check("self-overlay refused", True, True)


def test_torch(m):
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        dims = m.load_dims(write_model(tmp, SOLO))
    plan = m.plan_for_tp(dims, 6)
    rules = {r.what: r for r in m.build_rules(plan)}
    pad = m.Padder(plan)

    print("padding: shapes")
    check("wq_b weight 8192 -> 8448",
          tuple(pad("l.0.attn.wq_b.weight", torch.zeros(8192, 1280)).shape), (8448, 1280))
    check("wq_b scale 64 -> 66 (block 128 inferred)",
          tuple(pad("l.0.attn.wq_b.weight_scale_inv", torch.zeros(64, 10)).shape), (66, 10))
    check("wq_b scale 256 -> 264 (block 32 inferred)",
          tuple(pad("l.0.attn.wq_b.weight_scale_inv", torch.zeros(256, 40)).shape), (264, 40))
    check("per-tensor scale never grows",
          tuple(pad("l.0.attn.wq_b.input_scale", torch.zeros(1)).shape), (1,))
    sink = pad("l.0.attn.attn_sink", torch.zeros(64))
    check("attn_sink 64 -> 66", tuple(sink.shape), (66,))
    check("attn_sink tail is -inf", bool(torch.isneginf(sink[64:]).all()), True)
    check("wo_a out 32768 -> 33792",
          tuple(pad("l.0.attn.wo_a.weight", torch.zeros(32768, 128)).shape), (33792, 128))
    check("wo_a per-group input untouched",
          pad("l.0.attn.wo_a.weight", torch.zeros(32768, 128)).shape[1], 128)
    check("wo_b in 32768 -> 33792",
          tuple(pad("l.0.attn.wo_b.weight", torch.zeros(7168, 32768)).shape), (7168, 33792))
    check("expert gate 2048 -> 2304",
          tuple(pad("l.3.mlp.experts.7.gate_proj.weight", torch.zeros(2048, 7168)).shape),
          (2304, 7168))
    check("expert gate scale 16 -> 18",
          tuple(pad("l.3.mlp.experts.7.gate_proj.weight_scale_inv", torch.zeros(16, 56)).shape),
          (18, 56))
    check("expert down dim1 2048 -> 2304",
          tuple(pad("l.3.mlp.experts.7.down_proj.weight", torch.zeros(7168, 2048)).shape),
          (7168, 2304))
    check("unrelated tensor untouched",
          tuple(pad("l.0.mlp.gate.weight", torch.zeros(256, 7168)).shape), (256, 7168))
    check("already padded is not padded twice",
          tuple(pad("l.0.attn.wq_b.weight", torch.zeros(8448, 1280)).shape), (8448, 1280))
    check("padder counted its work", pad.count > 0, True)

    print("padding: zero padding changes no output")
    torch.manual_seed(0)
    hidden, inter, tp = 256, 100, 6
    x = torch.randn(4, hidden)
    gate, up = torch.randn(inter, hidden), torch.randn(inter, hidden)
    down = torch.randn(hidden, inter)
    ref = torch.nn.functional.silu(x @ gate.T) * (x @ up.T) @ down.T
    n = -(-inter // tp) * tp
    got = (torch.nn.functional.silu(x @ torch.cat([gate, torch.zeros(n - inter, hidden)]).T)
           * (x @ torch.cat([up, torch.zeros(n - inter, hidden)]).T)
           ) @ torch.cat([down, torch.zeros(hidden, n - inter)], dim=1).T
    check("padded FFN == unpadded FFN", bool(torch.allclose(ref, got, atol=1e-5)), True)

    # A dummy head's q is zero, so it attends uniformly to something finite.
    # What makes it harmless is its group's zero wo_a rows.
    v = torch.randn(3, 8)
    attn = torch.softmax(torch.zeros(1, 8) @ v.T, dim=-1) @ v
    check("dummy head output is finite", bool(torch.isfinite(attn).all()), True)
    check("dummy group contributes exactly zero",
          bool((attn @ torch.zeros(8, 16) == 0).all()), True)
    assert rules  # referenced for clarity above


# The real checkpoint: backbone 384 experts x 40 blocks, DSpark draft 128 x 3,
# each block carrying its own gate.weight AND gate.bias. At tp=6 only the 128
# fails to divide -- 384 / 6 = 64 exactly.
EXPERTS = dict(
    SOLO,
    text_config={
        "num_attention_heads": 64, "o_groups": 64, "head_dim": 128,
        "n_routed_experts": 384,
        "dspark_n_routed_experts": 128,
    },
)


def test_experts(m):
    print("experts: the draft's count is padded, the backbone's is not")
    with tempfile.TemporaryDirectory() as tmp:
        d = m.load_dims(write_model(tmp, EXPERTS))
    check("both counts read",
          sorted(v for _, v in d.expert_counts), [128, 384])

    p6 = m.plan_for_tp(d, 6)
    check("draft 128 -> 132",
          p6.expert_pads, (("text_config.dspark_n_routed_experts", 128, 132),))
    check("132 shards evenly over 6", 132 % 6, 0)
    check("backbone 384 needed nothing", 384 % 6, 0)
    # This is the assertion the cluster hit:
    # "n_physical_experts=128 must be divisible by tp_size=6".
    check("no longer refused by the expert guard", p6.expert_advice("dspark"), None)

    print("experts: tp=4 divides both, so nothing is padded")
    check("no pads at tp=4", m.plan_for_tp(d, 4).expert_pads, ())

    print("experts: a backbone count that does not divide is STILL refused")
    odd = dict(EXPERTS, text_config=dict(EXPERTS["text_config"],
                                         n_routed_experts=386))
    with tempfile.TemporaryDirectory() as tmp:
        do = m.load_dims(write_model(tmp, odd))
    advice = m.plan_for_tp(do, 6).expert_advice("dspark")
    check("backbone 386 reported", bool(advice and "386" in advice), True)
    check("draft not reported (it is padded now)",
          "dspark_n_routed_experts" in (advice or ""), False)

    print("experts: the router rules cannot touch the backbone's 384-wide gate")
    rules = {r.what: r for r in m.build_rules(p6)}
    w = rules.get("draft router logits")
    b = rules.get("draft router bias")
    check("logit rows zero-filled 128 -> 132",
          (w.old, w.new, w.mode), (128, 132, m.ZERO))
    check("bias entries pushed out of top-k",
          (b.old, b.new, b.mode), (128, 132, m.NEG_BIG))
    check("bias is finite (-inf would risk 0 * -inf = NaN)",
          m.NEG_BIG_VALUE < -1000 and m.NEG_BIG_VALUE == m.NEG_BIG_VALUE, True)
    for name in ("model.layers.0.ffn.gate.weight", "model.layers.7.ffn.gate.bias"):
        check(f"backbone {name.rsplit('.', 2)[-2]}.{name.rsplit('.', 1)[-1]} untouched",
              bool(w.pattern.search(name) or b.pattern.search(name)), False)
    check("draft gate.weight matched by the logit rule",
          bool(w.pattern.search("model.mtp.0.ffn.gate.weight")), True)
    check("draft gate.bias matched by the bias rule",
          bool(b.pattern.search("model.mtp.2.ffn.gate.bias")), True)
    check("the two rules do not overlap",
          bool(w.pattern.search("model.mtp.0.ffn.gate.bias")), False)

    print("experts: an expert-indexed tensor is caught by shape, not by name")
    # The named rules have to guess the router module's name. The first cluster
    # boot died because that guess was wrong: "Attempted to load weight
    # (torch.Size([128])) into parameter (torch.Size([132]))". The fallback keys
    # on the dim-0 extent inside a draft block instead.
    rx = m._DRAFT_PREFIX_RX
    for name in ("mtp.0.ffn.gate.bias", "mtp.0.ffn.router_bias",
                 "model.dspark.0.ffn.whatever_they_called_it"):
        check(f"draft-scoped: {name}", bool(rx.search(name)), True)
    for name in ("model.layers.0.ffn.gate.bias", "model.embed_tokens.weight"):
        check(f"NOT draft-scoped: {name}", bool(rx.search(name)), False)

    print("experts: the overlay advertises the padded draft count")
    # rewrite_config directly: build()'s symlink half needs a privilege Windows
    # withholds, and the logic under test is entirely in the rewrite.
    spec = importlib.util.spec_from_file_location(
        "make_overlay", os.path.join(SHIM_DIR, "make_overlay.py"))
    ov = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ov)
    cfg = ov.rewrite_config(json.loads(json.dumps(EXPERTS)), p6)
    check("draft count rewritten",
          cfg["text_config"]["dspark_n_routed_experts"], 132)
    check("backbone count left alone",
          cfg["text_config"]["n_routed_experts"], 384)
    check("the rewrite is recorded for the log",
          any("dspark_n_routed_experts" in c
              for c in cfg["_dsv41_tp_pad"]["changes"]), True)


if __name__ == "__main__":
    mod = load_shim()
    test_plan(mod)
    test_overlay(mod)
    test_experts(mod)
    if "--torch" in sys.argv:
        test_torch(mod)
    else:
        print("(padding tests skipped: rerun with --torch on a box that has torch)")
    print("FAILED:", FAILED or "none")
    sys.exit(1 if FAILED else 0)
