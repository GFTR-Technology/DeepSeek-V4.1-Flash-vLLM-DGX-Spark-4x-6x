"""Read-only probe for the vLLM build inside the DeepSeek-V4.1-Flash image.

The dsv41-feat branch moves between releases and this repo bind-mounts seven
patched files over it, so the exact seams the padding shim hooks cannot be
checked from a dev box. Run this in the container before the first TP!=4 boot and
read the report.

    python3 /opt/dsv41/probe.py --tp 6 --model /model

Touches nothing. Everything it prints is a filesystem read, an attribute lookup
or arithmetic.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys

SECTION = "=" * 72

# What we want to know about this build, and why.
GREPS = [
    ("tp-divide", r"\bdivide\s*\(", "every divisibility assert TP=6 can trip"),
    ("tp-size", r"tensor_model_parallel_world_size|\btp_size\b", "TP-aware code"),
    ("heads", r"num_attention_heads", "does it read the padded head count?"),
    ("o-groups", r"o_groups|n_local_groups", "the wo_a bmm group split"),
    ("disable-tp", r"disable_tp\s*=", "layers that opt OUT of sharding entirely"),
    ("engram-cols", r"part_n_hash_cols|n_hash_cols", "Engram hash-column split"),
]


def _print_header(title: str) -> None:
    print(f"\n{SECTION}\n== {title}\n{SECTION}")


def _vllm_root() -> str | None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        return None
    return list(spec.submodule_search_locations)[0]


def _walk_py(root: str):
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(base, name)


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def section_env() -> None:
    _print_header("environment")
    try:
        import torch
        print(f"torch      {torch.__version__}")
        print(f"cuda       {getattr(torch.version, 'cuda', '?')}")
    except Exception as exc:                                   # noqa: BLE001
        print(f"torch      unavailable ({exc})")
    try:
        import vllm
        print(f"vllm       {vllm.__version__}")
    except Exception as exc:                                   # noqa: BLE001
        print(f"vllm       unavailable ({exc})")
    for key in ("DSV41_TP_PAD", "DSV41_MODEL_SRC", "DSV41_MODEL_DIR",
                "DSV41_ENGRAM_DISK", "DSV41_ENGRAM_DIR", "PYTHONPATH"):
        print(f"{key:22s} {os.environ.get(key, '')!r}")


def section_greps(root: str) -> None:
    """Where in this build does TP divisibility actually bite?"""
    _print_header("TP seams in the model code")
    files = [p for p in _walk_py(root)
             if "deepseek_v4" in p or "/layers/" in p or "/attention/" in p]
    print(f"scanned {len(files)} file(s) under {root}")
    for label, pattern, why in GREPS:
        rx = re.compile(pattern)
        hits = []
        for path in files:
            for n, line in enumerate(_read(path).splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{os.path.relpath(path, root)}:{n}: {line.strip()[:110]}")
        print(f"\n[{label}] {why} — {len(hits)} hit(s)")
        for hit in hits[:12]:
            print(f"   {hit}")
        if len(hits) > 12:
            print(f"   ... {len(hits) - 12} more")


def section_seams() -> None:
    """Do the modules the shim hooks exist, and do they carry the seam?"""
    _print_header("shim attachment points")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dsv41_tp_pad as pad

    for name in pad.HOOKS:
        spec = None
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, AttributeError, ValueError):
            pass
        if spec is None:
            print(f"  MISSING  {name}")
            continue
        module = importlib.import_module(name)
        iters = [a for a in dir(module) if a.endswith("_weights_iterator")]
        print(f"  present  {name}")
        if iters:
            print(f"           {len(iters)} weight iterator(s): {', '.join(sorted(iters))}")
        else:
            print("           no *_weights_iterator attribute — the shim would "
                  "wrap nothing here")


def section_engram() -> None:
    """Engram is not padded; confirm this build can survive a zero-column rank."""
    _print_header("Engram under a TP that does not divide the hash columns")
    try:
        from vllm.model_executor.models.deepseek_v4_1.common import engram
    except Exception as exc:                                   # noqa: BLE001
        print(f"  cannot import engram module: {exc}")
        return
    src = _read(getattr(engram, "__file__", "") or "")
    checks = [
        ("all-gather then slice back to n_hash_cols",
         r"tensor_model_parallel_all_gather.*\n.*\[:, *: *self\.n_hash_cols\]"),
        ("stager clamps head_end to >= head_start", r"max\(\s*self\.head_start"),
        ("zero-owned read is skipped", r"if not bool\(owned\.any\(\)\)"),
    ]
    for what, pattern in checks:
        ok = re.search(pattern, src, re.MULTILINE) is not None
        print(f"  [{'ok ' if ok else 'NO '}] {what}")
    if not all(re.search(p, src, re.MULTILINE) for _, p in checks[1:]):
        print("  -> patch/engram.py is not the mounted one, or it is out of date.")


def section_plan(tp: int, model: str) -> None:
    _print_header(f"padding plan for TP={tp}")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dsv41_tp_pad as pad

    try:
        dims = pad.load_dims(model)
    except SystemExit as exc:
        print(f"  {exc}")
        return
    plan = pad.plan_for_tp(dims, tp)
    print(f"  checkpoint  heads={dims.heads} o_groups={dims.groups} "
          f"heads/group={dims.heads_per_group}")
    print(f"              head_dim={dims.head_dim} o_lora_rank={dims.o_lora_rank}")
    print(f"              moe_intermediate={dims.moe_intermediate} "
          f"intermediate={dims.intermediate} block={dims.block}")
    print(f"  active={plan.active}  {plan.summary()}")
    for rule in pad.build_rules(plan):
        print(f"    {rule.group:6s} {rule.what:22s} dim{rule.dim} "
              f"{rule.old} -> {rule.new}  [{rule.mode}]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, default=int(os.environ.get("DSV41_TP_PAD", "6") or 6))
    parser.add_argument("--model", default=os.environ.get("DSV41_MODEL_SRC", "/model"))
    args = parser.parse_args()

    section_env()
    root = _vllm_root()
    if root:
        section_greps(root)
    else:
        print("\n(vllm not importable — skipping the source scan)")
    section_seams()
    section_engram()
    section_plan(args.tp, args.model)
    print(f"\n{SECTION}\nprobe done — nothing was modified\n{SECTION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
