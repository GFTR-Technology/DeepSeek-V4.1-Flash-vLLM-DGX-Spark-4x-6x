"""Build a symlink overlay of the model directory with a TP-padded config.json.

vLLM derives every parallel layer's shape from the config, so the config has to
agree with what `dsv41_tp_pad` does to the weight stream. Rather than
monkeypatching the config classes, we hand vLLM a directory that symlinks the 48
real shards (zero copy, no extra disk) and carries one rewritten file..

    python3 /opt/dsv41/make_overlay.py --src /model --dst /model-tp6 --tp 6

Engram fields are deliberately left alone. The tables are split by hash column
and the forward all-gathers then slices back to `n_hash_cols`, so a rank past the
last column owns nothing and needs no config change; a synthetic column would
also have no rows on disk. Likewise `vocab_size`: vLLM pads the vocab itself with
`pad_vocab_size`, and the lm_head/embed weights on disk still have the original
row count.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dsv41_tp_pad as pad  # noqa: E402

CONFIG_NAME = "config.json"


def _container(config: dict) -> tuple[dict, str]:
    """Where the text dims live: some releases nest them under text_config."""
    for key in ("text_config", "language_config", "llm_config"):
        sub = config.get(key)
        if isinstance(sub, dict) and "num_attention_heads" in sub:
            return sub, key
    if "num_attention_heads" in config:
        return config, "top level"
    raise SystemExit(f"[dsv41-overlay] {CONFIG_NAME} has no num_attention_heads")


def _expect(container: dict, key: str, want: int, where: str) -> None:
    got = container.get(key)
    if got != want:
        raise SystemExit(
            f"[dsv41-overlay] {where}.{key} is {got!r}, expected {want!r}. "
            "The plan and the config disagree — re-read the source config.json "
            "before padding anything."
        )


def _walk(config: dict, path: str) -> tuple[dict, str]:
    """Resolve a dotted path to (containing dict, final key)."""
    parts = path.split(".")
    node = config
    for part in parts[:-1]:
        node = node.get(part)
        if not isinstance(node, dict):
            raise SystemExit(f"[dsv41-overlay] {CONFIG_NAME} has no section {part!r} "
                             f"on the way to {path}")
    return node, parts[-1]


def rewrite_config(config: dict, plan: pad.PadPlan) -> dict:
    text, where = _container(config)
    d = plan.dims

    # The plan was built from this same file, so a mismatch here means the source
    # moved under us between planning and overlay build.
    _expect(text, "num_attention_heads", d.heads, where)
    if d.groups is not None:
        _expect(text, "o_groups", d.groups, where)

    changes = []

    def put(container, key, value, place):
        if container.get(key) != value:
            changes.append(f"{place}.{key}: {container.get(key)} -> {value}")
            container[key] = value

    if "attn" in plan.groups:
        if plan.o_groups is not None and d.groups is not None:
            put(text, "o_groups", plan.o_groups, where)
        put(text, "num_attention_heads", plan.heads, where)
        # Only follow the head count when this config really means "one KV head
        # per attention head". MLA keeps a single latent KV head
        # (num_key_value_heads=1), vLLM sizes the KV cache from it, and 1 divides
        # every TP — rewriting it would inflate the KV cache, not pad anything.
        if text.get("num_key_value_heads") == d.heads:
            put(text, "num_key_value_heads", plan.heads, where)
    if "moe" in plan.groups and plan.moe_intermediate != d.moe_intermediate:
        put(text, "moe_intermediate_size", plan.moe_intermediate, where)
    if "dense" in plan.groups and plan.intermediate != d.intermediate:
        put(text, "intermediate_size", plan.intermediate, where)
    # The draft's expert count. _expert_counts records the full dotted path it
    # was found at, because these keys nest (text_config.dspark_n_routed_experts)
    # and the backbone's own count sits right beside it under a different name.
    for path, old, new in plan.expert_pads:
        holder, leaf = _walk(config, path)
        if holder.get(leaf) != old:
            raise SystemExit(
                f"[dsv41-overlay] {path} is {holder.get(leaf)!r}, expected {old!r}. "
                "The plan and the config disagree — re-read the source "
                "config.json before padding anything."
            )
        put(holder, leaf, new, path.rsplit(".", 1)[0] if "." in path else "top level")

    if not changes:
        raise SystemExit(
            "[dsv41-overlay] the plan says padding is needed but nothing in "
            f"{CONFIG_NAME} changed. Serving this overlay would hit vLLM's own "
            "divisibility check. Check that the keys live where _container() "
            "looks for them."
        )

    config["_dsv41_tp_pad"] = {
        "tp": plan.tp,
        "groups": sorted(plan.groups),
        "changes": changes,
        "note": "generated by make_overlay.py; weights are padded at load time",
    }
    for change in changes:
        print(f"[dsv41-overlay] {change}")
    return config


def build(src: str, dst: str, plan: pad.PadPlan) -> None:
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if not os.path.isdir(src):
        raise SystemExit(f"[dsv41-overlay] source {src} is not a directory")
    if os.path.realpath(src) == os.path.realpath(dst):
        raise SystemExit("[dsv41-overlay] refusing to overlay a directory onto itself")

    config_path = os.path.join(src, CONFIG_NAME)
    if not os.path.isfile(config_path):
        raise SystemExit(f"[dsv41-overlay] {config_path} not found")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    config = rewrite_config(config, plan)

    # Rebuild from scratch so a restart cannot inherit a stale config.
    if os.path.islink(dst) or os.path.isfile(dst):
        os.unlink(dst)
    elif os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)

    linked = 0
    for entry in sorted(os.listdir(src)):
        if entry == CONFIG_NAME:
            continue
        os.symlink(os.path.join(src, entry), os.path.join(dst, entry))
        linked += 1

    with open(os.path.join(dst, CONFIG_NAME), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    print(f"[dsv41-overlay] {dst}: {linked} symlink(s) + rewritten {CONFIG_NAME}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="real model directory (read only)")
    parser.add_argument("--dst", required=True, help="overlay directory to create")
    parser.add_argument("--tp", type=int, required=True, help="tensor parallel size")
    parser.add_argument(
        "--groups",
        default=os.environ.get("DSV41_TP_PAD_GROUPS", ""),
        help="comma list from %s (default: %s)"
        % (",".join(pad.ALL_GROUPS), ",".join(pad.DEFAULT_GROUPS)),
    )
    args = parser.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()] or None
    plan = pad.plan_for_tp(pad.load_dims(args.src), args.tp, groups)
    if not plan.active:
        raise SystemExit(
            f"[dsv41-overlay] TP={args.tp} needs no padding; serve /model directly"
        )
    print(f"[dsv41-overlay] TP={plan.tp}: {plan.summary()}")
    build(args.src, args.dst, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
