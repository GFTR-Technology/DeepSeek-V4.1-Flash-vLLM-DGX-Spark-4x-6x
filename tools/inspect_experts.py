#!/usr/bin/env python3
"""What do this checkpoint's MoE tensors actually look like?

Reads only model.safetensors.index.json (a few hundred KB), never the shards, so
it is instant and safe to run against a live NFS export.

    python3 tools/inspect_experts.py /var/tmp/models/DeepSeek-V4.1-Flash

It answers the three questions that decide whether an expert count can be
changed at all:

  1. Does the DSpark draft carry its OWN expert weights, or does it reuse the
     backbone's? Separate names mean separate weights mean real surgery.
  2. Is there an `e_score_correction_bias` (or similar) added to the routing
     score BEFORE top-k? That is the only thing that could make an added expert
     genuinely unselectable, i.e. inert.
  3. What is the router (`gate`) shaped like? Its first dimension is the expert
     count the router can address, and it has to move with any change.
"""
import collections
import json
import os
import re
import sys

EXPERT_RX = re.compile(r"\.experts\.(\d+)\.")
GATE_RX = re.compile(r"(gate|router)\.(weight|e_score_correction_bias|bias)$")
BIAS_RX = re.compile(r"e_score_correction_bias|expert_bias|routed_scaling", re.I)


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    model = argv[1]
    idx = os.path.join(model, "model.safetensors.index.json")
    with open(idx, encoding="utf-8") as f:
        names = list(json.load(f)["weight_map"])
    print(f"{len(names)} tensors in {idx}\n")

    # ---- 1. expert ids per module prefix -------------------------------------
    groups = collections.defaultdict(set)
    for n in names:
        m = EXPERT_RX.search(n)
        if m:
            groups[n[: m.start()]].add(int(m.group(1)))
    print("== expert blocks (prefix -> how many expert ids)")
    by_count = collections.defaultdict(list)
    for prefix, ids in groups.items():
        by_count[(len(ids), min(ids), max(ids))].append(prefix)
    for (count, lo, hi), prefixes in sorted(by_count.items()):
        sample = sorted(prefixes)[:2]
        print(f"   {count:4d} experts (ids {lo}..{hi})  x{len(prefixes):4d} blocks"
              f"   e.g. {sample[0]}")
        if len(sample) > 1:
            print(f"{'':>44}{sample[1]}")
    if not groups:
        print("   none found — experts may be fused into one tensor per layer")

    # ---- 2. a bias applied to the routing score ------------------------------
    print("\n== routing bias (the only candidate for making an expert unselectable)")
    biases = [n for n in names if BIAS_RX.search(n)]
    if biases:
        for n in biases[:6]:
            print(f"   {n}")
        print(f"   ({len(biases)} total)")
        print("   -> if this is added BEFORE top-k, a very negative entry would")
        print("      exclude an added expert. Verify in the model source first.")
    else:
        print("   NONE. Without it an added expert cannot be made unselectable:")
        print("   a zero router row scores exactly 0.0 and top-k can pick it.")

    # ---- 3. the router itself -------------------------------------------------
    print("\n== router / gate tensors")
    gates = [n for n in names if GATE_RX.search(n) and ".experts." not in n]
    for n in sorted(gates)[:8]:
        print(f"   {n}")
    print(f"   ({len(gates)} total; the gate's first dim must match the expert count)")

    # ---- 4. anything that looks like the draft --------------------------------
    print("\n== tensors that look like the speculative draft")
    draft = [n for n in names if re.search(r"dspark|draft|mtp|eagle", n, re.I)]
    if draft:
        dg = collections.defaultdict(set)
        for n in draft:
            m = EXPERT_RX.search(n)
            if m:
                dg[n[: m.start()]].add(int(m.group(1)))
        print(f"   {len(draft)} tensors, {len(dg)} of them expert blocks")
        for prefix, ids in sorted(dg.items())[:3]:
            print(f"     {prefix}  -> {len(ids)} experts")
        if not dg:
            print("   no expert ids under a draft prefix: the draft may REUSE the")
            print("   backbone's experts, in which case its count is a config knob,")
            print("   not a weight layout.")
    else:
        print("   none by name — the draft likely reuses backbone layers")
        print("   (the log line 'Using Eagle3 auxiliary layers from config: (37, 38, 39)'")
        print("    points that way)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
