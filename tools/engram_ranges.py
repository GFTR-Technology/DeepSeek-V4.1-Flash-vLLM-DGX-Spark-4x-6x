#!/usr/bin/env python3
"""Per-rank Engram row ranges for a given TP, straight from config.json.

The Engram tables are split by hash column, so every rank owns a different row
range -- and the ranges MOVE when TP changes. tools/engram_local.py needs them
as LAYER:START:END. Previously they were copied out of a boot log; this derives
them without booting, by reproducing the prime layout in patch/engram.py
(EngramLayout: engram_vocab_size - 1 -> find_next_prime, n_heads per n-gram
order, max_ngram_size - 1 orders per layer, one shared "seen" set across layers).

usage: engram_ranges.py <model_dir> <tp> [rank ...] [--json]
       engram_ranges.py /var/tmp/models/DeepSeek-V4.1-Flash 4 1
       engram_ranges.py /mnt/reddie-models/DeepSeek-V4.1-Flash 6      # all ranks

With a rank it prints only that rank's arguments, ready to paste:
       1:96000564:192001740 14:96003054:192007016
"""
import json
import os
import sys


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 2**32, same as patch/engram.py."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen: set) -> int:
    c = start + 1
    while not _is_prime(c) or c in seen:
        c += 1
    return c


def layout(cfg):
    """[(layer_id, [head_size, ...]), ...] in the order engram.py builds them."""
    layer_ids = list(cfg["engram_layer_ids"])
    n_heads = int(cfg["engram_n_heads"])
    orders = int(cfg["engram_max_ngram_size"]) - 1
    base = int(cfg["engram_vocab_size"]) - 1
    seen, out = set(), []
    for layer_id in layer_ids:
        flat = []
        for _ in range(orders):
            current = base
            for _ in range(n_heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                flat.append(current)
        out.append((layer_id, flat))
    return out


def main(argv):
    flags = [a for a in argv[1:] if a.startswith("--")]
    argv = [argv[0]] + [a for a in argv[1:] if not a.startswith("--")]
    as_json = "--json" in flags
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    model_dir, tp = argv[1], int(argv[2])
    ranks = [int(x) for x in argv[3:]] or list(range(tp))
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    for key in ("text_config", "language_config"):
        if isinstance(cfg.get(key), dict):
            cfg = {**cfg[key], **{k: v for k, v in cfg.items() if not isinstance(v, dict)}}
            break

    tables = layout(cfg)
    n_cols = len(tables[0][1])
    part = -(-n_cols // tp)                        # cdiv, exactly as engram.py
    if n_cols % tp:
        print(f"WARNING: {n_cols} hash columns do not divide TP={tp}: ranks "
              f"{(n_cols + part - 1) // part}..{tp - 1} would own no rows. "
              f"That is fine: patch/engram.py clamps those ranks to zero rows "
              f"and the forward slices their output away. They just need no "
              f"local copy.",
              file=sys.stderr)
    out = {"tp": tp, "hash_columns": n_cols, "ranks": {}}
    for rank in ranks:
        specs, total, layers = [], 0, {}
        for layer_id, sizes in tables:
            lo = sum(sizes[:min(rank * part, n_cols)])
            hi = sum(sizes[:min((rank + 1) * part, n_cols)])
            specs.append(f"{layer_id}:{lo}:{hi}")
            layers[str(layer_id)] = [lo, hi]
            total += hi - lo
        gib = total * (int(cfg["engram_head_dim"]) + int(cfg["engram_head_dim"]) // 32) / 2**30
        out["ranks"][str(rank)] = {"specs": specs, "layers": layers,
                                   "rows": total, "gib": round(gib, 1)}
        if as_json:
            continue
        if len(ranks) == 1:
            print(" ".join(specs))
        else:
            print(f"rank {rank}: {' '.join(specs)}   # {total:,} rows, ~{gib:.0f} GiB local")
    if as_json:
        json.dump(out, sys.stdout)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
