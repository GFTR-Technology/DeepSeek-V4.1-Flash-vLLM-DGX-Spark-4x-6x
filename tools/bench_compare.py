#!/usr/bin/env python3
"""Compare fixed-prompt bench runs (bench-<label>.json from bench/v41bench.py) side by side.

usage: bench_compare.py results/boot6/bench-boot6.json results/boot7/bench-boot7.json [...]
Prints Markdown: headline by concurrency, per-stream tok/s by category at C1, cold prefill,
and the ratio of the last run to the first (e.g. DSpark vs no speculation).
"""
import json, statistics as st, sys


def load(p):
    d = json.load(open(p))
    d["_by"] = {(b["c"], b["category"]): b for b in d["batches"]}
    return d


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    runs = [load(p) for p in sys.argv[1:]]
    labels = [r["label"] for r in runs]
    levels = sorted({b["c"] for r in runs for b in r["batches"]})
    cats = []
    for r in runs:
        for b in r["batches"]:
            if b["category"] not in cats:
                cats.append(b["category"])
    print("### Throughput by concurrency (8 categories; counting ceiling excluded)\n")
    rows = []
    for c in levels:
        row = [f"C{c}"]
        for r in runs:
            h = next((x for x in r.get("headline", []) if x["level"] == f"C{c}"), None)
            row += [h["agg_tok_s"] if h else None, h["per_stream_tok_s"] if h else None, h["ttft_mean_s"] if h else None]
        rows.append(row)
    hdr = ["C"] + [f"{l} {k}" for l in labels for k in ("agg tok/s", "per-stream", "TTFT s")]
    print(table(hdr, rows) + "\n")
    if len(runs) > 1:
        a, b = runs[0], runs[-1]
        print(f"### {b['label']} / {a['label']} ratio\n")
        rr = []
        for c in levels:
            ha = next((x for x in a["headline"] if x["level"] == f"C{c}"), None)
            hb = next((x for x in b["headline"] if x["level"] == f"C{c}"), None)
            if ha and hb and ha["agg_tok_s"]:
                rr.append([f"C{c}", f"{hb['agg_tok_s'] / ha['agg_tok_s']:.2f}x",
                           f"{hb['per_stream_tok_s'] / ha['per_stream_tok_s']:.2f}x" if ha["per_stream_tok_s"] else None])
        print(table(["C", "aggregate", "per-stream"], rr) + "\n")
    for c in (1, max(levels)):
        print(f"### Per-stream tok/s by category at C{c}\n")
        rows = [[cat] + [r["_by"].get((c, cat), {}).get("per_stream_tok_s") for r in runs] for cat in cats]
        print(table(["category"] + labels, rows) + "\n")
    print("### Cold prefill (unique prefix)\n")
    tg = sorted({p["target"] for r in runs for p in r.get("prefill", [])})
    rows = []
    for t in tg:
        row = [t]
        for r in runs:
            p = next((x for x in r.get("prefill", []) if x["target"] == t), {})
            row += [p.get("prompt_tokens"), p.get("ttft_s"), p.get("prefill_tok_s")]
        rows.append(row)
    print(table(["target"] + [f"{l} {k}" for l in labels for k in ("tokens", "TTFT s", "tok/s")], rows) + "\n")
    for r in runs:
        c1 = [r["_by"][(1, cat)]["per_stream_tok_s"] for cat in cats if (1, cat) in r["_by"] and r["_by"][(1, cat)]["per_stream_tok_s"]]
        top = max(levels)
        agg = [b["agg_tok_s"] for b in r["batches"] if b["c"] == top]
        print(f"- {r['label']}: C1 per-stream peak {max(c1):.1f}, median {st.median(c1):.1f} tok/s; "
              f"C{top} aggregate peak {max(agg):.1f} tok/s")


if __name__ == "__main__":
    main()
