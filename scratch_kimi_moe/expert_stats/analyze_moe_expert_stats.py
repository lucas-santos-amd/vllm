# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize the expert_stats_<pid>.txt files written by sitecustomize.py.

Usage:
    python analyze_moe_expert_stats.py /tmp/moe_stats [--tokens 64]
    python analyze_moe_expert_stats.py /tmp/moe_stats --by-tag [--csv experts.csv]

Default: per tokens-per-call value, how many distinct routed experts the calls hit
(mean / median / p10 / p90 / min / max), and for --tokens a per-layer breakdown,
taking layer = call_index % --moe-layers.

--by-tag: one block per tag written to MOE_EXPERT_STATS_DIR/tag. A tag "c<N>" (for
example c16) is a run at concurrency N: calls with N tokens are its decode steps,
calls with more than N tokens are its prefill (or mixed prefill + decode) steps.
--csv writes that summary, one row per (tag, phase, tokens).

TP ranks route identically, so only the file with the most calls is used unless
--all-files is given. Counts exclude the shared expert; add 1 for distinct experts
touched.
"""

import argparse
import csv
import glob
import os
import re
import statistics
from collections import defaultdict


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) not in (7, 8):
                continue
            tag = parts[7] if len(parts) == 8 else "-"
            rows.append((*(int(p) for p in parts[:7]), tag))
    return rows


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]


def summary(values):
    return {
        "calls": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p10": pct(values, 0.1),
        "p90": pct(values, 0.9),
        "min": min(values),
        "max": max(values),
    }


HEADER = (
    f"  {'tokens':>7} {'calls':>7} {'mean':>7} {'median':>7} "
    f"{'p10':>5} {'p90':>5} {'min':>5} {'max':>5}   distinct routed experts"
)


def print_row(tokens, s):
    print(
        f"  {tokens!s:>7} {s['calls']:7d} {s['mean']:7.1f} {s['median']:7.1f} "
        f"{s['p10']:5d} {s['p90']:5d} {s['min']:5d} {s['max']:5d}"
    )


def by_tokens(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r[1]].append(r[5])
    return groups


def report_plain(rows, args):
    print(HEADER)
    groups = by_tokens(rows)
    for tokens in sorted(groups):
        print_row(tokens, summary(groups[tokens]))
    for tokens in args.tokens:
        per_layer = defaultdict(list)
        for r in rows:
            if r[1] == tokens:
                per_layer[r[0] % args.moe_layers].append(r[5])
        if not per_layer:
            continue
        print(f"  per layer at tokens={tokens} (layer = call % {args.moe_layers}):")
        line = [f"{k}:{statistics.mean(v):.0f}" for k, v in sorted(per_layer.items())]
        for i in range(0, len(line), 12):
            print("    " + "  ".join(line[i : i + 12]))


def report_by_tag(rows, csv_rows):
    tags = defaultdict(list)
    for r in rows:
        tags[r[7]].append(r)
    for tag in sorted(tags, key=lambda t: (len(t), t)):
        m = re.fullmatch(r"c(\d+)", tag)
        conc = int(m.group(1)) if m else None
        print(f"\n  tag={tag} ({len(tags[tag])} calls)")
        if conc is None:
            print("    no c<N> tag: listing every token count")
            groups = by_tokens(tags[tag])
            print(HEADER)
            for tokens in sorted(groups):
                print_row(tokens, summary(groups[tokens]))
            continue
        groups = by_tokens(tags[tag])
        print(HEADER)
        if conc in groups:
            s = summary(groups[conc])
            print("    decode (tokens == conc):")
            print_row(conc, s)
            csv_rows.append([tag, conc, "decode", conc, *s.values()])
            dec = groups[conc]
            q = max(1, len(dec) // 4)
            quarters = [dec[i : i + q] for i in range(0, q * 4, q)]
            print(
                "    decode mean by quarter of the run: "
                + "  ".join(f"{statistics.mean(v):.1f}" for v in quarters if v)
            )
        else:
            print(f"    no decode calls with {conc} tokens")
        prefill = {t: v for t, v in groups.items() if t > conc}
        if prefill:
            print("    prefill / mixed (tokens > conc):")
            for tokens in sorted(prefill):
                s = summary(prefill[tokens])
                print_row(tokens, s)
                csv_rows.append([tag, conc, "prefill", tokens, *s.values()])
            all_pref = [x for v in prefill.values() for x in v]
            weighted = sum(t * len(v) for t, v in prefill.items()) / len(all_pref)
            s = summary(all_pref)
            print(f"    all prefill calls (call-weighted mean tokens {weighted:.0f}):")
            print_row("all", s)
            csv_rows.append([tag, conc, "prefill_all", round(weighted), *s.values()])
        ramp = {t: v for t, v in groups.items() if t < conc}
        if ramp:
            print(
                "    ramp (tokens < conc): "
                + ", ".join(
                    f"{t}:{statistics.mean(v):.0f}" for t, v in sorted(ramp.items())
                )
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stats_dir")
    p.add_argument("--moe-layers", type=int, default=60)
    p.add_argument("--tokens", type=int, nargs="*", default=[64])
    p.add_argument("--by-tag", action="store_true")
    p.add_argument("--csv", default=None, help="with --by-tag: write the summary")
    p.add_argument("--all-files", action="store_true")
    args = p.parse_args()

    files = glob.glob(os.path.join(args.stats_dir, "expert_stats_*.txt"))
    data = {f: load(f) for f in files}
    data = {f: r for f, r in data.items() if r}
    if not data:
        raise SystemExit(f"no stats in {args.stats_dir}")
    if not args.all_files:
        best = max(data, key=lambda f: len(data[f]))
        data = {best: data[best]}
    csv_rows = []
    for f, rows in data.items():
        print(f"{os.path.basename(f)}: {len(rows)} calls")
        topk = {r[2] for r in rows}
        n_exp = {r[3] for r in rows}
        shared = sum(r[6] for r in rows) / len(rows)
        print(f"  topk={sorted(topk)} n_experts={sorted(n_exp)}")
        print(f"  shared expert in every row: {shared:.3f} of calls")
        if args.by_tag:
            report_by_tag(rows, csv_rows)
        else:
            report_plain(rows, args)
    if args.csv and csv_rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["tag", "conc", "phase", "tokens", "calls", "mean", "median"]
                + ["p10", "p90", "min", "max"]
            )
            w.writerows(csv_rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
