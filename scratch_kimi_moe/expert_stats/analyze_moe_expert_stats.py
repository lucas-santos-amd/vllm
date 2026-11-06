# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize the expert_stats_<pid>.txt files written by sitecustomize.py.

Usage:
    python analyze_moe_expert_stats.py /tmp/moe_stats [--moe-layers 60] [--tokens 64]

Prints, per tokens-per-call value, how many distinct routed experts the calls hit
(mean / median / p10 / p90 / min / max), and for --tokens a per-layer breakdown,
taking layer = call_index % --moe-layers. TP ranks route identically, so only the
file with the most calls is used unless --all-files is given.
"""

import argparse
import glob
import os
import statistics
from collections import defaultdict


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 7:
                rows.append(tuple(int(p) for p in parts))
    return rows


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stats_dir")
    p.add_argument("--moe-layers", type=int, default=60)
    p.add_argument("--tokens", type=int, nargs="*", default=[64])
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
    for f, rows in data.items():
        print(f"{os.path.basename(f)}: {len(rows)} calls")
        topk = {r[2] for r in rows}
        n_exp = {r[3] for r in rows}
        shared = sum(r[6] for r in rows) / len(rows)
        print(f"  topk={sorted(topk)} n_experts={sorted(n_exp)}")
        print(f"  shared expert in every row: {shared:.3f} of calls")

        by_tokens = defaultdict(list)
        for call, tokens, _, _, _, routed, _ in rows:
            by_tokens[tokens].append(routed)
        print(
            f"  {'tokens':>7} {'calls':>7} {'mean':>7} {'median':>7} "
            f"{'p10':>5} {'p90':>5} {'min':>5} {'max':>5}   distinct routed experts"
        )
        for tokens in sorted(by_tokens):
            v = by_tokens[tokens]
            print(
                f"  {tokens:7d} {len(v):7d} {statistics.mean(v):7.1f} "
                f"{statistics.median(v):7.1f} {pct(v, 0.1):5d} {pct(v, 0.9):5d} "
                f"{min(v):5d} {max(v):5d}"
            )

        for tokens in args.tokens:
            per_layer = defaultdict(list)
            for call, t, _, _, _, routed, _ in rows:
                if t == tokens:
                    per_layer[call % args.moe_layers].append(routed)
            if not per_layer:
                continue
            print(f"  per layer at tokens={tokens} (layer = call % {args.moe_layers}):")
            line = []
            for layer in sorted(per_layer):
                line.append(f"{layer}:{statistics.mean(per_layer[layer]):.0f}")
            for i in range(0, len(line), 12):
                print("    " + "  ".join(line[i : i + 12]))


if __name__ == "__main__":
    main()
