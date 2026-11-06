# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K2.6 MXFP4 MoE at TP8 on MI355X: per-kernel times of GEMM1 providers.

Providers:
  aiter   AITER 2-stage fused_moe (FlyDSL GEMM1 / GEMM2), as vLLM's AITER_MXFP4_MXFP4
          backend runs it. Times every kernel of the call.
  triton  AITER Triton moe_gemm_a4w4 GEMM1 on gfx950 with the work the FlyDSL GEMM1
          does in one launch: mxfp4_quant(x) -> GEMM1 with fused SiLU(gate) x up ->
          mxfp4_quant(intermediate). Routing is built outside the timed region.

Shape: 384 routed experts + 1 fused shared expert (id 384), top-k 8 + 1, hidden 7168,
moe intermediate 256 per rank. Weights are built like AITER's
op_tests/test_moe_2stage.py and vLLM's weight prep (per-1x32 fp4x2 quant,
shuffle_weight((16, 16)), e8m0_shuffle).
The Triton provider reads the same fp4 bytes and scales in its own layout: gate / up
rows interleaved for the fused activation, [E, K/2, N] column-major, scales
[E, K/32, N] with the CDNA4 swizzle (--no-swizzle to skip it).

Each token picks its 8 routed experts among --active-experts experts (default: all
384), because decode GEMM1 is weight-bandwidth bound and its time follows the number
of distinct experts hit. The shared expert is always added. Both providers get the
same routing. --copies layers rotate so every call reads its weights from HBM.

--check compares the Triton GEMM1 output with an fp32 torch reference on the same
dequantized fp4 input and weights (run from the AITER repo root, or put it on
PYTHONPATH, for op_tests).

Example:
    python bench_moe_kimi.py -M 1 16 32 64 128 256 512 16384 --check
    python bench_moe_kimi.py -M 64 --active-experts 139 --provider triton

"""

import argparse
import csv
from collections import defaultdict

import aiter
import torch
import torch.nn.functional as F
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_a4w4, mxfp4_quant
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.utils.shuffle import shuffle_scale_moe
from aiter.utility import fp4_utils

E_ROUTED, HIDDEN, INTER, TOPK = 384, 7168, 256, 8
E = E_ROUTED + 1
N_ACT = TOPK + 1
SHARED_ID = E_ROUTED


def make_layer(device, with_triton, swizzle):
    """One MoE layer's weights in the layout fused_moe reads, plus the Triton GEMM1
    view of the same W1 bytes when requested."""
    quant = aiter.get_torch_quant(QuantType.per_1x32)
    layer = {}
    for name, shape in (("w1", (E, 2 * INTER, HIDDEN)), ("w2", (E, HIDDEN, INTER))):
        w = torch.randn(shape, dtype=torch.bfloat16, device=device) * 0.05
        w_q, w_scale = quant(w, quant_dtype=dtypes.fp4x2)
        del w
        w_q = w_q.view(shape[0], shape[1], shape[2] // 2)
        if name == "w1" and with_triton:
            layer.update(triton_w1(w_q, w_scale, swizzle))
        layer[name] = shuffle_weight(w_q, layout=(16, 16))
        layer[name + "_scale"] = fp4_utils.e8m0_shuffle(w_scale)
    return layer


def triton_w1(w_q, w_scale, swizzle):
    """W1 bytes [E, 2I, K/2] + scales -> moe_gemm_a4w4 layout. Gate / up rows are
    interleaved because the fused activation reads (gate, up) pairs along N."""
    w = w_q.view(torch.uint8)
    s = w_scale.view(torch.uint8).view(E, 2 * INTER, HIDDEN // 32)
    w = torch.stack((w[:, :INTER], w[:, INTER:]), dim=2).reshape(E, 2 * INTER, -1)
    s = torch.stack((s[:, :INTER], s[:, INTER:]), dim=2).reshape(E, 2 * INTER, -1)
    w_t, s_plain = w.transpose(1, 2), s.transpose(1, 2)
    if swizzle:
        s_t, layout = shuffle_scale_moe(
            s_plain,
            arch="gfx950",
            preshuffle_factor=32,
            scale_kwidth=8,
            return_layout=True,
        )
    else:
        s_t, layout = s_plain, None
    return {"tw1": w_t, "tw1_scale": s_t, "tw1_scale_plain": s_plain, "layout": layout}


def make_routing(M, active, device, seed=0):
    """topk_ids [M, 9] (8 routed among `active` experts + the shared expert) and
    normalized topk_weights [M, 9] fp32."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    active = max(TOPK, min(active, E_ROUTED))
    pool = torch.randperm(E_ROUTED, generator=gen)[:active]
    picks = torch.rand((M, active), generator=gen).topk(TOPK, dim=1).indices
    routed = pool[picks]
    ids = torch.cat([routed, torch.full((M, 1), SHARED_ID, dtype=routed.dtype)], dim=1)
    weights = torch.rand((M, TOPK), generator=gen)
    weights = torch.cat([weights / weights.sum(1, keepdim=True), torch.ones((M, 1))], 1)
    distinct = torch.unique(routed).numel() + 1
    return (
        ids.to(device=device, dtype=torch.int32),
        weights.to(device=device, dtype=torch.float32),
        distinct,
    )


def triton_routing(ids):
    """RoutingData whose top-9 per token is exactly `ids`."""
    M = ids.shape[0]
    logits = torch.full((M, E), -30.0, dtype=torch.float16, device=ids.device)
    ranks = torch.arange(N_ACT, 0, -1, dtype=torch.float16, device=ids.device)
    logits.scatter_(1, ids.long(), ranks.expand(M, N_ACT).contiguous())
    rdata, gather_idx, _ = routing(logits, N_ACT)
    got = rdata.topk_ids.long().sort(dim=1).values
    assert torch.equal(got, ids.long().sort(dim=1).values), "routing changed experts"
    return rdata, gather_idx


def run_triton(hidden, layer, rdata, gather_idx):
    x_q, x_scale = mxfp4_quant(hidden)
    inter = moe_gemm_a4w4(
        x_q,
        layer["tw1"],
        x_scale,
        layer["tw1_scale"],
        routing_data=rdata,
        gather_indx=gather_idx,
        swizzle_mx_scale=layer["layout"],
        out_dtype=torch.bfloat16,
        apply_swiglu=True,
        alpha=1.0,
        limit=None,
        swiglu_add_residual=False,
    )
    mxfp4_quant(inter)
    return x_q, x_scale, inter


def check_triton(hidden, layer, rdata, gather_idx):
    """Relative and max error of the Triton GEMM1 (+ SiLU x up) vs fp32 torch."""
    from op_tests.triton_tests.utils.mxfp_ref import upcast_from_mxfp

    x_q, x_scale, inter = run_triton(hidden, layer, rdata, gather_idx)
    x = upcast_from_mxfp(x_q, x_scale, torch.bfloat16, axis=-1).float()
    rows = gather_idx.long()
    tok = rows // N_ACT
    expert = rdata.topk_ids.reshape(-1).long()[rows]
    ref = torch.empty_like(inter, dtype=torch.float32)
    for e in torch.unique(expert).tolist():
        sel = expert == e
        w = upcast_from_mxfp(
            layer["tw1"][e : e + 1],
            layer["tw1_scale_plain"][e : e + 1],
            torch.bfloat16,
            axis=1,
        )[0].float()
        h = x[tok[sel]] @ w
        ref[sel] = F.silu(h[:, 0::2]) * h[:, 1::2]
    diff = inter.float() - ref
    return (diff.norm() / ref.norm()).item(), diff.abs().max().item()


def classify(name):
    n = name.lower()
    if "sort" in n:
        return "sort"
    if "moe_gemm_a4w4" in n or "gemm1" in n or "moe1" in n or "_g1_" in n:
        return "gemm1"
    if "gemm2" in n or "moe2" in n or "_g2_" in n:
        return "gemm2"
    if "quant" in n:
        return "quant"
    return "other"


def kernel_times(fn, iters):
    """Average device time per call of each kernel fn launches."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for i in range(iters):
            fn(i)
        torch.cuda.synchronize()
    totals = defaultdict(float)
    counts = defaultdict(int)
    for item in prof.key_averages():
        if item.device_type != torch.autograd.DeviceType.CUDA:
            continue
        total = getattr(item, "device_time_total", None)
        if total is None:
            total = item.cuda_time_total
        totals[item.key] += total
        counts[item.key] += item.count
    return {k: (totals[k] / iters, counts[k] / iters) for k in totals}


def report(provider, M, active, distinct, times, rows):
    by_class = defaultdict(float)
    print(f"\n[{provider}] M={M} active_routed={active} distinct_experts={distinct}")
    for name, (us, per_call) in sorted(times.items(), key=lambda kv: -kv[1][0]):
        kind = classify(name)
        by_class[kind] += us
        print(f"  {kind:6s} {us:9.2f} us  {per_call:4.1f}x/call  {name[:150]}")
        rows.append([provider, M, active, distinct, kind, f"{us:.3f}", per_call, name])
    total = sum(by_class.values())
    summary = "  ".join(f"{k}={v:.2f}" for k, v in sorted(by_class.items()))
    print(f"  total={total:.2f} us  {summary}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-M", type=int, nargs="+", default=[1, 16, 32, 64, 128, 256, 512])
    p.add_argument("--active-experts", type=int, default=E_ROUTED)
    p.add_argument(
        "--provider",
        nargs="+",
        choices=("aiter", "triton"),
        default=["aiter", "triton"],
    )
    p.add_argument("--no-swizzle", action="store_true", help="plain Triton W1 scales")
    p.add_argument("--check", action="store_true", help="check Triton GEMM1 output")
    p.add_argument("--copies", type=int, default=4, help="layers rotated per call")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    device = "cuda"
    torch.manual_seed(0)
    with_triton = "triton" in args.provider
    layers = [
        make_layer(device, with_triton, not args.no_swizzle) for _ in range(args.copies)
    ]
    rows = []
    for M in args.M:
        ids, weights, distinct = make_routing(M, args.active_experts, device)
        hidden = torch.randn((M, HIDDEN), dtype=torch.bfloat16, device=device)
        fns = {}
        if "aiter" in args.provider:

            def fn_aiter(i, hidden=hidden, weights=weights, ids=ids):
                layer = layers[i % len(layers)]
                fused_moe(
                    hidden,
                    layer["w1"],
                    layer["w2"],
                    weights,
                    ids,
                    None,
                    ActivationType.Silu,
                    QuantType.per_1x32,
                    False,
                    layer["w1_scale"],
                    layer["w2_scale"],
                )

            fns["aiter"] = fn_aiter
        if with_triton:
            rdata, gather_idx = triton_routing(ids)
            if args.check:
                rel, mx = check_triton(hidden, layers[0], rdata, gather_idx)
                print(f"\n[triton check] M={M} rel_err={rel:.3e} max_abs_err={mx:.3e}")

            def fn_triton(i, hidden=hidden, rdata=rdata, gather_idx=gather_idx):
                run_triton(hidden, layers[i % len(layers)], rdata, gather_idx)

            fns["triton"] = fn_triton

        for provider, fn in fns.items():
            for i in range(args.warmup):
                fn(i)
            torch.cuda.synchronize()
            times = kernel_times(fn, args.iters)
            report(provider, M, args.active_experts, distinct, times, rows)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "provider",
                    "M",
                    "active_routed",
                    "distinct_experts",
                    "class",
                    "us",
                    "per_call",
                    "kernel",
                ]
            )
            w.writerows(rows)


if __name__ == "__main__":
    main()
