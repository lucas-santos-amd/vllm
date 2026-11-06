# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Log how many distinct experts each AITER fused_moe call routes to.

Put this directory on PYTHONPATH and set MOE_EXPERT_STATS_DIR. Every Python process
(the vLLM API server and each TP worker) loads sitecustomize at startup; when it later
imports aiter.fused_moe, fused_moe is wrapped so each eager call appends one line to
MOE_EXPERT_STATS_DIR/expert_stats_<pid>.txt:

    call_index tokens topk n_experts distinct_total distinct_routed
    shared_in_every_row tag

tag is the content of MOE_EXPERT_STATS_DIR/tag (re-read every 20 calls; "-" when the
file is missing), so one server can measure several workloads: write a new tag
between benchmark runs.

The wrapper only runs on eager calls (it skips CUDA-graph capture), so serve with
-cc.cudagraph_mode=NONE to see every step. It syncs on each call; use it for
measurement runs only.
"""

import os
import sys

_OUT_DIR = os.environ.get("MOE_EXPERT_STATS_DIR")


def _wrap(module):
    import atexit

    import torch

    orig = module.fused_moe
    path = os.path.join(_OUT_DIR, f"expert_stats_{os.getpid()}.txt")
    # Stays open for the whole process; closed at exit.
    out = open(path, "a", buffering=1 << 16)  # noqa: SIM115
    atexit.register(out.close)
    tag_path = os.path.join(_OUT_DIR, "tag")
    state = {"calls": 0, "tag": "-"}

    def read_tag():
        try:
            with open(tag_path) as f:
                return f.read().strip().replace(" ", "_") or "-"
        except OSError:
            return "-"

    def fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, *args, **kwargs):
        if not torch.cuda.is_current_stream_capturing():
            if state["calls"] % 20 == 0:
                state["tag"] = read_tag()
            n_experts = w1.shape[0]
            shared_id = n_experts - 1
            ids = topk_ids
            distinct_total = torch.unique(ids).numel()
            distinct_routed = torch.unique(ids[ids != shared_id]).numel()
            shared_every_row = bool((ids == shared_id).any(dim=1).all())
            out.write(
                f"{state['calls']} {ids.shape[0]} {ids.shape[1]} {n_experts} "
                f"{distinct_total} {distinct_routed} {int(shared_every_row)} "
                f"{state['tag']}\n"
            )
            state["calls"] += 1
            if state["calls"] % 60 == 0:
                out.flush()
        return orig(hidden_states, w1, w2, topk_weight, topk_ids, *args, **kwargs)

    module.fused_moe = fused_moe


if _OUT_DIR:
    import importlib.abc
    import importlib.util

    os.makedirs(_OUT_DIR, exist_ok=True)

    class _WrapFusedMoe(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != "aiter.fused_moe":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return spec
            exec_module = spec.loader.exec_module

            def exec_and_wrap(module):
                exec_module(module)
                _wrap(module)

            spec.loader.exec_module = exec_and_wrap
            return spec

    sys.meta_path.insert(0, _WrapFusedMoe())
