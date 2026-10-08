#!/usr/bin/env python3
"""tools/glm_oracle/compare_dumps.py - two dump directories side by side (llama.cpp's dump_tensors vs strata-glm5,
or either vs ref/glm5.py --save), per step and per token.

    python tools/glm_oracle/compare_dumps.py ORACLE_DIR ENGINE_DIR [--step 0] [--token 0] [--layer N] [--all]

Prints, for every tensor present in both, max |diff|, that relative to the oracle's largest magnitude, and the cosine.
Pools padded by llama.cpp (indexer_score) are cut to the engine's count; the MoE weights are compared as sorted sets
(ggml_top_k's order is unspecified).  Exit 1 when a cosine falls under --min-cos (default 0.999).
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import numpy as np


def load(d: pathlib.Path, step: int, token: int, name: str):
    f = d / (f"s{step}_{name}.npy" if token == 0 else f"s{step}_{name}__{token}.npy")
    return np.load(f) if f.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oracle")
    ap.add_argument("engine")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--token", type=int, default=0)
    ap.add_argument("--layer", type=int, help="only this layer's tensors")
    ap.add_argument("--all", action="store_true", help="also the per-layer tensors (default: the layer outputs only)")
    ap.add_argument("--min-cos", type=float, default=0.999)
    a = ap.parse_args()
    A, B = pathlib.Path(a.oracle), pathlib.Path(a.engine)
    suffix = ".npy" if a.token == 0 else f"__{a.token}.npy"
    names = []
    for f in sorted(B.glob(f"s{a.step}_*{suffix}")):
        n = f.name[len(f"s{a.step}_"):-len(suffix)]
        if a.token == 0 and re.search(r"__\d+$", n):
            continue
        names.append(n)

    def layer_of(n):
        m = re.search(r"-(\d+)$", n)
        return int(m.group(1)) if m else -1

    def order(n):
        base = re.sub(r"-\d+$", "", n)
        seq = ["inp_embd", "hc_init", "hc_attn_pre", "attn_norm", "kda_q_conv", "kda_k_conv", "kda_v_conv", "kda_g1", "kda_beta",
               "kda_scan_out", "kda_g2", "kda_normed", "kda_out", "q_resid", "kv_cmpr", "q_absorbed", "indexer_q", "indexer_k",
               "indexer_gate", "indexer_weights", "indexer_score", "attn_out", "hc_attn_post", "hc_ffn_pre", "ffn_norm",
               "ffn_moe_logits", "ffn_moe_probs", "ffn_moe_topk", "ffn_moe_weights", "ffn_moe_out", "ffn_shexp", "ffn_out",
               "l_out", "hc_head", "result_norm", "result_output"]
        return (layer_of(n), seq.index(base) if base in seq else 99)

    names.sort(key=order)
    bad = 0
    rows = []
    for n in names:
        il = layer_of(n)
        base = re.sub(r"-\d+$", "", n)
        if a.layer is not None and il != a.layer:
            continue
        if not a.all and a.layer is None and base not in ("inp_embd", "hc_attn_post", "l_out", "hc_head", "result_norm", "result_output"):
            continue
        ea = load(A, a.step, a.token, n)
        eb = load(B, a.step, a.token, n)
        if ea is None or eb is None:
            rows.append((n, "missing in " + ("oracle" if ea is None else "engine"), "", "", ""))
            continue
        x, y = np.squeeze(ea), np.squeeze(eb)
        note = ""
        if base == "indexer_score":
            x = x.reshape(-1)[: y.size]
            note = "visible pools"
        elif base == "ffn_moe_weights":
            x, y = -np.sort(-x.reshape(-1)), -np.sort(-y.reshape(-1))
            note = "sorted"
        elif base == "ffn_moe_topk":
            x, y = np.sort(x.reshape(-1)), np.sort(y.reshape(-1))
            rows.append((n, "ids", f"{np.mean(x == y) * 100:.0f}% equal", "", "sorted"))
            if not np.array_equal(x, y):
                bad += 1
            continue
        if x.shape != y.shape:
            rows.append((n, f"shape oracle {x.shape} vs engine {y.shape}", "", "", ""))
            bad += 1
            continue
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        ref = np.abs(x.astype(np.float64)).max() + 1e-30
        cos = float(np.dot(x.ravel(), y.ravel()) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))
        if not np.isfinite(y).all():
            note += " NON-FINITE"
            bad += 1
        elif cos < a.min_cos:
            note += " <-- "
            bad += 1
        rows.append((n, f"{d.max():.3e}", f"{d.max() / ref:.3e}", f"{cos:.6f}", note))
    w = max((len(r[0]) for r in rows), default=10)
    print(f"{'tensor':{w}s}  {'max|diff|':>10s}  {'rel to max':>10s}  {'cosine':>9s}")
    for r in rows:
        print(f"{r[0]:{w}s}  {r[1]:>10s}  {r[2]:>10s}  {r[3]:>9s}  {r[4]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
