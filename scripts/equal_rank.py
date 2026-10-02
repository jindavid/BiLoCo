"""Equal-rank control (paper appendix): binary rank 32 vs dense BF16 rank 32 in the same fused kernel
(M = 1, K = N = 5120), plus the dense rank-32 term run unfused (two BF16 GEMMs, then the FP4 GEMM with
the result added) and the padded FP4-only GEMM.

    python scripts/equal_rank.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from biloco.kernel import BiLoCoGemm, dense_bf16_env, ops, reference, time_graph  # noqa: E402
from kernel_table import check  # noqa: E402
from problem import random_problem  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--passes", type=int, default=15)
    ap.add_argument("--stress", type=int, default=50)
    a = ap.parse_args()
    M, K, N, r = 1, 5120, 5120, 32
    torch.manual_seed(0)
    p = random_problem(M, K, N, r)
    X = p["X"]
    # dense BF16 factors in SVDQuant's format (scale folded into V, S = 1)
    Vd = (torch.randn(r, K, device="cuda") / K ** 0.5).to(torch.bfloat16).contiguous()
    Ud = (torch.randn(r, N, device="cuda") * 0.25).to(torch.bfloat16).contiguous()
    S1 = torch.ones(r, device="cuda", dtype=torch.bfloat16)

    y_ref_bin = reference(p["A"], p["A_sf"], p["W"], p["W_sf"], p["alpha"], X, p["V_packed"], p["U_packed"], p["S"])
    yd = ((X.float() @ Vd.float().t()) @ Ud.float()).to(torch.bfloat16)
    y_ref_dense = ops.fp4_gemm_residual(p["A"], p["W"], p["A_sf"], p["W_sf"], p["alpha"], yd)

    g2 = BiLoCoGemm(p["W"], p["W_sf"], p["alpha"], p["V_packed"], p["U_packed"], p["S"], M=1, mode=2)
    g4 = BiLoCoGemm(p["W"], p["W_sf"], p["alpha"], p["V_packed"], p["U_packed"], p["S"], M=1, mode=4)
    gd = BiLoCoGemm(p["W"], p["W_sf"], p["alpha"], p["V_packed"], p["U_packed"], S1, M=1, mode=2)

    def dense_fused():
        gd.epoch += 1
        dense_bf16_env(True)
        try:
            return ops.biloco_gemm_mode2(p["A_pad"], p["W"], p["A_sf"], p["W_sf"], p["alpha"], X, p["V_packed"], S1,
                                         gd.U_col32, gd.z, gd.flags, gd.none, Vd, Ud, M, gd.epoch)
        finally:
            dense_bf16_env(False)

    zbuf = torch.empty(M, r, device="cuda", dtype=torch.bfloat16)
    ybuf = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    def dense_unfused():
        torch.mm(X, Vd.t(), out=zbuf); torch.mm(zbuf, Ud, out=ybuf)
        return ops.fp4_gemm_residual(p["A"], p["W"], p["A_sf"], p["W_sf"], p["alpha"], ybuf)

    arms = {
        "fp4_only_padded": (lambda: ops.fp4_gemm_n64(p["A_pad"], p["W"], p["A_sf"], p["W_sf"], p["alpha"]), None, None),
        "binary_mode2": (lambda: g2(p["A_pad"], p["A_sf"], X), g2.reset, y_ref_bin),
        "binary_mode4": (lambda: g4(p["A_pad"], p["A_sf"], X), g4.reset, y_ref_bin),
        "dense_fused": (dense_fused, gd.reset, y_ref_dense),
        "dense_unfused": (dense_unfused, None, y_ref_dense),
    }
    for k, (fn, pre, yr) in arms.items():
        if yr is None:
            continue
        if pre:
            pre()
        nonfin, worst, _ = check(fn, yr.float().view(M, N), M, N, a.stress)
        print(f"# {k}: runs={a.stress} nonfinite={nonfin} max|err|={worst:.4f} max|ref|={float(yr.abs().max()):.2f}", flush=True)
    samples = {k: [] for k in arms}
    for _ in range(a.passes):
        for k, (fn, pre, _) in arms.items():
            samples[k].append(time_graph(fn, pre))
    t = {k: sorted(v)[len(v) // 2] for k, v in samples.items()}
    for k in arms:
        print(f"{k:16s} {t[k]:7.2f} us  {t[k] / t['fp4_only_padded']:.3f}x", flush=True)


if __name__ == "__main__":
    main()
