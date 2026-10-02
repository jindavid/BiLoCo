"""Decode latency of the fused BiLoCo kernel vs the FP4-only GEMM (paper Table 5) and its error vs the
unfused reference. K = N = 5120 on B300.

For each (M, rank) every available mode is checked (--stress launches against the reference) and timed;
the table reports the fastest correct mode. The baseline is the FP4-only GEMM with the same 128x64 tiles
and the activation zero-padded to 128 rows (`pad`), measured in the same run; `s128` is the unpadded
128x128-tile FP4 GEMM. Times: median over --passes interleaved passes of a CUDA graph of 64 launches.

    python scripts/kernel_table.py --Ms 1,2,4,8,16,32 --ranks 16,32,64,128,256,512
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from biloco.kernel import BiLoCoGemm, modes_for, ops, reference, time_graph  # noqa: E402
from problem import random_problem  # noqa: E402


def check(fn, y_ref, M, N, runs):
    """Error of `runs` consecutive launches vs the reference: non-finite count, max |err|, max #elements differing."""
    nonfin, worst, ndiff = 0, 0.0, 0
    for _ in range(runs):
        y = fn().view(-1, N)[:M].float(); torch.cuda.synchronize()
        fin = torch.isfinite(y); nonfin += int((~fin).sum())
        worst = max(worst, float((y - y_ref).abs().masked_fill(~fin, 0).max()))
        ndiff = max(ndiff, int(((y != y_ref) | ~fin).sum()))
    return nonfin, worst, ndiff


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Ms", default="1,2,4,8,16,32")
    ap.add_argument("--ranks", default="16,32,64,128,256,512")
    ap.add_argument("--K", type=int, default=5120)
    ap.add_argument("--N", type=int, default=5120)
    ap.add_argument("--passes", type=int, default=15)
    ap.add_argument("--stress", type=int, default=50, help="launches checked against the reference per mode")
    ap.add_argument("--scales", default="zero", choices=["zero", "random"], help="NVFP4 block scales of A and W")
    a = ap.parse_args()
    K, N = a.K, a.N
    print(f"# {torch.cuda.get_device_name()} K={K} N={N} passes={a.passes} scales={a.scales} (us per launch)", flush=True)
    print(f"{'M':>3} {'r':>4} {'pad':>6} {'s128':>6} {'mode2':>6} {'mode4':>6} {'mode5':>6} {'best':>6} {'x pad':>6}  mode  "
          f"max|err|/max|ref|  frac_diff", flush=True)
    for M in map(int, a.Ms.split(",")):
        for r in map(int, a.ranks.split(",")):
            p = random_problem(M, K, N, r, scales=a.scales)
            y_ref = reference(p["A"], p["A_sf"], p["W"], p["W_sf"], p["alpha"], p["X"], p["V_packed"], p["U_packed"], p["S"])
            y_ref = y_ref.float().view(M, N)
            arms = {"pad": (lambda: ops.fp4_gemm_n64(p["A_pad"], p["W"], p["A_sf"], p["W_sf"], p["alpha"]), None),
                    "s128": (lambda: ops.fp4_gemm_n128(p["A"], p["W"], p["A_sf"], p["W_sf"], p["alpha"]), None)}
            errs = {}
            for mode in modes_for(M, r, K):
                g = BiLoCoGemm(p["W"], p["W_sf"], p["alpha"], p["V_packed"], p["U_packed"], p["S"], M=M, mode=mode)
                fn = (lambda g=g: g(p["A_pad"], p["A_sf"], p["X"]))
                g.reset()
                try:
                    nonfin, worst, ndiff = check(fn, y_ref, M, N, a.stress)
                except RuntimeError as ex:      # a host-side check rejected this configuration
                    print(f"# M={M} r={r} mode{mode}: {str(ex).splitlines()[0][:120]}", flush=True)
                    continue
                if nonfin == 0:
                    arms[f"mode{mode}"] = (fn, g.reset)
                    errs[f"mode{mode}"] = (worst / float(y_ref.abs().max()), ndiff / (M * N))
                else:
                    print(f"# M={M} r={r} mode{mode}: {nonfin} non-finite outputs, excluded", flush=True)
            samples = {k: [] for k in arms}
            for _ in range(a.passes):
                for k, (fn, pre) in arms.items():
                    samples[k].append(time_graph(fn, pre))
            t = {k: sorted(v)[len(v) // 2] for k, v in samples.items()}
            best = min((k for k in t if k.startswith("mode")), key=t.get)
            col = lambda k: f"{t[k]:6.2f}" if k in t else f"{'-':>6}"
            print(f"{M:>3} {r:>4} {t['pad']:6.2f} {t['s128']:6.2f} {col('mode2')} {col('mode4')} {col('mode5')} "
                  f"{t[best]:6.2f} {t[best] / t['pad']:6.3f}  {best}  {errs[best][0]:.2e}  {errs[best][1]:.5f}", flush=True)


if __name__ == "__main__":
    main()
