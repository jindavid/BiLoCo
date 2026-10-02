"""Fused BiLoCo kernel at M = 1 on the Qwen3-14B projection shapes it supports (paper Table 6).

FP4-only baselines: the padded 128x64-tile GEMM (activation zero-padded to 128 rows) and three unpadded
configurations (128x64 and 128x128 tiles, and the default 256x256 2-SM GEMM). Ratios are to the fastest
FP4-only GEMM of the shape. Mode 2 exists only for K = 5120, so the other shapes use mode 4.

    python scripts/kernel_shapes.py --shapes 5120x5120,5120x1024,17408x5120 --ranks 128,256,512
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from biloco.kernel import BiLoCoGemm, modes_for, ops, reference, time_graph  # noqa: E402
from kernel_table import check  # noqa: E402
from problem import random_problem  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", default="5120x5120,5120x1024,17408x5120", help="KxN list")
    ap.add_argument("--ranks", default="128,256,512")
    ap.add_argument("--passes", type=int, default=15)
    ap.add_argument("--stress", type=int, default=20)
    ap.add_argument("--scales", default="zero", choices=["zero", "random"])
    a = ap.parse_args()
    M = 1
    print(f"# {torch.cuda.get_device_name()} M=1 passes={a.passes} scales={a.scales} (us per launch)", flush=True)
    for shp in a.shapes.split(","):
        K, N = map(int, shp.split("x"))
        p = random_problem(M, K, N, 16, scales=a.scales)
        fp4 = {"pad": lambda: ops.fp4_gemm_n64(p["A_pad"], p["W"], p["A_sf"], p["W_sf"], p["alpha"]),
               "n64": lambda: ops.fp4_gemm_n64(p["A"], p["W"], p["A_sf"], p["W_sf"], p["alpha"]),
               "n128": lambda: ops.fp4_gemm_n128(p["A"], p["W"], p["A_sf"], p["W_sf"], p["alpha"]),
               "default": lambda: ops.fp4_gemm(p["A"], p["W"], p["A_sf"], p["W_sf"], p["alpha"])}
        samples = {k: [] for k in fp4}
        for _ in range(a.passes):
            for k, fn in fp4.items():
                samples[k].append(time_graph(fn))
        tf = {k: sorted(v)[len(v) // 2] for k, v in samples.items()}
        fastest = min(tf.values())
        print(f"{shp:>11} FP4-only: " + "  ".join(f"{k} {v:.2f}" for k, v in tf.items()), flush=True)
        for r in map(int, a.ranks.split(",")):
            p = random_problem(M, K, N, r, scales=a.scales)
            y_ref = reference(p["A"], p["A_sf"], p["W"], p["W_sf"], p["alpha"], p["X"], p["V_packed"], p["U_packed"], p["S"])
            y_ref = y_ref.float().view(M, N)
            arms = {}
            for mode in modes_for(M, r, K):
                g = BiLoCoGemm(p["W"], p["W_sf"], p["alpha"], p["V_packed"], p["U_packed"], p["S"], M=M, mode=mode)
                fn = (lambda g=g: g(p["A_pad"], p["A_sf"], p["X"]))
                g.reset()
                try:
                    nonfin, worst, _ = check(fn, y_ref, M, N, a.stress)
                except RuntimeError as ex:      # a host-side check rejected this configuration
                    print(f"# {shp} r={r} mode{mode}: {str(ex).splitlines()[0][:120]}", flush=True)
                    continue
                if nonfin == 0:
                    arms[f"mode{mode}"] = (fn, g.reset)
            samples = {k: [] for k in arms}
            for _ in range(a.passes):
                for k, (fn, pre) in arms.items():
                    samples[k].append(time_graph(fn, pre))
            t = {k: sorted(v)[len(v) // 2] for k, v in samples.items()}
            best = min(t, key=t.get)
            print(f"{shp:>11} r={r:>3} " + "  ".join(f"{k} {v:.2f}" for k, v in t.items())
                  + f"  best {t[best]:.2f} ({t[best] / fastest:.3f}x fastest FP4-only, {best})", flush=True)


if __name__ == "__main__":
    main()
