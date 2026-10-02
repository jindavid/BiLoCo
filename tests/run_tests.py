"""Tests: the decomposition (CPU), and the fused kernel against the unfused reference for every batch size,
rank and mode (needs a B300 and the built extension; skipped otherwise).

    python tests/run_tests.py
"""
import os
import sys
import traceback

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import torch  # noqa: E402

from biloco.peel import binary_peel, decompose, refit_alphas  # noqa: E402


def test_peel():
    torch.manual_seed(0)
    W = torch.randn(256, 384)
    U, V, S, R = binary_peel(W, r=32, T=5)
    assert set(U.unique().tolist()) <= {-1.0, 1.0} and set(V.unique().tolist()) <= {-1.0, 1.0}
    assert torch.allclose(W, (U * S) @ V.T + R, atol=1e-4)
    # each atom removes (u^T R v)^2 / (mn) of the squared norm, so the residual shrinks
    assert R.norm() < W.norm()
    S2, R2 = refit_alphas(W, U, V, S)
    assert torch.allclose(W, (U * S2) @ V.T + R2, atol=1e-3)
    assert R2.norm() <= R.norm() + 1e-4          # the joint least-squares refit cannot increase the residual
    U3, V3, S3, R3 = decompose(W, r=32, T=5)
    assert torch.equal(U3, U) and torch.allclose(S3, S2) and torch.allclose(R3, R2)


def test_kernel(kernel, random_problem, M, r):
    K = N = 5120
    p = random_problem(M, K, N, r)
    y_ref = kernel.reference(p["A"], p["A_sf"], p["W"], p["W_sf"], p["alpha"], p["X"], p["V_packed"], p["U_packed"], p["S"])
    y_ref = y_ref.float().view(M, N)
    for mode in kernel.modes_for(M, r, K):
        g = kernel.BiLoCoGemm(p["W"], p["W_sf"], p["alpha"], p["V_packed"], p["U_packed"], p["S"], M=M, mode=mode)
        for _ in range(5):                       # consecutive launches reuse the epoch-tagged scratch
            y = g(p["A_pad"], p["A_sf"], p["X"]).view(-1, N)[:M].float()
            torch.cuda.synchronize()
            assert torch.isfinite(y).all(), f"mode {mode}: non-finite output"
            err = float((y - y_ref).abs().max())
            assert err <= 0.01 * float(y_ref.abs().max()), f"mode {mode}: max |err| {err:.4f}"


def main():
    results = []

    def run(name, fn, *args):
        try:
            fn(*args)
            results.append(True)
            print(f"ok    {name}", flush=True)
        except Exception:  # noqa: BLE001
            results.append(False)
            print(f"FAIL  {name}", flush=True)
            traceback.print_exc()

    run("peel", test_peel)
    try:
        import biloco.kernel as kernel
        from problem import random_problem
    except ImportError as ex:
        kernel = None
        print(f"skip  kernel tests (extension not built: {ex})")
    if kernel is not None and not torch.cuda.is_available():
        kernel = None
        print("skip  kernel tests (no GPU)")
    if kernel is not None:
        for M in (1, 2, 4, 8, 16, 32):
            for r in kernel.RANKS:
                run(f"kernel M={M} r={r}", test_kernel, kernel, random_problem, M, r)
    n_fail = results.count(False)
    print(f"{len(results) - n_fail} passed, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
