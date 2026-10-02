"""Synthetic layer operands for the kernel timing scripts (same construction as the paper's measurements)."""
import torch

from biloco.kernel import nvfp4_sf_bytes, pack_signs


def random_problem(M: int, K: int, N: int, R: int, scales: str = "zero", seed: int = None):
    """Random activation, +-1 factors, coefficients S ~ 0.01 N(0, 1), and random NVFP4 codes for A and W.
    scales="zero" (paper): the NVFP4 block scales are zero, so the FP4 GEMM contributes 0 to y and the
    correctness check isolates the low-rank term; scales="random": block scales drawn from [0.25, 2]."""
    torch.manual_seed(R if seed is None else seed)
    X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    V = torch.sign(torch.randn(R, K, device="cuda", dtype=torch.bfloat16)); V[V == 0] = 1
    U = torch.sign(torch.randn(R, N, device="cuda", dtype=torch.bfloat16)); U[U == 0] = 1
    V_packed, U_packed = pack_signs(V), pack_signs(U)
    S = torch.randn(R, device="cuda", dtype=torch.bfloat16) * 0.01
    A = torch.randint(0, 256, (1, M, K // 2), device="cuda", dtype=torch.uint8)
    W = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)
    A_sf = torch.zeros(nvfp4_sf_bytes(M, K), device="cuda", dtype=torch.float8_e4m3fn)
    W_sf = torch.zeros(nvfp4_sf_bytes(N, K), device="cuda", dtype=torch.float8_e4m3fn)
    if scales == "random":
        A_sf.copy_((torch.rand(A_sf.numel(), device="cuda") * 1.75 + 0.25).to(torch.float8_e4m3fn))
        W_sf.copy_((torch.rand(W_sf.numel(), device="cuda") * 1.75 + 0.25).to(torch.float8_e4m3fn))
    alpha = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    A_pad = torch.zeros(1, 128, K // 2, device="cuda", dtype=torch.uint8)
    A_pad[:, :M] = A
    return dict(X=X, V_packed=V_packed, U_packed=U_packed, S=S, A=A, A_pad=A_pad, W=W, A_sf=A_sf, W_sf=W_sf, alpha=alpha)
