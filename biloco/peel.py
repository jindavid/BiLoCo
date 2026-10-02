"""BiLoCo decomposition W ~ U diag(alpha) V^T + R with U, V in {+-1}: greedy sign peeling and a joint coefficient refit.
R is the residual that is then quantized to NVFP4."""
import torch

_PEEL_SEED = 42


def set_peel_seed(seed: int):
    """Seed of the random sign initialization. Every binary_peel call restarts the same CUDA stream."""
    global _PEEL_SEED
    _PEEL_SEED = int(seed)


def binary_peel(W: torch.Tensor, r: int, T: int = 5, generator: torch.Generator = None):
    """Greedy rank-one peeling W ~ sum_i alpha_i u_i v_i^T with u_i, v_i in {+-1}.

    Each atom starts from random signs v, alternates u = sign(R v), v = sign(R^T u) for T steps,
    takes the Frobenius-optimal alpha = u^T R v / (m n), and deflates R <- R - alpha u v^T.
    Returns U (m, r), V (n, r), S (r,) and the residual R, all in W's dtype.
    """
    device = W.device
    m, n = W.shape
    U = torch.empty(m, r, device=device, dtype=W.dtype)
    V = torch.empty(n, r, device=device, dtype=W.dtype)
    S = torch.empty(r, device=device, dtype=W.dtype)
    R = W.clone()
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(_PEEL_SEED)
    for i in range(r):
        v = torch.sign(torch.randn(n, generator=generator, device=device, dtype=W.dtype))
        v[v == 0] = 1.0
        for _ in range(T):
            u = torch.sign(R @ v)
            u[u == 0] = 1.0
            v = torch.sign(R.T @ u)
            v[v == 0] = 1.0
        alpha = (u @ R @ v) / (m * n)
        U[:, i] = u
        V[:, i] = v
        S[i] = alpha
        R = R - alpha * torch.outer(u, v)
    return U, V, S, R


def refit_alphas(W, U, V, S):
    """Least-squares coefficients for fixed signs: solve A alpha = b with
    A_ij = (u_i^T u_j)(v_i^T v_j), b_i = u_i^T W v_i (plus a 1e-6 relative ridge).
    Returns the new alpha and residual W - U diag(alpha) V^T."""
    r = U.shape[1]
    UTU = U.T.float() @ U.float()
    VTV = V.T.float() @ V.float()
    A = UTU * VTV
    b = torch.diag(U.T.float() @ W.float() @ V.float())
    eps = 1e-6 * A.diagonal().abs().max()
    A = A + eps * torch.eye(r, device=A.device, dtype=A.dtype)
    alpha_new = torch.linalg.solve(A, b)
    R_new = W - U.float() @ torch.diag(alpha_new) @ V.T.float()
    return alpha_new.to(S.dtype), R_new.to(W.dtype)


def decompose(W: torch.Tensor, r: int, T: int = 5):
    """Greedy peeling followed by the joint refit (the paper's solver). W should be fp32
    (and already smoothed if SmoothQuant is used). Returns U, V, alpha, R."""
    U, V, S, _ = binary_peel(W, r=r, T=T)
    S, R = refit_alphas(W, U, V, S)
    return U, V, S, R
