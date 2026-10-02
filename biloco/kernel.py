"""Python side of the fused Blackwell kernel: sign packing, operand layouts, and a launcher.

One launch computes  y = FP4(x) FP4(R)^T + sum_i S_i (x . v_i) u_i  for a decode batch of M <= 32 rows:
the FP4 GEMM runs on tensor cores (128x64 tiles, activation zero-padded to 128 rows) and the binary
low-rank term runs on CUDA cores of the same launch from the bit-packed signs, added in the GEMM epilogue.
Three schedules (modes) exist; `DEFAULT_MODE` is the fastest one per (M, rank) on B300 at K = N = 5120.
"""
import os

import torch

from . import _C  # noqa: F401  (registers torch.ops.biloco)

ops = torch.ops.biloco

RANKS = (16, 32, 64, 128, 256, 512)

# Fastest mode per (M, rank) on B300, K = N = 5120 (paper kernel table).
DEFAULT_MODE = {
    1: {16: 4, 32: 4, 64: 2, 128: 2, 256: 2, 512: 2},
    2: {16: 2, 32: 2, 64: 4, 128: 4, 256: 4, 512: 4},
    4: {16: 2, 32: 2, 64: 4, 128: 4, 256: 4, 512: 4},
    8: {16: 2, 32: 2, 64: 2, 128: 4, 256: 4, 512: 4},
    16: {16: 2, 32: 2, 64: 4, 128: 4, 256: 5, 512: 5},
    32: {16: 2, 32: 2, 64: 2, 128: 5, 256: 5, 512: 5},
}


def pack_signs(signs: torch.Tensor) -> torch.Tensor:
    """[..., n] +-1 -> uint8 [..., n/8]; bit i of byte j is set iff signs[..., 8j + i] >= 0."""
    assert signs.shape[-1] % 8 == 0
    bits = (signs >= 0).to(torch.uint8).reshape(*signs.shape[:-1], signs.shape[-1] // 8, 8)
    weights = 1 << torch.arange(8, device=signs.device, dtype=torch.uint8)
    return (bits * weights).sum(dim=-1).to(torch.uint8)


def pack_u_col32(U_packed: torch.Tensor, n_cols: int, rank: int) -> torch.Tensor:
    """U_packed [R, N/8] -> int32 [N, ceil(R/32)]: word w of column n holds the signs of ranks 32w..32w+31."""
    shifts = torch.arange(8, device=U_packed.device, dtype=torch.int16)
    bits_nr = ((U_packed.to(torch.int16).unsqueeze(-1) >> shifts) & 1).reshape(rank, n_cols).t().contiguous()
    words = (rank + 31) // 32
    out = torch.empty((n_cols, words), device=U_packed.device, dtype=torch.int32)
    for w in range(words):
        r0, r1 = 32 * w, min(rank, 32 * w + 32)
        weights = 1 << torch.arange(r1 - r0, device=U_packed.device, dtype=torch.int64)
        out[:, w] = (bits_nr[:, r0:r1].to(torch.int64) * weights).sum(dim=1).to(torch.int32)
    return out


def pack_v_perm(V_packed: torch.Tensor, K: int = 5120, split: int = 7) -> torch.Tensor:
    """Mode 2 layout of V (K = 5120 only): [R][2 words][128 threads]; thread t owns the K nibbles 128j + t (j < 10),
    word 0 holds nibbles j < 7 at bits 4j, word 1 holds j >= 7 at bits 4(j - 7)."""
    R = V_packed.shape[0]
    assert V_packed.shape[1] * 8 == K == 5120, "mode 2 is built for K = 5120"
    v = V_packed.view(R, K // 8).to(torch.int64)
    nib = torch.stack([v & 15, v >> 4], dim=-1).view(R, K // 4).view(R, 10, 128).permute(0, 2, 1).contiguous()
    w0 = torch.zeros(R, 128, dtype=torch.int64, device=V_packed.device)
    w1 = torch.zeros(R, 128, dtype=torch.int64, device=V_packed.device)
    for j in range(split):
        w0 |= nib[..., j] << (4 * j)
    for j in range(10 - split):
        w1 |= nib[..., split + j] << (4 * j)
    return torch.stack([w0, w1], dim=1).to(torch.int32).contiguous().view(torch.uint8).view(-1).contiguous()


def modes_for(M: int, rank: int, K: int = 5120):
    """Modes with a kernel instantiation for this (M, rank, K). The launch can still reject a shape
    in its host-side checks (e.g. mode 4 at K x N = 5120 x 1024 with rank 256)."""
    out = []
    if K == 5120 and (M <= 16 or (M <= 32 and rank <= 128)):
        out.append(2)
    if (M == 1 and rank in RANKS) or (2 <= M <= 8 and rank >= 64 and not (M > 4 and rank == 64)) or (M == 16 and rank in (64, 128)):
        out.append(4)   # the 8-row r = 64 instantiation is disabled (it produces NaN)
    if 2 <= M <= 64 and rank % 128 == 0:
        out.append(5)
    return out


class BiLoCoGemm:
    """Fused FP4 GEMM + binary low-rank term for one layer at a fixed decode batch M.

    W_fp4 [N, K/2] uint8 and W_sf: NVFP4 weight codes and block scales; alpha: fp32 [1];
    V_packed [R, K/8], U_packed [R, N/8]: packed signs; S: bf16 [R] coefficients.
    Call with the NVFP4 activation zero-padded to 128 rows (A [1, 128, K/2] uint8, rows >= M zero), its block
    scales A_sf, and the BF16 activation X [M, K] (the low-rank term uses the unquantized activation).
    Scratch buffers are tagged with the call's epoch (a launch argument), so consecutive calls need no clearing.
    A CUDA graph bakes the epochs in: call reset() at the start of the captured region so that every replay
    starts from zeroed scratch and epoch 1 (the paper's timing does this; it adds two memsets per graph).
    """

    def __init__(self, W_fp4, W_sf, alpha, V_packed, U_packed, S, M: int, mode: int = None):
        self.N, self.K = W_fp4.shape[0], W_fp4.shape[1] * 2
        self.R, self.M = V_packed.shape[0], M
        self.mode = mode or DEFAULT_MODE.get(M, {}).get(self.R) or modes_for(M, self.R, self.K)[0]
        assert self.mode in modes_for(M, self.R, self.K), f"mode {self.mode} not available for M={M} R={self.R} K={self.K}"
        self.W_fp4, self.W_sf, self.alpha, self.V, self.S = W_fp4, W_sf, alpha, V_packed, S
        dev = W_fp4.device
        self.U_col32 = pack_u_col32(U_packed, n_cols=self.N, rank=self.R)
        self.none = torch.zeros(0, device=dev, dtype=torch.int64)
        R, N = self.R, self.N
        if self.mode == 2:
            self.V_perm, self.U_rw = pack_v_perm(V_packed, K=self.K), self.U_col32.t().contiguous()
            self.z = torch.zeros(8, M, R, 2, device=dev)
            self.flags = torch.zeros(4096, device=dev, dtype=torch.int32)
        elif self.mode == 4:
            rows = 8 * M if M <= 8 else max(8 * M, -(-(80 * (R // 32) * 1024) // (2 * R * M)) * M)
            self.z = torch.zeros(rows, 2 * R, device=dev)
        else:
            Mp = (M + 15) // 16 * 16
            self.z = torch.zeros(2 * Mp, R, device=dev)
            self.corr = torch.zeros(N // 64, Mp, 64, device=dev)
            self.cnt = torch.zeros(1 + N // 64, device=dev, dtype=torch.int32)
        self.epoch = 0

    def reset(self):
        for t in (getattr(self, n, None) for n in ("z", "flags", "cnt")):
            if t is not None:
                t.zero_()
        self.epoch = 0

    def __call__(self, A_pad, A_sf, X_bf16):
        """Returns bf16 [1, 128, N]; rows >= M are padding."""
        self.epoch += 1
        a = (A_pad, self.W_fp4, A_sf, self.W_sf, self.alpha, X_bf16, self.V, self.S, self.U_col32)
        if self.mode == 2:
            return ops.biloco_gemm_mode2(*a, self.z, self.flags, self.none, self.V_perm, self.U_rw, self.M, self.epoch)
        if self.mode == 4:
            return ops.biloco_gemm_mode4(*a, self.z, self.none, self.M, self.epoch)
        return ops.biloco_gemm_mode5(*a, self.z, self.corr, self.cnt, self.M, self.epoch)


def nvfp4_sf_bytes(rows: int, k: int) -> int:
    """Bytes of the NVFP4 block-scale tensor for a [rows, k] operand (rows padded to 128, k/16 padded to 4)."""
    return (rows + 127) // 128 * 128 * ((k // 16 + 3) // 4 * 4)


def reference(A_fp4, A_sf, W_fp4, W_sf, alpha, X_bf16, V_packed, U_packed, S):
    """Unfused reference: z = S * (X sign(V)^T), y_lr = z sign(U) (bf16), then the FP4 GEMM with y_lr added."""
    z = ops.binary_z_reference(X_bf16, V_packed, S).contiguous()
    y_lr = torch.empty(X_bf16.shape[0], U_packed.shape[1] * 8, device=X_bf16.device, dtype=torch.bfloat16)
    ops.binary_expand_reference(z, U_packed, y_lr)
    return ops.fp4_gemm_residual(A_fp4, W_fp4, A_sf, W_sf, alpha, y_lr)


def time_graph(fn, pre=None, iters: int = 30, batch: int = 64):
    """Median time per launch (us) of fn, replaying a CUDA graph of pre() followed by `batch` launches."""
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        if pre:
            pre()
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        if pre:
            pre()
        for _ in range(batch):
            fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for a, b in ev:
        a.record(); g.replay(); b.record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) / batch * 1e3 for a, b in ev)
    return ts[len(ts) // 2]


def dense_bf16_env(enabled: bool):
    """Mode 2 at M = 1, R = 32 runs the dense BF16 rank-32 control while BILOCO_DENSE_BF16 is set."""
    if enabled:
        os.environ["BILOCO_DENSE_BF16"] = "1"
    else:
        os.environ.pop("BILOCO_DENSE_BF16", None)
