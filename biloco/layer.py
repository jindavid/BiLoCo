"""BiLoCoLinear: y = (x/s) V_scaled^T U_T + FP4(x/s) FP4(R)^T + b, with SmoothQuant scale s.

V_scaled = diag(alpha) V^T and U_T = U^T hold the +-1 signs (alpha folded into V). With method "fp4"
(the NVFP4-only baseline) the low-rank term is absent.
This is the fake-quant accuracy path; the fused Blackwell kernel is in biloco.kernel.
"""
import torch
import torch.nn as nn

from . import fp4
from .peel import decompose

METHODS = ("biloco", "fp4")


class BiLoCoLinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, method: str = "biloco", smooth_factor: float = 0.5):
        super().__init__()
        assert method in METHODS, method
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.method = method
        self.rank = rank if method != "fp4" else 0
        self.smooth_factor = smooth_factor
        dev = linear.weight.device
        kw = dict(device=dev, dtype=torch.bfloat16)
        # The original weight is consumed by quantize() and then freed.
        self.weight = nn.Parameter(linear.weight.data.to(torch.bfloat16), requires_grad=False)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.data.to(torch.bfloat16), requires_grad=False)
        self.register_buffer("V_scaled", torch.zeros(self.rank, self.in_features, **kw))
        self.register_buffer("U_T", torch.zeros(self.rank, self.out_features, **kw))
        self.register_buffer("smooth_scale", torch.ones(self.in_features, **kw))
        self.register_buffer("weight_fp4", torch.empty(0, **kw))   # FP4(R) as bf16, filled by quantize()

    @torch.no_grad()
    def quantize(self, act_col_max: torch.Tensor = None, T: int = 5, device=None):
        """Smooth, split off the low-rank component, and fake-quantize the residual to NVFP4."""
        device = device or self.weight.device
        W = self.weight.data.to(device=device, dtype=torch.float32)          # (out, in)
        if act_col_max is not None:
            a = act_col_max.to(device=device, dtype=torch.float32).clamp(min=1e-6)
            w_col_max = W.abs().max(dim=0).values.clamp(min=1e-6)
            s = (a ** self.smooth_factor) / (w_col_max ** (1 - self.smooth_factor))
            self.smooth_scale.copy_(s.to(torch.bfloat16))
            W = W * s
        if self.rank > 0:
            U, V, S, R = decompose(W, r=self.rank, T=T)
            self.V_scaled.copy_((V * S.unsqueeze(0)).T.contiguous().to(torch.bfloat16))
            self.U_T.copy_(U.T.contiguous().to(torch.bfloat16))
        else:
            R = W
        self.weight_fp4 = fp4.fake_quantize_fp4(R.to(torch.bfloat16), config=fp4.CONFIG_W).to(torch.bfloat16).contiguous()
        self.weight = None

    def forward(self, x):
        x_s = x.to(torch.bfloat16) / self.smooth_scale
        y = torch.zeros((*x_s.shape[:-1], self.out_features), dtype=torch.bfloat16, device=x_s.device)
        if self.rank > 0:
            # Low-rank term on the unquantized (smoothed) BF16 activation.
            y = y + (x_s @ self.V_scaled.T) @ self.U_T
        x_fq = fp4.fake_quantize_fp4(x_s, config=fp4.CONFIG_X)
        y = y + x_fq @ self.weight_fp4.T
        if self.bias is not None:
            y = y + self.bias
        return y.to(torch.bfloat16)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, method={self.method}, rank={self.rank}"
