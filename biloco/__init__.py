"""BiLoCo: a binary (+-1) low-rank component plus an NVFP4 residual, with a fused Blackwell decode kernel.
The kernel lives in biloco.kernel (needs the built extension)."""
from .fp4 import fake_quantize_fp4, set_scale_rules
from .layer import BiLoCoLinear
from .model import load_quantized, quantize_model, replace_linears, save_quantized
from .peel import binary_peel, decompose, refit_alphas, set_peel_seed
