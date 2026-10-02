"""NVFP4 fake quantization (16-element blocks, E4M3 block scales) via fouroversix.

The paper uses the "four over six" per-block MSE scale search (`mse`) for weights and activations.
Activations are quantized as one [tokens, K] tensor, so the global scale is shared by the batch.
"""
import torch
from fouroversix import quantize_to_fp4 as _quantize
from fouroversix.quantize import dequantize as _dequantize
from fouroversix.quantize.config import QuantizationConfig
from fouroversix.utils import ScaleRule

SCALE_RULES = {
    "static_4": ScaleRule.static_4,
    "static_6": ScaleRule.static_6,
    "abs_max": ScaleRule.abs_max,
    "mse": ScaleRule.mse,
    "mae": ScaleRule.mae,
}

CONFIG_W = QuantizationConfig(scale_rule=ScaleRule.mse)
CONFIG_X = QuantizationConfig(scale_rule=ScaleRule.mse)


def set_scale_rules(weight_rule: str = "mse", act_rule: str = "mse"):
    global CONFIG_W, CONFIG_X
    CONFIG_W = QuantizationConfig(scale_rule=SCALE_RULES[weight_rule])
    CONFIG_X = QuantizationConfig(scale_rule=SCALE_RULES[act_rule])


def fake_quantize_fp4(x: torch.Tensor, config=None) -> torch.Tensor:
    """Quantize to NVFP4 and dequantize back (same dtype and shape)."""
    config = CONFIG_W if config is None else config
    shape = x.shape
    qt = _quantize(x.reshape(-1, x.shape[-1]), config=config)
    return _dequantize(qt).reshape(shape)
