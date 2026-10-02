"""Whole-model PTQ: replace every nn.Linear (including lm_head) with BiLoCoLinear, collect SmoothQuant
activation statistics on a BF16 reference copy of the model, and quantize layer by layer."""
import gc

import torch
import torch.nn as nn

from .layer import BiLoCoLinear


def replace_linears(model: nn.Module, rank: int, method: str = "biloco", smooth_factor: float = 0.5):
    """Swap every nn.Linear for a BiLoCoLinear in place. Returns {name: module}."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            setattr(model, name, BiLoCoLinear(child, rank=rank, method=method, smooth_factor=smooth_factor))
        else:
            replace_linears(child, rank=rank, method=method, smooth_factor=smooth_factor)
    return {n: m for n, m in model.named_modules() if isinstance(m, BiLoCoLinear)}


@torch.no_grad()
def collect_act_col_max(ref_model, names, tokenizer, dataloader, calib_batches: int = 4, max_length: int = 2048):
    """Per-input-channel max |x| of every named linear of ref_model over the first calib_batches batches."""
    col_max = {}
    hooks = []
    for name, mod in ref_model.named_modules():
        if isinstance(mod, nn.Linear) and name in names:
            def hook(_m, inp, _out, name=name):
                x = inp[0] if isinstance(inp, tuple) else inp
                mx = x.reshape(-1, x.shape[-1]).float().abs().max(dim=0).values.cpu()
                col_max[name] = mx if name not in col_max else torch.max(col_max[name], mx)
            hooks.append(mod.register_forward_hook(hook))
    dev = next(ref_model.parameters()).device
    for i, batch in enumerate(dataloader):
        if i >= calib_batches:
            break
        enc = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length).to(dev)
        ref_model(**enc)
    for h in hooks:
        h.remove()
    return col_max


@torch.no_grad()
def quantize_model(model, ref_model=None, tokenizer=None, dataloader=None, rank: int = 512, method: str = "biloco",
                   T: int = 5, smooth_factor: float = 0.5, calib_batches: int = 4, max_length: int = 2048, device=None):
    """PTQ of model in place. Smoothing statistics come from ref_model (an unmodified BF16 copy) on the
    calibration dataloader; pass ref_model=None to skip smoothing."""
    device = device or torch.device("cuda")
    layers = replace_linears(model, rank=rank, method=method, smooth_factor=smooth_factor)
    col_max = {}
    if ref_model is not None and dataloader is not None:
        col_max = collect_act_col_max(ref_model, set(layers), tokenizer, dataloader, calib_batches, max_length)
    for i, (name, m) in enumerate(layers.items()):
        m.quantize(act_col_max=col_max.get(name), T=T, device=device)
        if (i + 1) % 50 == 0:
            gc.collect(); torch.cuda.empty_cache()
    gc.collect(); torch.cuda.empty_cache()
    return model


def save_quantized(model, path: str, manifest: dict = None):
    """Save the quantized tensors of every BiLoCoLinear (the rest of the model is the original checkpoint)."""
    layers = {n: {k: v for k, v in m.state_dict().items() if v.numel() > 0}
              for n, m in model.named_modules() if isinstance(m, BiLoCoLinear)}
    torch.save({"layers": layers, "manifest": manifest or {}}, path)


@torch.no_grad()
def load_quantized(model, path: str, rank: int, method: str = "biloco"):
    """Replace the linears of a freshly loaded BF16 model and restore the saved quantized tensors."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    layers = replace_linears(model, rank=rank, method=method)
    for name, m in layers.items():
        st = blob["layers"][name]
        dev = m.V_scaled.device
        for k in ("V_scaled", "U_T", "smooth_scale"):
            getattr(m, k).copy_(st[k])
        m.weight_fp4 = st["weight_fp4"].to(dev)
        if m.bias is not None:
            m.bias.copy_(st["bias"])
        m.weight = None
    return blob["manifest"]
