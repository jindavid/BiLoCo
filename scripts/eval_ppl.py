"""Quantize a model and report perplexity on WikiText-2 and C4 (paper Table 2).

    python scripts/eval_ppl.py --model_name Qwen/Qwen3-4B --method biloco   # BiLoCo, binary rank 512 + NVFP4 residual
    python scripts/eval_ppl.py --model_name Qwen/Qwen3-4B --method fp4      # NVFP4-only baseline
    python scripts/eval_ppl.py --model_name Qwen/Qwen3-4B --method bf16     # BF16 reference, no quantization

Every nn.Linear (including lm_head) is replaced. Defaults follow the paper: SmoothQuant alpha = 0.5 from
C4-train calibration, T = 5 sign-alternation steps per atom, the MSE ("four over six") NVFP4 scale rule for
weights and activations, and the joint coefficient refit.
"""
import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import biloco  # noqa: E402
from biloco.data import collate_text, load_text_dataset  # noqa: E402
from biloco.evaluate import perplexity, prepare_eval_tokens  # noqa: E402

torch.backends.cuda.enable_cudnn_sdp(False)


def load_bf16(name, device, eager=False):
    kw = dict(attn_implementation="eager") if eager else {}
    m = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, dtype=torch.bfloat16,
                                             device_map={"": device}, **kw)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True)
    p.add_argument("--method", default="biloco", choices=["biloco", "fp4", "bf16"],
                   help="biloco: binary rank-r + NVFP4 residual; fp4: NVFP4 only; bf16: no quantization")
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--peel_seed", type=int, default=42, help="sign-initialization seed")
    p.add_argument("--peel_iters", type=int, default=5, help="sign-alternation steps per atom (T)")
    p.add_argument("--smooth_factor", type=float, default=0.5, help="SmoothQuant alpha")
    p.add_argument("--scale_rule", default="mse", choices=list(biloco.fp4.SCALE_RULES),
                   help="NVFP4 block-scale rule for weights and activations (mse = four over six)")
    p.add_argument("--calib_dataset", default="c4_train")
    p.add_argument("--calib_samples", type=int, default=128, help="documents loaded for calibration")
    p.add_argument("--calib_batches", type=int, default=4, help="batches of --calib_batch_size documents actually run")
    p.add_argument("--calib_batch_size", type=int, default=2)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--ref_device", type=int, default=0, help="GPU for the BF16 reference copy used for calibration")
    p.add_argument("--dataset", nargs="+", default=["wikitext", "c4"], choices=["wikitext", "c4", "fineweb"])
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--c4_windows", type=int, default=256, help="random windows for c4 / fineweb")
    p.add_argument("--output", default=None, help="write results as JSON")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token        # pad positions enter the calibration max and the activation scale
    biloco.set_scale_rules(args.scale_rule, args.scale_rule)
    biloco.set_peel_seed(args.peel_seed)

    t0 = time.time()
    if args.method == "bf16":
        model = load_bf16(args.model_name, 0, eager=True)
    else:
        model = load_bf16(args.model_name, 0)
        ref = load_bf16(args.model_name, args.ref_device, eager=True)
        calib = DataLoader(load_text_dataset(args.calib_dataset, max_samples=args.calib_samples),
                           batch_size=args.calib_batch_size, shuffle=False, collate_fn=collate_text)
        biloco.quantize_model(model, ref_model=ref, tokenizer=tok, dataloader=calib, rank=args.rank,
                              method=args.method, T=args.peel_iters, smooth_factor=args.smooth_factor,
                              calib_batches=args.calib_batches, max_length=args.max_length,
                              device=torch.device("cuda:0"))
        del ref; gc.collect(); torch.cuda.empty_cache()
    model.eval()
    out = {"args": vars(args), "ptq_time_s": time.time() - t0, "ppl": {}}

    for ds_name in args.dataset:
        n_docs = 0 if ds_name == "wikitext" else 4 * args.c4_windows
        tokens = prepare_eval_tokens(ds_name, load_text_dataset(ds_name, max_samples=n_docs), tok,
                                     seqlen=args.seqlen, n_samples_c4=args.c4_windows)
        _, ppl = perplexity(model, tokens, torch.device("cuda:0"), seqlen=args.seqlen)
        out["ppl"][ds_name] = ppl
        tag = f"r={args.rank} seed={args.peel_seed}" if args.method == "biloco" else ""
        print(f"{args.model_name} {args.method} {tag} {ds_name}: PPL {ppl:.4f}", flush=True)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
