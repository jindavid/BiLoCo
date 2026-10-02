"""Perplexity on fixed token windows (GPTQ-style)."""
import random

import torch
from tqdm import tqdm


def prepare_eval_tokens(dataset_name, dataset, tokenizer, seqlen: int = 2048, n_samples_c4: int = 256, seed: int = 0):
    """wikitext: all documents joined with blank lines and tokenized once (split into seqlen windows later).
    c4 / fineweb: n_samples_c4 random seqlen-token windows from random documents of at least seqlen tokens."""
    if dataset_name in ("c4", "fineweb"):
        random.seed(seed)
        torch.manual_seed(seed)
        windows, attempts, n = [], 0, len(dataset)
        while len(windows) < n_samples_c4 and attempts < n_samples_c4 * 100:
            attempts += 1
            text = dataset[random.randint(0, n - 1)].get("text", "")
            if not text:
                continue
            tok = tokenizer(text, return_tensors="pt")
            if tok.input_ids.shape[1] < seqlen:
                continue
            start = random.randint(0, tok.input_ids.shape[1] - seqlen)
            windows.append(tok.input_ids[:, start:start + seqlen])
        if len(windows) < n_samples_c4:
            raise RuntimeError(f"only {len(windows)}/{n_samples_c4} documents of >= {seqlen} tokens in {dataset_name}")
        return torch.hstack(windows)
    return tokenizer("\n\n".join(d.get("text", "") for d in dataset), return_tensors="pt").input_ids


@torch.no_grad()
def perplexity(model, eval_tokens, device, seqlen: int = 2048):
    """Mean token NLL over non-overlapping seqlen windows; returns (loss, ppl)."""
    model.eval()
    n = eval_tokens.shape[1] // seqlen
    nlls = []
    for i in tqdm(range(n), desc="PPL"):
        chunk = eval_tokens[:, i * seqlen:(i + 1) * seqlen].to(device)
        nlls.append(model(chunk, labels=chunk).loss.float() * seqlen)
    loss = torch.stack(nlls).sum() / (n * seqlen)
    return loss.item(), torch.exp(loss).item()
