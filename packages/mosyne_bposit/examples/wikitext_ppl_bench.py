"""Wikitext-2 perplexity comparison: baseline vs bposit-W8A8 (FFN linears).

Compares Qwen2.5-Coder-0.5B-Instruct on a fixed slice of WikiText-2-raw test,
first in bf16 baseline, then with the FFN gate/up/down nn.Linear modules
replaced by mosyne_bposit.torch_compat.BPositLinear (W8A8 via INT8 IMMA).

This is the downstream-task accuracy claim the whitepaper currently lacks.
"""
from __future__ import annotations
import argparse
import math
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_perplexity(model, tokenizer, texts, max_seqs, ctx_len, device):
    model.eval()
    encoded = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids[0]
    n_tok = encoded.shape[0]
    n_seqs = min(max_seqs, n_tok // ctx_len)
    print(f"  evaluating {n_seqs} sequences × {ctx_len} tokens "
          f"({n_seqs * ctx_len:,} tokens)")

    nll_sum, tok_sum = 0.0, 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(n_seqs):
            input_ids = encoded[i * ctx_len:(i + 1) * ctx_len].unsqueeze(0).to(device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            n = input_ids.shape[1] - 1  # next-token loss masks out 1
            nll_sum += outputs.loss.item() * n
            tok_sum += n
            if (i + 1) % 10 == 0:
                avg_nll = nll_sum / tok_sum
                ppl = math.exp(avg_nll)
                elapsed = time.perf_counter() - t0
                print(f"    [{i+1}/{n_seqs}] ppl={ppl:.4f} "
                      f"({tok_sum/elapsed:.0f} tok/s)")
    avg_nll = nll_sum / tok_sum
    return math.exp(avg_nll), tok_sum, time.perf_counter() - t0


def replace_ffn_linears(model):
    """Convert just the FFN gate_proj/up_proj/down_proj nn.Linear modules to
    BPositLinear. Leave attention (q/k/v/o) and lm_head in fp32 for now —
    those have known issues with naïve W8A8 (KV cache, low-rank projections).
    Reports the count + total params replaced."""
    from mosyne_bposit.torch_compat import BPositLinear
    import torch.nn as nn

    targets = []
    for name, module in model.named_modules():
        for attr, child in module._modules.items():
            full = f"{name}.{attr}" if name else attr
            if (
                isinstance(child, nn.Linear)
                and not isinstance(child, BPositLinear)
                and any(k in full for k in (".gate_proj", ".up_proj", ".down_proj"))
            ):
                targets.append((module, attr, child, full))
    n_params = 0
    for parent, attr, child, full in targets:
        n_params += child.weight.numel()
        new_mod = BPositLinear.from_linear(child)
        setattr(parent, attr, new_mod)
    print(f"  replaced {len(targets)} FFN linears "
          f"({n_params/1e6:.1f}M params, ~{n_params/1e6 * 1:.0f} MB at int8)")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--n-seqs", type=int, default=64)
    ap.add_argument("--ctx-len", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print(f"loading dataset wikitext-2-raw-v1 test split...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if t.strip()]
    print(f"  {len(texts)} non-empty paragraphs")

    print(f"loading tokenizer for {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("\n=== baseline: bf16 ===")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device,
        low_cpu_mem_usage=True,
    )
    print(f"  loaded {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    base_ppl, base_tok, base_dt = evaluate_perplexity(
        model, tokenizer, texts, args.n_seqs, args.ctx_len, args.device,
    )
    print(f"  baseline PPL = {base_ppl:.4f}  ({base_tok} tokens, {base_dt:.1f}s)")

    print("\n=== bposit W8A8 (FFN only) ===")
    # BPositLinear handles the fp32 cast on each individual weight inside
    # its constructor — no need to cast the whole model to fp32 (which would
    # double its memory). Surrounding modules stay in bf16; BPositLinear
    # casts inputs to fp32 inside forward and outputs back to x.dtype.
    replace_ffn_linears(model)
    bp_ppl, bp_tok, bp_dt = evaluate_perplexity(
        model, tokenizer, texts, args.n_seqs, args.ctx_len, args.device,
    )
    print(f"  bposit-W8A8 PPL = {bp_ppl:.4f}  ({bp_tok} tokens, {bp_dt:.1f}s)")

    print("\n=== summary ===")
    print(f"  model       : {args.model}")
    print(f"  context     : {args.n_seqs} × {args.ctx_len} tokens")
    print(f"  baseline    : PPL={base_ppl:.4f}")
    print(f"  bposit-W8A8 : PPL={bp_ppl:.4f}  (Δ={bp_ppl-base_ppl:+.4f}, "
          f"{(bp_ppl/base_ppl-1)*100:+.2f}%)")
    if bp_ppl < base_ppl * 1.05:
        print(f"  → W8A8 preserves perplexity within 5% — strong claim")
    elif bp_ppl < base_ppl * 1.20:
        print(f"  → W8A8 within 20% — usable; calibration would tighten")
    else:
        print(f"  → W8A8 degrades >20% — needs investigation")


if __name__ == "__main__":
    sys.exit(main())
