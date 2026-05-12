"""Wikitext-2 perplexity comparison: baseline vs bposit-W8A8.

Compares Qwen2.5-Coder-0.5B-Instruct on a fixed slice of WikiText-2-raw test,
first in bf16/fp16 baseline, then with selected nn.Linear modules replaced
by mosyne_bposit.torch_compat.BPositLinear (W8A8 via INT8 IMMA). Default
swaps FFN only (the whitepaper §4.5 deployment-recommended scope); pass
``--swap-attention --skip-kv-proj`` to also extend to the attention Q/O
projections under the iter-58 narrow-N skip policy.

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


FFN_PROJ_NAMES = (".gate_proj", ".up_proj", ".down_proj")
ATTN_PROJ_NAMES = (".q_proj", ".k_proj", ".v_proj", ".o_proj")


def replace_linears(
    model,
    *,
    swap_ffn: bool = True,
    swap_attention: bool = False,
    skip_kv_proj: bool = False,
):
    """Replace selected nn.Linear modules with BPositLinear. Surface
    matches qwen_generate_bench.replace_linears so users get a coherent
    story across throughput and perplexity benches.

    Default swaps FFN only (whitepaper §4.5 scope: bit-identical bf16
    parity). With ``swap_attention=True`` also extends to attention
    projections; ``skip_kv_proj=True`` is the iter-58 narrow-N policy
    that leaves the GQA K/V projections on bf16 — bposit loses ~3× on
    those at prefill, see
    docs/research/bposit_attn_regression_breakdown_2026-05-09.md.
    """
    from mosyne_bposit.torch_compat import BPositLinear
    import torch.nn as nn

    needles: list[str] = []
    if swap_ffn:
        needles.extend(FFN_PROJ_NAMES)
    if swap_attention:
        needles.extend(ATTN_PROJ_NAMES)
    if not needles:
        return model

    targets = []
    for name, module in model.named_modules():
        for attr, child in module._modules.items():
            full = f"{name}.{attr}" if name else attr
            if (
                isinstance(child, nn.Linear)
                and not isinstance(child, BPositLinear)
                and any(k in full for k in needles)
            ):
                if skip_kv_proj and full.rsplit(".", 1)[-1] in ("k_proj", "v_proj"):
                    continue
                targets.append((module, attr, child, full))

    n_params = 0
    for parent, attr, child, _full in targets:
        n_params += child.weight.numel()
        setattr(parent, attr, BPositLinear.from_linear(child))

    surface = "+".join(
        s for s, on in [("FFN", swap_ffn), ("attention", swap_attention)] if on
    )
    print(f"  replaced {len(targets)} {surface} linears "
          f"({n_params/1e6:.1f}M params, ~{n_params/1e6:.0f} MB at int8)")
    return model


# Back-compat alias for any external callers of the iter-9 name.
def replace_ffn_linears(model):
    return replace_linears(model, swap_ffn=True, swap_attention=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--n-seqs", type=int, default=64)
    ap.add_argument("--ctx-len", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-dtype", choices=["bf16", "fp16"], default="bf16",
                    help="dtype to load the HF model in (default: bf16)")
    ap.add_argument("--swap-ffn", dest="swap_ffn", action="store_true", default=True,
                    help="swap FFN gate/up/down linears to bposit (default on)")
    ap.add_argument("--no-swap-ffn", dest="swap_ffn", action="store_false")
    ap.add_argument("--swap-attention", dest="swap_attention",
                    action="store_true", default=False,
                    help="also swap attention q/k/v/o projections")
    ap.add_argument("--skip-kv-proj", action="store_true", default=False,
                    help="skip .k_proj and .v_proj by name (iter-58 narrow-N "
                         "policy — recommended when --swap-attention is on)")
    args = ap.parse_args()
    if not (args.swap_ffn or args.swap_attention):
        ap.error("must enable at least one of --swap-ffn / --swap-attention")

    print(f"loading dataset wikitext-2-raw-v1 test split...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if t.strip()]
    print(f"  {len(texts)} non-empty paragraphs")

    print(f"loading tokenizer for {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model_dtype = torch.bfloat16 if args.model_dtype == "bf16" else torch.float16
    print(f"\n=== baseline: {args.model_dtype} ===")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=model_dtype, device_map=args.device,
        low_cpu_mem_usage=True,
    )
    print(f"  loaded {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    base_ppl, base_tok, base_dt = evaluate_perplexity(
        model, tokenizer, texts, args.n_seqs, args.ctx_len, args.device,
    )
    print(f"  baseline PPL = {base_ppl:.4f}  ({base_tok} tokens, {base_dt:.1f}s)")

    surface = "+".join(
        s for s, on in [("FFN", args.swap_ffn), ("attention", args.swap_attention)] if on
    )
    if args.skip_kv_proj:
        surface += " (skip_kv_proj)"
    print(f"\n=== bposit W8A8 ({surface}) ===")
    # BPositLinear handles the fp32 cast on each individual weight inside
    # its constructor — no need to cast the whole model to fp32 (which would
    # double its memory). Surrounding modules stay in the model dtype;
    # BPositLinear dispatches on input dtype to native bf16/fp16 paths.
    replace_linears(model,
                    swap_ffn=args.swap_ffn,
                    swap_attention=args.swap_attention,
                    skip_kv_proj=args.skip_kv_proj)
    bp_ppl, bp_tok, bp_dt = evaluate_perplexity(
        model, tokenizer, texts, args.n_seqs, args.ctx_len, args.device,
    )
    print(f"  bposit-W8A8 PPL = {bp_ppl:.4f}  ({bp_tok} tokens, {bp_dt:.1f}s)")

    print("\n=== summary ===")
    print(f"  model       : {args.model}")
    print(f"  dtype       : {args.model_dtype}")
    print(f"  swapped     : {surface}")
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
