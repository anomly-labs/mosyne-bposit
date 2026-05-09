"""End-to-end token generation throughput: baseline bf16 vs bposit-W8A8.

Measures tokens/sec on real Qwen2.5-Coder generation, the metric customers
deploying autoregressive inference actually buy on. Same model, same
prompt, deterministic decoding (do_sample=False), so the only difference
between runs is the FFN matmul path.

Order matters here: we load the baseline first, measure, then swap FFN
linears in place to BPositLinear (freeing the originals). We can't easily
flip back without reloading.
"""
from __future__ import annotations
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FFN_PROJ_NAMES = (".gate_proj", ".up_proj", ".down_proj")
ATTN_PROJ_NAMES = (".q_proj", ".k_proj", ".v_proj", ".o_proj")


def replace_linears(model, swap_ffn: bool = True, swap_attention: bool = False):
    """Replace nn.Linear modules with BPositLinear, gated by which sublayers
    to target. ``swap_ffn`` covers gate / up / down projections;
    ``swap_attention`` covers q / k / v / o projections. Pass both to swap
    every linear in the transformer block (everything except embeddings,
    layer norms, and the lm_head)."""
    from mosyne_bposit.torch_compat import BPositLinear
    import torch.nn as nn
    needles = []
    if swap_ffn:
        needles.extend(FFN_PROJ_NAMES)
    if swap_attention:
        needles.extend(ATTN_PROJ_NAMES)
    if not needles:
        return 0
    targets = []
    for name, module in model.named_modules():
        for attr, child in module._modules.items():
            full = f"{name}.{attr}" if name else attr
            if (
                isinstance(child, nn.Linear)
                and not isinstance(child, BPositLinear)
                and any(k in full for k in needles)
            ):
                targets.append((module, attr, child, full))
    for parent, attr, child, full in targets:
        setattr(parent, attr, BPositLinear.from_linear(child))
    return len(targets)


# Back-compat alias for any external callers of the old name.
def replace_ffn_linears(model):
    return replace_linears(model, swap_ffn=True, swap_attention=False)


def time_generation(model, tokenizer, prompt, n_new_tokens, n_trials, warmup):
    """Returns (mean_tok_per_sec, generated_text)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Warmup so we're not measuring CUDA kernel compilation.
    for _ in range(warmup):
        _ = model.generate(
            **inputs,
            max_new_tokens=n_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    torch.cuda.synchronize()

    times = []
    text = None
    for i in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            **inputs,
            max_new_tokens=n_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        if text is None:
            text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                                    skip_special_tokens=True)

    mean_dt = sum(times) / len(times)
    tok_per_sec = n_new_tokens / mean_dt
    return tok_per_sec, mean_dt, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--n-new-tokens", type=int, default=128)
    ap.add_argument("--n-trials", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-dtype", choices=["bf16", "fp16"], default="bf16",
                    help="dtype to load the HF model in (default: bf16, the realistic deployment dtype)")
    ap.add_argument("--swap-ffn", dest="swap_ffn", action="store_true", default=True,
                    help="swap FFN gate/up/down linears to BPositLinear (default: on)")
    ap.add_argument("--no-swap-ffn", dest="swap_ffn", action="store_false",
                    help="disable FFN swap (use with --swap-attention to bench attention alone)")
    ap.add_argument("--swap-attention", dest="swap_attention", action="store_true",
                    default=False,
                    help="also swap attention q/k/v/o projections to BPositLinear")
    args = ap.parse_args()
    if not (args.swap_ffn or args.swap_attention):
        ap.error("must enable at least one of --swap-ffn / --swap-attention")

    prompt = (
        "Write a Python function that takes a list of integers and returns "
        "the second-largest value, handling edge cases like empty lists and "
        "lists with duplicates. Include type hints and a docstring."
    )

    model_dtype = torch.bfloat16 if args.model_dtype == "bf16" else torch.float16
    print(f"loading {args.model} ({args.model_dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=model_dtype, device_map=args.device,
        low_cpu_mem_usage=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.0f}M params")

    print(f"\n=== baseline {args.model_dtype}: {args.n_new_tokens} new tokens × "
          f"{args.n_trials} trials ===")
    base_tps, base_dt, base_text = time_generation(
        model, tokenizer, prompt,
        args.n_new_tokens, args.n_trials, args.warmup,
    )
    print(f"  {base_tps:.2f} tok/s  ({base_dt*1000:.1f} ms / generation)")

    swap_label = "+".join(
        s for s, on in [("FFN", args.swap_ffn), ("attention", args.swap_attention)] if on
    )
    print(f"\n=== swapping {swap_label} linears to BPositLinear ===")
    n_replaced = replace_linears(
        model, swap_ffn=args.swap_ffn, swap_attention=args.swap_attention
    )
    print(f"  replaced {n_replaced} {swap_label} linears")

    print(f"\n=== bposit-W8A8 ({swap_label}): {args.n_new_tokens} new tokens × "
          f"{args.n_trials} trials ===")
    bp_tps, bp_dt, bp_text = time_generation(
        model, tokenizer, prompt,
        args.n_new_tokens, args.n_trials, args.warmup,
    )
    print(f"  {bp_tps:.2f} tok/s  ({bp_dt*1000:.1f} ms / generation)")

    print("\n=== summary ===")
    print(f"  model         : {args.model}")
    print(f"  swapped       : {swap_label} ({n_replaced} linears)")
    print(f"  generation    : {args.n_new_tokens} tokens, "
          f"deterministic decoding")
    print(f"  baseline {args.model_dtype:5s}: {base_tps:6.2f} tok/s ({base_dt*1000:6.1f} ms)")
    print(f"  bposit-W8A8   : {bp_tps:6.2f} tok/s ({bp_dt*1000:6.1f} ms)")
    delta_pct = (bp_tps / base_tps - 1) * 100
    print(f"  Δ throughput  : {delta_pct:+.1f}% (bposit vs baseline)")

    if base_text == bp_text:
        print("  outputs match : ✓ identical text generated")
    else:
        # Find first diverging position so we can print a snippet.
        i = 0
        while i < min(len(base_text), len(bp_text)) and base_text[i] == bp_text[i]:
            i += 1
        print(f"  outputs differ : first divergence at char {i}")
        print(f"    baseline[:50] : {base_text[:50]!r}")
        print(f"    bposit  [:50] : {bp_text[:50]!r}")


if __name__ == "__main__":
    main()
