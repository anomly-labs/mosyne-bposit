# Attention quantization for bposit-W8A8 — scope

Date: 2026-05-08
Status: scoping only; no code yet.

> **Forward pointer (iter-57, 2026-05-10):** the −13 to −16 % framing
> below was wrong on both ends. Naive whole-attention swap is actually
> −45 %; the right K/V-only-skip pattern (`skip_kv_proj=True`, shipped
> iter-58) reaches −7.1 % attention-only with bit-identical output.
> The per-shape diagnostic that pinned this down lives at
> `bposit_attn_regression_breakdown_2026-05-09.md`. Whitepaper §4.5 has
> been updated to reflect the corrected story. This doc is preserved
> as the iter-30 measurement record.

## Context

The end-to-end Qwen2.5-Coder-1.5B parity result (whitepaper §4.5,
+0.3% vs bf16) replaces only FFN linears with `BPositLinear`. Attention
projections (Q / K / V / O) and `lm_head` stay in bf16. A natural next
extension is quantizing attention too — but it's not a one-line
predicate change; the shapes and bias path differ enough to deserve
its own validation pass.

## Qwen2.5-Coder-1.5B attention shape inventory

Per layer, observed live on the model:

| Module | shape           | bias   | Notes                           |
|--------|-----------------|--------|---------------------------------|
| q_proj | 1536 → 1536     | **yes** | full-head Q projection          |
| k_proj | 1536 → 256      | **yes** | GQA: only 2 KV heads × 128 dim  |
| v_proj | 1536 → 256      | **yes** | GQA                             |
| o_proj | 1536 → 1536     | no     | output projection                |

`hidden_size = 1536`, `num_attention_heads = 12`,
`num_key_value_heads = 2`, `head_dim = 128`. So the model uses
grouped-query attention with 6× KV-head compression.

Counts: 28 layers × 4 attention linears = **112 attention linears**
per generated token, vs **84 FFN linears** today (28 layers × 3
{gate, up, down}). Adding attention more than doubles the number of
bposit calls per token (196 total).

## Three concerns that distinguish this from FFN

1. **Bias path is exercised for the first time.** `BPositLinear` has
   the bias branch (`torch_compat.py` line ~127-130) but the e2e
   parity bench has never hit it — FFN linears are bias-free. The
   bias is added in fp32 after the dequant; for the bf16-native fast
   path the addition needs to land in bf16 too without an
   intermediate cast back to fp32. Verify this works correctly,
   especially for the perplexity/numerics path.

2. **K/V are skinny-N shapes (1536→256).** Per whitepaper §4.1
   (tab:matmul), cuBLASLt's INT8 IMMA heuristic underperforms BF16
   HMMA at decode-with-skinny-K shapes. Skinny-N is likely similar:
   not enough work to amortise the per-call overhead. Per-call decode
   bench at 1×1536×256 will show whether bposit is at parity, faster,
   or slower for K/V projections specifically.

3. **KV cache compatibility.** When K and V projections produce
   bposit-output via `linear_w8a8_dev_*_io`, the result tensor goes
   into the model's KV cache. The cache stores activations across
   tokens within a generation. If the rest of the pipeline expects
   the cache dtype to match the model's compute dtype (bf16), and
   bposit produces bf16 output, this should "just work" — but it's
   the kind of integration assumption worth verifying with a
   determinism test (same prompt × 2, identical output bytes).

## Validation gates before shipping

Before extending `replace_linear_modules` to attention modules:

- **Per-call bench at attention shapes** (1×1536×1536, 1×1536×256,
  1×1536×1536) under `--dtype bf16`. Confirm bposit isn't worse than
  bf16 nn.Linear by more than a few µs per call at decode. If skinny-N
  underperforms, that's the same matmul-level limitation as decode
  FFN-down (whitepaper §6) and a custom CUTLASS template would be the
  fix path; deploying without it would regress overall throughput.
- **Perplexity**: run `wikitext_ppl_bench.py` with attention also
  swapped. Acceptable if PPL gap stays under 1% (current bound,
  AWQ/SmoothQuant band).
- **End-to-end Qwen2.5-Coder-1.5B autoregressive throughput** with
  attention also quantized. The FFN-only number is +0.3% vs bf16; the
  attention-also number could go either way depending on how skinny-N
  IMMA dispatch performs vs HMMA. If it regresses below -1% overall,
  the deployment story would be "FFN-only is the recommended config;
  attention quantization is opt-in."
- **Memory**: weight-memory savings should approximately double for
  attention layers' contribution. Quantify in MB.

## Implementation footprint

Concretely, only `replace_linear_modules` callers change:

```python
def is_ffn_or_attention(name, lin):
    return any(k in name for k in (
        ".gate_proj", ".up_proj", ".down_proj",     # existing FFN
        ".q_proj", ".k_proj", ".v_proj", ".o_proj", # new: attention
    ))
replace_linear_modules(model, predicate=is_ffn_or_attention)
```

`BPositLinear` itself shouldn't need to change — it already supports
bias and arbitrary shapes. The only library-side risk is whether
`linear_w8a8_dev_bf16_io` performs reasonably at attention's shapes
(which the per-call bench would surface).

## Next steps when an iter picks this up

1. Add an attention-only mode to `qwen_generate_bench.py`
   (`--swap-attention`) so the bench can isolate attention-only,
   FFN-only, or both, and compare e2e throughput across the three
   configs.
2. Run perplexity validation with attention swapped on
   Qwen2.5-Coder-1.5B / 0.5B / 3B.
3. Per-call bench at attention shapes (M=1, M=128 prefill).
4. If all three gates clear: extend `transformers_qwen_integration.py`
   to demonstrate the both-FFN-and-attention path. Update whitepaper
   §4.5 footnote noting that the e2e parity result generalises (or
   doesn't) to attention-included swap.

## Outcome — gate 1 measured (2026-05-09)

Step 1 landed in iter-29 (`--swap-ffn` / `--swap-attention` flags).
Iter-30 ran the three configs and the e2e-throughput gate **fails** for
both attention-touching configs:

| Config           | Linears swapped | bposit tok/s | BF16 baseline | gap     | outputs vs bf16 |
|------------------|-----------------|--------------|---------------|---------|-----------------|
| **FFN only**     | 84              | 133.31       | 133.38        | **−0.1%** (parity) | identical |
| FFN + attention  | 196             | 112.08       | 133.07        | −15.8%  | diverge @ char 6 |
| Attention only   | 112             | 115.34       | 132.91        | −13.2%  | diverge @ char 44 |

Same bench harness as the iter-9..iter-13 work
(`packages/mosyne_bposit/examples/qwen_generate_bench.py`),
RTX 3090, Qwen2.5-Coder-1.5B-Instruct, deterministic decode, 128
tokens × 3 trials.

The −13–16% regression confirms the skinny-N hypothesis from §3
above: K/V projections at 1536→256 land squarely in the cuBLASLt-12.8
INT8 IMMA decode-with-skinny-K regime (whitepaper §4.1's `0.27×` /
`0.51×` matmul-ratio rows on the 5090). Attention quantization is
matmul-bound below the wrapper level we already optimised; the
overhead won't close without the work that's queued under "future
directions" in whitepaper §7 — custom CUTLASS INT8 templates tuned
for small-M / skinny-N decode.

The `outputs differ` lines for both attention-touching configs aren't
a separate quality fail — W8A8 quantising 112-196 linears
accumulates the per-block 3.5% L2 error into a deterministic-decoding
divergence. Perplexity validation (gate 2) is the right metric for
quality; throughput gate already gates the deployment decision.

### Deployment recommendation (revised)

FFN-only is the recommended deployed config and the one
whitepaper §4.5 documents at +0.3% parity. Do NOT extend to
attention until CUTLASS small-M templates close the matmul-level
gap. The `--swap-attention` flag exists for future-experiment
reproducibility, not as a production toggle.

Gate 2 (perplexity) and gate 3 (per-call attention-shape bench) are
no longer load-bearing for the deployment decision since gate 1
fails. They remain useful as "is the matmul-level fix actually
moving the needle" instrumentation when CUTLASS templates land.
