# mosyne-bposit

Bounded-posit (`bposit`) W8A8 matrix multiplication on NVIDIA tensor cores
via INT8 IMMA, with bit-exact reproducibility across runs that IEEE float
on tensor cores cannot deliver.

> **For AI model builders** (eval / interpretability / fleet-deployment
> leads): this is the numerics layer of [Mosyne](https://gitlab.com/anomly/mosyne),
> a framework for synthetic-data generation + eval reproducibility on
> commodity hardware. If your eval pipeline needs *"did this model say X
> on date D given input Y"* answerable as a SHA-256 hash check across
> heterogeneous GPU fleets — that's what `BPositLinear` gives you as a
> drop-in `nn.Linear` replacement. If your pre-train data pipeline needs
> difficulty-graded synthetic Q&A with auditable per-datum provenance —
> that's what `mosyne_forge` (private repo) gives you on top. This
> package is the deployable piece you can pip-install today.
>
> Five-minute walkthrough of the wider framework (math / code / chat /
> RAG adapters, HF dataset integration, 24/7 autopilot, SFT export):
> [`docs/MODEL_BUILDER_QUICKSTART.md`](https://gitlab.com/anomly/mosyne/-/blob/main/docs/MODEL_BUILDER_QUICKSTART.md)
> in the parent repo.

## 30-second demo

```bash
pip install mosyne-bposit[torch]
mosyne-bposit-build                          # one-time .so build (needs nvcc 12.x)
mosyne-bposit-demo                           # human output, all four claims
mosyne-bposit-demo --strict --json           # machine-readable, non-zero exit on any regression
```

The `--strict --json` form is the canonical CI-gating one-liner — pipe
the stdout into `jq '.ok'` to assert the headline claims still hold
after any change. Human narration is on stderr in JSON mode so it
doesn't pollute the parseable output.

Output (RTX 3090, FFN-gate shape):

```
[1/4] Throughput at the Llama FFN-gate shape (M=128 K=4096 N=11008)
      BPositLinear (W8A8 via INT8 IMMA) : 363 µs / call
      fp32 nn.Linear                    : 599 µs / call
      strict-mode check (bp < fp32)     : PASS

[2/4] Numerical accuracy on a synthetic W8A8 matmul
      bposit-W8A8 vs fp32 L2 rel. error : 1.22%
      strict-mode check (rel.err < 5%)  : PASS

[3/4] Reproducibility across 5 runs of the same forward pass
      → 5/5 runs produced bit-identical output (sha256 = 1cbf3da9492e1af8)
      strict-mode check (1 distinct hash): PASS

[4/4] Weight memory at the Llama FFN-gate shape (K=4096 N=11008)
      bf16 nn.Linear weight             :   86.00 MB
      BPositLinear weight + scales      :   43.04 MB
      ratio bposit / bf16               : 0.500× (expected ~0.50)
      strict-mode check (ratio < 0.55)  : PASS

  → all 4 claims passed.
```

## Install

```bash
pip install mosyne-bposit                  # PyPI (pure-Python wrapper + .cu source)
mosyne-bposit-build                        # one-time: builds libmosyne_bposit.so via nvcc
```

The build step needs `nvcc` (CUDA 12.x) on PATH and writes the compiled
shared library next to the package.  `pip install mosyne-bposit[torch]`
also pulls in PyTorch for users who want it; the core library doesn't
require it.

### GPU compatibility note (important for Blackwell users)

The default `pip install mosyne-bposit[torch]` pulls a PyTorch
build with support for sm_50 through sm_90 (Volta → Hopper). If
you're on an **RTX 5090 / RTX PRO 6000 / any Blackwell (sm_120)**
card, you need PyTorch ≥ 2.7 with CUDA ≥ 12.8 — install via:

```bash
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch
```

Both `mosyne-bposit-demo` and `mosyne-bposit-probe` detect this
case at startup and print an actionable install hint instead of
the cryptic CUDA error PyTorch produces by default. The
compiled `libmosyne_bposit.so` itself supports both sm_86 and
sm_120 — only the PyTorch wrapper layer needs the version bump.

## Usage

```python
import numpy as np
from mosyne_bposit import linear_w8a8

x = np.random.randn(128, 2048).astype(np.float32)
w = (np.random.randn(2048, 11008) * 0.02).astype(np.float32)

y = linear_w8a8(x, w)         # x @ w via cuBLASLt INT8 IMMA, W8A8 PTQ
```

The pipeline:

1. Per-output-column scale on `w` → INT8 weights
2. Per-token scale on `x` → INT8 input
3. cuBLASLt INT8 IMMA matmul → INT32 accumulator
4. Outer-product dequantisation back to float32

Every step is deterministic. Two runs of the same call return identical
bits, on the same GPU and across different GPUs of the same architecture.

## Why use this

* **Bit-exact reproducibility, by construction.** IEEE float reductions
  on tensor cores are non-deterministic by design (NVIDIA documents
  this). Bposit's compute path is integer-only — INT8 IMMA →
  256-bit fixed-point quire → deterministic round — so there is no
  float reduction order for SM count, occupancy, or scheduling to
  perturb. Five runs of the same forward pass return identical output
  bits, and the property is **structurally deterministic across
  NVIDIA GPUs** rather than empirically lucky. Verified bit-exact on
  RTX 3090 (Ampere) and RTX 5090 (Blackwell); the mechanism should
  hold on Hopper / Ada / Turing but those are still to confirm
  empirically. This is the property current `bf16` / `fp16` / `fp32`
  paths cannot deliver at production tensor-core throughput.

* **Downstream-task accuracy preserved within 1%.** WikiText-2-raw
  perplexity, baseline `bf16` vs FFN-only bposit-W8A8:
  Qwen2.5-Coder-0.5B-Instruct +0.21% over 131K tokens;
  Qwen2.5-Coder-1.5B-Instruct +0.56% over 65K tokens. Squarely within
  the SmoothQuant / AWQ acceptable band, calibration-free. Reproduce
  with `examples/wikitext_ppl_bench.py`.

* **2× lighter weight memory on the modules we replace** versus `fp16` /
  `bf16` (the FFN linears, by default). With attention and `lm_head`
  kept in `bf16` — the default safe configuration — that translates to
  roughly a 30–35% whole-model memory reduction on typical transformer
  shapes. Quantising attention as well closes the model-wide number
  toward 2× (`--swap-attention --skip-kv-proj` in the bench script;
  see "Extending past FFN to attention projections" below).

* **Throughput characterisation (honest).** Measured on RTX 3090 vs
  native PyTorch `nn.Linear`:

  | Shape (M × K × N)            | fp32   | bf16   | bposit | bp/bf16        |
  |------------------------------|--------|--------|--------|----------------|
  | Llama FFN-gate 128×4096×11008| 611 µs | 237 µs | 333 µs | 1.40× slower   |
  | Decode-class   1×4096×4096   |  79 µs |  48 µs |  43 µs | **0.91×** (faster) |
  | Small square   32×2048×2048  |  39 µs |  25 µs |  43 µs | 1.72× slower   |

  Against `bf16` (the realistic deployment baseline), bposit is faster
  at autoregressive decode and slower at prefill / small shapes. With
  the bf16-native hot path (no fp32 buffers in the inner loop), the
  decode-shape advantage compounds: end-to-end Qwen2.5-Coder-1.5B
  token generation is 133.2 tok/s for bposit-W8A8 vs 132.8 tok/s for
  the bf16 baseline (+0.3%, within run-to-run noise). The
  reproducibility + accuracy + memory pitch now lands without any
  throughput trade — bposit matches bf16 end-to-end.

  Against `fp32` (less common deployment baseline), bposit is 1.5–1.9×
  faster on the same shapes — but most production inference runs in
  `bf16` or `fp16`, so the bf16 column is the column that matters.

See the white paper at
<https://github.com/anomly-labs/mosyne-bposit/tree/main/docs/whitepaper>
for the full set of measurements (matmul shape sweep, perplexity table,
reproducibility head-to-head, real Qwen layer). White paper §4.1 has the
authoritative `bposit-via-IMMA` vs `bf16 HMMA` shape sweep on the RTX 5090.

## What this doesn't claim

Inverse of the "Why use this" section — the corners we have *not*
measured, so a careful reader can size the project's actual envelope
without having to read between marketing lines.

* **Not "every model gets 2× memory" out of the box.** The 2× weight
  reduction is per-module on the FFN linears we replace; the default
  configuration keeps attention and `lm_head` in `bf16`, giving roughly
  a 30–35% whole-model memory reduction on typical transformer shapes.
  Closing the gap to a true model-wide 2× requires also quantising
  attention (Q / K / V / O), which is supported via flags but has not
  been characterised at the same rigour as the FFN-only path.

* **Reproducibility is verified on Ampere + Blackwell only.** Empirical
  cross-GPU bit-exactness has been measured on RTX 3090 (sm_86, Ampere)
  and RTX 5090 (sm_120, Blackwell). The structural argument — integer
  arithmetic is associative, so reduction order cannot perturb the
  output — applies to any NVIDIA GPU running the same INT8 IMMA path,
  but we have not yet run the probe on H100 (Hopper), A100 (Ampere
  datacentre), 4090 (Ada), or older Turing/Volta cards. **If you have
  access to one and want to help close this gap empirically**, run
  `mosyne-bposit-probe --json > probe_yourgpu.json` and send us the
  output; the comparison against the 3090/5090 reference is a single
  command (`--compare-with`) and takes about 5 minutes of GPU time.

* **Not faster than `bf16` on every shape.** Across an 11-shape
  deployment sweep on the RTX 5090, bposit-IMMA wins on the
  big compute-bound shapes (2048³, 4096³, attention QK at 1.14–1.33×)
  and ties on most prefill shapes, but loses by up to 4× on small
  decode shapes (decode FFN-down at 0.27×) where the IMMA dequant
  overhead dominates. Geomean across the 11 shapes is 0.76× of native
  BF16 HMMA. The loss is recoverable through a deeper cast-fusion path
  (engineering, not arithmetic), but that work is not shipped yet.

* **No FP8 head-to-head benchmark.** Anomly does not have an H100 or
  H200 in-house. The whitepaper compares against BF16 (today's
  deployment baseline) rigorously; FP8 (E4M3 / E5M2) on Hopper-class
  hardware is a comparison we want to run and would welcome
  collaboration on.

* **Inference only.** This package quantises weights post-training and
  runs the matmul through INT8 IMMA at inference. It is not a training
  quantisation scheme. Posit-aware training is an open research
  direction we are tracking, not a shipped capability.

* **Not a substitute for IEEE float in scientific computing.** The
  bposit-16 precision floor is approximately 3 × 10⁻³ relative error
  at the magnitudes seen in transformer activations — fine for LLM
  inference and within the SmoothQuant / AWQ acceptable band, not fine
  for general scientific compute that expects IEEE-equivalent precision.

* **Does not accelerate non-`nn.Linear` paths.** Scaled-dot-product
  attention, softmax, RoPE, activation functions, `lm_head`, and any
  custom CUDA kernels in a model are unchanged. We replace `nn.Linear`
  modules; everything else runs in its original dtype.

## PyTorch / HuggingFace integration

For users with PyTorch:

```python
from mosyne_bposit.torch_compat import BPositLinear, replace_linear_modules

# Wrap a single layer:
import torch.nn as nn
linear = nn.Linear(2048, 11008)
bposit_linear = BPositLinear.from_linear(linear)
y = bposit_linear(x)            # x @ w via INT8 IMMA, W8A8

# Or replace selected nn.Linear modules across an entire model:
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct",
                                             torch_dtype=torch.bfloat16)
replace_linear_modules(
    model,
    predicate=lambda name, lin: ".mlp." in name and ".layers.10." in name,
)
# Now layer 10's gate/up/down linears run via the bposit-IMMA pipeline.
```

A complete demo (load Qwen2.5-Coder-3B, swap in W8A8 for layers 10–12,
compare logits to the BF16 baseline) lives at
``examples/transformers_qwen_integration.py``.

### Extending past FFN to attention projections

The whitepaper §4.5 deployment recommendation is FFN-only because the
GQA K/V projection (e.g. 1536→256 on Qwen2.5-Coder-1.5B) hits a
narrow-N regime where cuBLASLt INT8 IMMA loses ~3× to bf16. The
`skip_kv_proj=True` policy lets you swap the rest of attention (Q, O)
without that penalty:

```python
replace_linear_modules(model, skip_kv_proj=True)        # whole model
# Or via the bench CLI for end-to-end timing:
# python qwen_generate_bench.py --swap-ffn --swap-attention --skip-kv-proj
```

Verified on Qwen2.5-Coder-1.5B (3090): attention-only with this flag
is -7.1% throughput with bit-identical output to bf16; FFN+attention
combined is -8.1% (the bposit decode-step wins on FFN-down and Q/O
roughly cancel the attention penalty). Per-shape diagnostic at
`docs/research/bposit_attn_regression_breakdown_2026-05-09.md`.

Supported activation dtypes: ``torch.bfloat16``, ``torch.float16``, and
``torch.float32``. ``BPositLinear.forward`` dispatches on input dtype:
``bfloat16`` and ``float16`` route through fully-native hot paths that
keep both quantize and dequantize kernels in the input dtype (no fp32
intermediate buffers in the inner loop). ``float32`` and any other
dtype go through the cast-to-fp32 path.

For the perplexity-on-real-LLM accuracy benchmark cited above, see
``examples/wikitext_ppl_bench.py`` — runs Qwen2.5-Coder against
WikiText-2-raw test, baseline vs FFN-only bposit-W8A8, and prints the
PPL delta.

## Repo

<https://github.com/anomly-labs/mosyne-bposit> — public source +
reproducibility scripts. Every result in the white paper is reproducible
from the included runners.

## Tuning

| Env var | Default | Range | Purpose |
|---|---|---|---|
| `MOSYNE_BPOSIT_WS_MB` | 64 | 16–4096 | cuBLASLt INT8 IMMA workspace size (MB). cuBLASLt's algorithm picker respects this cap; larger workspace gives more algorithm options. The 64 MB default fits 7B–30B model FFN shapes; bump to 256+ if deploying Llama-70B-class shapes (FFN-down at K=28672 N=8192). Values outside the safe range silently fall back to the default. |

Set at process startup:

```bash
MOSYNE_BPOSIT_WS_MB=256 python my_inference.py
```

## Contributing

We are looking for help on five specific things, in priority order
(highest-leverage first):

1. **Cross-GPU bit-exactness verification** on hardware we don't
   have (H100, A100, 4090, L40, T4, V100). Five minutes of GPU
   time and a JSON paste; we credit you in the next release.
2. **FP8 head-to-head benchmark** on H100 / H200 — we don't have
   one in-house.
3. **Verified HuggingFace integration recipes** for Llama-3 /
   Mistral / Gemma / DeepSeek (Qwen is what the whitepaper
   validates against).
4. **Attention quantisation accuracy characterisation** for the
   `--swap-attention --skip-kv-proj` path.
5. **Substrate ports** — RISC-V / Tenstorrent / AMD CDNA /
   Intel Gaudi (research docs in the parent repo's
   `docs/research/`).

Full details, what we're *not* looking for help with, dev setup,
and PR conventions in
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0.
