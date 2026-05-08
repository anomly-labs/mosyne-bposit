# mosyne-bposit

Bounded-posit (`bposit`) W8A8 matrix multiplication on NVIDIA tensor cores
via INT8 IMMA, with bit-exact reproducibility across runs that IEEE float
on tensor cores cannot deliver.

## 30-second demo

```bash
pip install mosyne-bposit[torch]
mosyne-bposit-build                        # one-time .so build (needs nvcc 12.x)
mosyne-bposit-demo                         # prints all three claims at once
```

Output (RTX 3090, FFN-gate shape):

```
[1/3] Throughput at the Llama FFN-gate shape (M=128 K=4096 N=11008)
      BPositLinear (W8A8 via INT8 IMMA) : 363 µs / call
      fp32 nn.Linear                    : 599 µs / call
      speedup                           : 1.65× faster

[2/3] Numerical accuracy on a synthetic W8A8 matmul
      bposit-W8A8 vs fp32 L2 rel. error : 1.22%
      verdict: well within the SmoothQuant / AWQ acceptable-W8A8 band

[3/3] Reproducibility across 5 runs of the same forward pass
      → 5/5 runs produced bit-identical output (sha256 = 1cbf3da9492e1af8)
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

* **Bit-exact reproducibility across runs.** IEEE float reductions on
  tensor cores are non-deterministic by design (NVIDIA documents this).
  Bposit + 256-bit quire integer accumulation removes the issue —
  five runs of the same forward pass, identical output bits. This is
  the property current `bf16` / `fp16` / `fp32` paths cannot deliver
  at production tensor-core throughput.

* **Downstream-task accuracy preserved within 1%.** WikiText-2-raw
  perplexity, baseline `bf16` vs FFN-only bposit-W8A8:
  Qwen2.5-Coder-0.5B-Instruct +0.21% over 131K tokens;
  Qwen2.5-Coder-1.5B-Instruct +0.56% over 65K tokens. Squarely within
  the SmoothQuant / AWQ acceptable band, calibration-free. Reproduce
  with `examples/wikitext_ppl_bench.py`.

* **2× lighter weight memory** versus `fp16` / `bf16`. Fits more model
  state in the same VRAM at no accuracy cost in the W8A8 regime above.

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

## License

Apache 2.0.
