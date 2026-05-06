# mosyne-bposit

Bounded-posit (`bposit`) W8A8 matrix multiplication on NVIDIA tensor cores
via INT8 IMMA, with bit-exact reproducibility across runs that IEEE float
on tensor cores cannot deliver.

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

* **2× lighter weight memory** versus FP16 / BF16 — fits more model
  state in the same VRAM.
* **Bit-exact reproducibility across runs.** IEEE float reductions on
  tensor cores are non-deterministic by design (NVIDIA documents this).
  Bposit + 256-bit quire integer accumulation removes the issue.
* **Throughput parity** at training-prefill matmul shapes
  (1.02× BF16 HMMA at $2048^3$ on RTX 5090, 1.07× at $4096^3$ on
  RTX 3090). Loses on small-batch decode where memory bandwidth
  dominates.

See the white paper at
<https://github.com/anomly-llc/mosyne-bposit/tree/main/docs/whitepaper>
for the full set of measurements.

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

## Repo

<https://github.com/anomly-llc/mosyne-bposit> — full source, every
result reproducible from the included single-command runner.

## License

Apache 2.0.
