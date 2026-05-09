# mosyne-bposit

**Bounded posits at IEEE-tensor-core throughput on commodity NVIDIA Blackwell, with bit-exact reproducibility.**

This repository accompanies the white paper *Bounded Posits at IEEE
Tensor-Core Throughput on Commodity NVIDIA Blackwell, with Bit-Exact
Reproducibility* (`docs/whitepaper/`). It contains:

- A complete `bposit16` ALU running as CUDA `__device__` functions
  (`add`, `mul`, `neg`, `recip`, `sqrt`, `log2`, `exp2`, plus a
  `quire256` exact accumulator and the keystone `quire256 → bposit16`
  encoder).
- The Path C pipeline that routes posit arithmetic through the existing
  INT8 IMMA tensor cores (`bposit8 → INT8 → IMMA → INT32 → quire256 →
  bposit32`).
- Multi-trial throughput benches across 11 transformer-class matmul
  shapes, with mean ± stddev reporting and a cuBLASLt algorithm sweep.
- A bit-exact reproducibility head-to-head between IEEE FP32 with
  `atomicAdd` and bposit + quire (the headline result of the paper).
- A real `Qwen2.5-Coder-3B` feed-forward layer running through the
  pipeline with W8A8 post-training quantisation, plus the W8A8 PTQ
  progression (per-tensor → per-channel → per-token).
- A pip-installable Python package (`mosyne-bposit`) wrapping the
  pipeline as a clean ctypes API with optional PyTorch / HuggingFace
  integration.

## Headline numbers (RTX 5090, mean of 8 trials × 30 reps each)

| Bench | BF16 HMMA | bposit-IMMA | ratio |
|---|---:|---:|---:|
| 2048³ matmul | 167.1 ± 0.5 TFLOPS | **167.8 ± 0.3 TPOPS** | 1.00× |
| 4096³ matmul | 173.1 ± 0.1 TFLOPS | **197.4 ± 0.3 TPOPS** | **1.14×** |
| Prefill FFN gate (2048×14336×4096) | 200.6 ± 0.4 TFLOPS | **217.8 ± 0.1 TPOPS** | **1.09×** |
| Attention QK (2048×2048×128) | 65.5 TFLOPS | **87.2 TPOPS** | **1.33×** |

| Reproducibility (5 runs, identical input) | distinct bit patterns |
|---|---:|
| IEEE FP32 with `atomicAdd` | **5** |
| bposit16 + quire256 | **1** |

See `docs/whitepaper/mosyne_bposit_whitepaper.tex` for the full set of
measurements, methodology, error analysis, and discussion.

## 60-second reproducibility demo

If you're here to verify the bit-exact-reproducibility claim — the
property that drives the eval / interpretability / regulatory-audit
use case — the fastest path is:

```bash
cd scripts/repro_bench/min_demo
make demo
```

This builds `kernels/test_determinism_headline.cu` for `sm_86 + sm_120`,
runs it on every NVIDIA GPU on your host, and prints a comparison table.
On a host with both an RTX 3090 (Ampere) and RTX 5090 (Blackwell), the
output reads:

```
GPU  Name                       bposit (5 runs)     FP32 atomicAdd (5 runs, unique)
---  ----                       ---------------     -------------------------------
0    NVIDIA GeForce RTX 5090    bposit16=0x7627     bits=0x4a93ec2e … 0x4a93ec70  (5 unique)
1    NVIDIA GeForce RTX 3090    bposit16=0x7627     bits=0x4a93ec40 … 0x4a93ec4a  (5 unique)

✓ bposit produced the SAME hash on all 2 GPU(s) —
  bit-exact reproducibility across hardware generations.
  IEEE FP32 atomicAdd produced 9 distinct hashes across
  10 runs total (5 per GPU × 2 GPUs).
```

No PyTorch, no model download, no Python beyond what `nvcc` itself
needs — just the synthetic 65 K-element log-uniform sum from
whitepaper §4.2 turned into a turnkey artifact. Full design notes:
`scripts/repro_bench/min_demo/README.md`.

## Quick start

```bash
git clone https://github.com/anomly-labs/mosyne-bposit.git
cd mosyne-bposit

# 1. Build & validate the bposit ALU (about 30 s on a 5090 / 3090)
cd kernels
python3 build_bposit_luts.py    # bakes the ~9.6 MB device-const header
bash run_public_tests.sh        # validates the 8 ALU primitives + 4 demos
bash bench_robust_sweep.sh      # multi-trial matmul sweep across 11 shapes

# 2. Reproduce the determinism head-to-head
nvcc -O3 -std=c++17 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_120,code=sm_120 \
  -lcublasLt -lcudart \
  test_determinism_headline.cu -o test_determinism
./test_determinism              # → "5 distinct" vs "1 distinct"

# 3. Use the pip-installable package
cd ../packages/mosyne_bposit
pip install -e .                # pure-Python wrapper + .cu source
mosyne-bposit-build             # one-time, requires nvcc on PATH
python -c "
import numpy as np
from mosyne_bposit import linear_w8a8
x = np.random.randn(128, 2048).astype(np.float32)
w = (np.random.randn(2048, 11008) * 0.02).astype(np.float32)
y = linear_w8a8(x, w)            # x @ w via cuBLASLt INT8 IMMA, W8A8
print(y.shape, y.std())
"

# 4. (optional) Run the real Qwen2.5 FFN bench
#    First download the checkpoint (Hugging Face Qwen/Qwen2.5-Coder-3B-Instruct);
#    then extract one layer's FFN weights to /tmp/qwen_layer_weights/:
python3 kernels/extract_qwen_layer.py \
    --model ~/path/to/Qwen2.5-Coder-3B-Instruct/snapshots/SHA \
    --layer 10 --out /tmp/qwen_layer_weights
nvcc -O3 -std=c++17 -gencode arch=compute_120,code=sm_120 -lcublasLt -lcudart \
    kernels/test_ffn_layer_real_qwen.cu -o /tmp/test_ffn_qwen
/tmp/test_ffn_qwen
```

## Repository layout

```
.
├── docs/whitepaper/            LaTeX whitepaper (compiles on Overleaf)
├── kernels/                    CUDA kernels + benches + ALU primitives
│   ├── bposit16_reference.py   pure-Python rational-arithmetic reference
│   ├── build_bposit_luts.py    LUT generator (recip, sqrt, log2, exp2, ...)
│   ├── bposit16_encode.cuh     keystone quire256 → bposit16 encoder + ALU
│   ├── libmosyne_bposit.cu     C-ABI shared-library version of the pipeline
│   ├── bench_*.cu, bench_*.sh  matmul throughput benches
│   ├── test_*.cu               ALU primitive validation + AI/ML demos
│   └── extract_qwen_layer.py   pure-numpy safetensors → fp32 binary loader
└── packages/
    └── mosyne_bposit/          pip-installable package
        ├── pyproject.toml
        ├── README.md
        ├── examples/
        │   └── transformers_qwen_integration.py
        └── src/mosyne_bposit/
            ├── _api.py             ctypes wrapper, lazy library load
            ├── build_so.py         mosyne-bposit-build CLI
            ├── torch_compat.py     optional PyTorch / HF integration
            └── _cuda/libmosyne_bposit.cu
```

## Hardware tested

- RTX 5090 (Blackwell sm_120), CUDA 12.8, cuBLASLt 12.8.5.5
- RTX 3090 (Ampere sm_86), same toolchain

Single fatbin builds for both architectures
(`-gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120`).

## Citing

If you use this work, please cite the whitepaper:

```bibtex
@techreport{bruscoe2026bposit,
  title  = {Bounded Posits at IEEE Tensor-Core Throughput on Commodity
            NVIDIA Blackwell, with Bit-Exact Reproducibility},
  author = {Bruscoe, Ry},
  institution = {Anomly, Inc.},
  year   = {2026},
  url    = {https://github.com/anomly-labs/mosyne-bposit}
}
```

## License

Copyright (c) 2026 Ry Bruscoe and Anomly, Inc.

Licensed under the Apache License, Version 2.0. See `LICENSE` for the
full license text and `NOTICE` for attribution requirements.

## Background

Posit arithmetic was introduced by John L. Gustafson as an alternative to
IEEE 754 floating point. The bounded variant used here (`bposit`) caps
the regime field for predictable dynamic range. See *The End of Error*
(CRC Press, 2015) for the full specification.

This repository complements rather than competes with Stanford's
[MINOTAUR](https://ieeexplore.ieee.org/document/10916649/) (IEEE JSSC,
April 2025), which demonstrates posit's energy advantage on a custom
40 nm ASIC. The present work shows that the *throughput* property
holds on commodity tensor-core silicon, and that the `quire`
accumulator delivers bit-exact reproducibility IEEE float on tensor
cores fundamentally cannot.
