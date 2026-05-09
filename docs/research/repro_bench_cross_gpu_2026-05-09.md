# Cross-GPU reproducibility — measured

Date: 2026-05-09
Hardware: RTX 5090 (sm_120) and RTX 3090 (sm_86) on the same host (`anomly-lab1`).
Test: `deploy/5dst_cuda/test_determinism_headline.cu` — sum 65,536 log-uniform
values across 6 decades, FP32-atomicAdd path vs bposit16+quire256 path, 5 runs each.

## Result

**bposit + quire256 path is byte-identical across hardware generations.**

| Path                      | RTX 3090 (sm_86)             | RTX 5090 (sm_120)            | Same?   |
|---------------------------|------------------------------|------------------------------|---------|
| bposit16 + quire256       | `quire_hi=0x0049ec1830468000`<br>`bp16=0x7627` | `quire_hi=0x0049ec1830468000`<br>`bp16=0x7627` | **YES** |
| FP32 + atomicAdd, run 0   | `0x4a93ec53`                 | `0x4a93ec51`                 | no      |
| FP32 + atomicAdd, run 1   | `0x4a93ec63`                 | `0x4a93ec2f`                 | no      |
| FP32 + atomicAdd, run 2   | `0x4a93ec44`                 | `0x4a93ec22`                 | no      |
| FP32 + atomicAdd, run 3   | `0x4a93ec60`                 | `0x4a93ec64`                 | no      |
| FP32 + atomicAdd, run 4   | `0x4a93ec46`                 | `0x4a93ec44`                 | no      |
| FP32 unique hashes (5 runs) | 5                          | 5                            | —       |
| **bposit unique hashes (5 runs)** | **1**                | **1**                        | **YES** |

**Headline:** Across 10 total runs on two different NVIDIA generations (Ampere
3090 + Blackwell 5090), the bposit16 + quire256 path produced exactly **one**
output bit pattern. The IEEE FP32 + atomicAdd path produced **9 distinct**
output bit patterns (10 runs, with one accidental collision — `0x4a93ec44`
appears once on 3090 run 2 and once on 5090 run 4 by coincidence; the *sets*
of values produced on each GPU are otherwise disjoint).

This is the bit-exact reproducibility property whitepaper §4.2 claims, now
empirically extended from "same hardware, 5 runs" to "different hardware
generations, 10 runs."

## Why it matters

For any production deployment that needs to answer "did this model produce
this output for this input on this date" as a hash check — eval reproducibility,
mechanistic interpretability, regulatory audit, constitutional-AI training
reward signals, fleet-wide A/B tests on heterogeneous hardware — IEEE float
on tensor cores cannot deliver this. bposit + quire256 can, and now we have
cross-generation evidence.

## What's next

The `scripts/repro_bench/repro_bench.py` MVP from iter-41 hashes LLM-level
logits + token IDs through `BPositLinear`. Single-GPU it shows
within-process determinism but not the cross-GPU differential — at the LLM
forward level on a single GPU, cuBLASLt picks the same algo each time, so
even IEEE is locally deterministic. The differential surfaces with
multi-block parallel reductions (the atomicAdd case here) or with cross-GPU
algo divergence (different SM counts → different reduction trees → different
IEEE roundoff, even when the same heuristic algo number is selected).

Future iterations:
1. Extend `repro_bench.py` to drive a *parallel-reduction* layer (the kind of
   sum that hits atomicAdd in cuBLAS, e.g., a reduction across an attention
   batch dimension) so the LLM-level bench surfaces the same differential
   the synthetic sum already does.
2. Reproduce the cross-GPU comparison on a peer machine with a different
   driver / cuBLAS version pair, to extend "different hardware" to "different
   software stack" too.
3. Publish the bench as `github.com/anomly-labs/repro-bench` (separate repo
   so it's a standalone artifact other teams can run).
