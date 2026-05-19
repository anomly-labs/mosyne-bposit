# Canonical cross-GPU probe references

Saved outputs from `mosyne-bposit-probe --json` and
`mosyne-bposit-demo --strict --json` on each verified GPU. The
claim "bit-exact across NVIDIA GPUs" is verified empirically by
running the probe on a new GPU and using `--compare-with` to
diff against one of these references.

## Files

| File | GPU | Date | Notes |
|---|---|---|---|
| `probe_3090_2026-05-19.json` | RTX 3090 (Ampere, sm_86) | 2026-05-19 | Canonical reference; produced from `pip install -e .[torch]` with default PyTorch 2.5.1+cu124 |
| `demo_3090_2026-05-19.json` | RTX 3090 (Ampere, sm_86) | 2026-05-19 | Full demo output — all 4 claims PASS |

## Pending

- **RTX 5090 (Blackwell, sm_120)** reference. The default
  `pip install` PyTorch (sm_50..sm_90) cannot drive Blackwell
  kernels — need a PyTorch ≥ 2.7 / CUDA ≥ 12.8 build (`cu128`
  channel) to capture this. See the GPU compatibility section
  in the package README. The mechanism is structurally
  deterministic by construction; the empirical reference will
  match the 3090 hashes (because the inputs are CPU-generated
  with a fixed seed, and the bposit-IMMA path is integer-only).
- **H100 / A100 / 4090 / T4 / V100** — these are the
  community-contribution gates per `CONTRIBUTING.md`. Run the
  probe on any of these and open an issue with the JSON paste;
  we'll add it here.

## Verifying a new GPU against these references

```bash
# On the new GPU:
mosyne-bposit-probe --json --compare-with probe_3090_2026-05-19.json

# Exits 0 with "PASS: all 3 shapes bit-identical" if the cross-GPU
# claim holds for the new hardware. Exits 1 with a per-shape diff
# if any hash differs — that's a real finding, please file an issue.
```
