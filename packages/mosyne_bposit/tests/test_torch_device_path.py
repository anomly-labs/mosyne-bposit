"""Device-native PyTorch forward-path tests.

Skips cleanly when:
  - PyTorch is not installed
  - CUDA is not available to PyTorch
  - libmosyne_bposit.so is not built / present

so that hosts without nvcc + GPU continue to pass the suite.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available; device-path tests require a GPU",
                allow_module_level=True)


def _library_available() -> bool:
    from mosyne_bposit import library_path
    return library_path().exists()


pytestmark = pytest.mark.skipif(
    not _library_available(),
    reason="libmosyne_bposit.so not built; run mosyne-bposit-build",
)


def test_device_path_matches_host_path_bit_exact():
    """The device-pointer forward path must produce the same result as the
    host-pointer forward path on the same numerical inputs. Both go through
    the same underlying CUDA kernels, so the bit-pattern of the output
    should be identical."""
    from mosyne_bposit.torch_compat import BPositLinear

    torch.manual_seed(0)
    M, K, N = 32, 256, 512
    w_cpu = torch.randn(N, K, dtype=torch.float32) * (1.0 / K ** 0.5)
    x_cpu = torch.randn(M, K, dtype=torch.float32) * 0.5
    w_gpu = w_cpu.cuda()
    x_gpu = x_cpu.cuda()

    mod_dev = BPositLinear(weight=w_gpu)
    mod_host = BPositLinear(weight=w_cpu)

    y_dev = mod_dev(x_gpu).cpu()
    y_host = mod_host(x_cpu)

    # Same kernels, same data → bit-exact.
    assert (y_dev - y_host).abs().max().item() == 0.0, (
        "device-path output must be bit-exact against host-path output"
    )


def test_device_path_w8a8_error_in_expected_band():
    """Sanity check: W8A8 quant error on a well-conditioned random matmul
    should be in the few-percent range, matching the published whitepaper
    numbers. If this regresses badly, the layout/quant pipeline is wrong."""
    from mosyne_bposit.torch_compat import BPositLinear

    torch.manual_seed(0)
    M, K, N = 64, 512, 1024
    x = torch.randn(M, K, device="cuda", dtype=torch.float32) * 0.5
    w = torch.randn(N, K, device="cuda", dtype=torch.float32) * (1.0 / K ** 0.5)
    ref = x @ w.t()

    mod = BPositLinear(weight=w)
    y = mod(x)

    err = (y - ref).pow(2).mean().sqrt().item()
    ref_l2 = ref.pow(2).mean().sqrt().item()
    rel_err = err / ref_l2
    # Whitepaper §4.3 reports 3.5–4% L2 on real Qwen layers; well-conditioned
    # random matrices come in tighter (~1%). If we're above 5% something is
    # structurally wrong (likely a layout bug like the one fixed in this
    # commit).
    assert rel_err < 0.05, (
        f"W8A8 relative error {rel_err*100:.2f}% is outside the expected band "
        f"(<5%); layout or quant pipeline likely broken"
    )


def test_bf16_input_matches_fp32_input_within_quant_noise():
    """The bf16-direct device path should produce numerically equivalent
    results to the fp32 device path on bf16-castable inputs. Difference
    should be within bf16 round-trip tolerance — bf16 has ~3 decimal
    digits of precision, so we expect agreement to ~1e-2 on a moderately
    sized matmul."""
    from mosyne_bposit.torch_compat import BPositLinear

    torch.manual_seed(0)
    M, K, N = 32, 512, 1024
    w = torch.randn(N, K, device="cuda", dtype=torch.float32) * (1.0 / K ** 0.5)
    x_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32) * 0.5
    x_bf16 = x_fp32.to(torch.bfloat16)

    mod = BPositLinear(weight=w)
    y_fp32 = mod(x_fp32)
    y_bf16 = mod(x_bf16).to(torch.float32)

    rel = (y_fp32 - y_bf16).abs().max() / y_fp32.abs().max()
    assert rel.item() < 5e-2, (
        f"bf16-input path diverged from fp32-input path by {rel.item()*100:.2f}%; "
        f"both should produce results within bf16 round-trip noise"
    )


def test_fp16_input_matches_fp32_input_within_quant_noise():
    """The fp16-direct device path should produce results within fp16
    round-trip tolerance of the fp32 path. fp16 has more mantissa bits
    than bf16 (10 vs 7) so the tolerance is tighter."""
    from mosyne_bposit.torch_compat import BPositLinear

    torch.manual_seed(0)
    M, K, N = 32, 512, 1024
    w = torch.randn(N, K, device="cuda", dtype=torch.float32) * (1.0 / K ** 0.5)
    x_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32) * 0.5
    x_fp16 = x_fp32.to(torch.float16)

    mod = BPositLinear(weight=w)
    y_fp32 = mod(x_fp32)
    y_fp16 = mod(x_fp16).to(torch.float32)

    rel = (y_fp32 - y_fp16).abs().max() / y_fp32.abs().max()
    assert rel.item() < 1e-2, (
        f"fp16-input path diverged from fp32-input path by {rel.item()*100:.2f}%; "
        f"both should produce results within fp16 round-trip noise"
    )


def test_device_path_module_repr_indicates_device():
    """extra_repr should advertise which path the module is wired to use,
    so users can confirm at a glance that they're getting the no-host-roundtrip
    fast path."""
    from mosyne_bposit.torch_compat import BPositLinear

    w_gpu = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    w_cpu = w_gpu.cpu()

    mod_dev = BPositLinear(weight=w_gpu)
    mod_host = BPositLinear(weight=w_cpu)

    assert "path=device" in repr(mod_dev)
    assert "path=host-fallback" in repr(mod_host)
