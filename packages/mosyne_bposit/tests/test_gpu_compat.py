"""iter-299: surface tests for mosyne_bposit._gpu_compat.

The shared GPU/PyTorch compat helper catches the "fresh install
on a 5090, PyTorch ships sm_50..sm_90, demo crashes cryptically
with no kernel image" failure mode and prints an actionable
install hint instead. This is the path 100% of Blackwell users
will hit on first install, so a regression here is one of the
worst possible UX-on-first-impression bugs.

CPU-only — no real GPU access required."""
from __future__ import annotations

from unittest.mock import patch

import pytest


def test_module_imports_without_torch() -> None:
    import importlib
    mod = importlib.import_module("mosyne_bposit._gpu_compat")
    assert hasattr(mod, "check_gpu_compat")


def test_returns_1_with_actionable_hint_when_torch_missing(capsys) -> None:
    """First failure path: torch isn't installed at all."""
    from mosyne_bposit import _gpu_compat

    real_import = __builtins__["__import__"] if isinstance(
        __builtins__, dict) else __builtins__.__import__

    def _block_torch(name, *a, **kw):
        if name == "torch":
            raise ImportError("simulated")
        return real_import(name, *a, **kw)

    with patch("builtins.__import__", _block_torch):
        rc = _gpu_compat.check_gpu_compat()
    assert rc == 1
    err = capsys.readouterr().err
    assert "requires PyTorch" in err
    assert "mosyne-bposit[torch]" in err


def test_returns_1_when_cuda_unavailable(monkeypatch, capsys) -> None:
    """Second failure path: torch is installed but no CUDA GPU."""
    pytest.importorskip("torch")
    import torch
    from mosyne_bposit import _gpu_compat

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rc = _gpu_compat.check_gpu_compat()
    assert rc == 1
    assert "CUDA not available" in capsys.readouterr().err


def test_returns_1_with_blackwell_hint_when_sm120_unsupported(
    monkeypatch, capsys
) -> None:
    """The actual case this helper exists for: GPU is sm_120
    (Blackwell, RTX 5090) but PyTorch was built without sm_120
    support. Must print the cu128 install hint."""
    pytest.importorskip("torch")
    import torch
    from mosyne_bposit import _gpu_compat

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name",
                        lambda i=0: "NVIDIA GeForce RTX 5090")
    monkeypatch.setattr(torch.cuda, "get_arch_list",
                        lambda: ["sm_50", "sm_60", "sm_70", "sm_75",
                                 "sm_80", "sm_86", "sm_90"])
    rc = _gpu_compat.check_gpu_compat()
    assert rc == 1
    err = capsys.readouterr().err
    # GPU + sm + what we have + what's needed all in the error.
    assert "RTX 5090" in err
    assert "sm_120" in err
    assert "sm_90" in err
    assert ("Blackwell" in err or "cu128" in err), (
        "error must point Blackwell users at the cu128 install path"
    )


def test_returns_0_when_arch_supported(monkeypatch) -> None:
    """The happy path: PyTorch supports the GPU's arch. No error."""
    pytest.importorskip("torch")
    import torch
    from mosyne_bposit import _gpu_compat

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 6))
    monkeypatch.setattr(torch.cuda, "get_device_name",
                        lambda i=0: "NVIDIA GeForce RTX 3090")
    monkeypatch.setattr(torch.cuda, "get_arch_list",
                        lambda: ["sm_80", "sm_86", "sm_90"])
    rc = _gpu_compat.check_gpu_compat()
    assert rc == 0


def test_returns_0_when_arch_list_introspection_fails(
    monkeypatch
) -> None:
    """Defensive: if PyTorch's arch-list introspection itself
    blows up, don't false-positive-abort. Let the first kernel
    launch fail with PyTorch's native error rather than ours."""
    pytest.importorskip("torch")
    import torch
    from mosyne_bposit import _gpu_compat

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 6))
    monkeypatch.setattr(torch.cuda, "get_device_name",
                        lambda i=0: "NVIDIA GeForce RTX 3090")

    def _raise():
        raise RuntimeError("simulated introspection failure")

    monkeypatch.setattr(torch.cuda, "get_arch_list", _raise)
    rc = _gpu_compat.check_gpu_compat()
    assert rc == 0


def test_returns_0_when_arch_list_is_empty(monkeypatch) -> None:
    """Defensive: PyTorch built without CUDA arch support
    metadata. Don't abort — let the first kernel launch
    self-report."""
    pytest.importorskip("torch")
    import torch
    from mosyne_bposit import _gpu_compat

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 6))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: [])
    rc = _gpu_compat.check_gpu_compat()
    assert rc == 0


def test_old_arch_missing_points_at_pytorch_matrix_not_cu128(
    monkeypatch, capsys
) -> None:
    """If GPU is sm < 12 (i.e. not Blackwell) but the running
    PyTorch happens to lack support for it, the error should
    point at the general PyTorch install matrix, not the
    Blackwell-specific cu128 hint (which would be misleading)."""
    pytest.importorskip("torch")
    import torch
    from mosyne_bposit import _gpu_compat

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    monkeypatch.setattr(torch.cuda, "get_device_name",
                        lambda i=0: "Tesla T4")
    monkeypatch.setattr(torch.cuda, "get_arch_list",
                        lambda: ["sm_86", "sm_90"])
    rc = _gpu_compat.check_gpu_compat()
    assert rc == 1
    err = capsys.readouterr().err
    assert "sm_75" in err
    # Don't claim "Blackwell" or push cu128 for a non-Blackwell GPU.
    assert "Blackwell" not in err
    assert "cu128" not in err
    assert "install matrix" in err.lower() or "get-started" in err
