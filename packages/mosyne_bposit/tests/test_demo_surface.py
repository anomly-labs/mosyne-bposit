"""mosyne-bposit-demo --strict / --json / --skip-memory surface tests.

The demo can't actually run its four GPU claims in CI (no CUDA in
the test runner), but its argparse surface, ClaimResult dataclass,
and no-CUDA error path are all CPU-testable. These tests are the
drift-guards that catch:

  - A flag being silently dropped from argparse (--strict, --json,
    --skip-memory)
  - The new claim_memory() function disappearing from the module
  - ClaimResult drift (a name/passed/details field renamed)
  - Tne JSON exit path not emitting parseable output

What we can NOT test here: the actual numerical claims. Those need
CUDA + a built libmosyne_bposit.so.
"""
from __future__ import annotations

import argparse
import importlib
import json
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------
# Module imports + ClaimResult shape
# ---------------------------------------------------------------

def test_module_imports_without_cuda() -> None:
    """Demo module must be importable without torch / CUDA — argparse
    inspection has to work in any environment, not just GPU boxes."""
    mod = importlib.import_module("mosyne_bposit.demo")
    assert hasattr(mod, "main")
    assert hasattr(mod, "ClaimResult")


def test_claim_result_has_expected_fields() -> None:
    from mosyne_bposit.demo import ClaimResult
    r = ClaimResult(name="x", passed=True, details={"k": "v"})
    assert r.name == "x"
    assert r.passed is True
    assert r.details == {"k": "v"}
    # Must be dataclass-serialisable for --json mode.
    from dataclasses import asdict
    assert asdict(r) == {"name": "x", "passed": True, "details": {"k": "v"}}


def test_all_four_claim_functions_exposed() -> None:
    """A regression where one of the four claims is silently dropped
    from the demo would invalidate the README's '[N/4]' headers."""
    from mosyne_bposit import demo
    for name in ("claim_throughput", "claim_accuracy",
                 "claim_reproducibility", "claim_memory"):
        assert hasattr(demo, name), f"demo missing {name}"


# ---------------------------------------------------------------
# Argparse surface
# ---------------------------------------------------------------

def _capture_parser() -> argparse.ArgumentParser:
    """Get the ArgumentParser instance built inside main() by
    intercepting parse_args. Pattern reused from
    tests/test_forge_cli_on_error.py (iter-290)."""
    from mosyne_bposit import demo

    captured: dict = {}
    real_init = argparse.ArgumentParser.__init__

    def _capture(self, *a, **kw):
        real_init(self, *a, **kw)
        captured["parser"] = self

    def _abort(self, *a, **kw):  # noqa: ARG001
        raise SystemExit(0)

    with patch.object(argparse.ArgumentParser, "__init__", _capture), \
         patch.object(argparse.ArgumentParser, "parse_args", _abort):
        with pytest.raises(SystemExit):
            demo.main([])
    return captured["parser"]


def test_strict_flag_declared() -> None:
    parser = _capture_parser()
    dests = {a.dest for a in parser._actions}
    assert "strict" in dests, "--strict missing (CI-gating regression)"


def test_json_flag_declared() -> None:
    parser = _capture_parser()
    dests = {a.dest for a in parser._actions}
    assert "json" in dests, "--json missing (CI parseability regression)"


def test_skip_memory_flag_declared() -> None:
    parser = _capture_parser()
    dests = {a.dest for a in parser._actions}
    assert "skip_memory" in dests, "--skip-memory missing"


def test_all_four_skip_flags_present() -> None:
    """One --skip-X per claim, kept in lockstep with the four
    claim_* functions."""
    parser = _capture_parser()
    dests = {a.dest for a in parser._actions}
    for flag in ("skip_throughput", "skip_accuracy",
                 "skip_reproducibility", "skip_memory"):
        assert flag in dests, f"--{flag.replace('_', '-')} missing"


# ---------------------------------------------------------------
# No-CUDA error path
# ---------------------------------------------------------------

def test_returns_1_when_torch_missing(monkeypatch, capsys) -> None:
    """If PyTorch isn't installed, demo must print an actionable
    install hint and return 1 — not crash with ImportError.
    iter-299: error now goes via the shared _gpu_compat helper to
    stderr (more correct routing for errors)."""
    from mosyne_bposit import demo

    real_import = __builtins__["__import__"] if isinstance(
        __builtins__, dict) else __builtins__.__import__

    def _block_torch(name, *a, **kw):
        if name == "torch":
            raise ImportError("simulated: torch not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", _block_torch)
    rc = demo.main([])
    assert rc == 1
    assert "requires PyTorch" in capsys.readouterr().err


def test_returns_1_when_cuda_unavailable(monkeypatch, capsys) -> None:
    """The demo measures GPU behaviour — must error cleanly when no
    CUDA device is present, not produce misleading CPU results."""
    pytest.importorskip("torch")
    import torch
    from mosyne_bposit import demo

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rc = demo.main([])
    assert rc == 1
    assert "CUDA not available" in capsys.readouterr().err


# ---------------------------------------------------------------
# --json output structure (with all claims stubbed)
# ---------------------------------------------------------------

def test_json_mode_emits_parseable_summary(
    monkeypatch, capsys
) -> None:
    """End-to-end: --json produces a valid JSON object on stdout with
    the expected top-level shape, regardless of human narration on
    stderr. Catches drift in the payload schema."""
    pytest.importorskip("torch")
    import torch
    from mosyne_bposit import demo

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "STUB")
    # iter-299: short-circuit the new pre-flight GPU compat check
    # — it would otherwise hit the real GPU and refuse to proceed
    # on hardware where our PyTorch lacks the arch (sm_120).
    from mosyne_bposit import _gpu_compat
    monkeypatch.setattr(_gpu_compat, "check_gpu_compat", lambda: 0)

    # Stub every claim to a known PASS so we hit the JSON-emit path
    # without needing the .so or a real GPU.
    def _stub_throughput():
        return demo.ClaimResult(name="throughput", passed=True,
                                details={"stub": True})
    def _stub_accuracy():
        return demo.ClaimResult(name="accuracy", passed=True,
                                details={"stub": True})
    def _stub_reproducibility():
        return demo.ClaimResult(name="reproducibility", passed=True,
                                details={"stub": True})
    def _stub_memory():
        return demo.ClaimResult(name="memory", passed=True,
                                details={"stub": True})

    monkeypatch.setattr(demo, "claim_throughput", _stub_throughput)
    monkeypatch.setattr(demo, "claim_accuracy", _stub_accuracy)
    monkeypatch.setattr(demo, "claim_reproducibility", _stub_reproducibility)
    monkeypatch.setattr(demo, "claim_memory", _stub_memory)

    rc = demo.main(["--json"])
    assert rc == 0
    out = capsys.readouterr().out
    # Find the JSON object (whole stdout in JSON mode is exactly the
    # JSON object, no other content).
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["passed"] == 4
    assert payload["total"] == 4
    assert payload["gpu"] == "STUB"
    assert "claims" in payload
    assert {c["name"] for c in payload["claims"]} == {
        "throughput", "accuracy", "reproducibility", "memory",
    }
    assert all(c["passed"] for c in payload["claims"])


def test_strict_mode_returns_1_on_any_failure(
    monkeypatch, capsys
) -> None:
    """The whole point of --strict is non-zero exit on regression.
    Stub one claim to FAIL and confirm rc=1."""
    pytest.importorskip("torch")
    import torch
    from mosyne_bposit import demo

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "STUB")
    # iter-299: short-circuit the new pre-flight GPU compat check
    # — it would otherwise hit the real GPU and refuse to proceed
    # on hardware where our PyTorch lacks the arch (sm_120).
    from mosyne_bposit import _gpu_compat
    monkeypatch.setattr(_gpu_compat, "check_gpu_compat", lambda: 0)

    monkeypatch.setattr(demo, "claim_throughput",
                        lambda: demo.ClaimResult("throughput", True, {}))
    monkeypatch.setattr(demo, "claim_accuracy",
                        lambda: demo.ClaimResult("accuracy", True, {}))
    monkeypatch.setattr(demo, "claim_reproducibility",
                        lambda: demo.ClaimResult("reproducibility", False, {}))
    monkeypatch.setattr(demo, "claim_memory",
                        lambda: demo.ClaimResult("memory", True, {}))

    rc = demo.main(["--strict"])
    assert rc == 1
    # Human output names the regression count.
    err = capsys.readouterr().out
    assert "3/4 passed" in err or "1 regressed" in err


def test_strict_mode_returns_0_when_all_pass(
    monkeypatch
) -> None:
    pytest.importorskip("torch")
    import torch
    from mosyne_bposit import demo

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "STUB")
    # iter-299: short-circuit the new pre-flight GPU compat check
    # — it would otherwise hit the real GPU and refuse to proceed
    # on hardware where our PyTorch lacks the arch (sm_120).
    from mosyne_bposit import _gpu_compat
    monkeypatch.setattr(_gpu_compat, "check_gpu_compat", lambda: 0)
    for fn in ("claim_throughput", "claim_accuracy",
               "claim_reproducibility", "claim_memory"):
        monkeypatch.setattr(demo, fn,
                            lambda name=fn.replace("claim_", ""):
                            demo.ClaimResult(name, True, {}))
    rc = demo.main(["--strict"])
    assert rc == 0
