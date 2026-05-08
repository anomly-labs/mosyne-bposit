"""Validation/structural tests that don't require a built libmosyne_bposit.so.

These exercise the public API surface: import, version exposure, library_path
resolution, and shape-validation in linear_w8a8 (which must raise ValueError
on bad inputs before any GPU work).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_package_imports_and_version():
    import mosyne_bposit

    assert isinstance(mosyne_bposit.__version__, str)
    assert mosyne_bposit.__version__
    assert {"linear_w8a8", "shutdown", "library_path", "build_library"} <= set(mosyne_bposit.__all__)


def test_library_path_resolves_next_to_module():
    import mosyne_bposit
    from mosyne_bposit import _api

    p = mosyne_bposit.library_path()
    assert isinstance(p, Path)
    assert p.name == "libmosyne_bposit.so"
    assert p.parent == Path(_api.__file__).resolve().parent


def test_linear_w8a8_rejects_non_2d_inputs():
    """Shape validation must run before library load so the error is
    actionable on hosts without nvcc/CUDA."""
    from mosyne_bposit import linear_w8a8

    x_1d = np.zeros(8, dtype=np.float32)
    w_2d = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 2-D"):
        linear_w8a8(x_1d, w_2d)

    x_2d = np.zeros((8, 8), dtype=np.float32)
    w_3d = np.zeros((8, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 2-D"):
        linear_w8a8(x_2d, w_3d)


def test_linear_w8a8_rejects_shape_mismatch():
    from mosyne_bposit import linear_w8a8

    x = np.zeros((4, 8), dtype=np.float32)
    w = np.zeros((16, 4), dtype=np.float32)  # K mismatch: 8 vs 16
    with pytest.raises(ValueError, match="shape mismatch"):
        linear_w8a8(x, w)


def test_shutdown_is_safe_when_library_not_loaded():
    """shutdown() before any linear_w8a8() call must not raise — it's
    documented as safe-to-call-multiple-times."""
    from mosyne_bposit import _api, shutdown

    saved = _api._lib
    _api._lib = None
    try:
        shutdown()
    finally:
        _api._lib = saved
