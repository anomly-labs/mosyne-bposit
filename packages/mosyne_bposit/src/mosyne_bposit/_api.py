# Copyright (c) 2026 Ry Bruscoe and Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ctypes wrapper around libmosyne_bposit.so.

The shared library is built once at install time by mosyne-bposit-build,
which writes it next to this module.  We locate it lazily on first call
so that import-time errors are limited to "library missing → run build"
rather than "CUDA driver not available" surprises.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SO_NAME = "libmosyne_bposit.so"
_lib = None


def library_path() -> Path:
    """Return the absolute path the .so should live at."""
    return _HERE / _SO_NAME


def _ensure_library() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    p = library_path()
    if not p.exists():
        raise RuntimeError(
            f"libmosyne_bposit.so not found at {p}. "
            f"Run `mosyne-bposit-build` (one-time, requires nvcc on PATH) "
            f"or set MOSYNE_BPOSIT_SO to a prebuilt .so path."
        )
    _lib = ctypes.CDLL(str(p))
    _lib.mosyne_bposit_init.restype = ctypes.c_int
    _lib.mosyne_bposit_init.argtypes = []
    _lib.mosyne_bposit_shutdown.restype = None
    _lib.mosyne_bposit_shutdown.argtypes = []
    _lib.mosyne_bposit_linear_w8a8_host.restype = ctypes.c_int
    _lib.mosyne_bposit_linear_w8a8_host.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    rc = _lib.mosyne_bposit_init()
    if rc != 0:
        raise RuntimeError(f"mosyne_bposit_init failed (rc={rc})")
    return _lib


def linear_w8a8(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute ``y = x @ w`` via the bposit-IMMA W8A8 pipeline.

    Inputs accepted as row-major numpy arrays of shape ``(M, K)`` and
    ``(K, N)``.  Internally the library expects column-major layout, so we
    transpose-and-contiguize.  Returns a row-major float32 array of shape
    ``(M, N)``.
    """
    lib = _ensure_library()
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError(f"x and w must be 2-D (got x.shape={x.shape}, w.shape={w.shape})")
    if x.shape[1] != w.shape[0]:
        raise ValueError(
            f"shape mismatch: x.shape[1]={x.shape[1]} != w.shape[0]={w.shape[0]}"
        )
    M, K = x.shape
    _, N = w.shape

    x_cm = np.asfortranarray(x.astype(np.float32, copy=False))
    w_cm = np.asfortranarray(w.astype(np.float32, copy=False))
    y_cm = np.zeros((M, N), dtype=np.float32, order="F")
    rc = lib.mosyne_bposit_linear_w8a8_host(
        x_cm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), M, K,
        w_cm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N,
        y_cm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise RuntimeError(f"mosyne_bposit_linear_w8a8_host failed (rc={rc})")
    return np.ascontiguousarray(y_cm)


def shutdown() -> None:
    """Free GPU resources held by the library.  Safe to call multiple times."""
    global _lib
    if _lib is not None:
        _lib.mosyne_bposit_shutdown()