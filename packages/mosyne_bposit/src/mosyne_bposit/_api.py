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
    # Device-pointer entry points — accept raw void* (uintptr_t) so callers
    # can pass torch.Tensor.data_ptr() directly without ctypes-cast gymnastics.
    _lib.mosyne_bposit_quantize_weight_per_channel.restype = ctypes.c_int
    _lib.mosyne_bposit_quantize_weight_per_channel.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    _lib.mosyne_bposit_linear_w8a8.restype = ctypes.c_int
    _lib.mosyne_bposit_linear_w8a8.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p,
    ]
    _lib.mosyne_bposit_linear_w8a8_bf16.restype = ctypes.c_int
    _lib.mosyne_bposit_linear_w8a8_bf16.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p,
    ]
    _lib.mosyne_bposit_linear_w8a8_bf16_io.restype = ctypes.c_int
    _lib.mosyne_bposit_linear_w8a8_bf16_io.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p,
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
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError(f"x and w must be 2-D (got x.shape={x.shape}, w.shape={w.shape})")
    if x.shape[1] != w.shape[0]:
        raise ValueError(
            f"shape mismatch: x.shape[1]={x.shape[1]} != w.shape[0]={w.shape[0]}"
        )
    lib = _ensure_library()
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


def quantize_weight_per_channel_dev(
    w_fp32_dev_ptr: int, K: int, N: int,
    w_i8_dev_ptr: int, w_scale_dev_ptr: int,
) -> None:
    """Quantize a column-major float32 weight ``[K, N]`` already resident on
    the GPU to int8 with per-output-channel scales (also on the GPU). All
    pointer arguments are integer device addresses — typically obtained from
    a torch tensor's ``.data_ptr()``.

    Layout matches the host-side library: ``w_fp32`` is column-major, the
    scale buffer is ``[N]`` floats, and the output ``w_i8`` is column-major
    ``[K, N]`` int8.
    """
    lib = _ensure_library()
    rc = lib.mosyne_bposit_quantize_weight_per_channel(
        ctypes.c_void_p(w_fp32_dev_ptr), K, N,
        ctypes.c_void_p(w_i8_dev_ptr), ctypes.c_void_p(w_scale_dev_ptr),
    )
    if rc != 0:
        raise RuntimeError(f"mosyne_bposit_quantize_weight_per_channel failed (rc={rc})")


def linear_w8a8_dev(
    x_fp32_dev_ptr: int, M: int, K: int,
    w_i8_dev_ptr: int, w_scale_dev_ptr: int, N: int,
    y_fp32_dev_ptr: int,
) -> None:
    """W8A8 linear with all buffers already on the GPU. Computes
    ``y[M,N] = x[M,K] @ w[K,N]`` via the bposit-IMMA pipeline. Input is
    quantized per-token internally; weights are expected pre-quantized
    (call :func:`quantize_weight_per_channel_dev` first).

    Buffer layout: ``x`` and ``y`` are column-major float32; ``w_i8`` is
    column-major int8; ``w_scale`` is ``[N]`` float32. Pointers are
    integer device addresses (e.g., ``tensor.data_ptr()``).
    """
    lib = _ensure_library()
    rc = lib.mosyne_bposit_linear_w8a8(
        ctypes.c_void_p(x_fp32_dev_ptr), M, K,
        ctypes.c_void_p(w_i8_dev_ptr), ctypes.c_void_p(w_scale_dev_ptr), N,
        ctypes.c_void_p(y_fp32_dev_ptr),
    )
    if rc != 0:
        raise RuntimeError(f"mosyne_bposit_linear_w8a8 failed (rc={rc})")


def linear_w8a8_dev_bf16(
    x_bf16_dev_ptr: int, M: int, K: int,
    w_i8_dev_ptr: int, w_scale_dev_ptr: int, N: int,
    y_fp32_dev_ptr: int,
) -> None:
    """W8A8 linear with bf16 input — same contract as :func:`linear_w8a8_dev`
    but ``x`` is column-major bfloat16 instead of float32, saving a cast for
    callers whose activations are already bf16. Output stays float32.
    """
    lib = _ensure_library()
    rc = lib.mosyne_bposit_linear_w8a8_bf16(
        ctypes.c_void_p(x_bf16_dev_ptr), M, K,
        ctypes.c_void_p(w_i8_dev_ptr), ctypes.c_void_p(w_scale_dev_ptr), N,
        ctypes.c_void_p(y_fp32_dev_ptr),
    )
    if rc != 0:
        raise RuntimeError(f"mosyne_bposit_linear_w8a8_bf16 failed (rc={rc})")


def linear_w8a8_dev_bf16_io(
    x_bf16_dev_ptr: int, M: int, K: int,
    w_i8_dev_ptr: int, w_scale_dev_ptr: int, N: int,
    y_bf16_dev_ptr: int,
) -> None:
    """W8A8 linear with bf16 input AND bf16 output — fully bf16-native hot
    path. ``x`` and ``y`` are column-major bfloat16. Skips both the
    bf16→fp32 input cast and the fp32→bf16 output cast for callers whose
    activations are bf16 end-to-end (e.g. PyTorch transformers in bf16).
    """
    lib = _ensure_library()
    rc = lib.mosyne_bposit_linear_w8a8_bf16_io(
        ctypes.c_void_p(x_bf16_dev_ptr), M, K,
        ctypes.c_void_p(w_i8_dev_ptr), ctypes.c_void_p(w_scale_dev_ptr), N,
        ctypes.c_void_p(y_bf16_dev_ptr),
    )
    if rc != 0:
        raise RuntimeError(f"mosyne_bposit_linear_w8a8_bf16_io failed (rc={rc})")


def shutdown() -> None:
    """Free GPU resources held by the library.  Safe to call multiple times."""
    global _lib
    if _lib is not None:
        _lib.mosyne_bposit_shutdown()
