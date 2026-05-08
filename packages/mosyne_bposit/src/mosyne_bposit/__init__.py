"""mosyne_bposit — bposit-via-IMMA W8A8 matrix multiplication on NVIDIA tensor cores.

After ``pip install mosyne-bposit``, run ``mosyne-bposit-build`` once to
compile the CUDA shared library (requires ``nvcc`` on PATH).  Then::

    from mosyne_bposit import linear_w8a8
    y = linear_w8a8(x, w)
"""
from __future__ import annotations

from ._api import (
    linear_w8a8,
    linear_w8a8_dev,
    quantize_weight_per_channel_dev,
    shutdown,
    library_path,
)
from .build_so import build as build_library

__version__ = "0.1.0"

__all__ = [
    "linear_w8a8",
    "linear_w8a8_dev",
    "quantize_weight_per_channel_dev",
    "shutdown",
    "library_path",
    "build_library",
    "__version__",
]
