"""mosyne_bposit.torch_compat — drop-in PyTorch integration.

This module is imported only when ``torch`` is available; importing it
on a system without torch raises a clean ImportError so the rest of
``mosyne_bposit`` continues to work numpy-only.

Public API
----------

``BPositLinear(weight, bias=None)``
    A ``torch.nn.Module`` that quantises a single ``nn.Linear`` weight
    matrix at construction time and runs the forward pass via the
    bposit-IMMA W8A8 pipeline.

``replace_linear_modules(model, predicate=None)``
    Traverse a ``torch.nn.Module`` and replace every ``nn.Linear``
    (optionally filtered by ``predicate``) with a ``BPositLinear``.
    Returns the modified module.  Use to convert FFN / projection
    layers of a Hugging Face transformers model in place.

Caveats
-------

The current implementation routes through the host-side ``linear_w8a8``
entry point, which means every forward pass copies the input from GPU
to CPU and back.  This is fine for accuracy validation but adds latency
that the underlying CUDA library does not have.  A future tick will
expose the device-pointer entry point so PyTorch tensors stay on the
GPU end-to-end.
"""
from __future__ import annotations

from typing import Callable, Optional

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover — only triggered without torch
    raise ImportError(
        "mosyne_bposit.torch_compat requires PyTorch. "
        "Install with `pip install mosyne-bposit[torch]` or `pip install torch`."
    ) from e

import numpy as np

from ._api import linear_w8a8


class BPositLinear(nn.Module):
    """Drop-in W8A8 replacement for ``nn.Linear`` using bposit-IMMA.

    Weight quantisation is done at construction time and stored on the
    host side as ``self._w_fp32`` (float32, contiguous).  Inputs are
    copied to host on every forward pass — see *Caveats* above.

    Parameters
    ----------
    weight : Tensor of shape (out_features, in_features), any dtype that
        casts to float32.
    bias : optional Tensor of shape (out_features,).
    """

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2-D (got shape {tuple(weight.shape)})")
        # PyTorch nn.Linear stores weight as [out, in].  Our linear_w8a8
        # expects (M,K) @ (K,N), where K = in_features and N = out_features.
        # So we take the transpose and store as [in, out] float32.
        w_fp32 = weight.detach().to(torch.float32).t().contiguous().cpu().numpy()
        # Register as a non-trainable buffer so .to() / .cuda() etc. don't move it
        # (it lives on host — the bposit-IMMA pipeline allocates its own device buffers).
        self.register_buffer("_w_fp32_buf", torch.from_numpy(w_fp32), persistent=False)
        if bias is not None:
            b = bias.detach().to(torch.float32).cpu().numpy()
            self.register_buffer("_b_fp32_buf", torch.from_numpy(b.copy()), persistent=False)
        else:
            self._b_fp32_buf = None
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "BPositLinear":
        return cls(weight=linear.weight, bias=linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) → flatten to (M, K), apply, reshape
        orig_shape = x.shape
        K = orig_shape[-1]
        if K != self.in_features:
            raise ValueError(
                f"input last dim {K} does not match in_features {self.in_features}"
            )
        x_flat = x.reshape(-1, K)
        x_np = x_flat.detach().to(torch.float32).cpu().numpy()
        w_np = self._w_fp32_buf.numpy()
        y_np = linear_w8a8(x_np, w_np)            # (M, out_features)
        y = torch.from_numpy(y_np).to(x.device).to(x.dtype)
        if self._b_fp32_buf is not None:
            y = y + self._b_fp32_buf.to(x.device).to(x.dtype)
        return y.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, has_bias={self._b_fp32_buf is not None}, dtype=W8A8(bposit-IMMA)"


def replace_linear_modules(
    model: nn.Module,
    predicate: Optional[Callable[[str, nn.Linear], bool]] = None,
) -> nn.Module:
    """Walk *model* and replace every ``nn.Linear`` with a ``BPositLinear``.

    If *predicate* is given, it's called as ``predicate(qualified_name,
    linear)`` and only modules where it returns truthy are replaced.

    Returns the (now-modified) model.

    Example::

        from mosyne_bposit.torch_compat import replace_linear_modules

        # Replace only the MLP linears in layers 10..15:
        def keep(name, mod):
            return ".mlp." in name and any(f".layers.{i}." in name for i in range(10, 16))

        replace_linear_modules(model, predicate=keep)
    """
    # First collect (parent, attr_name, child) pairs; we can't mutate while iterating.
    # Use the canonical _modules dict — it's where PyTorch tracks named submodules
    # and avoids triggering property side effects from a generic dir() walk.
    targets: list[tuple[nn.Module, str, nn.Linear, str]] = []
    for qual_name, module in model.named_modules():
        for attr, child in module._modules.items():
            if isinstance(child, nn.Linear) and not isinstance(child, BPositLinear):
                full = f"{qual_name}.{attr}" if qual_name else attr
                if predicate is None or predicate(full, child):
                    targets.append((module, attr, child, full))
    for parent, attr, _linear, _full in targets:
        new_mod = BPositLinear.from_linear(_linear)
        setattr(parent, attr, new_mod)
    return model


__all__ = ["BPositLinear", "replace_linear_modules"]
