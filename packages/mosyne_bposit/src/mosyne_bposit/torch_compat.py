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

Behaviour
---------

When a ``BPositLinear`` is constructed from a CUDA-resident weight, the
weight is quantized to int8 on the device once at construction time and
stored as device-resident buffers. Forward passes whose input tensor
also lives on CUDA route through the device-pointer entry point — no
host roundtrip, no per-call weight re-upload — so PyTorch tensors stay
on the GPU end-to-end.

Forward dispatches on input dtype:

* ``bfloat16`` input → fully bf16-native hot path
  (``linear_w8a8_dev_bf16_io``). The per-token quantize kernel reads
  bf16 directly and the dequantize kernel emits bf16 directly, so no
  fp32 buffer is allocated in the inner loop. This is the realistic
  deployment path; on a Llama-class FFN-gate shape
  (M=128, K=4096, N=11008, RTX 3090) it runs at ~0.33 ms/call.

* Any other floating dtype (fp32, fp16, fp64) → cast-to-fp32 path
  (``linear_w8a8_dev``). The input is cast and transposed in a single
  fused PyTorch kernel before entering the library; the output is
  produced in fp32 and cast back to the caller's dtype. Slightly
  slower than the bf16-native path but bit-exactly equivalent.

End-to-end on autoregressive ``Qwen2.5-Coder-1.5B-Instruct`` token
generation with all 84 FFN linears swapped, the bf16 path matches the
``bf16`` ``nn.Linear`` baseline within run-to-run noise (133.2 vs 132.8
tok/s, +0.3%; see ``examples/qwen_generate_bench.py``).

The host-fallback path is meaningfully slower — not because of host
roundtrip per se, but because the underlying ``linear_w8a8_host`` C
entry point uploads + per-channel-quantizes the fp32 weight on every
call (that's its accuracy-validation contract). That path is for
validating numerics on hosts where you don't want to keep the weight
resident; it is not a fair latency baseline.
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

from ._api import (
    linear_w8a8,
    linear_w8a8_dev,
    linear_w8a8_dev_bf16_io,
    linear_w8a8_dev_fp16_io,
    quantize_weight_per_channel_dev,
)


class BPositLinear(nn.Module):
    """Drop-in W8A8 replacement for ``nn.Linear`` using bposit-IMMA.

    Weight quantisation is done at construction time. If the source weight
    lives on a CUDA device, the int8 weight + scale buffers are kept on
    that device, and forward passes with CUDA-resident inputs route
    through the device-pointer pipeline (no host roundtrip). For
    host-resident weights or inputs the module falls back to the
    host-pointer entry point — useful for accuracy validation.

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
        # So we take the transpose and store as [in, out] (column-major
        # is what the underlying library expects, equivalent to row-major
        # of the transposed shape).
        K = weight.shape[1]
        N = weight.shape[0]
        self.in_features = K
        self.out_features = N
        self._cuda_device: Optional[torch.device] = None

        if weight.is_cuda:
            # Quantize weights on the device once at construction time.
            # Library expects column-major [K, N], which is identical in
            # memory to row-major [N, K] — the storage shape of the source
            # nn.Linear weight. So we just keep weight.contiguous() and
            # treat its data_ptr as col-major [K, N] for the library.
            w_fp32 = weight.detach().to(torch.float32).contiguous()  # [N, K] row-major
            w_i8 = torch.empty((N, K), dtype=torch.int8, device=weight.device)
            w_scale = torch.empty((N,), dtype=torch.float32, device=weight.device)
            quantize_weight_per_channel_dev(
                w_fp32.data_ptr(), K, N,
                w_i8.data_ptr(), w_scale.data_ptr(),
            )
            self.register_buffer("_w_i8_buf", w_i8, persistent=False)
            self.register_buffer("_w_scale_buf", w_scale, persistent=False)
            # Stash a host-side fp32 weight in the host-path layout (numpy
            # [K, N] row-major) so the host-fallback path can reuse it.
            self.register_buffer(
                "_w_fp32_buf",
                weight.detach().to(torch.float32).t().contiguous().cpu(),
                persistent=False,
            )
            self._cuda_device = weight.device
        else:
            # Host-only path: keep fp32 weight in host-path layout (numpy
            # [K, N] row-major), fall back to host entry point.
            w_fp32 = weight.detach().to(torch.float32).t().contiguous().cpu()
            self.register_buffer("_w_fp32_buf", w_fp32, persistent=False)
            self._w_i8_buf = None
            self._w_scale_buf = None

        if bias is not None:
            b = bias.detach().to(torch.float32)
            self.register_buffer("_b_fp32_buf", b.contiguous(), persistent=False)
        else:
            self._b_fp32_buf = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "BPositLinear":
        return cls(weight=linear.weight, bias=linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        K = orig_shape[-1]
        if K != self.in_features:
            raise ValueError(
                f"input last dim {K} does not match in_features {self.in_features}"
            )
        x_flat = x.reshape(-1, K)
        M = x_flat.shape[0]
        N = self.out_features

        if (
            x_flat.is_cuda
            and self._w_i8_buf is not None
            and self._w_scale_buf is not None
            and x_flat.device == self._cuda_device
        ):
            # Device-native path — no host roundtrip.
            # Library expects column-major inputs/outputs, which is
            # equivalent in memory to row-major [K, M] / [N, M].
            #
            # Use empty + copy_(transposed source) instead of .t().contiguous()
            # so the dtype cast and the layout swap fuse into a single CUDA
            # kernel each, saving ~17 µs per forward at FFN-gate (measured).
            #
            # bf16 input + bf16 output is the no-cast hot path; fp16 has
            # a symmetric path. Anything else (fp32, fp64) goes through
            # the cast-to-fp32 fallback.
            if x_flat.dtype == torch.bfloat16:
                x_cm = torch.empty((K, M), dtype=torch.bfloat16, device=x_flat.device)
                x_cm.copy_(x_flat.t())
                y_cm = torch.empty((N, M), dtype=torch.bfloat16, device=x_flat.device)
                linear_w8a8_dev_bf16_io(
                    x_cm.data_ptr(), M, K,
                    self._w_i8_buf.data_ptr(), self._w_scale_buf.data_ptr(), N,
                    y_cm.data_ptr(),
                )
            elif x_flat.dtype == torch.float16:
                x_cm = torch.empty((K, M), dtype=torch.float16, device=x_flat.device)
                x_cm.copy_(x_flat.t())
                y_cm = torch.empty((N, M), dtype=torch.float16, device=x_flat.device)
                linear_w8a8_dev_fp16_io(
                    x_cm.data_ptr(), M, K,
                    self._w_i8_buf.data_ptr(), self._w_scale_buf.data_ptr(), N,
                    y_cm.data_ptr(),
                )
            else:
                x_cm = torch.empty((K, M), dtype=torch.float32, device=x_flat.device)
                x_cm.copy_(x_flat.t())  # cast (any → fp32) + transpose, fused
                y_cm = torch.empty((N, M), dtype=torch.float32, device=x_flat.device)
                linear_w8a8_dev(
                    x_cm.data_ptr(), M, K,
                    self._w_i8_buf.data_ptr(), self._w_scale_buf.data_ptr(), N,
                    y_cm.data_ptr(),
                )
            y = torch.empty((M, N), dtype=x.dtype, device=x_flat.device)
            y.copy_(y_cm.t())  # col-major → x.dtype row-major, fused
        else:
            # Host fallback (host weight, or input on a different device).
            x_np = x_flat.detach().to(torch.float32).cpu().numpy()
            w_np = self._w_fp32_buf.numpy()
            y_np = linear_w8a8(x_np, w_np)
            y = torch.from_numpy(y_np).to(x.device).to(x.dtype)

        if self._b_fp32_buf is not None:
            y = y + self._b_fp32_buf.to(y.device).to(y.dtype)
        return y.reshape(*orig_shape[:-1], N)

    def extra_repr(self) -> str:
        path = "device" if self._w_i8_buf is not None else "host-fallback"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"has_bias={self._b_fp32_buf is not None}, dtype=W8A8(bposit-IMMA), "
            f"path={path}"
        )


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
