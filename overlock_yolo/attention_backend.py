"""NATTEN backend resolution + a real, memory-bounded, differentiable ``na2d_av`` reference.

Design contract (DESIGN_V2.md 7.2 / 7.5)
----------------------------------------
* ``attention_backend: auto|natten|torch_reference`` is explicit and the *resolved* backend is
  always reported.
* Resolution is **lazy and per device** (:func:`resolve_backend` is called with the device of
  the tensor actually flowing into the op).  Building a model on CPU and later calling
  ``.to("cuda")`` therefore re-resolves instead of reusing a stale CPU answer, and an explicit
  ``natten`` request with no CUDA/NATTEN fails loudly at forward time rather than silently
  falling back to the reference kernel.
* On CPU, ``auto`` selects the real (non-mocked, differentiable) PyTorch reference.
* ``na2d_av`` is **never** replaced by average pooling, a zero function, a plain conv or
  identity.

Implemented subset: ``na2d_av(attn, value, kernel_size)`` with ``dilation=1`` and
``is_causal=False``:

* ``attn``  : ``[B, heads, H, W, K*K]``
* ``value`` : ``[B, heads, H, W, D]``
* output    : ``[B, heads, H, W, D]``

with the legacy NATTEN full-neighbourhood-shift boundary rule (``H, W >= K``)::

    start_y = clamp(y - K // 2, 0, H - K)
    start_x = clamp(x - K // 2, 0, W - K)
    out[b,h,y,x,d] = sum_{i,j} attn[b,h,y,x,i*K + j] * value[b,h,start_y+i,start_x+j,d]

The reference gathers neighbours **inside** the row-block loop, so the largest attention-shaped
intermediate is ``[heads*B, rows, K*K]`` rather than ``[B, heads, D, H*W, K*K]``; the circular
padding is a vectorised index map (no pixel copy) and the neighbour indices are precomputed
once per forward.  ``row_chunk=0`` processes every row in a single block and is then exactly
equivalent modulo floating-point summation order.

Accumulation runs in float32 (matching NATTEN's fp32 accumulation); the result is cast back to
``value.dtype``.  The reference is differentiable w.r.t. both ``attn`` and ``value`` and never
re-applies a softmax (the official dynamic block applies its own softmax).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "AttentionBackendError",
    "ResolvedBackend",
    "resolve_backend",
    "na2d_av",
    "na2d_av_reference",
    "na2d_av_native",
    "natten_available",
    "natten_import_error",
    "row_ranges",
    "BackendCache",
    "backend_cache_report",
    "na2d_av_macs",
    "BACKENDS",
]

BACKENDS = ("auto", "natten", "torch_reference")


class AttentionBackendError(RuntimeError):
    """Raised for unusable/unavailable attention backends or invalid ``na2d_av`` input."""


# --------------------------------------------------------------------------------------
# backend resolution
# --------------------------------------------------------------------------------------
def natten_available() -> bool:
    """True when the real ``natten.functional.na2d_av`` can be imported."""
    return natten_import_error() is None


def natten_import_error() -> Optional[str]:
    """The import error string for NATTEN, or ``None`` when importable."""
    try:  # pragma: no cover - environment dependent
        from natten.functional import na2d_av as _na2d_av  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"{type(exc).__name__}: {exc}"
    return None


def _cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive
        return False


@dataclass(frozen=True)
class ResolvedBackend:
    """The backend actually used for a specific device, plus the reason it was selected."""

    requested: str
    resolved: str
    reason: str
    cuda_available: bool
    device: str = "cpu"

    @property
    def is_native(self) -> bool:
        return self.resolved == "natten"

    def as_dict(self) -> dict:
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "reason": self.reason,
            "device": self.device,
            "cuda_available": self.cuda_available,
            "natten_importable": natten_available(),
            "natten_import_error": natten_import_error(),
        }


def resolve_backend(backend: str = "auto", device: Optional[torch.device] = None) -> ResolvedBackend:
    """Resolve the requested backend for ``device`` without ever silently faking NATTEN.

    ``auto``           : native NATTEN when importable *and* the device is CUDA, else the CPU
                         reference (reported as ``torch_reference``).
    ``natten``         : hard failure when NATTEN is missing **or** the device is not CUDA
                         (the native kernel is CUDA-only).  No fallback.
    ``torch_reference``: always the PyTorch reference (test/CPU comparison only).
    """
    if backend not in BACKENDS:
        raise AttentionBackendError(f"unknown attention backend {backend!r}; expected one of {BACKENDS}")

    cuda = _cuda_available()
    dev = torch.device(device) if device is not None else None
    on_cuda = bool(dev is not None and dev.type == "cuda")
    dev_name = str(dev) if dev is not None else "<unresolved>"

    if backend == "torch_reference":
        return ResolvedBackend(backend, "torch_reference", "explicitly requested PyTorch reference", cuda, dev_name)

    err = natten_import_error()
    if backend == "natten":
        if err is not None:
            raise AttentionBackendError(
                "attention_backend='natten' requested but NATTEN is not importable "
                f"({err}). Refusing to silently fall back; install NATTEN or use "
                "'torch_reference'/'auto'."
            )
        if not on_cuda:
            raise AttentionBackendError(
                "attention_backend='natten' requested but the target device is not CUDA "
                f"(device={dev_name}, cuda_available={cuda}). The native NATTEN kernel is "
                "CUDA-only; refusing to silently fall back to the reference."
            )
        return ResolvedBackend(backend, "natten", "native NATTEN available on CUDA", cuda, dev_name)

    # auto
    if err is None and on_cuda:
        return ResolvedBackend(backend, "natten", "auto: NATTEN importable and device is CUDA", cuda, dev_name)
    if err is not None:
        reason = f"auto: NATTEN not importable ({err}); using the differentiable PyTorch reference"
    else:
        reason = (
            "auto: NATTEN importable but the device is not CUDA "
            f"(device={dev_name}, cuda_available={cuda}); using the differentiable PyTorch reference"
        )
    return ResolvedBackend(backend, "torch_reference", reason, cuda, dev_name)


# --------------------------------------------------------------------------------------
# native op (thin, lazily imported)
# --------------------------------------------------------------------------------------
def na2d_av_native(attn: torch.Tensor, value: torch.Tensor, kernel_size: int) -> torch.Tensor:  # pragma: no cover
    """Call the real NATTEN kernel. Only reachable when NATTEN is installed and on CUDA."""
    from natten.functional import na2d_av as _native

    return _native(attn, value, kernel_size)


# --------------------------------------------------------------------------------------
# cost accounting hook (DESIGN_V2.md 6.3)
# --------------------------------------------------------------------------------------
def na2d_av_macs(b: int, heads: int, h: int, w: int, d: int, k: int) -> int:
    """MACs of one ``na2d_av`` call: ``B * heads * H * W * D * K * K`` multiply-accumulates."""
    return int(b) * int(heads) * int(h) * int(w) * int(d) * int(k) * int(k)


# --------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------
def _validate(attn: torch.Tensor, value: torch.Tensor, kernel_size: int) -> Tuple[int, int, int, int, int]:
    if not torch.is_tensor(attn) or not torch.is_tensor(value):
        raise AttentionBackendError("na2d_av expects tensor inputs")
    if attn.dim() != 5:
        raise AttentionBackendError(f"attn must be [B,heads,H,W,K*K] (5D), got shape {tuple(attn.shape)}")
    if value.dim() != 5:
        raise AttentionBackendError(f"value must be [B,heads,H,W,D] (5D), got shape {tuple(value.shape)}")
    if not isinstance(kernel_size, int) or isinstance(kernel_size, bool):
        raise AttentionBackendError(f"kernel_size must be an int, got {type(kernel_size).__name__}")
    if kernel_size < 1:
        raise AttentionBackendError(f"kernel_size must be >= 1, got {kernel_size}")
    if attn.shape[:4] != value.shape[:4]:
        raise AttentionBackendError(
            f"attn/value spatial-head-batch dims differ: attn {tuple(attn.shape)} vs value {tuple(value.shape)}"
        )
    b, heads, h, w, kk = attn.shape
    if kk != kernel_size * kernel_size:
        raise AttentionBackendError(
            f"attn last dim must be K*K={kernel_size * kernel_size} for kernel_size={kernel_size}, got {kk}"
        )
    if h < kernel_size or w < kernel_size:
        raise AttentionBackendError(
            f"na2d_av requires H,W >= kernel_size (the official dynamic block upsamples first); "
            f"got H={h}, W={w}, K={kernel_size}"
        )
    if attn.dtype != value.dtype:
        raise AttentionBackendError(f"attn/value dtype mismatch: {attn.dtype} vs {value.dtype}")
    if not (attn.dtype.is_floating_point and value.dtype.is_floating_point):
        raise AttentionBackendError(f"na2d_av requires floating point tensors, got {attn.dtype}")
    if attn.device != value.device:
        raise AttentionBackendError(f"attn/value device mismatch: {attn.device} vs {value.device}")
    return b, heads, h, w, kk


def row_ranges(total: int, chunk: int = 0) -> Iterable[Tuple[int, int]]:
    """Yield ``(start, end)`` row blocks covering ``range(total)``.

    ``chunk <= 0`` yields a single ``(0, total)`` block.  Exposed so tests can drive the
    reference with arbitrary chunk boundaries (including ones that do not divide ``total``).
    """
    if chunk is None or chunk <= 0 or chunk >= total:
        yield (0, total)
        return
    for start in range(0, total, chunk):
        yield (start, min(start + chunk, total))


# --------------------------------------------------------------------------------------
# differentiable, memory-bounded reference
# --------------------------------------------------------------------------------------
def na2d_av_reference(
    attn: torch.Tensor,
    value: torch.Tensor,
    kernel_size: int,
    row_chunk: int = 0,
    *,
    fp32_accumulate: bool = True,
) -> torch.Tensor:
    """Exact, memory-bounded, differentiable ``na2d_av`` reference (see module docstring)."""
    _validate(attn, value, kernel_size)

    b, heads, h, w, kk = attn.shape
    d = value.shape[-1]
    k = kernel_size
    out_dtype = value.dtype
    work_dtype = torch.float32 if (fp32_accumulate and out_dtype in (torch.float16, torch.bfloat16)) else out_dtype
    a = attn.to(work_dtype)
    v = value.to(work_dtype)
    device = v.device

    # --- neighbour indices: legacy NATTEN full-neighbourhood shift, expressed as a circular
    # --- index map. ``start_y = clamp(y - K//2, 0, H-K)`` and the circular index
    # --- ``(start_y + i) mod H`` coincide for every i in [0,K), corners included.
    iy = torch.arange(k, device=device)
    ix = torch.arange(k, device=device)
    start_y = (torch.arange(h, device=device) - k // 2).clamp(0, h - k)  # [H]
    start_x = (torch.arange(w, device=device) - k // 2).clamp(0, w - k)  # [W]
    rows = (start_y.view(h, 1) + iy.view(1, k)) % h  # [H,K]
    cols = (start_x.view(w, 1) + ix.view(1, k)) % w  # [W,K]
    flat = (rows.view(h, 1, k, 1) * w + cols.view(1, w, 1, k)).reshape(h * w, kk)  # [H*W,K*K]

    a_flat = a.permute(1, 0, 2, 3, 4).reshape(heads * b, h * w, kk)  # [heads*B,H*W,K*K]
    v_bdhw = v.permute(1, 0, 4, 2, 3).reshape(heads * b, d, h * w)  # [heads*B,D,H*W]

    outs: List[torch.Tensor] = []
    for start, end in row_ranges(h * w, row_chunk):
        idx = flat[start:end]  # [rows,K*K]
        # gather only this block's neighbours: [heads*B, D, rows, K*K]
        block = v_bdhw[:, :, idx]
        # [heads*B, rows, 1, K*K] @ [heads*B, rows, K*K, D] -> [heads*B, rows, 1, D]
        outs.append(torch.matmul(a_flat[:, start:end, None, :], block.permute(0, 2, 3, 1)).squeeze(-2))
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)

    out = out.reshape(heads, b, h, w, d).permute(1, 0, 2, 3, 4)
    return out.to(out_dtype)


def na2d_av(
    attn: torch.Tensor,
    value: torch.Tensor,
    kernel_size: int,
    *,
    dilation: int = 1,
    is_causal: bool = False,
    backend: str = "torch_reference",
    resolved: Optional[ResolvedBackend] = None,
    row_chunk: int = 0,
) -> torch.Tensor:
    """Dispatch ``na2d_av`` to the resolved backend.

    Only the subset required by the official OverLoCK dynamic block is implemented
    (``dilation=1``, ``is_causal=False``); any other value fails fast instead of being ignored.
    ``backend``/``resolved`` come from :func:`resolve_backend`; the default keeps the
    pure-function behaviour (CPU reference) so the op can be used standalone.
    """
    if dilation != 1:
        raise AttentionBackendError(f"na2d_av reference supports dilation=1 only, got {dilation}")
    if is_causal:
        raise AttentionBackendError("na2d_av reference supports is_causal=False only")
    effective = resolved.resolved if resolved is not None else backend
    if effective == "natten":
        return na2d_av_native(attn, value, kernel_size)
    if effective == "torch_reference":
        return na2d_av_reference(attn, value, kernel_size, row_chunk=row_chunk)
    raise AttentionBackendError(f"unresolved attention backend {effective!r}")


# --------------------------------------------------------------------------------------
# per-device cache (never a stale single answer)
# --------------------------------------------------------------------------------------
class BackendCache:
    """Device-keyed backend resolution cache.

    The previous implementation cached one :class:`ResolvedBackend` at construction time, so a
    model built on CPU and moved with ``.to("cuda")`` kept using the reference kernel.  This
    cache keys on the *actual* device of the input tensor and is therefore invalidated
    automatically by ``.to(...)``; :meth:`resolve` is called on every forward.
    """

    def __init__(self, requested: str = "auto"):
        if requested not in BACKENDS:
            raise AttentionBackendError(f"unknown attention backend {requested!r}; expected one of {BACKENDS}")
        self.requested = requested
        self._by_key: Dict[str, ResolvedBackend] = {}

    def resolve(self, device) -> ResolvedBackend:
        dev = torch.device(device)
        key = f"{dev.type}:{dev.index}"
        cached = self._by_key.get(key)
        if cached is None:
            cached = resolve_backend(self.requested, device=dev)
            self._by_key[key] = cached
        return cached

    def invalidate(self) -> None:
        self._by_key.clear()

    def report(self) -> dict:
        return {
            "requested": self.requested,
            "resolved_by_device": {k: v.as_dict() for k, v in sorted(self._by_key.items())},
            "n_resolutions": len(self._by_key),
        }


def backend_cache_report(caches: Iterable[BackendCache]) -> dict:
    """Merge several device caches into one report block."""
    merged: Dict[str, dict] = {}
    requested = set()
    for cache in caches:
        requested.add(cache.requested)
        for key, value in cache.report()["resolved_by_device"].items():
            merged[key] = value
    return {
        "requested": sorted(requested),
        "resolved_by_device": merged,
        "any_native": any(v["resolved"] == "natten" for v in merged.values()),
        "any_reference": any(v["resolved"] == "torch_reference" for v in merged.values()),
    }


def env_report() -> dict:
    """Compact environment facts for ``reports/v2/environment.json``."""
    return {
        "natten_importable": natten_available(),
        "natten_import_error": natten_import_error(),
        "cuda_available": _cuda_available(),
        "torch_num_threads": torch.get_num_threads(),
        "cwd": os.getcwd(),
    }
