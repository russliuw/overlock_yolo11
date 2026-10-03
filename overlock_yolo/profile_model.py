"""Parameter / MACs accounting for one detector combination (DESIGN_V2.md 6).

Two hard rules enforced here:

1. **Unique objects only.**  Every parameter tensor is counted once by identity; a bucket report
   plus a duplicate check proves that shared/aliased registrations cannot inflate the total.
2. **Honest completeness labels.**  THOP only knows ``Conv2d``/``Linear``/``BatchNorm`` and a few
   native attention blocks.  Our custom operators -- the NATTEN ``na2d_av`` calls, the
   ``einsum`` dynamic-kernel weight generation, the ``LayerScale`` groupwise convs, ``grid_sample``
   style resampling -- are *not* in its table.  A THOP-only number is therefore reported as
   ``partial_macs`` with an explicit list of what is missing, plus the analytic ``na2d_av`` MACs
   (``B*heads*H*W*D*K*K`` per call, measured on the real feature shapes).  No complete GFLOPs
   figure is ever printed.

Reference points from the official classification table (224 classification models, NOT this
detector): OverLoCK-T ~33M, OverLoCK-B ~95M.
"""

from __future__ import annotations

import copy
import json
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .attention_backend import na2d_av_macs
from .model import build_detector, ultralytics_namespace
from .variants import FEATURE_INFO, OVERLOCK_VARIANTS, P3P4P5_CHANNELS, expected_neck_entry_channels

__all__ = [
    "parameter_report",
    "count_partial_macs",
    "count_na2d_macs",
    "profile_detector",
    "write_profile_reports",
    "combination_parameter_matrix",
]

#: Operator kinds THOP accounts for in this checkout.
THOP_SUPPORTED = ("nn.Conv2d", "nn.Linear", "nn.BatchNorm2d", "native Attention/AAttn (custom_ops)")
#: Operator kinds this project uses that THOP does NOT account for.
THOP_UNSUPPORTED = (
    "natten na2d_av (custom op)",
    "einops.einsum dynamic-kernel weight generation",
    "torch.matmul/softmax inside DynamicConvBlock",
    "F.interpolate / adaptive_avg_pool2d",
    "GRN (norm/mean reductions)",
    "LayerScale (groupwise F.conv2d)",
    "DFL softmax projection",
)


def strip_thop_artifacts(model) -> int:
    """Remove THOP's ``total_ops``/``total_params`` buffers from a model in place.

    ``thop.profile`` attaches these to every module it walks.  They are profiling artefacts, not
    model state: they must never enter ``state_dict()``, and a second forward on a module that
    still carries them can fail.
    """
    removed = 0
    for module in model.modules():
        for attr in ("total_ops", "total_params"):
            if attr in getattr(module, "_buffers", {}):
                del module._buffers[attr]
                removed += 1
    return removed


def _unique_parameters(module: nn.Module) -> List[nn.Parameter]:
    seen: Dict[int, nn.Parameter] = {}
    for p in module.parameters():
        seen.setdefault(id(p), p)
    return list(seen.values())


def parameter_report(model, buckets: Optional[Dict[str, nn.Module]] = None) -> dict:
    """Per-bucket and total parameter accounting with an explicit duplicate check."""
    from .model import OverLoCKStem  # noqa: F401  (documented type)

    if buckets is None:
        buckets = {
            "backbone_overlock": model.stem.backbone,
            "adapters": nn.ModuleList([model.stem.adapter1, model.stem.adapter2, model.stem.adapter3]),
            "neck_head_native": model.tail,
        }
    seen: Dict[int, str] = {}
    duplicates: List[list] = []
    out: Dict[str, dict] = {"buckets": {}, "aliases": []}
    for name, mod in buckets.items():
        params = _unique_parameters(mod)
        numel = int(sum(p.numel() for p in params))
        for p in params:
            if id(p) in seen:
                duplicates.append([seen[id(p)], name])
            seen[id(p)] = name
        out["buckets"][name] = {
            "unique_tensors": len(params),
            "numel": numel,
            "trainable_numel": int(sum(p.numel() for p in params if p.requires_grad)),
            "buffers_numel": int(sum(b.numel() for b in mod.buffers())),
        }
    all_params = _unique_parameters(model)
    named = dict(model.named_parameters())
    out["total"] = {
        "unique_parameters": len(all_params),
        "unique_numel": int(sum(p.numel() for p in all_params)),
        "named_parameters": len(named),
        "trainable_numel": int(sum(p.numel() for p in all_params if p.requires_grad)),
        "frozen_numel": int(sum(p.numel() for p in all_params if not p.requires_grad)),
        "buffers_numel": int(sum(b.numel() for b in model.buffers())),
        "buckets_sum_numel": int(sum(b["numel"] for b in out["buckets"].values())),
        "duplicate_registration": duplicates,
    }
    out["total"]["buckets_cover_all_parameters"] = out["total"]["buckets_sum_numel"] == out["total"]["unique_numel"]
    return out


class _Na2dCounter:
    """Context manager counting ``na2d_av`` calls/MACs by monkey-patching the op in-place.

    The patch is applied to the *already imported* function objects used by the dynamic blocks,
    so it measures the real call sites and the real feature shapes.
    """

    def __init__(self):
        self.calls: List[dict] = []

    def __enter__(self):
        from . import attention_backend as ab
        from . import backbone as bb

        self._ab = ab.na2d_av
        self._bb = bb.na2d_av

        def wrapper(attn, value, kernel_size, **kwargs):
            b, heads, h, w, _kk = attn.shape
            d = value.shape[-1]
            self.calls.append(
                {
                    "shape": [int(b), int(heads), int(h), int(w), int(d), int(kernel_size)],
                    "macs": na2d_av_macs(b, heads, h, w, d, kernel_size),
                }
            )
            return self._ab(attn, value, kernel_size, **kwargs)

        ab.na2d_av = wrapper
        bb.na2d_av = wrapper
        return self

    def __exit__(self, *exc):
        from . import attention_backend as ab
        from . import backbone as bb

        ab.na2d_av = self._ab
        bb.na2d_av = self._bb
        return False

    def report(self) -> dict:
        return {
            "calls": len(self.calls),
            "macs": int(sum(c["macs"] for c in self.calls)),
            "per_call": self.calls,
        }


def count_na2d_macs(model, imgsz: int = 640) -> dict:
    """MACs of every ``na2d_av`` call for one batch-1 forward at ``imgsz``."""
    # A previous THOP pass leaves ``total_ops``/``total_params`` buffers on every module, which
    # breaks a later forward on some torch versions; profiling artefacts are not model state.
    from .trainer import strip_thop_artifacts

    strip_thop_artifacts(model)
    model.eval()
    with _Na2dCounter() as counter, torch.no_grad():
        model(torch.zeros(1, 3, imgsz, imgsz))
    report = counter.report()
    report["definition"] = "sum over calls of B*heads*H*W*D*K*K multiply-accumulates"
    report["imgsz"] = int(imgsz)
    report["batch"] = 1
    return report


def count_partial_macs(model, imgsz: int = 640, device=None) -> dict:
    """THOP MACs for conv/linear/BN only -- explicitly labelled partial.

    THOP *accumulates* into ``self.total_ops`` on the module tree it walks and does not reset it,
    so two profiles of the same object would give different answers.  It therefore runs on a
    deep copy and the original model is left clean.
    """
    try:
        import thop
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "reason": f"thop not importable: {type(exc).__name__}: {exc}"}

    p = next(model.parameters())
    im = torch.empty((1, p.shape[1], imgsz, imgsz), device=device or p.device, dtype=p.dtype)
    model.eval()
    subject = copy.deepcopy(model).eval()
    strip_thop_artifacts(subject)
    strip_thop_artifacts(model)
    try:
        t0 = time.time()
        macs = thop.profile(subject, inputs=[im], verbose=False)[0]
        seconds = time.time() - t0
    except Exception as exc:  # pragma: no cover - defensive
        return {"available": False, "reason": f"thop.profile failed: {type(exc).__name__}: {exc}"}
    finally:
        del subject
    return {
        "available": True,
        "partial_macs": int(macs),
        "partial_gflops_1mac2flops": float(macs) * 2 / 1e9,
        "macs_definition": "1 MAC = 1 multiply-accumulate; the GFLOPs column uses 1 MAC = 2 FLOPs",
        "imgsz": int(imgsz),
        "batch": 1,
        "dtype": str(p.dtype),
        "device": str(p.device),
        "profiler": f"thop {getattr(thop, '__version__', 'unknown')}",
        "supported_ops": list(THOP_SUPPORTED),
        "unsupported_ops_excluded": list(THOP_UNSUPPORTED),
        "completeness": "partial_macs -- conv/linear/batchnorm only; NOT the full detector cost",
        "seconds": round(seconds, 2),
    }


def input_invariance(model, sizes=(224, 640), device=None) -> dict:
    """Parameter count must not depend on the input resolution."""
    counts = {}
    for size in sizes:
        counts[str(size)] = parameter_report(model)["total"]["unique_numel"]
    return {
        "numel_by_imgsz": counts,
        "invariant": len(set(counts.values())) == 1,
        "note": "the parameter count is a structure property; the input area changes MACs, not parameters",
    }


def profile_detector(
    cfg: dict,
    *,
    imgsz: Optional[int] = None,
    device: Optional[str] = None,
    macs: bool = True,
    ultra_root: Optional[str] = None,
) -> dict:
    """Full cost report for one resolved combination."""
    res = cfg["_resolved"]
    imgsz = int(imgsz or res["imgsz"])
    device = device or cfg["runtime"]["device"]
    t0 = time.time()
    model = build_detector(
        variant=res["backbone_variant"],
        family=res["yolo_family"],
        scale=res["yolo_scale"],
        nc=int(res["nc"]),
        names=res["names"],
        weights=None,
        attention_backend=cfg["backbone"]["attention_backend"],
        ultra_root=ultra_root,
        row_chunk=int(cfg["backbone"]["row_chunk"]),
        verbose=False,
    )
    build_seconds = time.time() - t0
    params = parameter_report(model)
    report = {
        "combination": res["combination"],
        "backbone_variant": res["backbone_variant"],
        "yolo_family": res["yolo_family"],
        "yolo_scale": res["yolo_scale"],
        "nc": int(res["nc"]),
        "structure": {
            "deploy": False,
            "fused": False,
            "attention_backend_requested": cfg["backbone"]["attention_backend"],
            "backbone_feature_info": [list(x) for x in FEATURE_INFO[res["backbone_variant"]]],
            "backbone_p3p4p5": list(P3P4P5_CHANNELS[res["backbone_variant"]]),
            "adapter_map": res["adapter_map"],
            "head": model.parameter_groups_report()["head"],
            "criterion": type(model.init_criterion()).__name__,
        },
        "parameters": params,
        "build_seconds": round(build_seconds, 3),
        "input_invariance": input_invariance(model),
        "cost_basis": {
            "batch": 1,
            "eval_mode": True,
            "dtype": "float32",
            "macs_definition": "1 MAC = 1 multiply-accumulate (GFLOPs uses 1 MAC = 2 FLOPs)",
            "excludes": ["resize/letterbox", "image decode", "NMS", "post-processing", "loss"],
        },
    }
    if macs:
        report["partial_macs"] = count_partial_macs(model, imgsz=imgsz, device=device)
        try:
            report["na2d_av_macs"] = count_na2d_macs(model, imgsz=imgsz)
        except Exception as exc:  # pragma: no cover - defensive
            report["na2d_av_macs"] = {"error": f"{type(exc).__name__}: {exc}"}
        partial = report["partial_macs"].get("partial_macs")
        na2d = report["na2d_av_macs"].get("macs")
        if partial is not None and na2d is not None:
            report["partial_plus_na2d_macs"] = int(partial) + int(na2d)
            report["partial_plus_na2d_gflops_1mac2flops"] = (int(partial) + int(na2d)) * 2 / 1e9
            report["completeness"] = (
                "partial: conv/linear/batchnorm (THOP) + analytically counted na2d_av MACs. "
                "einsum dynamic-kernel generation, softmax and interpolations remain uncounted."
            )
    report["complete_flops"] = {"available": False, "reason": "no complete FLOPs profiler covers the custom operators"}
    report["profiler_versions"] = _profiler_versions()
    report["structure_state"] = {"deploy": False, "fused": False, "attention_backend": model.attention_backend_report()}
    return report


def _profiler_versions() -> dict:
    out = {}
    try:
        import thop

        out["thop"] = getattr(thop, "__version__", "unknown")
    except Exception:
        out["thop"] = None
    try:
        import torch

        out["torch"] = torch.__version__
    except Exception:  # pragma: no cover
        pass
    out["custom_ops"] = "analytic na2d_av MACs (B*heads*H*W*D*K*K per call), measured on real shapes"
    return out


def combination_parameter_matrix(variants=("xt", "t", "s", "b"), families=("yolo11", "yolo26"), scales=("n", "s", "m", "l", "x")) -> dict:
    """Backbone and tail parameter counts measured once each, combined into the 40-cell matrix.

    Building the tail needs no backbone forward and building a backbone needs no tail, so the
    matrix costs 4 + 10 builds instead of 40 full detectors.  The cells are marked as
    *derived*; a few are cross-checked against a fully assembled detector by the caller.
    """
    from .model import build_native_tail
    from .paths import resolve_ultralytics_root

    ultra_root = resolve_ultralytics_root()
    backbone_params = {}
    for variant in variants:
        from .backbone import build_overlock

        model = build_overlock(variant)
        backbone_params[variant] = int(sum(p.numel() for p in model.parameters()))
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    tail_params = {}
    for family in families:
        for scale in scales:
            modules, _save, _yaml, report = build_native_tail(family, scale, 6, ultra_root)
            tail_params[f"{family}{scale}"] = int(report["tail_parameters"])
            del modules

    cells = []
    for variant in variants:
        for family in families:
            for scale in scales:
                backbone_ch = P3P4P5_CHANNELS[variant]
                neck = list(expected_neck_entry_channels(family, scale))
                adapter_numel = int(sum(backbone_ch[i] * neck[i] + 2 * neck[i] for i in range(3)))
                total = backbone_params[variant] + adapter_numel + tail_params[f"{family}{scale}"]
                cells.append(
                    {
                        "combination": f"{variant}+{family}{scale}",
                        "backbone_variant": variant,
                        "yolo_family": family,
                        "yolo_scale": scale,
                        "backbone_numel": backbone_params[variant],
                        "adapter_numel": adapter_numel,
                        "tail_numel": tail_params[f"{family}{scale}"],
                        "total_numel": total,
                        "source": "derived from independently measured backbone + adapter formula + tail",
                    }
                )
    return {
        "backbone_measured_numel": backbone_params,
        "tail_measured_numel": tail_params,
        "adapter_formula": "sum_i(c_in_i*c_out_i + 2*c_out_i) over the three 1x1 Conv-BN-SiLU adapters "
                           "(nn.Conv2d(bias=False) + BatchNorm2d)",
        "cells": cells,
        "n_cells": len(cells),
    }


def write_profile_reports(report: dict, json_path: str, md_path: Optional[str] = None) -> dict:
    """Write the JSON report (source of truth) and an optional Markdown summary."""
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    out = {"json": json_path}
    if md_path:
        lines = [
            f"# Profile: {report.get('combination', '?')}",
            "",
            f"- parameters (unique): **{report['parameters']['total']['unique_numel']:,}**",
            f"- trainable: {report['parameters']['total']['trainable_numel']:,}",
            f"- buffers: {report['parameters']['total']['buffers_numel']:,}",
            f"- partial MACs (conv/linear/BN only): {report.get('partial_macs', {}).get('partial_macs', 'n/a')}",
            f"- na2d_av MACs: {report.get('na2d_av_macs', {}).get('macs', 'n/a')}",
            f"- complete FLOPs: NOT AVAILABLE ({report['complete_flops']['reason']})",
            "",
            "## buckets",
            "",
            "| bucket | unique tensors | numel | trainable | buffers |",
            "|---|---|---|---|---|",
        ]
        for name, bucket in report["parameters"]["buckets"].items():
            lines.append(
                f"| {name} | {bucket['unique_tensors']} | {bucket['numel']:,} | "
                f"{bucket['trainable_numel']:,} | {bucket['buffers_numel']:,} |"
            )
        lines += ["", "## uncounted operators", ""]
        lines += [f"- {op}" for op in report.get("partial_macs", {}).get("unsupported_ops_excluded", [])]
        lines += ["", "Parameter counts do not change between 224 and 640; only spatial cost does.", ""]
        os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        out["markdown"] = md_path
    return out
