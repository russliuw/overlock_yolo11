#!/usr/bin/env python
"""Write ``reports/v2/environment.json`` and ``reports/v2/compatibility.json`` (offline, read-only).

Everything here is derived from files already on disk plus the vendored source; nothing is
downloaded and no cluster/GPU is contacted.  Wheel-availability entries are *recorded research
facts* with their source URLs, not local verification.

Usage::

    python scripts/report_environment.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from overlock_yolo.paths import install_ultralytics_root, prepare_ultralytics_env, resolve_ultralytics_root  # noqa: E402

# The pinned source must be installed before anything imports ultralytics (the site-packages
# copy has no E2ELoss, which is exactly the failure this ordering prevents).
prepare_ultralytics_env()
install_ultralytics_root(resolve_ultralytics_root())

#: Target server stack from DESIGN_V2.md 3.1 (recorded facts + their sources).
TARGET_STACK = {
    "os": "Ubuntu 22.04, Linux x86_64",
    "gpu": "RTX 4090 24GB (not rented yet; nothing was measured on it)",
    "python": "3.12",
    "torch": "2.5.1 (CUDA 12.4 wheel)",
    "torchvision": "0.20.1",
    "natten": "0.17.4+torch250cu124 (cp312 manylinux x86_64 wheel)",
    "timm": "0.6.13 minimal-fix candidate (or the modern interface already adapted here)",
    "large_kernel_conv": "use_gemm=False -> real nn.Conv2d, no iGEMM extension build",
    "mmcv_mmdet": "not required by this isolated implementation",
}

WHEEL_EVIDENCE = [
    {
        "package": "torch 2.5.1 / torchvision 0.20.1 (cu124)",
        "source": "https://pytorch.org/get-started/previous-versions/",
        "claim": "a cu124 install combination exists for 2.5.1 + 0.20.1",
        "verified_how": "official install matrix (documented); not downloaded here",
    },
    {
        "package": "NATTEN 0.17.4",
        "source": "https://github.com/SHI-Labs/NATTEN/releases/tag/v0.17.4",
        "claim": "the torch250cu124 build targets Torch 2.5.X, not only 2.5.0",
        "verified_how": "release notes",
    },
    {
        "package": "natten-0.17.4+torch250cu124-cp312-cp312-linux_x86_64.whl",
        "source": "https://whl.natten.org/old/",
        "claim": "the exact CPython 3.12 / Linux x86_64 wheel is published",
        "verified_how": "wheel index listing; the ~475 MB file was NOT downloaded",
    },
    {
        "package": "NATTEN 0.17.4 API",
        "source": "https://github.com/SHI-Labs/NATTEN/blob/v0.17.4/src/natten/functional.py",
        "claim": "na2d_av is still exported in 0.17.4",
        "verified_how": "tagged source file",
    },
    {
        "package": "timm py3.11+ dataclass issue",
        "source": "https://github.com/huggingface/pytorch-image-models/issues/1723",
        "claim": "0.6.12 breaks on Python 3.11+; 0.6.13 carries the fix",
        "verified_how": "issue + v0.6.13 source",
        "caveat": "0.6.13 is a *candidate*, not a target-GPU verification result",
    },
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_digest(root: str) -> dict:
    entries = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            p = os.path.join(dirpath, name)
            entries.append(f"{os.path.relpath(p, root)} {sha256_file(p)}")
    return {
        "root": root,
        "files": len(entries),
        "combined_sha256": hashlib.sha256("\n".join(entries).encode()).hexdigest(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write environment.json + compatibility.json (offline)")
    ap.add_argument("--reports", default=os.path.join(ROOT, "reports", "v2"))
    args = ap.parse_args(argv)
    prepare_ultralytics_env()
    os.makedirs(args.reports, exist_ok=True)

    import torch

    env = {
        "generated_by": "scripts/report_environment.py",
        "role": "LOCAL verification environment -- not the target server",
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cwd": os.getcwd(),
        "project_root": ROOT,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "torchvision": __import__("torchvision").__version__,
        "numpy": __import__("numpy").__version__,
        "yaml": __import__("yaml").__version__,
        "torch_threads": torch.get_num_threads(),
        "device_used_for_all_local_checks": "cpu",
        "dtype_used_for_all_local_checks": "float32",
        "amp": False,
        "workers": 0,
        "target_server_stack": TARGET_STACK,
        "differences_from_target": [
            "Python 3.10 here vs 3.12 on the server",
            "torch 2.11 (CPU-only build) here vs 2.5.1+cu124 on the server",
            "no CUDA device locally",
            "NATTEN is not installed locally, so every local attention call used the bundled "
            "differentiable PyTorch reference (reported per model as torch_reference)",
        ],
        "not_verified": [
            "target GPU availability, memory and latency",
            "NATTEN native kernel numerics/latency",
            "AMP",
            "training convergence and any AP number",
            "DDP / multi-GPU",
        ],
    }
    try:
        import importlib.metadata as md

        env["packages"] = {
            name: md.version(name)
            for name in ("torch", "torchvision", "numpy", "PyYAML", "einops", "timm", "thop", "Pillow", "scipy")
            if _has(md, name)
        }
    except Exception:  # pragma: no cover
        env["packages"] = {}

    # --- compatibility report ---------------------------------------------------------
    from overlock_yolo import __version__
    from overlock_yolo.backbone import OVERLOCK_SOURCE
    from overlock_yolo.model import build_native_tail
    from overlock_yolo.variants import (
        FEATURE_INFO,
        OVERLOCK_VARIANTS,
        P3P4P5_CHANNELS,
        YOLO_FAMILY_CONTRACT,
        YOLO_YAML_RELPATH,
        adapter_mapping_report,
        all_combinations,
    )
    from overlock_yolo.paths import resolve_ultralytics_root

    ultra = resolve_ultralytics_root()
    install_ultralytics_root(ultra)
    vendor = os.path.join(ROOT, "vendor", "ultralytics")
    vendor_present = os.path.isdir(vendor)

    native_tails = {}
    for family in ("yolo11", "yolo26"):
        for scale in ("n", "s", "m", "l", "x"):
            _m, _s, _y, rep = build_native_tail(family, scale, 6, ultra)
            native_tails[f"{family}{scale}"] = {
                "tail_start": rep["tail_start"],
                "neck_entry_channels": rep["neck_entry_channels"],
                "detect_entry_channels": rep["detect_entry_channels"],
                "reg_max": rep["detect"]["reg_max"],
                "has_one2one": rep["detect"]["has_one2one"],
                "tail_parameters": rep["tail_parameters"],
                "c3k_per_layer": [f["c3k"] for f in rep["c3k_flags"]],
                "block_kinds_per_layer": [f["block_kinds"] for f in rep["c3k_flags"]],
            }

    compat = {
        "generated_by": "scripts/report_environment.py",
        "package_version": __version__,
        "design": "DESIGN_V2.md",
        "verdict": (
            "structure + native interface implemented and CPU-verified; "
            "target GPU, AMP, full-data training and every AP number remain UNVERIFIED"
        ),
        "overlock_source": OVERLOCK_SOURCE,
        "backbone_variant_table": OVERLOCK_VARIANTS,
        "backbone_feature_info": {k: [list(x) for x in v] for k, v in FEATURE_INFO.items()},
        "backbone_p3p4p5": {k: list(v) for k, v in P3P4P5_CHANNELS.items()},
        "yolo_family_contract": YOLO_FAMILY_CONTRACT,
        "native_yaml_relpath": {k: list(v) for k, v in YOLO_YAML_RELPATH.items()},
        "native_tail_measurements": native_tails,
        "adapter_map_examples": {
            "t+yolo11s": adapter_mapping_report("t", "yolo11", "s"),
            "t+yolo11n": adapter_mapping_report("t", "yolo11", "n"),
            "b+yolo26x": adapter_mapping_report("b", "yolo26", "x"),
        },
        "combinations": all_combinations(),
        "n_combinations": len(all_combinations()),
        "ultralytics": {
            "resolved_root": ultra,
            "vendored_snapshot": vendor_present,
            "vendor_tree": tree_digest(os.path.join(vendor, "ultralytics")) if vendor_present else None,
            "import_origin": __import__("ultralytics").__file__,
            "version": __import__("ultralytics").__version__,
            "license": "AGPL-3.0 (see THIRD_PARTY_NOTICES.md)",
            "default_resolution_order": [
                "--ultralytics-root",
                "$OVERLOCK_ULTRALYTICS_ROOT",
                "<project>/vendor/ultralytics (used by default)",
                "<project>/../ultralytics-main (author's sibling checkout, development only)",
            ],
        },
        "target_stack": TARGET_STACK,
        "wheel_evidence": WHEEL_EVIDENCE,
        "not_installed_here": {
            "natten": "the native CUDA kernel is unavailable locally; CUDA 'auto' would select it, "
            "CPU 'auto' selects the differentiable PyTorch reference",
            "mmcv/mmengine/mmdet": "deliberately not installed; the COCO bridge reuses only the parsing",
        },
        "losses": {
            "yolo11": "ultralytics.utils.loss.v8DetectionLoss (native)",
            "yolo26": "ultralytics.utils.loss.E2ELoss (native; O2M topk10 / O2O topk7+topk2=1, schedule, L1)",
            "not_used": ["E2EDetectLoss (older API)", "any hand-written loss/assigner"],
        },
        "metrics": {
            "primary": "pinned native DetectionValidator + DetMetrics",
            "keys": ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"],
            "conf_default": 0.001,
            "save_json_default": False,
            "cocoeval_role": "not used as the primary metric",
        },
        "protocol": {
            "imgsz": 640,
            "resize": "equal ratio + letterbox, padding value 114",
            "rect": False,
            "multi_scale": False,
            "scale_fill": False,
            "auto": False,
            "scaleup": False,
            "normalisation": "float/255 once in the loader, ImageNet mean/std once inside the stem",
            "p3_p4_p5_at_640": [[80, 80], [40, 40], [20, 20]],
            "anchors": 8400,
        },
        "resume": {
            "supported": True,
            "scope": "only checkpoints written by this project (format overlock-yolo-state-v1)",
            "stock_ultralytics_checkpoint": "explicitly rejected with an explanatory error",
            "restores": ["model tensors", "optimizer state", "epoch", "criterion.updates and branch weights"],
        },
    }

    for name, payload in (("environment.json", env), ("compatibility.json", compat)):
        path = os.path.join(args.reports, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"[overlock] -> {path}")
    return 0


def _has(md, name: str) -> bool:
    try:
        md.version(name)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
