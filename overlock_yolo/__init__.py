"""OverLoCK (xt/t/s/b) + native YOLO11/YOLO26 (n/s/m/l/x) detector, isolated implementation.

Contract: ``DESIGN_V2.md`` (copy of
``/Users/lw/Desktop/ieeeaccessyolo/OverLoCK_YOLO11_YOLO26_V2_MultiVariant_640_Git_Design_20261003.md``)::

    RGB -> native letterbox to exactly imgsz x imgsz -> float/255        (native dataloader)
        -> ONE ImageNet mean/std normalisation (buffers on the stem)
        -> isolated official OverLoCK *detection* backbone (variant xt/t/s/b)
        -> x1/x2/x3 at stride 8/16/32
        -> three 1x1 Conv-BN-SiLU adapters -> the native neck entry widths
        -> the native neck+head of the family YAML at the requested scale (YOLO11 or YOLO26)
        -> the family's own native criterion (v8DetectionLoss / E2ELoss)

The package never mutates the original repository: the official detection source is vendored
with a provenance header and the NATTEN dependency is behind an explicit backend interface with
a real differentiable reference implementation.

Entry points live in ``scripts/``; every CLI resolves its paths through
:mod:`overlock_yolo.paths` (no machine-specific path is hardcoded) and installs the pinned
Ultralytics source *before* importing it.
"""

from __future__ import annotations

from .variants import (
    ADAPTER_CHANNELS,
    FEATURE_INFO,
    OVERLOCK_VARIANTS,
    P3P4P5_CHANNELS,
    VARIANTS_ORDER,
    YOLO_FAMILY_CONTRACT,
    adapter_mapping_report,
    all_combinations,
    expected_channels,
)

__all__ = [
    "__version__",
    "OVERLOCK_VARIANTS",
    "VARIANTS_ORDER",
    "P3P4P5_CHANNELS",
    "FEATURE_INFO",
    "ADAPTER_CHANNELS",
    "YOLO_FAMILY_CONTRACT",
    "adapter_mapping_report",
    "all_combinations",
    "expected_channels",
]

__version__ = "2.0.0"

#: Backwards-compatible alias: the V1 package exposed ``BACKBONE_CONFIGS['overlock_b']``.
BACKBONE_CONFIGS = {f"overlock_{k}": v for k, v in OVERLOCK_VARIANTS.items()}

#: Backwards-compatible alias for the OverLoCK-Base detection output signature.
BACKBONE_OUTPUT_SPEC = (
    (4, OVERLOCK_VARIANTS["b"]["embed_dim"][0]),
    (8, P3P4P5_CHANNELS["b"][0]),
    (16, P3P4P5_CHANNELS["b"][1]),
    (32, P3P4P5_CHANNELS["b"][2]),
)
