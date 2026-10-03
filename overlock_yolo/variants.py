"""OverLoCK variant table + YOLO11/YOLO26 tail contract (DESIGN_V2.md 4.2/4.3/6).

This is the single place where the 4 x 2 x 5 = 40 structure combinations are enumerated and
checked.  Values are transcribed from the *official detection source* that is vendored as
:mod:`overlock_yolo.backbone`, not extrapolated from OverLoCK-Base:

``/Users/lw/Documents/CNN-Mamba/OverLoCK-main/detection/models/overlock.py`` lines 861-946.

Only ``depth``/``sub_depth``/``embed_dim``/``sub_num_heads`` differ between variants; the
official factories keep ``kernel_size=[17, 15, 13, 7]``, ``mlp_ratio=[4, 4, 4, 4]``,
``sub_mlp_ratio=[3, 3]`` and the ``smk_size=5`` / ``ls_init_value=[None, None, 1, 1]`` /
``res_scale=True`` defaults for all four.

Derived detection output channels (``x1``/``x2``/``x3`` consumed by the adapter):

    P3 = embed_dim[1]
    P4 = embed_dim[2] + embed_dim[3] // 4
    P5 = embed_dim[3] + embed_dim[3] // 4

with the second term coming from the context channel concatenated in
``forward_sub_features`` (``extra_norm[2]``) and the ``h_proj`` fusion at ``extra_norm[3]``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

__all__ = [
    "OVERLOCK_VARIANTS",
    "ADAPTER_CHANNELS",
    "P3P4P5_CHANNELS",
    "FEATURE_INFO",
    "YOLO_CHANNEL_TABLE",
    "YOLO_YAML_RELPATH",
    "variant_config",
    "variant_ids",
    "expected_channels",
    "yolo_yaml_relpath",
    "yolo_scale_row",
    "expected_neck_entry_channels",
    "expected_detect_entry_channels",
    "all_combinations",
    "adapter_channels_for",
    "adapter_mapping_report",
    "YOLO_FAMILY_CONTRACT",
    "normalize_variant",
    "normalize_family",
    "normalize_scale",
    "FAMILY_ALIASES",
    "SCALE_ALIASES",
    "VARIANTS_ORDER",
]

#: Deterministic enumeration order of the four variants.
VARIANTS_ORDER: Tuple[str, ...] = ("xt", "t", "s", "b")

#: Per-variant official construction arguments (detection factory).
OVERLOCK_VARIANTS: Dict[str, dict] = {
    "xt": dict(
        depth=[2, 2, 3, 2],
        sub_depth=[6, 2],
        embed_dim=[56, 112, 256, 336],
        kernel_size=[17, 15, 13, 7],
        mlp_ratio=[4, 4, 4, 4],
        sub_num_heads=[4, 6],
        sub_mlp_ratio=[3, 3],
    ),
    "t": dict(
        depth=[4, 4, 6, 2],
        sub_depth=[12, 2],
        embed_dim=[64, 128, 256, 512],
        kernel_size=[17, 15, 13, 7],
        mlp_ratio=[4, 4, 4, 4],
        sub_num_heads=[4, 8],
        sub_mlp_ratio=[3, 3],
    ),
    "s": dict(
        depth=[6, 6, 8, 3],
        sub_depth=[16, 3],
        embed_dim=[64, 128, 320, 512],
        kernel_size=[17, 15, 13, 7],
        mlp_ratio=[4, 4, 4, 4],
        sub_num_heads=[8, 16],
        sub_mlp_ratio=[3, 3],
    ),
    "b": dict(
        depth=[8, 8, 10, 4],
        sub_depth=[20, 4],
        embed_dim=[80, 160, 384, 576],
        kernel_size=[17, 15, 13, 7],
        mlp_ratio=[4, 4, 4, 4],
        sub_num_heads=[6, 9],
        sub_mlp_ratio=[3, 3],
    ),
}

#: Adapter output widths for the default YOLO11s tail (kept for backwards compatibility with
#: V1 call sites); per-combination widths come from :func:`adapter_channels_for`.
ADAPTER_CHANNELS: Tuple[int, int, int] = (256, 256, 512)


def _p3p4p5(embed_dim: List[int]) -> Tuple[int, int, int]:
    return (int(embed_dim[1]), int(embed_dim[2] + embed_dim[3] // 4), int(embed_dim[3] + embed_dim[3] // 4))


#: Detection backbone output channels actually used by YOLO, keyed by variant.
P3P4P5_CHANNELS: Dict[str, Tuple[int, int, int]] = {v: _p3p4p5(c["embed_dim"]) for v, c in OVERLOCK_VARIANTS.items()}

#: Full four-level signature ``(stride, channels)`` of ``forward_features`` per variant.
FEATURE_INFO: Dict[str, Tuple[Tuple[int, int], ...]] = {
    v: ((4, int(c["embed_dim"][0])), (8, P3P4P5_CHANNELS[v][0]), (16, P3P4P5_CHANNELS[v][1]), (32, P3P4P5_CHANNELS[v][2]))
    for v, c in OVERLOCK_VARIANTS.items()
}

#: DESIGN_V2.md 4.3 -- scale -> (depth, width, max_channels) and the resulting native neck /
#: Detect entry widths.  Both YOLO11 and YOLO26 currently share this table, but the tables are
#: still kept per family so a future divergence cannot silently reuse the wrong numbers.
YOLO_CHANNEL_TABLE: Dict[str, Dict[str, dict]] = {
    "yolo11": {
        "n": dict(depth=0.50, width=0.25, max_channels=1024, neck=(128, 128, 256), detect=(64, 128, 256)),
        "s": dict(depth=0.50, width=0.50, max_channels=1024, neck=(256, 256, 512), detect=(128, 256, 512)),
        "m": dict(depth=0.50, width=1.00, max_channels=512, neck=(512, 512, 512), detect=(256, 512, 512)),
        "l": dict(depth=1.00, width=1.00, max_channels=512, neck=(512, 512, 512), detect=(256, 512, 512)),
        "x": dict(depth=1.00, width=1.50, max_channels=512, neck=(768, 768, 768), detect=(384, 768, 768)),
    },
    "yolo26": {
        "n": dict(depth=0.50, width=0.25, max_channels=1024, neck=(128, 128, 256), detect=(64, 128, 256)),
        "s": dict(depth=0.50, width=0.50, max_channels=1024, neck=(256, 256, 512), detect=(128, 256, 512)),
        "m": dict(depth=0.50, width=1.00, max_channels=512, neck=(512, 512, 512), detect=(256, 512, 512)),
        "l": dict(depth=1.00, width=1.00, max_channels=512, neck=(512, 512, 512), detect=(256, 512, 512)),
        "x": dict(depth=1.00, width=1.50, max_channels=512, neck=(768, 768, 768), detect=(384, 768, 768)),
    },
}

#: Relative path of the native model YAML inside the pinned Ultralytics source root.
YOLO_YAML_RELPATH: Dict[str, str] = {
    "yolo11": ("ultralytics", "cfg", "models", "11", "yolo11.yaml"),
    "yolo26": ("ultralytics", "cfg", "models", "26", "yolo26.yaml"),
}

#: Family-specific head/criterion contract (DESIGN_V2.md 10.2).
YOLO_FAMILY_CONTRACT: Dict[str, dict] = {
    "yolo11": dict(reg_max=16, branches="one-to-many", criterion="v8DetectionLoss", nms=True),
    "yolo26": dict(reg_max=1, branches="one-to-many+one-to-one", criterion="E2ELoss", nms=False),
}


def variant_ids() -> Tuple[str, ...]:
    return VARIANTS_ORDER


def normalize_variant(value: str) -> str:
    """Accept a variant id (``xt``/``t``/``s``/``b``) case-insensitively, else fail fast."""
    key = str(value).strip().lower()
    if key not in OVERLOCK_VARIANTS:
        raise ValueError(f"unknown backbone variant {value!r}; expected one of {VARIANTS_ORDER}")
    return key


#: Accepted spellings for the YOLO family selector.
FAMILY_ALIASES = {
    "yolo11": "yolo11",
    "11": "yolo11",
    "yolov11": "yolo11",
    "yolo26": "yolo26",
    "26": "yolo26",
    "yolov26": "yolo26",
}

#: Accepted spellings for the YOLO scale selector (bare letter or ``<family><letter>``).
SCALE_ALIASES = {s: s for s in ("n", "s", "m", "l", "x")}
SCALE_ALIASES.update({f"{fam}{s}": s for fam in ("yolo11", "yolo26", "yolov11", "yolov26") for s in ("n", "s", "m", "l", "x")})


def normalize_family(value: str) -> str:
    key = str(value).strip().lower()
    if key not in FAMILY_ALIASES:
        raise ValueError(f"unknown yolo family {value!r}; expected one of ('yolo11', 'yolo26')")
    return FAMILY_ALIASES[key]


def normalize_scale(value: str) -> str:
    key = str(value).strip().lower()
    if key not in SCALE_ALIASES:
        raise ValueError(f"unknown yolo scale {value!r}; expected one of ('n', 's', 'm', 'l', 'x')")
    return SCALE_ALIASES[key]


def variant_config(variant: str) -> dict:
    """Deep copy of the official construction arguments for ``variant``."""
    if variant not in OVERLOCK_VARIANTS:
        raise KeyError(f"unknown backbone variant {variant!r}; expected one of {variant_ids()}")
    return {k: (list(v) if isinstance(v, list) else v) for k, v in OVERLOCK_VARIANTS[variant].items()}


def expected_channels(variant: str) -> Tuple[int, int, int]:
    """``(P3, P4, P5)`` backbone output channels for ``variant`` (stride 8/16/32)."""
    if variant not in P3P4P5_CHANNELS:
        raise KeyError(f"unknown backbone variant {variant!r}; expected one of {variant_ids()}")
    return P3P4P5_CHANNELS[variant]


def yolo_yaml_relpath(family: str) -> Tuple[str, ...]:
    if family not in YOLO_YAML_RELPATH:
        raise KeyError(f"unknown yolo family {family!r}; expected one of {tuple(YOLO_YAML_RELPATH)}")
    return YOLO_YAML_RELPATH[family]


def yolo_scale_row(family: str, scale: str) -> dict:
    if family not in YOLO_CHANNEL_TABLE:
        raise KeyError(f"unknown yolo family {family!r}")
    if scale not in YOLO_CHANNEL_TABLE[family]:
        raise KeyError(f"unknown yolo scale {scale!r}; expected one of {tuple(YOLO_CHANNEL_TABLE[family])}")
    return dict(YOLO_CHANNEL_TABLE[family][scale])


def expected_neck_entry_channels(family: str, scale: str) -> Tuple[int, int, int]:
    """Adapters must produce exactly these widths (native neck consumers, yaml rows 4/6/10)."""
    return tuple(yolo_scale_row(family, scale)["neck"])


def expected_detect_entry_channels(family: str, scale: str) -> Tuple[int, int, int]:
    """Detect head input widths (native yaml rows 16/19/22)."""
    return tuple(yolo_scale_row(family, scale)["detect"])


def adapter_channels_for(variant: str, family: str, scale: str) -> Tuple[int, int, int]:
    """The three 1x1 adapter output widths for one combination (== neck entry widths)."""
    return expected_neck_entry_channels(family, scale)


def adapter_mapping_report(variant: str, family: str, scale: str) -> List[dict]:
    """Per-level ``backbone_ch -> adapter_ch`` mapping, used in reports and assertions."""
    src = expected_channels(variant)
    dst = expected_neck_entry_channels(family, scale)
    strides = (8, 16, 32)
    return [
        {"level": f"P{i + 3}", "stride": strides[i], "backbone_channels": int(src[i]), "adapter_channels": int(dst[i])}
        for i in range(3)
    ]


def all_combinations() -> List[dict]:
    """The 40 legal ``variant x family x scale`` combinations."""
    return [
        {"backbone_variant": v, "yolo_family": f, "yolo_scale": s, "combination": f"{v}+{f}{s}"}
        for v in VARIANTS_ORDER
        for f in ("yolo11", "yolo26")
        for s in ("n", "s", "m", "l", "x")
    ]
