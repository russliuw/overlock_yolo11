"""Strict experiment configuration (DESIGN_V2.md 4.4).

One YAML file fully describes an experiment::

    backbone:  {family, variant, weights, pretrained, deploy, attention_backend, row_chunk, ...}
    yolo:      {family, scale, nc, names}
    runtime:   {device, workers, seed}
    train:     {imgsz, rect, multi_scale, scale, amp, epochs, batch, ...}

Rules enforced here (not in the CLI, not in the model):

* Duplicate YAML keys are a hard error -- a later ``scale:`` must never silently win.
* Unknown keys / unknown ``variant`` / ``family`` / ``scale`` fail fast, naming the path.
* ``yolo.scale`` (an architecture spec: n/s/m/l/x) and ``train.scale`` (the native geometric
  augmentation factor, a float) are separate namespaces.  A bare top-level ``scale:`` is
  rejected so neither meaning can be inferred by accident.
* CLI values override file values **explicitly**; a CLI default never overwrites the file.

Every entry point writes the *fully resolved* configuration next to its outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import yaml

from .paths import expand_path_vars
from .paths import project_root as project_root_default
from .paths import resolve_checkpoint_path, resolve_data_yaml_with_candidates
from .variants import YOLO_FAMILY_CONTRACT, adapter_mapping_report, expected_channels
from .variants import normalize_family, normalize_scale, normalize_variant

__all__ = [
    "ConfigError",
    "VARIANTS",
    "FAMILIES",
    "SCALES",
    "DEFAULT_VARIANT",
    "DEFAULT_FAMILY",
    "DEFAULT_SCALE",
    "EPOCHS_SENTINEL",
    "load_yaml_strict",
    "load_experiment",
    "validate_config",
    "apply_cli_overrides",
    "resolved_summary",
    "write_resolved_config",
    "build_arg_parser",
    "experiment_parser",
]

VARIANTS = ("xt", "t", "s", "b")
FAMILIES = ("yolo11", "yolo26")
SCALES = ("n", "s", "m", "l", "x")

DEFAULT_VARIANT = "t"
DEFAULT_FAMILY = "yolo11"
DEFAULT_SCALE = "s"

#: Native default epoch count.  ``yolo26.yaml``'s O2M/O2O schedule divides by ``epochs - 1``,
#: so a resolved ``train.epochs`` must be >= 2 whenever a YOLO26 tail is built.
EPOCHS_SENTINEL = 100


class ConfigError(ValueError):
    """Raised for duplicated, unknown, or contradictory configuration entries."""


# --------------------------------------------------------------------------------------
# strict YAML
# --------------------------------------------------------------------------------------
class _StrictLoader(yaml.SafeLoader):
    """``SafeLoader`` that refuses duplicate mapping keys."""

    name = "<yaml>"


def _no_duplicates(loader: _StrictLoader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigError(
                f"duplicate key {key!r} in {getattr(loader, 'name', '<yaml>')} line "
                f"{key_node.start_mark.line + 1}; refusing to let a later value silently win"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


def load_yaml_strict(path: str) -> dict:
    """Load a YAML mapping, rejecting duplicate keys."""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise ConfigError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    loader = _StrictLoader(text)
    loader.name = path
    try:
        data = loader.get_single_data()
    except ConfigError:
        raise
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    finally:
        loader.dispose()
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")
    return data


# --------------------------------------------------------------------------------------
# schema (defaults are what an absent key means -- never applied over a present key)
# --------------------------------------------------------------------------------------
SCHEMA: Dict[str, Dict[str, Any]] = {
    "backbone": {
        "family": ("str", "overlock"),
        "variant": ("str", DEFAULT_VARIANT),
        "weights": ("path?", None),
        "pretrained": ("bool", False),
        "deploy": ("bool", False),
        "attention_backend": ("str", "auto"),
        "row_chunk": ("int", 0),
        "strict_shape": ("bool", True),
        "freeze": ("bool", False),
        "lr_mult": ("float", 1.0),
        "bn_eval": ("bool", False),
        "strict_weights": ("bool", True),
    },
    "yolo": {
        "family": ("str", DEFAULT_FAMILY),
        "scale": ("str", DEFAULT_SCALE),
        "nc": ("int?", None),
        "names": ("dict?", None),
    },
    "runtime": {
        "device": ("str", "cpu"),
        "workers": ("int", 0),
        "seed": ("int", 0),
        "torch_threads": ("int?", None),
        "verbose": ("bool", False),
    },
    "train": {
        "imgsz": ("int", 640),
        "rect": ("bool", False),
        "multi_scale": ("boolorfloat", False),
        "scale": ("float", 0.5),
        "amp": ("bool", False),
        "epochs": ("int", EPOCHS_SENTINEL),
        "batch": ("int", 16),
        "optimizer": ("str", "auto"),
        "lr0": ("float?", None),
        "weight_decay": ("float?", None),
        "val_scaleup": ("bool", False),
        "conf": ("float?", None),
        "iou": ("float?", None),
        "max_det": ("int?", None),
        "nms": ("bool?", None),
    },
    "data": {
        "yaml": ("path?", None),
        "view": ("path?", None),
        "limit_per_split": ("int?", None),
    },
}

_TYPE_CHECKERS = {
    "str": lambda v: isinstance(v, str) and not isinstance(v, bool),
    "bool": lambda v: isinstance(v, bool),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "dict": lambda v: isinstance(v, dict),
    "boolorfloat": lambda v: (isinstance(v, bool)) or (isinstance(v, (int, float)) and not isinstance(v, bool)),
    "path": lambda v: isinstance(v, str) and bool(v),
}


def _check_type(kind: str, value: Any, key: str) -> Any:
    optional = kind.endswith("?")
    base = kind[:-1] if optional else kind
    if value is None and optional:
        return None
    if not _TYPE_CHECKERS[base](value):
        raise ConfigError(f"{key}: expected {base}{' or null' if optional else ''}, got {value!r} ({type(value).__name__})")
    return value


def _defaults() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for section, keys in SCHEMA.items():
        out[section] = {}
        for key, (kind, default) in keys.items():
            out[section][key] = default
    return out


# --------------------------------------------------------------------------------------
# load + validate
# --------------------------------------------------------------------------------------
def load_experiment(
    config_path: Optional[str] = None,
    *,
    project_root: Optional[str] = None,
    cli_overrides: Optional[dict] = None,
    require_weights_file: bool = False,
) -> dict:
    """Load ``config_path`` (optional), apply explicit CLI overrides, validate, resolve.

    Returns the fully resolved configuration as a plain dict with two extra top-level keys:
    ``_meta`` (provenance + resolution notes) and ``_resolved`` (absolute, backend-resolved
    values such as the checkpoint path or the data YAML).
    """
    raw = load_yaml_strict(config_path) if config_path else {}
    if "scale" in raw:
        raise ConfigError(
            "top-level 'scale:' is ambiguous: use yolo.scale for the architecture spec "
            "(n/s/m/l/x) and train.scale for the geometric augmentation factor"
        )
    known_sections = set(SCHEMA)
    unknown_sections = sorted(set(raw) - known_sections)
    if unknown_sections:
        raise ConfigError(f"unknown config section(s) {unknown_sections}; expected {sorted(known_sections)}")

    cfg = _defaults()
    for section, values in raw.items():
        if not isinstance(values, dict):
            raise ConfigError(f"section {section!r} must be a mapping, got {type(values).__name__}")
        unknown = sorted(set(values) - set(SCHEMA[section]))
        if unknown:
            raise ConfigError(f"unknown key(s) {unknown} in section {section!r}; allowed: {sorted(SCHEMA[section])}")
        for key, value in values.items():
            cfg[section][key] = _check_type(SCHEMA[section][key][0], value, f"{section}.{key}")

    cli = {k: v for k, v in (cli_overrides or {}).items() if v is not None}
    applied = apply_cli_overrides(cfg, cli)

    cfg["_meta"] = {
        "config_path": os.path.abspath(config_path) if config_path else None,
        "cli_overrides_applied": applied,
        "defaults_used_for_absent_keys": True,
    }
    resolved = validate_config(cfg, project_root=project_root, require_weights_file=require_weights_file)
    cfg["_resolved"] = resolved
    return cfg


#: CLI argument name -> dotted config path.  Arguments not listed here are entry-point specific
#: (``--config``, ``--out`` ...) and never touch the experiment config.
CLI_TO_CONFIG = {
    "variant": "backbone.variant",
    "weights": "backbone.weights",
    "pretrained": "backbone.pretrained",
    "attention_backend": "backbone.attention_backend",
    "row_chunk": "backbone.row_chunk",
    "freeze_backbone": "backbone.freeze",
    "backbone_bn_eval": "backbone.bn_eval",
    "backbone_lr_mult": "backbone.lr_mult",
    "yolo_family": "yolo.family",
    "yolo_scale": "yolo.scale",
    "nc": "yolo.nc",
    "device": "runtime.device",
    "workers": "runtime.workers",
    "seed": "runtime.seed",
    "torch_threads": "runtime.torch_threads",
    "verbose": "runtime.verbose",
    "imgsz": "train.imgsz",
    "rect": "train.rect",
    "multi_scale": "train.multi_scale",
    "aug_scale": "train.scale",
    "amp": "train.amp",
    "epochs": "train.epochs",
    "batch": "train.batch",
    "optimizer": "train.optimizer",
    "lr0": "train.lr0",
    "weight_decay": "train.weight_decay",
    "val_scaleup": "train.val_scaleup",
    "conf": "train.conf",
    "iou": "train.iou",
    "max_det": "train.max_det",
    "nms": "train.nms",
    "data_yaml": "data.yaml",
    "view": "data.view",
    "limit_per_split": "data.limit_per_split",
}


def apply_cli_overrides(cfg: dict, cli: dict) -> List[dict]:
    """Set only the config paths the user actually passed on the command line."""
    applied = []
    for arg, path in CLI_TO_CONFIG.items():
        if arg not in cli or cli[arg] is None:
            continue
        section, key = path.split(".")
        kind = SCHEMA[section][key][0]
        value = _check_type(kind, cli[arg], path)
        if cfg[section].get(key) != value:
            applied.append({"cli": f"--{arg.replace('_', '-')}", "path": path, "value": value})
        cfg[section][key] = value
    return applied


def validate_config(cfg: dict, *, project_root: Optional[str] = None, require_weights_file: bool = False) -> dict:
    """Normalise + cross-check the three independent selectors; returns the ``_resolved`` block."""
    b, y, r, t, d = cfg["backbone"], cfg["yolo"], cfg["runtime"], cfg["train"], cfg["data"]

    if str(b["family"]).lower() not in ("overlock",):
        raise ConfigError(f"backbone.family must be 'overlock', got {b['family']!r}")
    b["variant"] = normalize_variant(b["variant"])
    y["family"] = normalize_family(y["family"])
    y["scale"] = normalize_scale(y["scale"])

    if b["attention_backend"] not in ("auto", "natten", "torch_reference"):
        raise ConfigError(
            f"backbone.attention_backend must be auto|natten|torch_reference, got {b['attention_backend']!r}"
        )
    if not b["deploy"]:
        pass  # deploy=True is rejected by the backbone factory; kept explicit in the config
    if int(b["row_chunk"]) < 0:
        raise ConfigError("backbone.row_chunk must be >= 0 (0 = all rows at once)")

    if int(t["imgsz"]) <= 0:
        raise ConfigError(f"train.imgsz must be positive, got {t['imgsz']}")
    if t["rect"]:
        raise ConfigError(
            "train.rect must be False: this project trains/validates/predicts at exactly 640x640 "
            "(equal-ratio resize + letterbox). Rectangular batches are not a supported protocol."
        )
    ms = t["multi_scale"]
    if isinstance(ms, bool):
        if ms:
            raise ConfigError("train.multi_scale must be False; the 640x640 protocol is fixed")
    elif float(ms) != 0.0:
        raise ConfigError(f"train.multi_scale must be False or 0.0, got {ms!r}")
    if not (0.0 <= float(t["scale"]) <= 1.0) and not (1.0 <= float(t["scale"]) <= 2.0):
        raise ConfigError(f"train.scale (geometric augmentation) must be in [0,1] or [1,2], got {t['scale']}")
    if y["family"] == "yolo26" and int(t["epochs"]) < 2:
        raise ConfigError(
            "yolo26's native E2ELoss schedule divides by (epochs - 1); train.epochs must be >= 2 "
            f"(got {t['epochs']})"
        )
    if t["nms"] is not None and y["family"] == "yolo11" and t["nms"] is False:
        raise ConfigError("yolo11 has no one-to-one head: train.nms=false would request an unsupported end2end path")

    nc = 6 if y["nc"] is None else int(y["nc"])
    if nc <= 0:
        raise ConfigError(f"yolo.nc must be positive, got {nc}")

    names = y["names"]
    if names is not None:
        names = {int(k): str(v) for k, v in names.items()}
        if sorted(names) != list(range(nc)):
            raise ConfigError(f"yolo.names keys must be exactly 0..{nc - 1}, got {sorted(names)}")
    else:
        names = {i: f"{i}" for i in range(nc)}

    root_for_paths = project_root or project_root_default()
    data_yaml = resolve_data_yaml_with_candidates(
        expand_path_vars(d["yaml"], root_for_paths) if d["yaml"] else None, project_root_=project_root
    )
    weights = resolve_checkpoint_path(
        b["variant"], expand_path_vars(b["weights"], root_for_paths) if b["weights"] else None, project_root_=project_root
    )
    if d.get("view"):
        d["view"] = os.path.normpath(os.path.abspath(expand_path_vars(d["view"], root_for_paths)))
    if b["weights"] and weights is not None and not os.path.isfile(weights):
        raise ConfigError(
            f"backbone.weights points at {weights}, which does not exist. This project never "
            "downloads weights: place the file, fix the path, or set backbone.pretrained=false "
            "for an explicitly random structure test."
        )
    if b["pretrained"] and weights is None:
        probed = [p for p in (b["weights"], f"<project>/../OverLoCK-main/checkpoints/overlock_{b['variant']}_in1k_224.pth") if p]
        raise ConfigError(
            f"backbone.pretrained=true but no checkpoint found for variant {b['variant']!r}. "
            f"Probed: {probed}. This project never downloads weights: place the file or set "
            "backbone.pretrained=false for an explicitly random structure test."
        )
    if require_weights_file and not weights:
        raise ConfigError(f"a checkpoint file is required for this command (variant {b['variant']!r})")

    resolved = {
        "project_root": project_root,
        "backbone_variant": b["variant"],
        "yolo_family": y["family"],
        "yolo_scale": y["scale"],
        "combination": f"{b['variant']}+{y['family']}{y['scale']}",
        "nc": nc,
        "names": names,
        "backbone_weights": weights,
        "backbone_channels": list(expected_channels(b["variant"])),
        "adapter_map": adapter_mapping_report(b["variant"], y["family"], y["scale"]),
        "data_yaml": data_yaml,
        "imgsz": int(t["imgsz"]),
        "rect": bool(t["rect"]),
        "multi_scale": bool(t["multi_scale"]) if isinstance(t["multi_scale"], bool) else float(t["multi_scale"]),
        "aug_scale": float(t["scale"]),
        "family_contract": YOLO_FAMILY_CONTRACT[y["family"]],
    }
    return resolved


def resolved_summary(cfg: dict) -> dict:
    """Compact one-screen summary of what will actually be built/run."""
    res = cfg["_resolved"]
    t = cfg["train"]
    return {
        "combination": res["combination"],
        "backbone": {
            "family": cfg["backbone"]["family"],
            "variant": cfg["backbone"]["variant"],
            "weights": res["backbone_weights"],
            "pretrained": cfg["backbone"]["pretrained"],
            "deploy": cfg["backbone"]["deploy"],
            "attention_backend": cfg["backbone"]["attention_backend"],
        },
        "yolo": {"family": res["yolo_family"], "scale": res["yolo_scale"], "nc": res["nc"]},
        "adapter_map": res["adapter_map"],
        "protocol": {
            "imgsz": res["imgsz"],
            "rect": res["rect"],
            "multi_scale": res["multi_scale"],
            "aug_scale": res["aug_scale"],
            "amp": bool(t["amp"]),
            "device": cfg["runtime"]["device"],
            "workers": cfg["runtime"]["workers"],
        },
        "data_yaml": res["data_yaml"].get("resolved"),
        "family_contract": res["family_contract"],
    }


def write_resolved_config(cfg: dict, path: str) -> str:
    """Write the fully resolved configuration (one JSON object) next to the run outputs."""
    payload = {
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "meta": cfg.get("_meta", {}),
        "resolved": cfg.get("_resolved", {}),
        "summary": resolved_summary(cfg),
    }
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


# --------------------------------------------------------------------------------------
# shared CLI surface
# --------------------------------------------------------------------------------------
def add_experiment_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the shared ``--config`` + selector + protocol arguments.

    Every argument defaults to ``None`` so an absent CLI flag never overwrites a file value.
    """
    g = ap.add_argument_group("experiment")
    g.add_argument("--config", default=None, help="experiment YAML (see configs/*.yaml)")
    g.add_argument("--project-root", default=None, help="repository root (default: auto-detected)")
    g.add_argument("--ultralytics-root", default=None, help="pinned Ultralytics source root")
    g.add_argument("--variant", default=None, choices=VARIANTS, help="OverLoCK backbone variant")
    g.add_argument("--weights", default=None, help="OverLoCK .pth (loaded with weights_only=True)")
    g.add_argument("--pretrained", default=None, action="store_true", help="require the checkpoint to be present")
    g.add_argument("--random-backbone", default=None, action="store_true", help="explicitly allow a random backbone")
    g.add_argument("--attention-backend", default=None, choices=("auto", "natten", "torch_reference"))
    g.add_argument("--yolo-family", default=None, choices=FAMILIES)
    g.add_argument("--yolo-scale", default=None, choices=SCALES)
    g.add_argument("--nc", default=None, type=int)

    g = ap.add_argument_group("protocol")
    g.add_argument("--imgsz", default=None, type=int)
    g.add_argument("--rect", default=None, action="store_true", help="rejected: the protocol is exactly 640x640")
    g.add_argument("--multi-scale", default=None, action="store_true", help="rejected: fixed 640x640")
    g.add_argument("--aug-scale", default=None, type=float, dest="aug_scale", help="train.scale (augmentation only)")
    g.add_argument("--val-scaleup", default=None, action="store_true")
    g.add_argument("--device", default=None)
    g.add_argument("--workers", default=None, type=int)
    g.add_argument("--batch", default=None, type=int)
    g.add_argument("--epochs", default=None, type=int)
    g.add_argument("--seed", default=None, type=int)
    g.add_argument("--amp", default=None, action="store_true")
    g.add_argument("--conf", default=None, type=float)
    g.add_argument("--iou", default=None, type=float)
    g.add_argument("--max-det", default=None, type=int, dest="max_det")
    g.add_argument("--nms", default=None, action="store_true", help="force NMS on (default: native per family)")

    g = ap.add_argument_group("paths")
    g.add_argument("--data-yaml", default=None, dest="data_yaml", help="SODA10M COCO config")
    g.add_argument("--view", default=None, help="prebuilt native YOLO data view directory")
    g.add_argument("--limit-per-split", default=None, type=int, dest="limit_per_split")
    g.add_argument("--freeze-backbone", default=None, action="store_true", dest="freeze_backbone")
    g.add_argument("--backbone-bn-eval", default=None, action="store_true", dest="backbone_bn_eval")
    g.add_argument("--backbone-lr-mult", default=None, type=float, dest="backbone_lr_mult")
    g.add_argument("--row-chunk", default=None, type=int, dest="row_chunk")
    g.add_argument("--torch-threads", default=None, type=int, dest="torch_threads")
    g.add_argument("--verbose", default=None, action="store_true")
    return ap


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """``argparse`` parser carrying the shared experiment surface."""
    return add_experiment_arguments(argparse.ArgumentParser(description=description))


#: Alias kept for readability at call sites.
experiment_parser = build_arg_parser
