"""Shared CLI bootstrap: resolve paths, install the pinned Ultralytics, build/attach a detector.

Every entry point follows the same order, which is what removes the old "hardcoded Mac path at
import time, CLI flag read afterwards" problem (DESIGN_V2.md 3.2)::

    args = parse()
    bootstrap(args)                  # env + project root + ultralytics root + sys.path
    cfg  = load_experiment(...)      # strict config + explicit CLI overrides
    model = build_from_config(cfg)   # detector for the resolved combination

Nothing here imports ``ultralytics`` before :func:`bootstrap` has run.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from .config import ConfigError, load_experiment, resolved_summary, write_resolved_config
from .paths import (
    PathResolutionError,
    install_ultralytics_root,
    prepare_ultralytics_env,
    project_root,
    resolve_project_root,
    resolve_ultralytics_root,
)

__doc__ += """

Usage note: :func:`bootstrap` must run before anything imports ``ultralytics``.
"""

__all__ = [
    "cli_guard",
    "bootstrap",
    "BootstrapResult",
    "load_config_for_args",
    "build_from_config",
    "apply_thread_setting",
]


def cli_guard(fn):
    """Run an entry point, turning expected configuration/path errors into a clean exit.

    A missing checkpoint, a missing dataset or a rejected option must print one actionable line
    and exit non-zero -- never a raw traceback, and never a download attempt.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        from .config import ConfigError
        from .paths import PathResolutionError

        try:
            return fn(*args, **kwargs)
        except (ConfigError, PathResolutionError, FileNotFoundError) as exc:
            print(f"\n[overlock] ERROR: {exc}", file=sys.stderr)
            print(
                "[overlock] nothing was downloaded or modified. Fix the path/option above, then re-run.",
                file=sys.stderr,
            )
            return 2

    return wrapper


class BootstrapResult:
    """Resolved roots after :func:`bootstrap`."""

    def __init__(self, project_root_: str, ultralytics_root: str, env: dict, import_origin: str):
        self.project_root = project_root_
        self.ultralytics_root = ultralytics_root
        self.env = env
        self.import_origin = import_origin

    def as_dict(self) -> dict:
        return {
            "project_root": self.project_root,
            "ultralytics_root": self.ultralytics_root,
            "ultralytics_import": self.import_origin,
            "env": self.env,
            "cwd": os.getcwd(),
        }


def bootstrap(args) -> BootstrapResult:
    """Resolve the project root, prepare the cache env and install the pinned Ultralytics source."""
    root = resolve_project_root(getattr(args, "project_root", None))
    env = prepare_ultralytics_env(root)
    ultra = resolve_ultralytics_root(
        getattr(args, "ultralytics_root", None), project_root_=root, allow_sibling=True
    )
    origin = install_ultralytics_root(ultra)
    return BootstrapResult(root, ultra, env, origin)


def apply_thread_setting(cfg: dict) -> int:
    """Honour ``runtime.torch_threads`` / ``OMP_NUM_THREADS`` before heavy work starts."""
    import torch

    requested = cfg["runtime"].get("torch_threads")
    if requested is None:
        requested = int(os.environ.get("OMP_NUM_THREADS", "2"))
    torch.set_num_threads(int(requested))
    return int(torch.get_num_threads())


def load_config_for_args(args, *, require_weights_file: bool = False) -> dict:
    """Build the resolved experiment config from ``--config`` plus explicit CLI overrides."""
    if getattr(args, "random_backbone", False):
        if getattr(args, "pretrained", False):
            raise ConfigError("--pretrained and --random-backbone are mutually exclusive")
        args.pretrained = None
    cli = {k: v for k, v in vars(args).items() if k in _CLI_PASSTHROUGH}
    cfg = load_experiment(
        getattr(args, "config", None),
        project_root=getattr(args, "_project_root", None) or project_root(),
        cli_overrides=cli,
        require_weights_file=require_weights_file,
    )
    if getattr(args, "random_backbone", False):
        cfg["backbone"]["pretrained"] = False
        cfg["_resolved"]["backbone_weights"] = cfg["_resolved"]["backbone_weights"]
    if getattr(args, "view", None):
        cfg["data"]["view"] = os.path.abspath(os.path.expanduser(args.view))
        cfg["_resolved"]["view"] = cfg["data"]["view"]
    return cfg


#: Argument names forwarded into the config layer (see ``config.CLI_TO_CONFIG``).
_CLI_PASSTHROUGH = frozenset(
    {
        "variant",
        "weights",
        "pretrained",
        "attention_backend",
        "row_chunk",
        "freeze_backbone",
        "backbone_bn_eval",
        "backbone_lr_mult",
        "yolo_family",
        "yolo_scale",
        "nc",
        "device",
        "workers",
        "seed",
        "torch_threads",
        "verbose",
        "imgsz",
        "rect",
        "multi_scale",
        "aug_scale",
        "amp",
        "epochs",
        "batch",
        "optimizer",
        "lr0",
        "weight_decay",
        "val_scaleup",
        "conf",
        "iou",
        "max_det",
        "nms",
        "data_yaml",
        "limit_per_split",
    }
)


def build_from_config(cfg: dict, *, ultra_root: Optional[str] = None, weights: Optional[str] = None, seed: Optional[int] = None):
    """Instantiate the detector for a resolved config (weights default to the resolved path)."""
    from .model import build_detector

    res = cfg["_resolved"]
    b = cfg["backbone"]
    use_weights = weights if weights is not None else res["backbone_weights"]
    pretrained = bool(b["pretrained"]) or bool(weights)
    if not pretrained:
        use_weights = None
    return build_detector(
        variant=res["backbone_variant"],
        family=res["yolo_family"],
        scale=res["yolo_scale"],
        nc=int(res["nc"]),
        names=res["names"],
        weights=use_weights,
        pretrained=pretrained,
        attention_backend=b["attention_backend"],
        ultra_root=ultra_root,
        seed=int(cfg["runtime"]["seed"] if seed is None else seed),
        row_chunk=int(b["row_chunk"]),
        freeze=bool(b["freeze"]),
        lr_mult=float(b["lr_mult"]),
        bn_eval=bool(b["bn_eval"]),
        strict_shape=bool(b["strict_shape"]),
        verbose=bool(cfg["runtime"]["verbose"]),
    )


def data_or_view(cfg: dict, explicit_data: Optional[str] = None) -> Optional[str]:
    """The dataset entry point: an explicit view wins, then ``data.yaml``, then the COCO config.

    A resolved path that does not exist is reported here, with the command that creates it,
    instead of surfacing as an obscure failure deep inside the native dataloader.
    """
    if explicit_data:
        path = os.path.abspath(os.path.expanduser(explicit_data))
    elif cfg.get("data", {}).get("view"):
        path = os.path.join(cfg["data"]["view"], "data.yaml")
    else:
        path = cfg["_resolved"]["data_yaml"].get("resolved")
    if path and not os.path.isfile(path):
        raise FileNotFoundError(
            f"dataset entry point {path} does not exist. Build a view first, e.g. "
            f"'python scripts/prepare_data.py --out <dir>' (add --limit-per-split 2 for a smoke view), "
            f"then pass --view <dir> or --data <dir>/data.yaml."
        )
    return path


def emit_summary(cfg: dict, path: Optional[str] = None) -> dict:
    """Print the resolved summary and optionally persist the fully resolved configuration."""
    import json

    summary = resolved_summary(cfg)
    print("[overlock] resolved experiment:")
    print(json.dumps(summary, indent=2, default=str))
    if path:
        write_resolved_config(cfg, path)
        print(f"[overlock] resolved config -> {path}")
    return summary
