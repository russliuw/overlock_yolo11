#!/usr/bin/env python
"""Train the OverLoCK + YOLO11/YOLO26 detector on a native data view.

Examples::

    # local CPU smoke configuration (never a full training run)
    python scripts/train.py --config configs/overlock_t_yolo11s_soda.yaml --epochs 1 --batch 2

    # server (RTX 4090) template -- install the locked requirements first
    python scripts/train.py --config configs/overlock_t_yolo11s_soda.yaml \\
        --device 0 --batch 16 --workers 8 --amp --epochs 100

The dataset must be a *full* native view.  A view built with ``--limit-per-split`` carries an
``overlock_smoke_view.json`` marker; training on it requires ``--allow-smoke-view`` so a rented
GPU cannot silently train on two images.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from overlock_yolo.cli import bootstrap, data_or_view, emit_summary, load_config_for_args  # noqa: E402
from overlock_yolo.config import build_arg_parser  # noqa: E402
from overlock_yolo.paths import project_root  # noqa: E402


def _smoke_marker(view_dir) -> dict | None:
    if not view_dir:
        return None
    path = os.path.join(view_dir, "overlock_smoke_view.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main(argv: list | None = None) -> int:
    ap = build_arg_parser("Train OverLoCK (xt/t/s/b) + native YOLO11/YOLO26 on SODA10M")
    ap.add_argument("--allow-smoke-view", action="store_true", help="explicitly allow training on a --limit view")
    ap.add_argument("--init-checkpoint", default=None, help="start from a checkpoint written by this project")
    ap.add_argument("--resume", default=None, help="resume from a checkpoint written by this project")
    ap.add_argument("--project", default=None, help="Ultralytics run directory (default <project>/runs)")
    ap.add_argument("--name", default=None, help="run name")
    ap.add_argument("--data", default=None, help="explicit data.yaml to train on (overrides the config)")
    args = ap.parse_args(argv)

    boot = bootstrap(args)
    cfg = load_config_for_args(args, require_weights_file=bool(args.pretrained))
    print(f"[overlock] project_root={boot.project_root}")
    print(f"[overlock] ultralytics={boot.import_origin}")
    runs_dir = args.project or os.path.join(project_root(), "runs")
    emit_summary(cfg, os.path.join(runs_dir, "resolved_config.json"))

    data = data_or_view(cfg, args.data)
    if not data:
        raise SystemExit(
            "no dataset: pass --data <view>/data.yaml, or --view <view dir>, or --data-yaml <soda10m.yaml> "
            "(then build a view with scripts/prepare_data.py)"
        )
    marker = _smoke_marker(cfg["data"].get("view"))
    if marker and not args.allow_smoke_view:
        raise SystemExit(
            f"refusing to train on the smoke view {cfg['data']['view']} "
            f"({marker.get('limit_per_split')} images per split). Build the full view "
            "(scripts/prepare_data.py without --limit-per-split) or pass --allow-smoke-view for a debug run."
        )
    if args.resume and args.pretrained:
        raise SystemExit(
            "--resume and --pretrained are mutually exclusive: resuming must not re-initialise the "
            "backbone over the trained weights"
        )

    from overlock_yolo.trainer import trainer_class

    t = cfg["train"]
    overrides = {
        "data": data,
        "imgsz": int(cfg["_resolved"]["imgsz"]),
        "rect": False,
        "multi_scale": False,
        "epochs": int(args.epochs if args.epochs is not None else t["epochs"]),
        "batch": int(args.batch if args.batch is not None else t["batch"]),
        "device": cfg["runtime"]["device"],
        "workers": int(cfg["runtime"]["workers"]),
        "seed": int(cfg["runtime"]["seed"]),
        "optimizer": t["optimizer"],
        "amp": bool(args.amp if args.amp is not None else t["amp"]),
        "task": "detect",
        "mode": "train",
        "val": True,
        "plots": False,
    }
    if args.project:
        overrides["project"] = args.project
    if args.name:
        overrides["name"] = args.name
    for key in ("lr0", "weight_decay"):
        if t.get(key) is not None:
            overrides[key] = t[key]
    if args.resume:
        overrides["resume"] = args.resume

    trainer = trainer_class()(overrides=overrides, detector_config=cfg)
    if args.init_checkpoint:
        from overlock_yolo.trainer import load_project_checkpoint

        report = load_project_checkpoint(trainer.model, args.init_checkpoint, strict=True)
        print(f"[overlock] initialised from {args.init_checkpoint}: {report['loaded_tensors']} tensors")
    trainer.train()
    print(f"[overlock] run directory: {trainer.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
