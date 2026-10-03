#!/usr/bin/env python
"""Native Ultralytics detection validation (primary metric) at the fixed 640x640 protocol.

``P``/``R``/``mAP50``/``mAP50-95`` (plus per-class values and the native ``results_dict``) come
from the *pinned native* ``DetectionValidator``/``DetMetrics``.  YOLO11 runs the native NMS path,
YOLO26 the native end-to-end (top-k) path; no hand-written NMS and no COCOeval as the primary
number.  ``--conf`` defaults to the native detection default (0.001).

Examples::

    python scripts/val.py --config configs/overlock_t_yolo11s_soda.yaml --view data/soda_smoke
    python scripts/val.py --config configs/overlock_t_yolo26s_soda.yaml --device 0 --batch 8
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from overlock_yolo.cli import cli_guard, bootstrap, build_from_config, data_or_view, emit_summary, load_config_for_args  # noqa: E402
from overlock_yolo.config import build_arg_parser  # noqa: E402
from overlock_yolo.paths import project_root  # noqa: E402


@cli_guard
def main(argv: list | None = None) -> int:
    ap = build_arg_parser("Native Ultralytics detection validation (fixed square 640 protocol)")
    ap.add_argument("--out", default=os.path.join(project_root(), "reports", "v2", "val_result.json"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--data", default=None, help="explicit data.yaml (overrides the config)")
    args = ap.parse_args(argv)

    boot = bootstrap(args)
    cfg = load_config_for_args(args, require_weights_file=bool(args.pretrained))
    print(f"[overlock] project_root={boot.project_root}")
    emit_summary(cfg)

    data = data_or_view(cfg, args.data)
    if not data:
        raise SystemExit("no dataset: pass --view <view dir> or --data-yaml <soda10m.yaml>")

    model = build_from_config(cfg, ultra_root=boot.ultralytics_root)
    params = model.parameter_groups_report()

    from overlock_yolo.validator import run_native_validation

    result = run_native_validation(
        model,
        cfg,
        data=data,
        batch=int(args.batch if args.batch is not None else 1),
        split=args.split,
    )
    payload = {
        "combination": cfg["_resolved"]["combination"],
        "parameters": params["total"],
        "backbone_weights": cfg["_resolved"]["backbone_weights"],
        "weights_loaded": bool(cfg["_resolved"]["backbone_weights"]) and model.pretrained_backbone_report is not None,
        "neck_head_init": "random (no detection pretraining)",
        "data_yaml": data,
        "validation": result,
        "primary_metric_source": "native ultralytics DetectionValidator + DetMetrics",
        "notes": [
            "P/R/mAP50/mAP50-95 are the native Ultralytics numbers; COCOeval is not used as the primary metric.",
            "conf defaults to 0.001 (native detection default); save_json is off so DetMetrics stays primary.",
            "YOLO11 -> native NMS; YOLO26 -> native end-to-end path (no extra NMS).",
            "metrics computed on a small smoke view are an interface check, not an accuracy measurement",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print("[overlock] native metrics:", json.dumps(result["results_dict"], indent=2))
    print(f"[overlock] validation report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
