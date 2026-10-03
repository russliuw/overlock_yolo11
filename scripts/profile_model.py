#!/usr/bin/env python
"""Parameter / partial-MACs profiling for one detector combination (DESIGN_V2.md 6).

Examples::

    python scripts/profile_model.py --config configs/overlock_t_yolo11s_soda.yaml
    python scripts/profile_model.py --matrix --out reports/v2/variant_matrix.json
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from overlock_yolo.cli import cli_guard, bootstrap, emit_summary, load_config_for_args  # noqa: E402
from overlock_yolo.config import build_arg_parser  # noqa: E402
from overlock_yolo.paths import project_root  # noqa: E402


@cli_guard
def main(argv: list | None = None) -> int:
    ap = build_arg_parser("Profile parameters / partial MACs / na2d_av MACs for one combination")
    ap.add_argument("--out", default=None, help="JSON output (default reports/v2/profile_<combination>.json)")
    ap.add_argument("--markdown", default=None, help="optional Markdown summary")
    ap.add_argument("--no-macs", action="store_true", help="parameters only (skip the THOP forward)")
    ap.add_argument("--matrix", action="store_true", help="emit the 40-cell parameter matrix instead")
    ap.add_argument("--verify-matrix", action="store_true", help="cross-check matrix cells against assembled models")
    args = ap.parse_args(argv)

    boot = bootstrap(args)
    reports_dir = os.path.join(project_root(), "reports", "v2")

    from overlock_yolo.profile_model import (
        combination_parameter_matrix,
        profile_detector,
        write_profile_reports,
    )

    if args.matrix:
        matrix = combination_parameter_matrix()
        checks = []
        if args.verify_matrix:
            from overlock_yolo.cli import build_from_config
            from overlock_yolo.profile_model import parameter_report

            for combo in (
                ("t", "yolo11", "s"),
                ("t", "yolo11", "n"),
                ("b", "yolo11", "s"),
                ("t", "yolo26", "n"),
                ("t", "yolo26", "s"),
            ):
                cfg = load_config_for_args(args)
                cfg["_resolved"]["backbone_variant"], cfg["_resolved"]["yolo_family"], cfg["_resolved"]["yolo_scale"] = combo
                cfg["_resolved"]["combination"] = f"{combo[0]}+{combo[1]}{combo[2]}"
                cfg["backbone"]["variant"] = combo[0]
                model = build_from_config(cfg, ultra_root=boot.ultralytics_root, weights=None)
                measured = parameter_report(model)["total"]["unique_numel"]
                cell = next(c for c in matrix["cells"] if c["combination"] == cfg["_resolved"]["combination"])
                checks.append(
                    {
                        "combination": cell["combination"],
                        "matrix_total": cell["total_numel"],
                        "assembled_total": measured,
                        "match": int(measured) == int(cell["total_numel"]),
                        "delta": int(measured) - int(cell["total_numel"]),
                    }
                )
                del model
        matrix["assembled_cross_checks"] = checks
        matrix["all_cross_checks_match"] = all(c["match"] for c in checks) if checks else None
        out = args.out or os.path.join(reports_dir, "variant_matrix.json")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(matrix, fh, indent=2, default=str)
        print(f"[overlock] variant matrix ({matrix['n_cells']} cells) -> {out}")
        if checks:
            print(json.dumps(checks, indent=2))
        return 0

    cfg = load_config_for_args(args)
    emit_summary(cfg)
    report = profile_detector(cfg, imgsz=cfg["_resolved"]["imgsz"], device=cfg["runtime"]["device"], macs=not args.no_macs)
    out = args.out or os.path.join(reports_dir, f"profile_{cfg['_resolved']['combination']}.json")
    md = args.markdown or os.path.join(reports_dir, f"profile_{cfg['_resolved']['combination']}.md")
    written = write_profile_reports(report, out, md)
    print(f"[overlock] parameters (unique): {report['parameters']['total']['unique_numel']:,}")
    if report.get("partial_macs", {}).get("available"):
        print(f"[overlock] partial MACs (conv/linear/BN only): {report['partial_macs']['partial_macs']:,}")
    print(f"[overlock] na2d_av MACs: {report.get('na2d_av_macs', {}).get('macs', 'n/a')}")
    print(f"[overlock] profile -> {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
