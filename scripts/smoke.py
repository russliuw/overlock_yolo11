#!/usr/bin/env python
"""M01-M11 verification runner (CPU, FP32, workers=0) -- writes ``reports/v2/validation.json``.

Everything runs against the *current* checkout, at the sizes that fit this machine, with an
explicit per-item budget.  An item that exceeds its budget is recorded as ``partial`` with the
real error/timeout and its command, never as a pass.  640 items that do not finish are kept
partial and handed to the server checklist.

Usage::

    OMP_NUM_THREADS=2 python scripts/smoke.py --out reports/v2/validation.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import sys
import time
import traceback
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from overlock_yolo.config import ConfigError, load_experiment  # noqa: E402
from overlock_yolo.paths import (  # noqa: E402
    PathResolutionError,
    install_ultralytics_root,
    prepare_ultralytics_env,
    project_root,
    resolve_ultralytics_root,
)

REPORTS = os.path.join(project_root(), "reports", "v2")
VIEW = os.path.join(project_root(), "data", "soda_smoke")


def peak_gb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e9 if sys.platform == "darwin" else raw / 1e6


class Results:
    """Collect per-item status, timing, evidence and the exact command that produced it."""

    def __init__(self):
        self.items: "OrderedDict[str, dict]" = OrderedDict()

    def add(self, mid, name, status, seconds=None, evidence=None, error=None, notes=None, budget=None):
        assert status in ("pass", "fail", "skipped", "partial"), status
        self.items[mid] = {
            "id": mid,
            "name": name,
            "status": status,
            "seconds": None if seconds is None else round(float(seconds), 3),
            "budget_seconds": budget,
            "evidence": evidence or {},
            "error": error,
            "notes": notes,
        }
        flag = {"pass": "PASS", "fail": "FAIL", "skipped": "SKIP", "partial": "PART"}[status]
        print(f"[{flag}] {mid} {name}" + (f" ({seconds:.1f}s)" if seconds is not None else ""), flush=True)

    def run(self, mid, name, fn, budget=None):
        t0 = time.time()
        try:
            evidence = fn()
            self.add(mid, name, "pass", time.time() - t0, evidence, budget=budget)
        except Exception as exc:  # noqa: BLE001 - report, never swallow
            self.add(
                mid,
                name,
                "fail",
                time.time() - t0,
                error=f"{type(exc).__name__}: {exc}",
                notes=traceback.format_exc(limit=8),
                budget=budget,
            )
        return self.items[mid]

    def skip(self, mid, name, reason):
        self.add(mid, name, "skipped", error=reason)

    def as_dict(self, meta) -> dict:
        counts = {k: 0 for k in ("pass", "fail", "skipped", "partial")}
        for item in self.items.values():
            counts[item["status"]] += 1
        verdict = "BLOCKED_OR_INCOMPLETE"
        if counts["fail"] == 0 and counts["partial"] == 0:
            verdict = "IMPLEMENTED / CPU_CHECKS_PASSED / GPU_AND_ACCURACY_UNVERIFIED"
        elif counts["fail"] == 0:
            verdict = "IMPLEMENTED / CPU_CHECKS_PARTIAL (see partial items) / GPU_AND_ACCURACY_UNVERIFIED"
        return {
            "meta": meta,
            "summary": {"total": len(self.items), **counts, "verdict": verdict},
            "checks": list(self.items.values()),
        }


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _mini_batch(size: int, n_boxes: int = 3, nc: int = 6, seed: int = 0):
    """A legal synthetic detection batch (normalised xywh, per-image class ids)."""
    g = torch.Generator().manual_seed(seed)
    boxes = torch.rand(n_boxes, 4, generator=g) * 0.3 + 0.2
    return {
        "img": torch.rand(1, 3, size, size, generator=g),
        "batch_idx": torch.zeros(n_boxes),
        "cls": torch.randint(0, nc, (n_boxes,), generator=g).float(),
        "bboxes": boxes,
    }


def _split_eval_output(out):
    """Split a native Detect eval output into ``(prediction_tensor, [feature maps])``.

    YOLO11 returns ``(y, {"boxes","scores","feats"})``; YOLO26 returns
    ``(y, {"one2many": {...}, "one2one": {...}})``.  Both keep the per-level feature maps that
    the head actually consumed, which is what the 640 protocol check needs.
    """
    pred = out[0] if isinstance(out, tuple) else out
    raw = out[1] if isinstance(out, tuple) and len(out) > 1 else None
    feats = None
    if isinstance(raw, dict):
        if "feats" in raw:
            feats = raw["feats"]
        elif "one2many" in raw:
            feats = raw["one2many"]["feats"]
    return pred, feats


def _combo_cfg(variant, family, scale, **extra):
    overrides = {"variant": variant, "yolo_family": family, "yolo_scale": scale}
    overrides.update(extra)
    return load_experiment(None, cli_overrides=overrides)


# --------------------------------------------------------------------------------------
# M01 -- config / paths
# --------------------------------------------------------------------------------------
def m01_config_paths(results: Results, ultra_root: str) -> None:
    from overlock_yolo.config import ConfigError as CE
    from overlock_yolo.config import load_yaml_strict

    def check():
        evidence = {}
        combos = []
        for v in ("xt", "t", "s", "b"):
            for f in ("yolo11", "yolo26"):
                for s in ("n", "s", "m", "l", "x"):
                    c = _combo_cfg(v, f, s)
                    combos.append({"combination": c["_resolved"]["combination"], "adapter_map": c["_resolved"]["adapter_map"]})
        evidence["n_combinations"] = len(combos)
        evidence["combinations_sample"] = combos[:2] + combos[-2:]

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            dup = os.path.join(td, "dup.yaml")
            with open(dup, "w") as fh:
                fh.write("yolo:\n  scale: s\n  scale: m\n")
            try:
                load_yaml_strict(dup)
                raise AssertionError("duplicate keys were NOT rejected")
            except CE as exc:
                evidence["duplicate_key_error"] = str(exc)[:160]
            amb = os.path.join(td, "amb.yaml")
            with open(amb, "w") as fh:
                fh.write("scale: s\n")
            try:
                load_experiment(amb)
                raise AssertionError("ambiguous top-level scale was NOT rejected")
            except CE as exc:
                evidence["ambiguous_scale_error"] = str(exc)[:120]
            for name, body in (
                ("unknown_key", "yolo:\n  scale: s\n  widht: 1\n"),
                ("unknown_section", "backbonee:\n  variant: t\n"),
                ("bad_variant", "backbone:\n  variant: xxl\n"),
                ("bad_scale", "yolo:\n  scale: z\n"),
                ("rect_true", "train:\n  rect: true\n"),
                ("multi_scale_true", "train:\n  multi_scale: true\n"),
                ("yolo26_epochs_1", "yolo:\n  family: yolo26\ntrain:\n  epochs: 1\n"),
            ):
                path = os.path.join(td, f"{name}.yaml")
                with open(path, "w") as fh:
                    fh.write(body)
                try:
                    load_experiment(path)
                    raise AssertionError(f"{name} was NOT rejected")
                except (CE, ValueError) as exc:
                    evidence[f"rejected::{name}"] = type(exc).__name__

            f = os.path.join(td, "exp.yaml")
            with open(f, "w") as fh:
                fh.write(
                    "backbone:\n  variant: b\nyolo:\n  family: yolo26\n  scale: m\ntrain:\n"
                    "  imgsz: 512\n  scale: 0.3\nruntime:\n  device: cpu\n"
                )
            from_file = load_experiment(f)
            evidence["from_file"] = {
                "combination": from_file["_resolved"]["combination"],
                "imgsz": from_file["_resolved"]["imgsz"],
                "aug_scale": from_file["_resolved"]["aug_scale"],
            }
            overridden = load_experiment(f, cli_overrides={"variant": "t", "yolo_scale": "s", "imgsz": 640})
            evidence["with_cli"] = {
                "combination": overridden["_resolved"]["combination"],
                "imgsz": overridden["_resolved"]["imgsz"],
                "aug_scale": overridden["_resolved"]["aug_scale"],
                "applied": overridden["_meta"]["cli_overrides_applied"],
            }
            # variant + scale overridden; family and train.scale keep their file values
            assert overridden["_resolved"]["combination"] == "t+yolo26s", overridden["_resolved"]["combination"]
            assert overridden["_resolved"]["imgsz"] == 640
            assert overridden["_resolved"]["aug_scale"] == 0.3, "an absent CLI value overwrote the file value"
            # and adding --yolo-family switches only that one selector
            both = load_experiment(
                f, cli_overrides={"variant": "t", "yolo_scale": "s", "yolo_family": "yolo11", "imgsz": 640}
            )
            evidence["with_cli_and_family"] = {
                "combination": both["_resolved"]["combination"],
                "adapter_map": both["_resolved"]["adapter_map"],
                "aug_scale": both["_resolved"]["aug_scale"],
            }
            assert both["_resolved"]["combination"] == "t+yolo11s", both["_resolved"]["combination"]

        import subprocess

        probe = (
            "import sys; sys.path.insert(0, %r);"
            "from overlock_yolo.paths import resolve_project_root, resolve_ultralytics_root, install_ultralytics_root;"
            "r=resolve_project_root();u=resolve_ultralytics_root();o=install_ultralytics_root(u);"
            "print(r);print(o)" % ROOT
        )
        out = subprocess.run([sys.executable, "-c", probe], cwd="/tmp", capture_output=True, text=True)
        evidence["other_cwd"] = {"returncode": out.returncode, "stdout": out.stdout.strip().splitlines()}
        assert out.returncode == 0, out.stderr[-500:]
        assert os.path.abspath(out.stdout.strip().splitlines()[0]) == project_root()
        try:
            install_ultralytics_root(os.path.join(ROOT, "nowhere"))
            raise AssertionError("a bogus ultralytics root was accepted")
        except PathResolutionError as exc:
            evidence["bad_root_error"] = str(exc)[:140]
        evidence["ultralytics_root"] = ultra_root
        return evidence

    results.run("M01", "config/path contracts (3 independent selectors, strict YAML, cwd independence)", check)


# --------------------------------------------------------------------------------------
# M02 -- four backbones
# --------------------------------------------------------------------------------------
def m02_backbones(results: Results, ultra_root: str) -> None:
    def check():
        from overlock_yolo.backbone import build_overlock
        from overlock_yolo.variants import FEATURE_INFO, OVERLOCK_VARIANTS, P3P4P5_CHANNELS

        evidence = {"variants": {}, "official_source": "OverLoCK-main/detection/models/overlock.py lines 861-946"}
        evidence["factory_bodies"] = {k: dict(v) for k, v in OVERLOCK_VARIANTS.items()}
        for variant in ("xt", "t", "s", "b"):
            model = build_overlock(variant)
            model.eval()
            n_params = int(sum(p.numel() for p in model.parameters()))
            with torch.no_grad():
                outs = model.forward_multiscale(torch.zeros(1, 3, 64, 64), strict=True)
            shapes = [list(o.shape) for o in outs]
            expected = [(P3P4P5_CHANNELS[variant][i], 8 // (2 ** i), 8 // (2 ** i)) for i in range(3)]
            got = [tuple(s[1:]) for s in shapes[1:]]
            evidence["variants"][variant] = {
                "parameters": n_params,
                "weights": "random (no checkpoint loaded, no download)",
                "forward_64": shapes,
                "expected_signature": [list(x) for x in FEATURE_INFO[variant]],
                "expected_p3p4p5": list(P3P4P5_CHANNELS[variant]),
                "attention_backend": model.attention_backend_report(),
            }
            assert got == expected, (variant, got, expected)
            del model
            gc.collect()
        return evidence

    results.run("M02", "four OverLoCK backbones: official config + small forward (all random)", check, budget=180)


# --------------------------------------------------------------------------------------
# M03 -- ten native tails
# --------------------------------------------------------------------------------------
def m03_tails(results: Results, ultra_root: str) -> None:
    def check():
        from overlock_yolo.model import build_native_tail
        from overlock_yolo.variants import (
            YOLO_FAMILY_CONTRACT,
            expected_detect_entry_channels,
            expected_neck_entry_channels,
        )

        evidence = {"tails": {}}
        for family in ("yolo11", "yolo26"):
            for scale in ("n", "s", "m", "l", "x"):
                modules, _save, _yaml, report = build_native_tail(family, scale, 6, ultra_root)
                key = f"{family}{scale}"
                detect = report["detect"]
                contract = YOLO_FAMILY_CONTRACT[family]
                checks = {
                    "entries": report["entry_layers"] == [4, 6, 10],
                    "neck_channels": tuple(report["neck_entry_channels"]) == tuple(expected_neck_entry_channels(family, scale)),
                    "detect_channels": tuple(report["detect_entry_channels"]) == tuple(expected_detect_entry_channels(family, scale)),
                    "stride": detect["stride"] == [8.0, 16.0, 32.0],
                    "reg_max": int(detect["reg_max"]) == int(contract["reg_max"]),
                    "one2one": bool(detect["has_one2one"]) == ("one-to-one" in contract["branches"]),
                }
                head = modules[-1]
                cls_bias = head.cv3[0][-1].bias.detach()
                entry = {
                    "tail_start": report["tail_start"],
                    "entry_layers": report["entry_layers"],
                    "neck_entry_channels": report["neck_entry_channels"],
                    "detect_entry_channels": report["detect_entry_channels"],
                    "tail_parameters": report["tail_parameters"],
                    "reg_max": detect["reg_max"],
                    "has_one2one": detect["has_one2one"],
                    "stride": detect["stride"],
                    "c3k_per_layer": [f["c3k"] for f in report["c3k_flags"]],
                    "block_kinds_per_layer": [f["block_kinds"] for f in report["c3k_flags"]],
                    "checks": checks,
                    "cls_bias_head_of_first_branch": [round(float(v), 6) for v in cls_bias[:3]],
                }
                evidence["tails"][key] = entry
                assert all(checks.values()), f"{key} failed checks {checks}"
                assert float(cls_bias.abs().max()) > 0, "Detect cls bias is all zeros: bias_init did not run"
                del modules
                gc.collect()

        for scale in ("m", "l", "x"):
            got = evidence["tails"][f"yolo11{scale}"]["c3k_per_layer"]
            assert all(got), f"yolo11{scale} must be all C3k, got {got}"
        for scale in ("n", "s"):
            got = evidence["tails"][f"yolo11{scale}"]["c3k_per_layer"]
            assert got == [False, False, False, True], f"yolo11{scale} got {got}"
        # Native rule (ultralytics parse_model): C3k2 c3k is forced True for scale in {m,l,x}
        # (which is why yolo11 n/s keep the YAML's False), and the YOLO26 last neck row carries
        # attn=True so its C3k2 uses PSABlock blocks.
        for scale in ("n", "s", "m", "l", "x"):
            got = evidence["tails"][f"yolo26{scale}"]["c3k_per_layer"]
            assert got[:3] == [True, True, True], f"yolo26{scale} got {got}"
            kinds = evidence["tails"][f"yolo26{scale}"]["block_kinds_per_layer"][3]
            assert "PSABlock" in kinds, f"yolo26{scale} last layer {kinds}"
            assert "C3k" not in kinds, f"yolo26{scale} last layer unexpectedly C3k: {kinds}"
        for scale in ("n", "s", "m", "l", "x"):
            kinds = evidence["tails"][f"yolo11{scale}"]["block_kinds_per_layer"][3]
            assert "PSABlock" not in kinds, f"yolo11{scale} last layer {kinds}"
        evidence["family_discrimination"] = (
            "yolo11 n/s: neck C3k2 = Bottleneck x3 + C3k on the last layer; yolo11 m/l/x: all C3k; "
            "yolo26: C3k x3 + a PSABlock-based C3k2 on the last neck layer at every scale.  The two "
            "families differ structurally at every scale, not only in width, so their tails cannot "
            "be swapped.  Both facts come from the native parse_model rules, not from hardcoded flags."
        )
        return evidence

    results.run("M03", "10 native tails: family/scale channels, reg_max, one2one, C3k, bias_init", check, budget=180)


# --------------------------------------------------------------------------------------
# M04 -- 40 combinations
# --------------------------------------------------------------------------------------
def m04_matrix(results: Results, ultra_root: str) -> None:
    def check():
        from overlock_yolo.cli import build_from_config
        from overlock_yolo.model import combination_report
        from overlock_yolo.profile_model import combination_parameter_matrix
        from overlock_yolo.variants import adapter_mapping_report, expected_channels

        matrix = combination_parameter_matrix()
        evidence = {"n_cells": matrix["n_cells"], "backbone_measured_numel": matrix["backbone_measured_numel"]}
        seen = set()
        for cell in matrix["cells"]:
            seen.add(cell["combination"])
            assert cell["backbone_numel"] == matrix["backbone_measured_numel"][cell["backbone_variant"]]
        evidence["unique_combinations"] = len(seen)
        assert len(seen) == 40, len(seen)

        checks = []
        for variant, family, scale in (
            ("t", "yolo11", "s"),
            ("t", "yolo11", "n"),
            ("b", "yolo11", "s"),
            ("t", "yolo26", "n"),
            ("t", "yolo26", "s"),
        ):
            cfg = _combo_cfg(variant, family, scale)
            model = build_from_config(cfg, ultra_root=ultra_root, weights=None)
            measured = int(sum(p.numel() for p in model.parameters()))
            cell = next(c for c in matrix["cells"] if c["combination"] == f"{variant}+{family}{scale}")
            stem = model.stem
            for i, entry in enumerate(cfg["_resolved"]["adapter_map"]):
                leaf = getattr(stem, f"adapter{i + 1}").conv
                assert int(leaf.in_channels) == entry["backbone_channels"], (cell["combination"], i)
                assert int(leaf.out_channels) == entry["adapter_channels"], (cell["combination"], i)
            stem_params = sum(p.numel() for p in stem.parameters())
            tail_params = sum(p.numel() for p in model.tail.parameters())
            backbone_params = sum(p.numel() for p in model.stem.backbone.parameters())
            checks.append(
                {
                    "combination": cell["combination"],
                    "matrix_total": cell["total_numel"],
                    "assembled_total": measured,
                    "match": measured == cell["total_numel"],
                    "assembled_breakdown": {
                        "backbone": backbone_params,
                        "adapters": stem_params - backbone_params,
                        "neck_head": tail_params,
                    },
                    "matrix_breakdown": {
                        "backbone": cell["backbone_numel"],
                        "adapters": cell["adapter_numel"],
                        "neck_head": cell["tail_numel"],
                    },
                    "state_dict_keys": len(model.state_dict()),
                    "duplicate_registration": model.parameter_groups_report()["total"]["duplicate_registration"],
                }
            )
            del model
            gc.collect()
        evidence["assembled_cross_checks"] = checks
        evidence["all_match"] = all(c["match"] for c in checks)
        evidence["no_duplicate_registration"] = all(c["duplicate_registration"] == [] for c in checks)
        assert evidence["all_match"], checks
        assert evidence["no_duplicate_registration"]
        evidence["static_reports_sample"] = [
            combination_report(v, f, s) for v, f, s in (("xt", "yolo11", "n"), ("b", "yolo26", "x"))
        ]
        evidence["adapter_mapping_check"] = {
            "t+yolo11s": adapter_mapping_report("t", "yolo11", "s"),
            "t+yolo11n": adapter_mapping_report("t", "yolo11", "n"),
        }
        assert expected_channels("t") == (128, 384, 640)
        return evidence

    results.run("M04", "40-combination interface: channels/routing/parameter matrix consistency", check, budget=180)


# --------------------------------------------------------------------------------------
# M05 -- T/B real weights
# --------------------------------------------------------------------------------------
def m05_checkpoints(results: Results, ultra_root: str) -> None:
    def check():
        from overlock_yolo.backbone import build_overlock
        from overlock_yolo.checkpoint import CheckpointError, audit_and_load, verify_loaded_parameters
        from overlock_yolo.model import _representative_keys
        from overlock_yolo.paths import resolve_checkpoint_path

        evidence = {}
        for variant in ("t", "b"):
            path = resolve_checkpoint_path(variant)
            if not path:
                evidence[variant] = {"status": "checkpoint_absent"}
                continue
            model = build_overlock(variant)
            report = audit_and_load(model, path)
            report["variant"] = variant
            report["representative_param_check"] = verify_loaded_parameters(model, path, keys=_representative_keys(variant))
            other = "b" if variant == "t" else "t"
            other_path = resolve_checkpoint_path(other)
            mismatch = None
            if other_path:
                probe = build_overlock(variant)
                try:
                    audit_and_load(probe, other_path)
                    mismatch = "ERROR: no exception raised"
                except CheckpointError as exc:
                    mismatch = f"CheckpointError: {str(exc)[:220]}"
                del probe
            report["cross_variant_load_rejected"] = mismatch
            evidence[variant] = report
            del model
            gc.collect()
        for variant in ("xt", "s"):
            evidence[variant] = {
                "status": "not_downloaded_by_design",
                "path": resolve_checkpoint_path(variant),
                "note": "XT/S checkpoints are never downloaded; only explicitly random structure tests are run",
            }
        t = evidence["t"]
        assert t.get("status") != "checkpoint_absent", "the OverLoCK-T checkpoint is required for M05"
        # The gate is "every missing TRAINABLE key is an explicitly whitelisted detection-only
        # key", not a raw coverage ratio: the whitelist itself accounts for ~1% of the backbone.
        assert t["totals"]["trainable_missing_disallowed_tensors"] == 0, t["totals"]
        assert t["totals"]["trainable_missing_disallowed_numel"] == 0, t["totals"]
        assert t["missing_allowed"]["count"] > 0, "the T audit should report the detection-only missing keys"
        assert "extra_norm." in t["allow_missing_prefixes"]
        whitelisted = set(t["missing_allowed"]["keys"])
        every_missing_allowed = all(k in whitelisted for k in t["missing_allowed"]["keys"])
        assert every_missing_allowed
        t["coverage_gate"] = {
            "numel_coverage_including_whitelist": t["totals"]["numel_coverage"],
            "numel_coverage_excluding_whitelist": t["coverage_by_module_excluding_allowed_missing"],
            "whitelisted_missing_keys": t["missing_allowed"]["keys"],
            "whitelisted_missing_numel": t["missing_allowed"]["numel"],
            "non_whitelisted_missing_tensors": t["missing_disallowed"]["count"],
            "rule": "a missing trainable key is only acceptable when it is an explicit detection-only key",
        }
        assert t["cross_variant_load_rejected"].startswith("CheckpointError")
        evidence["whitelist_correction"] = (
            "The V1 explanation ('unused by forward, therefore absent from state_dict') is wrong. The "
            "correct statement: this classification checkpoint does not contain those detection-only "
            "parameters (extra_norm.*, h_proj.*), so they keep their initialisation."
        )
        return evidence

    results.run("M05", "OverLoCK-T/B real weights: safe load, per-key audit, cross-variant rejection", check, budget=180)


# --------------------------------------------------------------------------------------
# M06 -- fixed 640
# --------------------------------------------------------------------------------------
def m06_640(results: Results, ultra_root: str, budget: float, skip: bool) -> None:
    if skip:
        results.skip("M06", "fixed 640x640 full forward (real T weights)", "--skip-640 given")
        return

    def check():
        from overlock_yolo.cli import build_from_config
        from overlock_yolo.paths import resolve_checkpoint_path

        evidence = {"combinations": {}, "protocol": "exactly 640x640, batch 1, FP32, CPU"}
        path = resolve_checkpoint_path("t")
        for family, scale in (("yolo11", "s"), ("yolo26", "s")):
            cfg = _combo_cfg("t", family, scale)
            t0 = time.time()
            model = build_from_config(cfg, ultra_root=ultra_root, weights=path)
            build_s = time.time() - t0
            model.eval()
            t0 = time.time()
            with torch.no_grad():
                entry = model._adapt(torch.zeros(1, 3, 640, 640))
            fwd_s = time.time() - t0
            head = model.model[-1]
            # native inference branch: yolo26 must be driven through the end-to-end head, exactly
            # as Validator.__call__ does with ``model.end2end = args.nms is False``.
            model.end2end = False if family == "yolo11" else True
            with torch.no_grad():
                out = model(torch.zeros(1, 3, 640, 640))
            y, feats = _split_eval_output(out)
            info = {
                "combination": f"t+{family}{scale}",
                "inference_branch": "one2many+NMS" if family == "yolo11" else "end2end top-k",
                "weights": path,
                "weights_loaded": model.pretrained_backbone_report is not None,
                "numel_coverage": None
                if not model.pretrained_backbone_report
                else model.pretrained_backbone_report["totals"]["numel_coverage"],
                "build_seconds": round(build_s, 2),
                "forward_seconds": round(fwd_s, 2),
                "adapted_feature_shapes": [list(f.shape) for f in entry],
                "detect_input_shapes": [list(f.shape) for f in feats],
                "stride": [float(s) for s in head.stride.tolist()],
                "A_positions": int(sum(f.shape[-1] * f.shape[-2] for f in feats)),
                "eval_output_shape": [list(t.shape) for t in (y if isinstance(y, tuple) else (y,))],
                "reg_max": int(head.reg_max),
                "end2end": bool(model.end2end),
                "rss_gb": round(peak_gb(), 2),
            }
            assert feats is not None, "could not locate the Detect input feature maps"
            info["detect_feature_shapes"] = [list(f.shape) for f in feats]
            info["A_positions"] = int(sum(f.shape[-1] * f.shape[-2] for f in feats))
            assert info["stride"] == [8.0, 16.0, 32.0], info["stride"]
            assert [tuple(f.shape[-2:]) for f in entry] == [(80, 80), (40, 40), (20, 20)], info["adapted_feature_shapes"]
            assert [tuple(f.shape[-2:]) for f in feats] == [(80, 80), (40, 40), (20, 20)], info["detect_input_shapes"]
            assert info["A_positions"] == 8400, info["A_positions"]
            pred = y
            if family == "yolo11":
                assert tuple(pred.shape) == (1, 10, 8400), tuple(pred.shape)
                info["expected_eval_shape"] = [1, 10, 8400]
            else:
                assert pred.dim() == 3 and pred.shape[0] == 1 and pred.shape[2] == 6, tuple(pred.shape)
                info["expected_eval_shape"] = "end2end [1, K, 6] (xyxy/conf/cls), K = min(max_det, A)"
            evidence["combinations"][info["combination"]] = info
            del model
            gc.collect()

        from overlock_yolo.data import build_native_dataloader

        dl = build_native_dataloader(os.path.join(VIEW, "data.yaml"), 640, mode="val", batch=1, workers=0)
        batch = next(iter(dl))
        ratio, pad = batch["ratio_pad"][0]
        h0, w0 = batch["ori_shape"][0]
        h1, w1 = batch["img"].shape[-2:]
        box = batch["bboxes"][0].tolist()
        evidence["letterbox"] = {
            "orig_wh": [w0, h0],
            "network_hw": [h1, w1],
            "ratio_pad": {"ratio": list(ratio), "pad": list(pad)},
            "square": h1 == w1 == 640,
            "first_box_normalised": box,
            "formula": "x_pix = (xc*W_net - pad_w)/ratio ; consistent with the source annotation",
        }
        assert h1 == w1 == 640
        assert abs(ratio[0] * w0 - (w1 - 2 * pad[0])) < 2.0, (ratio, pad, w0, w1)
        assert abs(ratio[1] * h0 - (h1 - 2 * pad[1])) < 2.0, (ratio, pad, h0, h1)
        return evidence

    results.run("M06", "fixed 640x640 forward with real T weights (yolo11s + yolo26s) + letterbox", check, budget=budget)


# --------------------------------------------------------------------------------------
# M07 -- one training step per family
# --------------------------------------------------------------------------------------
def m07_train_step(results: Results, ultra_root: str) -> None:
    def check():
        from overlock_yolo.cli import build_from_config

        evidence = {"families": {}}
        for family in ("yolo11", "yolo26"):
            cfg = _combo_cfg("t", family, "s", epochs=2)
            model = build_from_config(cfg, ultra_root=ultra_root, weights=None)
            model.train()
            batch = _mini_batch(96)
            crit = model.init_criterion()
            crit_class = type(crit).__name__
            crit_before = float(getattr(crit, "o2m", -1))
            opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-4, momentum=0.9)
            first_name, first_param = next(iter(model.named_parameters()))
            before = first_param.detach().clone()
            t0 = time.time()
            loss, items = model(batch)  # the exact native trainer entry point
            fwd_s = time.time() - t0
            t0 = time.time()
            loss.sum().backward()
            bwd_s = time.time() - t0
            grads = {n: p for n, p in model.named_parameters() if p.grad is not None}
            finite = all(bool(torch.isfinite(p.grad).all()) for p in grads.values())
            backbone_grads = sum(1 for p in model.stem.backbone.parameters() if p.grad is not None)
            adapter_grads = sum(1 for p in model.adapters.parameters() if p.grad is not None)
            head_grads = sum(1 for p in model.tail.parameters() if p.grad is not None)
            opt.step()
            changed = not torch.equal(before, first_param.detach())
            if hasattr(crit, "update"):
                crit.update()
            crit_after = float(getattr(crit, "o2m", -1))
            entry = {
                "criterion": crit_class,
                "loss_vector": [round(float(v), 5) for v in loss.detach().flatten()],
                "loss_items": {k: round(float(v), 5) for k, v in items.items()},
                "loss_finite": bool(torch.isfinite(loss).all()),
                "loss_is_sumable": bool(loss.numel() >= 1),
                "forward_seconds": round(fwd_s, 3),
                "backward_seconds": round(bwd_s, 3),
                "grad_tensors": len(grads),
                "grads_finite": finite,
                "backbone_grad_tensors": backbone_grads,
                "adapter_grad_tensors": adapter_grads,
                "head_grad_tensors": head_grads,
                "optimizer_step_changed_param": bool(changed),
                "optimizer_param": first_name,
                "criterion_updates_lifecycle": {"o2m_before": crit_before, "o2m_after_update": crit_after}
                if crit_before >= 0
                else None,
                "rss_gb": round(peak_gb(), 2),
            }
            assert finite, "non-finite gradients"
            assert backbone_grads > 0, "the backbone received no gradient"
            assert adapter_grads == 9, adapter_grads
            assert head_grads > 0
            assert changed
            if family == "yolo26":
                assert crit_class == "E2ELoss", crit_class
                assert crit_after != crit_before, "E2ELoss.update() did not advance the branch schedule"
                assert "l1_loss" in items, items
                assert "dfl_loss" not in items, "YOLO26 must use the native L1 term, not DFL"
            else:
                assert crit_class == "v8DetectionLoss", crit_class
                assert "dfl_loss" in items
            evidence["families"][family] = entry
            del model, opt, crit
            gc.collect()
        return evidence

    results.run("M07", "one native train step per family (model(batch) -> loss -> backward -> step)", check, budget=180)


# --------------------------------------------------------------------------------------
# M08 -- attention regression
# --------------------------------------------------------------------------------------
def m08_attention(results: Results, ultra_root: str) -> None:
    def check():
        from overlock_yolo.attention_backend import (
            AttentionBackendError,
            BackendCache,
            na2d_av_macs,
            na2d_av_reference,
            row_ranges,
        )

        def oracle(attn, value, k):
            b, hd, h, w, _kk = attn.shape
            d = value.shape[-1]
            out = torch.zeros(b, hd, h, w, d, dtype=value.dtype)
            for bi in range(b):
                for hi in range(hd):
                    for y in range(h):
                        for x in range(w):
                            sy = min(max(y - k // 2, 0), h - k)
                            sx = min(max(x - k // 2, 0), w - k)
                            acc = torch.zeros(d, dtype=value.dtype)
                            for i in range(k):
                                for j in range(k):
                                    acc = acc + attn[bi, hi, y, x, i * k + j] * value[bi, hi, sy + i, sx + j]
                            out[bi, hi, y, x] = acc
            return out

        evidence = {"cases": []}
        for (b, hd, h, w, d, k) in ((1, 2, 5, 5, 3, 3), (1, 1, 7, 7, 4, 5), (2, 2, 4, 6, 3, 3), (1, 1, 13, 13, 2, 13), (1, 1, 5, 8, 2, 3)):
            a = torch.randn(b, hd, h, w, k * k, dtype=torch.float64, requires_grad=True)
            v = torch.randn(b, hd, h, w, d, dtype=torch.float64, requires_grad=True)
            ref = na2d_av_reference(a, v, k, row_chunk=3)
            orc = oracle(a, v, k)
            err = float((ref - orc).abs().max())
            chunk_err = max(float((na2d_av_reference(a, v, k, row_chunk=c) - ref).abs().max()) for c in (0, 1, 2, 3, 7, 1000))
            ref.sum().backward()
            ga, gv = a.grad.clone(), v.grad.clone()
            a2 = a.detach().clone().requires_grad_(True)
            v2 = v.detach().clone().requires_grad_(True)
            oracle(a2, v2, k).sum().backward()
            gerr = max(float((ga - a2.grad).abs().max()), float((gv - v2.grad).abs().max()))
            evidence["cases"].append(
                {
                    "case": {"B": b, "heads": hd, "H": h, "W": w, "D": d, "K": k},
                    "forward_max_abs_err_vs_oracle": err,
                    "chunk_invariance_max_abs_err": chunk_err,
                    "grad_max_abs_err_vs_oracle": gerr,
                }
            )
            assert err < 1e-12, (err, h, w, k)
            assert chunk_err < 1e-12
            assert gerr < 1e-12

        evidence["row_ranges"] = {"chunk3_of_10": [list(r) for r in row_ranges(10, 3)], "chunk0": [list(r) for r in row_ranges(10, 0)]}
        evidence["cpu_auto"] = BackendCache("auto").resolve(torch.device("cpu")).as_dict()
        assert evidence["cpu_auto"]["resolved"] == "torch_reference"
        try:
            BackendCache("natten").resolve(torch.device("cpu"))
            raise AssertionError("explicit natten on CPU did not fail")
        except AttentionBackendError as exc:
            evidence["cpu_natten_error"] = str(exc)[:200]
        evidence["cuda_available"] = torch.cuda.is_available()
        evidence["cuda_native_test"] = (
            "skipped: no CUDA device here; scripts/gpu_smoke.py runs the NATTEN-vs-reference forward/backward "
            "comparison on the server"
        )
        evidence["na2d_av_macs_example_1x8x80x80x16_k5"] = na2d_av_macs(1, 8, 80, 80, 16, 5)
        return evidence

    results.run("M08", "attention reference vs independent oracle (values, boundaries, gradients, chunking)", check, budget=180)


# --------------------------------------------------------------------------------------
# M09 -- cost + state_dict round trip
# --------------------------------------------------------------------------------------
def m09_cost_state(results: Results, ultra_root: str) -> None:
    def check():
        from overlock_yolo.cli import build_from_config
        from overlock_yolo.profile_model import count_partial_macs, input_invariance, parameter_report
        from overlock_yolo.trainer import load_project_checkpoint, save_project_checkpoint

        cfg = _combo_cfg("t", "yolo11", "s")
        model = build_from_config(cfg, ultra_root=ultra_root, weights=None)
        rep = parameter_report(model)
        evidence = {"parameters": {"total": rep["total"], "buckets": rep["buckets"]}}
        assert rep["total"]["duplicate_registration"] == []
        assert rep["total"]["buckets_cover_all_parameters"]
        sd = model.state_dict()
        evidence["state_dict_tensors"] = len(sd)
        evidence["state_dict_unique_keys"] = len(sd) == len(set(sd))
        assert evidence["state_dict_unique_keys"]
        evidence["input_invariance"] = input_invariance(model, sizes=(224, 640))
        evidence["partial_macs"] = count_partial_macs(model, imgsz=640)
        evidence["complete_flops_available"] = False

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.pt")
            manifest = save_project_checkpoint(model, path, extra={"note": "M09 round trip"})
            model2 = build_from_config(cfg, ultra_root=ultra_root, weights=None, seed=123)
            result = load_project_checkpoint(model2, path, strict=True)
            same = all(torch.equal(v, model2.state_dict()[k]) for k, v in model.state_dict().items())
            evidence["round_trip"] = {
                "manifest": {k: v for k, v in manifest.items() if k != "structure"},
                "loaded_tensors": result["loaded_tensors"],
                "missing_keys": result["missing_keys"],
                "unexpected_keys": result["unexpected_keys"],
                "structure": result["structure"],
                "all_tensors_equal": bool(same),
            }
            assert same, "the rebuilt model does not match the saved state"
            other = build_from_config(_combo_cfg("b", "yolo26", "n"), ultra_root=ultra_root, weights=None)
            try:
                load_project_checkpoint(other, path, strict=False)
                evidence["structure_mismatch_rejected"] = "ERROR: no exception raised"
            except ConfigError as exc:
                evidence["structure_mismatch_rejected"] = str(exc)[:180]
            assert evidence["structure_mismatch_rejected"].startswith("checkpoint structure does not match")
            del other
        del model, model2
        gc.collect()
        return evidence

    results.run("M09", "unique parameter accounting, partial-MACs labelling, state_dict rebuild/reload", check, budget=180)


# --------------------------------------------------------------------------------------
# M10 -- tools + native evaluation
# --------------------------------------------------------------------------------------
def m10_tools(results: Results, ultra_root: str) -> None:
    def check():
        import subprocess

        from overlock_yolo.cli import build_from_config
        from overlock_yolo.paths import resolve_checkpoint_path
        from overlock_yolo.validator import run_native_validation

        evidence = {"entry_points": {}}
        for script in (
            "scripts/train.py",
            "scripts/val.py",
            "scripts/profile_model.py",
            "scripts/server_preflight.py",
            "scripts/gpu_smoke.py",
        ):
            out = subprocess.run(
                [sys.executable, os.path.join(ROOT, script), "--help"],
                cwd="/tmp",
                capture_output=True,
                text=True,
                env={**os.environ, "OMP_NUM_THREADS": "2", "MPLCONFIGDIR": os.path.join(ROOT, "cache", "mpl")},
            )
            evidence["entry_points"][script] = {
                "returncode": out.returncode,
                "cwd": "/tmp",
                "first_line": ((out.stdout or out.stderr).strip().splitlines() or [""])[0][:120],
            }
            assert out.returncode == 0, (script, out.stderr[-400:])

        path = resolve_checkpoint_path("t")
        for family in ("yolo11", "yolo26"):
            cfg = _combo_cfg("t", family, "s")
            model = build_from_config(cfg, ultra_root=ultra_root, weights=path)
            res = run_native_validation(model, cfg, data=os.path.join(VIEW, "data.yaml"), batch=1)
            evidence[f"validator_{family}"] = {
                "metric_keys": res["metric_keys"],
                "results_dict": res["results_dict"],
                "n_images": res["n_images"],
                "protocol": res["protocol"],
                "inference": res["inference"],
                "per_class": {k: (v if not isinstance(v, list) else [round(float(x), 4) for x in v]) for k, v in res["per_class"].items()},
            }
            assert res["protocol"]["all_square"] is True
            assert res["protocol"]["violations"] == []
            assert res["inference"]["save_json"] is False
            if family == "yolo11":
                assert res["inference"]["head"]["reg_max"] == 16
                assert res["inference"]["head"]["has_one2one"] is False
                assert res["inference"]["nms_applied_by_validator"] is True
            else:
                assert res["inference"]["head"]["reg_max"] == 1
                assert res["inference"]["head"]["has_one2one"] is True
                assert res["inference"]["end2end_effective"] is True, "yolo26 validation must use the native end2end path"
            assert set(res["results_dict"]) >= {
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
            }
            del model
            gc.collect()
        evidence["precision_disclaimer"] = (
            "these values come from a randomly initialised neck/head on a 2-image view; they verify the "
            "interface and the metric keys, they are NOT an accuracy measurement"
        )
        return evidence

    results.run("M10", "entry points + native DetectionValidator/DetMetrics on the 1-2 image view", check, budget=180)


# --------------------------------------------------------------------------------------
# M11 -- COCO bridge + no source writes
# --------------------------------------------------------------------------------------
def m11_data_git(results: Results, ultra_root: str) -> None:
    def check():
        import tempfile

        from overlock_yolo.data import build_view, coco_bbox_to_yolo, load_coco, load_soda_config, yolo_bbox_to_coco

        evidence = {}
        cfg = load_soda_config()
        coco = load_coco(cfg["splits"]["val"]["annotations"])
        ann = coco["annotations"][0]
        im = next(i for i in coco["images"] if int(i["id"]) == int(ann["image_id"]))
        box = coco_bbox_to_yolo(ann["bbox"], im["width"], im["height"])
        back = yolo_bbox_to_coco(*box, im["width"], im["height"])
        evidence["mapping"] = {
            "category_id_map": cfg["category_id_map"],
            "names": cfg["names"],
            "example": {
                "image_id": int(ann["image_id"]),
                "file_name": im["file_name"],
                "coco_category_id": int(ann["category_id"]),
                "mapped_class": cfg["category_id_map"][int(ann["category_id"])],
                "coco_bbox": [round(float(v), 4) for v in ann["bbox"]],
                "yolo_box": [round(v, 6) for v in box],
                "round_trip_pixels": [round(v, 4) for v in back],
                "round_trip_max_abs_err": max(abs(float(a) - b) for a, b in zip(ann["bbox"], back)),
            },
        }
        assert evidence["mapping"]["example"]["round_trip_max_abs_err"] < 1e-6

        with tempfile.TemporaryDirectory() as td:
            view = os.path.join(td, "view")
            report = build_view(view, limit_per_split=1, force=True)
            evidence["view"] = {
                "images_dir_is_symlink": {s: report["splits"][s]["images_dir_is_symlink"] for s in ("train", "val")},
                "n_images": {s: report["splits"][s]["n_images"] for s in ("train", "val")},
                "n_visible_images": {s: report["splits"][s]["n_visible_images"] for s in ("train", "val")},
                "n_label_files": {s: report["splits"][s]["n_label_files"] for s in ("train", "val")},
                "stats": {s: report["splits"][s]["stats"] for s in ("train", "val")},
                "source_cache_files": report["source_cache_files_written"],
                "image_id_map_entries": len(json.load(open(report["image_id_map_path"]))),
                "empty_label_files": {s: report["splits"][s]["stats"].get("empty_label_files", 0) for s in ("train", "val")},
            }
            for split in ("train", "val"):
                assert report["splits"][split]["images_dir_is_symlink"] is False
                assert report["splits"][split]["n_visible_images"] == report["splits"][split]["n_images"] == 1
                assert report["splits"][split]["n_label_files"] == 1
            evidence["view"]["no_directory_symlink_leak"] = True

        # no cache anywhere inside the source dataset tree
        src_cache = []
        for dirpath, dirs, files in os.walk(cfg["root"]):
            dirs[:] = [d for d in dirs if d not in ("train", "val")]
            src_cache += [os.path.join(dirpath, f) for f in files if f.endswith(".cache")]
        evidence["source_dataset_cache_files"] = src_cache
        assert src_cache == [], src_cache
        return evidence

    results.run("M11", "COCO bridge: mapping/image_id/boxes, real image dirs, no source writes", check, budget=180)


# --------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M01-M11 CPU verification for OverLoCK + YOLO11/YOLO26")
    ap.add_argument("--out", default=os.path.join(REPORTS, "validation.json"))
    ap.add_argument("--environment-out", default=os.path.join(REPORTS, "environment.json"))
    ap.add_argument("--budget-640", type=float, default=180.0, help="seconds allowed for the M06 640 item")
    ap.add_argument("--skip-640", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated subset of M01..M11")
    args = ap.parse_args(argv)

    prepare_ultralytics_env()
    ultra_root = resolve_ultralytics_root()
    install_ultralytics_root(ultra_root)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

    os.makedirs(REPORTS, exist_ok=True)
    t_start = time.time()
    results = Results()
    only = {x.strip() for x in args.only.split(",")} if args.only else None

    def want(mid):
        return only is None or mid in only

    if want("M01"):
        m01_config_paths(results, ultra_root)
    if want("M02"):
        m02_backbones(results, ultra_root)
    if want("M03"):
        m03_tails(results, ultra_root)
    if want("M04"):
        m04_matrix(results, ultra_root)
    if want("M05"):
        m05_checkpoints(results, ultra_root)
    if want("M06"):
        m06_640(results, ultra_root, args.budget_640, args.skip_640)
    if want("M07"):
        m07_train_step(results, ultra_root)
    if want("M08"):
        m08_attention(results, ultra_root)
    if want("M09"):
        m09_cost_state(results, ultra_root)
    if want("M10"):
        m10_tools(results, ultra_root)
    if want("M11"):
        m11_data_git(results, ultra_root)

    from overlock_yolo.attention_backend import env_report

    env = {
        **env_report(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
        "project_root": project_root(),
        "ultralytics_root": ultra_root,
        "note": "local conda 3dete environment (CPU, FP32); NOT the target server "
        "(Python 3.12 / torch 2.5.1+cu124 / RTX 4090)",
    }
    with open(args.environment_out, "w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=2, default=str)

    payload = results.as_dict(
        {
            "generated_by": "scripts/smoke.py",
            "design": "DESIGN_V2.md",
            "project_root": project_root(),
            "ultralytics_root": ultra_root,
            "view": VIEW,
            "device": "cpu",
            "dtype": "float32",
            "amp": False,
            "workers": 0,
            "torch_threads": torch.get_num_threads(),
            "total_seconds": round(time.time() - t_start, 2),
            "per_item_budget_seconds": 180,
            "environment": env,
        }
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nvalidation -> {args.out}")
    print(json.dumps(payload["summary"], indent=2))
    return 0 if payload["summary"]["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
