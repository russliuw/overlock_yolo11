"""Native Ultralytics detection validation with a fixed 640x640 (square) protocol.

Contract (DESIGN_V2.md 5, 10.3)
------------------------------
* The primary metrics are the ones produced by the *pinned native* ``DetectionValidator`` and
  ``DetMetrics``: precision, recall, mAP50, mAP50-95, per-class values and ``results_dict``.
  Nothing here re-implements AP integration, class averaging or IoU matching, and COCOeval is
  never substituted for the primary number.
* The input protocol is an equal-ratio resize + letterbox to **exactly** ``imgsz x imgsz``
  (single ``float/255`` conversion; the ImageNet mean/std normalisation happens once inside the
  detector's stem).  ``rect``/``auto`` padding can silently produce a non-square batch, so the
  transform is pinned (``rect_shape`` removed, ``auto=False``) *and* every batch shape is
  asserted and recorded.
* YOLO11 uses the native NMS path; YOLO26 uses the native end-to-end (top-k) path with no extra
  hand-written NMS on top.  ``save_json`` stays off so the native COCO branch cannot swap the
  primary metric for a COCOeval number.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch

from .paths import prepare_ultralytics_env  # noqa: F401  (documented: CLIs call this first)

__all__ = [
    "SquareLetterBox",
    "FixedSquareDetectionValidator",
    "validator_namespace",
    "build_validation_args",
    "run_native_validation",
]


def validator_namespace() -> dict:
    """Native validation classes (requires the pinned source to be installed already)."""
    from ultralytics.data.augment import LetterBox
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils.metrics import DetMetrics

    return {"LetterBox": LetterBox, "DetectionValidator": DetectionValidator, "DetMetrics": DetMetrics}


def _square_letterbox_class():
    """Build the fixed-square ``LetterBox`` subclass once."""
    LetterBox = validator_namespace()["LetterBox"]

    class SquareLetterBox(LetterBox):
        """LetterBox pinned to a square output: equal-ratio resize + pad, never a rectangle.

        The parent honours a ``rect_shape`` entry that the native dataloader stores on the label
        dict when ``rect=True``; removing it (plus ``auto=False``) makes the effective target the
        configured ``(imgsz, imgsz)`` for every sample, and the result is verified afterwards so
        a future Ultralytics change cannot quietly turn the protocol rectangular.
        """

        def __init__(self, new_shape, auto: bool = False, scaleup: bool = False, padding_value: int = 114, **kwargs):
            super().__init__(
                new_shape=new_shape,
                auto=False,
                scale_fill=False,
                scaleup=scaleup,
                center=True,
                stride=int(kwargs.pop("stride", 32)),
                padding_value=int(padding_value),
            )
            self.requested_shape = tuple(int(v) for v in (new_shape if not isinstance(new_shape, int) else (new_shape, new_shape)))

        def get_params(self, labels: dict) -> dict:
            labels.pop("rect_shape", None)  # the only path to a rectangular batch
            params = super().get_params(labels)
            h, w = params["new_unpad"]
            if params["top"] < 0 or params["left"] < 0:
                raise AssertionError(f"letterbox produced negative padding: {params}")
            return params

        def __call__(self, labels=None, image=None):
            labels = super().__call__(labels=labels, image=image)
            if isinstance(labels, dict):
                h, w = labels["img"].shape[:2]
                if (h, w) != self.requested_shape:
                    raise AssertionError(
                        f"fixed-square letterbox produced {h}x{w}, expected {self.requested_shape[0]}x"
                        f"{self.requested_shape[1]}; rect/auto padding must not change the 640 protocol"
                    )
            return labels

    SquareLetterBox.__name__ = "SquareLetterBox"
    SquareLetterBox.__qualname__ = "SquareLetterBox"
    SquareLetterBox.__module__ = __name__
    return SquareLetterBox


_SQUARE_LETTERBOX = None


def SquareLetterBox(*args, **kwargs):  # noqa: N802 - mirrors the native class name
    """Factory for the pinned square ``LetterBox`` (cached per process)."""
    global _SQUARE_LETTERBOX
    if _SQUARE_LETTERBOX is None:
        _SQUARE_LETTERBOX = _square_letterbox_class()
    return _SQUARE_LETTERBOX(*args, **kwargs)


def _validator_class():
    DetectionValidator = validator_namespace()["DetectionValidator"]

    class FixedSquareDetectionValidator(DetectionValidator):
        """``DetectionValidator`` that pins the square letterbox and records protocol evidence."""

        def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
            # NOTE: this checkout's DetectionValidator signature is
            # ``(dataloader=None, save_dir=None, args=None, _callbacks=None)`` -- no pbar.
            super().__init__(dataloader, save_dir, args, _callbacks)
            self.protocol_records: List[dict] = []
            self.protocol_violations: List[str] = []

        # -------------------------------------------------------------- protocol
        def _pin_square_transform(self, dataset) -> dict:
            """Replace only the *validation* LetterBox; training augmentation is untouched."""
            compose = getattr(dataset, "transforms", None)
            if compose is None:
                return {"patched": False, "reason": "dataset has no transforms"}
            replaced = 0
            for i, tf in enumerate(getattr(compose, "transforms", [])):
                name = type(tf).__name__
                if name in ("LetterBox", "SquareLetterBox"):
                    new = SquareLetterBox(
                        new_shape=(int(self.args.imgsz), int(self.args.imgsz)),
                        auto=False,
                        scaleup=bool(validation_options(self.args)["val_scaleup"]),
                        stride=32,
                    )
                    compose.transforms[i] = new
                    replaced += 1
            return {
                "patched": replaced > 0,
                "n_replaced": replaced,
                "imgsz": int(self.args.imgsz),
                "auto": False,
                "scale_fill": False,
                "scaleup": bool(validation_options(self.args)["val_scaleup"]),
                "padding_value": 114,
                "rect_arg": bool(getattr(self.args, "rect", False)),
                "multi_scale_arg": getattr(self.args, "multi_scale", False),
            }

        def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
            dataset = super().build_dataset(img_path, mode, batch)
            self.square_transform_report = self._pin_square_transform(dataset)
            if mode == "val" and not self.square_transform_report.get("patched"):
                raise AssertionError(
                    "could not pin the validation LetterBox to a square shape; refusing to run a "
                    "validation whose protocol is not the fixed 640x640 one"
                )
            return dataset

        # -------------------------------------------------------------- evidence
        def preprocess(self, batch: dict):
            batch = super().preprocess(batch)
            img = batch["img"]
            h, w = int(img.shape[-2]), int(img.shape[-1])
            ratio_pad = batch.get("ratio_pad")
            record = {
                "batch": len(self.protocol_records),
                "shape": [int(v) for v in img.shape],
                "square": h == w,
                "dtype": str(img.dtype),
            }
            if ratio_pad is not None:
                recs = []
                for rp in ratio_pad:
                    ratio, pad = rp
                    recs.append({"ratio": [float(ratio[0]), float(ratio[1])], "pad": [float(pad[0]), float(pad[1])]})
                record["ratio_pad_first"] = recs[0] if recs else None
            self.protocol_records.append(record)
            if h != w:
                self.protocol_violations.append(f"batch {record['batch']} is {h}x{w}, not square")
            if h != int(self.args.imgsz):
                self.protocol_violations.append(f"batch {record['batch']} has height {h} != imgsz {self.args.imgsz}")
            return batch

        # -------------------------------------------------------------- reporting
        def protocol_report(self) -> dict:
            net = getattr(self, "square_transform_report", {})
            return {
                "transform": net,
                "batches": self.protocol_records,
                "all_square": all(r["square"] for r in self.protocol_records) if self.protocol_records else None,
                "all_imgsz": all(r["shape"][-1] == int(self.args.imgsz) for r in self.protocol_records)
                if self.protocol_records
                else None,
                "violations": self.protocol_violations,
            }

        def inference_report(self, native=None) -> dict:
            """What actually ran: thresholds, the selected branch and the head's real attributes.

            The native validator releases ``self.model`` after the pass (it is an inference-mode
            cleanup), so the caller passes the model back in for the head introspection.
            """
            native = native if native is not None else getattr(self, "model", None)
            head = native.model[-1] if native is not None and hasattr(native, "model") else None
            return {
                "conf": float(self.args.conf),
                "iou": float(self.args.iou),
                "max_det": int(self.args.max_det),
                "nms_arg": self.args.nms,
                "end2end_effective": bool(getattr(self, "end2end", False)),
                "save_json": bool(self.args.save_json),
                "imgsz": int(self.args.imgsz),
                "rect": bool(self.args.rect),
                "multi_scale": getattr(self.args, "multi_scale", False),
                "head": None
                if head is None
                else {
                    "class": type(head).__name__,
                    "reg_max": int(getattr(head, "reg_max", -1)),
                    "has_one2one": getattr(head, "one2one_cv2", None) is not None,
                    "max_det_attr": getattr(head, "max_det", None),
                    "agnostic_nms_attr": getattr(head, "agnostic_nms", None),
                },
                "nms_applied_by_validator": not bool(getattr(self, "end2end", False)),
            }

    FixedSquareDetectionValidator.__name__ = "FixedSquareDetectionValidator"
    FixedSquareDetectionValidator.__qualname__ = "FixedSquareDetectionValidator"
    FixedSquareDetectionValidator.__module__ = __name__
    return FixedSquareDetectionValidator


_VALIDATOR_CLASS = None


def validator_class():
    global _VALIDATOR_CLASS
    if _VALIDATOR_CLASS is None:
        _VALIDATOR_CLASS = _validator_class()
    return _VALIDATOR_CLASS


def build_validation_args(
    cfg: dict,
    *,
    data: Optional[str] = None,
    view: Optional[str] = None,
    batch: int = 1,
    split: str = "val",
):
    """Native ``IterableSimpleNamespace`` args for a detection validation run."""
    from ultralytics.cfg import get_cfg

    t = cfg["train"]
    res = cfg["_resolved"]
    overrides = {
        "task": "detect",
        "mode": "val",
        "data": data or res["data_yaml"].get("resolved") or view,
        "imgsz": int(t["imgsz"]),
        "batch": int(batch),
        "device": cfg["runtime"]["device"],
        "workers": int(cfg["runtime"]["workers"]),
        "rect": False,  # must stay False: the protocol is a fixed square letterbox
        "multi_scale": False,
        "conf": 0.001 if t["conf"] is None else float(t["conf"]),
        "iou": 0.7 if t["iou"] is None else float(t["iou"]),
        "max_det": 300 if t["max_det"] is None else int(t["max_det"]),
        "save_json": False,  # keep DetMetrics as the primary metric
        "plots": False,
        "augment": False,
        "split": split,
        "agnostic_nms": False,
        "single_cls": False,
    }
    # ``val_scaleup`` is ours, not a native YOLO argument: it is attached after validation so
    # ``get_cfg`` cannot reject it.  ``nms`` selects the native inference path: the family default
    # is YOLO11 -> NMS, YOLO26 -> end-to-end (top-k, no NMS), and an explicit train.nms overrides.
    nms = t["nms"]
    if nms is None:
        nms = cfg["_resolved"]["yolo_family"] != "yolo26"
    overrides["nms"] = bool(nms)
    args = get_cfg(overrides=overrides)
    # NOTE: BaseValidator re-runs ``get_cfg(overrides=vars(args))``, so the namespace may only
    # contain real YOLO keys -- our own options live in this side table instead.
    return args


#: Per-namespace side table for options that are ours and must not leak into ``get_cfg``.
_EXTRA_ARGS: "dict[int, dict]" = {}


def validation_options(args) -> dict:
    """Project-specific validation options for a native args namespace."""
    return _EXTRA_ARGS.get(id(args), {"val_scaleup": False})


def run_native_validation(
    model,
    cfg: dict,
    *,
    data: Optional[str] = None,
    view: Optional[str] = None,
    batch: int = 1,
    split: str = "val",
    max_batches: Optional[int] = None,
) -> dict:
    """Run the native validator over ``1..n`` batches and return metrics + protocol evidence."""
    args = build_validation_args(cfg, data=data, view=view, batch=batch, split=split)
    _EXTRA_ARGS[id(args)] = {"val_scaleup": bool(cfg["train"]["val_scaleup"])}
    validator = validator_class()(args=args)
    model.eval()
    model.args = args
    # The native engine only flips the head's inference branch inside the *training* validation
    # path (``Validator.__call__`` does ``model.end2end = args.nms is False`` when ``trainer`` is
    # set).  Running the validator directly on a model therefore has to select the same branch
    # explicitly, otherwise a YOLO26 detector would silently validate on the one-to-many output
    # that the native end-to-end path does not use.
    if hasattr(model, "end2end"):
        model.end2end = args.nms is False
    results = validator(model=model)
    report = {
        "metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in dict(results).items()},
        "results_dict": {k: float(v) for k, v in validator.metrics.results_dict.items()},
        "metric_keys": list(validator.metrics.keys),
        "per_class": {
            "names": {int(k): str(v) for k, v in validator.metrics.names.items()},
            "p": [float(v) for v in validator.metrics.box.p],
            "r": [float(v) for v in validator.metrics.box.r],
            "ap50": [float(v) for v in validator.metrics.box.ap50],
            "ap": [float(v) for v in validator.metrics.box.ap],
        },
        "protocol": validator.protocol_report(),
        "inference": validator.inference_report(native=model),
        "n_images": int(validator.seen),
        "n_batches": len(validator.protocol_records),
        "max_batches_requested": max_batches,
    }
    if max_batches is not None:
        report["note"] = (
            f"the dataloader ran to completion over the {report['n_images']} images of the view; "
            "this is an interface check on a 1-2 image smoke view, NOT an accuracy measurement"
        )
    return report
