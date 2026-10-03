"""SODA10M COCO -> isolated native-YOLO data view (DESIGN_V2.md 10.1).

Why a view exists
-----------------
``soda10m.yaml`` is a *custom COCO-format* config.  The official OverLoCK detection code reads
COCO through ``mmdet.CocoDataset`` + ``DefaultFormatBundle``/``Collect``, which emit MMCV
``DataContainer`` objects with ``gt_bboxes``/``gt_labels``/``gt_masks`` and its own
normalisation pipeline.  That is **not** the ``img``/``cls``/``bboxes``/``batch_idx`` batch an
Ultralytics detector consumes, and installing the old MMCV/MMDetection stack just to re-read the
same JSON would give two inconsistent augmentation/normalisation paths.

What we reuse instead
---------------------
The COCO parsing/mapping/box maths already implemented here (and nothing MMCV-related), plus the
*native* Ultralytics loader, augmentation and collate.  The source JSON and images stay the
single source of truth: nothing is ever written back into the source tree.

Fixes over the previous version (DESIGN_V2.md 10.1)
--------------------------------------------------
* ``images/{train,val}`` are always **real directories** containing one symlink per *selected*
  image.  The old code symlinked the whole ``images`` directory, so ``--limit 2`` produced two
  label files while the loader still saw every image of the split (and could write a
  ``labels.cache`` next to the source labels).
* ``--limit-per-split N`` restricts the visible images *and* the labels together.
* crowd/ignore/degenerate/out-of-range boxes are counted and reported, never silently turned
  into positives; unknown categories, missing files, duplicate ids and path escapes fail fast.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import yaml

from .paths import project_root, resolve_data_yaml_with_candidates

__all__ = [
    "DataViewError",
    "load_soda_config",
    "load_coco",
    "validate_coco",
    "coco_bbox_to_yolo",
    "yolo_bbox_to_coco",
    "build_view",
    "build_native_dataloader",
    "image_size_from_file",
    "resolve_data_yaml",
    "DEFAULT_VIEW_DIR",
    "SPLITS",
    "DEFAULT_YAML",
]

#: Backwards-compatible alias: the V1 package exposed ``DEFAULT_YAML`` as the COCO config path.
#: ``DEFAULT_YAML`` is a *lazy* value here because the path is machine dependent; importers that
#: need the resolved path should call :func:`resolve_data_yaml`.
def __getattr__(name):
    if name == "DEFAULT_YAML":
        return resolve_data_yaml()
    raise AttributeError(name)

SPLITS = ("train", "val")
DEFAULT_VIEW_DIR = os.path.join(project_root(), "data", "soda_smoke")


class DataViewError(RuntimeError):
    """Raised for any dataset inconsistency that must not be silently skipped."""


# --------------------------------------------------------------------------------------
# config + json
# --------------------------------------------------------------------------------------
def resolve_data_yaml(explicit: Optional[str] = None) -> str:
    """Locate ``soda10m.yaml``; fail loudly (with the probed locations) when absent."""
    found = resolve_data_yaml_with_candidates(explicit)
    if not found["resolved"]:
        raise DataViewError(f"SODA10M config not found. Probed: {found['probed']}")
    return found["resolved"]


def load_soda_config(yaml_path: Optional[str] = None) -> dict:
    """Load ``soda10m.yaml`` and resolve every path relative to the YAML's own directory."""
    yaml_path = os.path.abspath(resolve_data_yaml(yaml_path))
    with open(yaml_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if str(cfg.get("format", "")).lower() != "coco":
        raise DataViewError(f"expected 'format: coco' in {yaml_path}, got {cfg.get('format')!r}")

    root = os.path.dirname(yaml_path)
    resolved = {
        "yaml_path": yaml_path,
        "root": root,
        "nc": int(cfg["nc"]),
        "names": {int(k): str(v) for k, v in dict(cfg["names"]).items()},
        "category_id_map": {int(k): int(v) for k, v in dict(cfg["category_id_map"]).items()},
        "splits": {},
    }
    for split in SPLITS:
        resolved["splits"][split] = {
            "images_dir": os.path.normpath(os.path.join(root, cfg[split])),
            "annotations": os.path.normpath(os.path.join(root, cfg["annotations"][split])),
        }
    return resolved


def load_coco(annotation_path: str) -> dict:
    with open(annotation_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise DataViewError(f"{annotation_path} is missing the '{key}' key")
    return data


#: How many `<split>.cache` files the source scan reports before stopping.
_MAX_CACHE_REPORT = 20


def _source_cache_files(images_dir: str) -> List[str]:
    """``*.cache`` files directly inside the source images directory (never a recursive walk).

    Scanning the split directory of a real dataset (SODA10M: 5000 entries) is cheap, but a deep
    walk is not, so only the top level is inspected.
    """
    try:
        return sorted(
            os.path.join(images_dir, name)
            for name in os.listdir(images_dir)
            if name.endswith(".cache")
        )[:_MAX_CACHE_REPORT]
    except OSError:
        return []


def validate_coco(coco: dict, cfg: dict, images_dir: str, *, check_files_for: Optional[set] = None) -> dict:
    """Fail-fast annotation validation; returns a summary of every check performed.

    ``check_files_for`` restricts the "image file exists" check to a set of image ids.  A limited
    view (``--limit-per-split``) is built from the *full* annotation file, so demanding that every
    annotated file be present on disk would make a perfectly valid partial view impossible; the
    path-escape check always runs for every entry, and the selected images are still verified
    (here and again in :func:`build_view`).
    """
    img_ids = [int(im["id"]) for im in coco["images"]]
    dup = [i for i, n in Counter(img_ids).items() if n > 1]
    if dup:
        raise DataViewError(f"duplicate image ids in annotation file: {dup[:10]}")

    names_by_id = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
    mapping = cfg["category_id_map"]
    missing_cats = sorted(set(names_by_id) - set(mapping))
    if missing_cats:
        raise DataViewError(f"category_id_map is missing dataset category ids {missing_cats}")
    if sorted(mapping.values()) != list(range(cfg["nc"])):
        raise DataViewError(f"category_id_map values {sorted(mapping.values())} are not contiguous 0..nc-1")
    by_id_name = {int(k): str(v) for k, v in cfg["names"].items()}
    for cid, mapped in mapping.items():
        if names_by_id[cid] != by_id_name[mapped]:
            raise DataViewError(
                f"category {cid} name {names_by_id[cid]!r} != names[{mapped}]={by_id_name[mapped]!r}"
            )

    # COCO category ids are not required to be contiguous 0..n-1 -- that is exactly why the
    # explicit category_id_map exists (SODA: 1..6 -> 0..5).
    known_ids = set(img_ids)
    root = os.path.abspath(images_dir)
    stats: Counter = Counter()
    bad: List[str] = []
    missing_files: List[str] = []
    escaped: List[str] = []
    per_class: Counter = Counter()
    for ann in coco["annotations"]:
        stats["annotations"] += 1
        iid = int(ann["image_id"])
        if iid not in known_ids:
            bad.append(f"annotation {ann.get('id')} references unknown image_id {iid}")
            continue
        cid = int(ann["category_id"])
        if cid not in mapping:
            bad.append(f"annotation {ann.get('id')} has unknown category_id {cid}")
            continue
        per_class[names_by_id[cid]] += 1
        x, y, w, h = (float(v) for v in ann["bbox"])
        if w <= 0 or h <= 0:
            stats["non_positive_wh"] += 1
        if float(ann.get("area", w * h)) < 0:
            stats["negative_area"] += 1
        if int(ann.get("iscrowd", 0)) == 1:
            stats["iscrowd"] += 1
        if int(ann.get("ignore", 0)) == 1:
            stats["ignore"] += 1
        if x < 0 or y < 0:
            stats["negative_xy"] += 1
    checked_files = 0
    for im in coco["images"]:
        fname = str(im["file_name"])
        if os.path.isabs(fname) or ".." in fname.replace("\\", "/").split("/"):
            escaped.append(fname)
            continue
        full = os.path.normpath(os.path.join(root, fname))
        if not full.startswith(root + os.sep):
            escaped.append(fname)
            continue
        if check_files_for is not None and int(im["id"]) not in check_files_for:
            continue
        checked_files += 1
        if not os.path.isfile(full):
            missing_files.append(fname)

    if bad:
        raise DataViewError(f"{len(bad)} annotation problem(s): {bad[:5]}")
    if escaped:
        raise DataViewError(f"{len(escaped)} image file_name(s) escape the images dir: {escaped[:5]}")
    if missing_files:
        raise DataViewError(f"{len(missing_files)} image file(s) missing, e.g. {missing_files[:5]}")

    return {
        "image_files_checked": checked_files,
        "image_files_scope": "all annotated images"
        if check_files_for is None
        else f"selected images only ({len(check_files_for)} ids)",
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "categories": names_by_id,
        "category_id_map": {int(k): int(v) for k, v in mapping.items()},
        "per_class": dict(per_class),
        "flag_counts": {k: int(v) for k, v in stats.items()},
        "crowd_or_ignore_present": bool(stats["iscrowd"] or stats["ignore"]),
        "duplicate_ids": [],
        "missing_files": [],
        "path_escapes": [],
    }


# --------------------------------------------------------------------------------------
# box transforms (unit-checked in both directions)
# --------------------------------------------------------------------------------------
def coco_bbox_to_yolo(bbox: Sequence[float], width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    """COCO pixel ``[x, y, w, h]`` (top-left) -> normalised YOLO ``(xc, yc, w, h)``.

    Clipped to the image; returns ``None`` for a degenerate box after clipping.
    """
    if width <= 0 or height <= 0:
        raise DataViewError(f"non-positive image size {width}x{height}")
    x, y, w, h = (float(v) for v in bbox)
    x1, y1, x2, y2 = x, y, x + w, y + h
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    cw, ch = x2 - x1, y2 - y1
    if cw <= 0 or ch <= 0:
        return None
    values = ((x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height, cw / width, ch / height)
    for v in values:
        if not (0.0 <= v <= 1.0):
            raise DataViewError(f"normalised box component {v} outside [0,1] for bbox {bbox}")
    return values


def yolo_bbox_to_coco(xc: float, yc: float, w: float, h: float, width: int, height: int) -> List[float]:
    """Inverse of :func:`coco_bbox_to_yolo` (pixel ``[x, y, w, h]``, top-left origin)."""
    return [
        (float(xc) - float(w) / 2.0) * width,
        (float(yc) - float(h) / 2.0) * height,
        float(w) * width,
        float(h) * height,
    ]


def image_size_from_file(path: str) -> Tuple[int, int]:
    """Real decoded ``(width, height)`` via PIL (no torchvision import)."""
    from PIL import Image

    with Image.open(path) as im:
        return int(im.size[0]), int(im.size[1])


# --------------------------------------------------------------------------------------
# view builder
# --------------------------------------------------------------------------------------
def _select_images(images: List[dict], limit: Optional[int]) -> List[dict]:
    ordered = sorted(images, key=lambda im: int(im["id"]))
    return ordered[:limit] if limit else ordered


def build_view(
    out_dir: str,
    yaml_path: Optional[str] = None,
    limit_per_split: Optional[int] = None,
    force: bool = False,
    verify_image_sizes: bool = True,
    max_size_mismatches: int = 0,
) -> dict:
    """Build the native-YOLO view under ``out_dir`` and return a machine-readable report.

    ``images/{split}`` is always a real directory holding one symlink per selected image, so the
    Ultralytics ``YOLODataset`` sees exactly the selected samples and writes any ``labels.cache``
    inside this view -- never into the source dataset.
    """
    cfg = load_soda_config(yaml_path)
    out_dir = os.path.abspath(out_dir)
    if os.path.exists(out_dir) and force:
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    report = {
        "out_dir": out_dir,
        "source": {"yaml": cfg["yaml_path"], "root": cfg["root"], "splits": cfg["splits"]},
        "nc": cfg["nc"],
        "names": cfg["names"],
        "category_id_map": cfg["category_id_map"],
        "limit_per_split": limit_per_split,
        "image_access": "per-file-symlink (images/<split> is a real directory)",
        "splits": {},
    }

    for split in SPLITS:
        src = cfg["splits"][split]
        coco = load_coco(src["annotations"])
        images = _select_images(coco["images"], limit_per_split)
        selected = {int(im["id"]) for im in images}
        validation = validate_coco(coco, cfg, src["images_dir"], check_files_for=selected)
        anns_by_image: Dict[int, List[dict]] = {i: [] for i in selected}
        for ann in coco["annotations"]:
            iid = int(ann["image_id"])
            if iid in anns_by_image:
                anns_by_image[iid].append(ann)

        img_root = os.path.join(out_dir, "images", split)
        label_root = os.path.join(out_dir, "labels", split)
        # REAL directories, rebuilt from scratch so a previous (possibly directory-symlink) view
        # cannot leak the entire source split into this one.
        for path in (img_root, label_root):
            if os.path.islink(path) or os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

        split_stats: Counter = Counter()
        size_mismatches: List[dict] = []
        for im in images:
            iid = int(im["id"])
            fname = str(im["file_name"])
            src_path = os.path.normpath(os.path.join(src["images_dir"], fname))
            dst = os.path.join(img_root, fname)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.lexists(dst):
                os.symlink(src_path, dst)
            split_stats["images_linked"] += 1

            jw, jh = int(im["width"]), int(im["height"])
            if verify_image_sizes:
                aw, ah = image_size_from_file(dst)
                if (aw, ah) != (jw, jh):
                    size_mismatches.append({"file_name": fname, "json": [jw, jh], "actual": [aw, ah]})
                split_stats["size_checks"] += 1

            lines = []
            for ann in anns_by_image[iid]:
                cid = int(ann["category_id"])
                cls = cfg["category_id_map"][cid]
                split_stats["annotations_seen"] += 1
                # crowd / ignore: YOLO txt has no ignore-region semantics.  They are counted and
                # reported; the box is still clipped/normalised like any other annotation so the
                # count is explicit rather than a silent positive or a silent drop.
                if int(ann.get("iscrowd", 0)) == 1:
                    split_stats["iscrowd_annotations"] += 1
                if int(ann.get("ignore", 0)) == 1:
                    split_stats["ignore_annotations"] += 1
                box = coco_bbox_to_yolo(ann["bbox"], jw, jh)
                if box is None:
                    split_stats["degenerate_dropped"] += 1
                    continue
                lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in box))
                split_stats["boxes_written"] += 1
            label_path = os.path.join(label_root, os.path.splitext(fname)[0] + ".txt")
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            with open(label_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
            if not lines:
                split_stats["empty_label_files"] += 1

        per_class_written: Counter = Counter()
        n_label_files = 0
        for root, _dirs, files in os.walk(label_root):
            for name in files:
                if not name.endswith(".txt"):
                    continue
                n_label_files += 1
                with open(os.path.join(root, name), "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            per_class_written[cfg["names"][int(line.split()[0])]] += 1

        n_visible_images = sum(
            1
            for root, _dirs, files in os.walk(img_root)
            for name in files
            if not name.startswith(".") and not name.endswith((".cache", ".npy"))
        )
        if n_visible_images != len(images):
            raise DataViewError(
                f"view {split}: {n_visible_images} visible images but {len(images)} were selected; "
                "--limit-per-split must restrict the visible images and the labels together"
            )
        if len(size_mismatches) > max_size_mismatches:
            raise DataViewError(
                f"{len(size_mismatches)} image size mismatch(es) in split {split} exceed "
                f"max_size_mismatches={max_size_mismatches}: {size_mismatches[:5]}"
            )

        report["splits"][split] = {
            "annotations_file": src["annotations"],
            "images_dir": src["images_dir"],
            "image_access": "per-file-symlink",
            "images_dir_is_symlink": False,
            "labels_dir_is_symlink": False,
            "n_images": len(images),
            "n_visible_images": n_visible_images,
            "n_label_files": n_label_files,
            "image_id_min": min(selected) if selected else None,
            "image_id_max": max(selected) if selected else None,
            "selected_image_ids": sorted(selected) if limit_per_split else f"{len(selected)} ids (full split)",
            "labels_dir": label_root,
            "validation": validation,
            "stats": {k: int(v) for k, v in split_stats.items()},
            "per_class_written": dict(per_class_written),
            "size_mismatches": size_mismatches,
            "first_samples": [
                {
                    "image_id": int(im["id"]),
                    "file_name": str(im["file_name"]),
                    "size": [int(im["width"]), int(im["height"])],
                }
                for im in images[:3]
            ],
        }

    # stable image-id map (both directions) so a future COCO evaluation cannot mis-assign ids
    id_map = {}
    for split in SPLITS:
        images = _select_images(load_coco(cfg["splits"][split]["annotations"])["images"], limit_per_split)
        for im in images:
            id_map[f"{split}/{im['file_name']}"] = {
                "split": split,
                "image_id": int(im["id"]),
                "file_name": str(im["file_name"]),
            }
    report["image_id_map_path"] = os.path.join(out_dir, "image_id_map.json")
    with open(report["image_id_map_path"], "w", encoding="utf-8") as fh:
        json.dump(id_map, fh, indent=2, sort_keys=True)

    data_yaml = {
        "path": out_dir,
        "train": os.path.join("images", "train"),
        "val": os.path.join("images", "val"),
        "nc": cfg["nc"],
        "names": cfg["names"],
    }
    data_yaml_path = os.path.join(out_dir, "data.yaml")
    with open(data_yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data_yaml, fh, sort_keys=False, allow_unicode=True)
    report["data_yaml"] = data_yaml_path
    report["data_yaml_content"] = data_yaml
    # A limited view is a smoke view: mark it so training cannot silently run on 2 images.
    if limit_per_split:
        marker_path = os.path.join(out_dir, "overlock_smoke_view.json")
        with open(marker_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "kind": "smoke_view",
                    "limit_per_split": int(limit_per_split),
                    "images": {s: report["splits"][s]["n_images"] for s in SPLITS},
                    "warning": "this view is NOT a full dataset; full training must rebuild it without --limit-per-split",
                },
                fh,
                indent=2,
            )
        report["smoke_marker"] = marker_path
    else:
        legacy = os.path.join(out_dir, "overlock_smoke_view.json")
        if os.path.isfile(legacy):
            os.remove(legacy)
    report["source_cache_files_written"] = [
        path for split in SPLITS for path in _source_cache_files(cfg["splits"][split]["images_dir"])
    ]
    return report


def build_native_dataloader(
    data_yaml: str,
    imgsz: int,
    *,
    mode: str = "val",
    batch: int = 1,
    workers: int = 0,
    stride: int = 32,
    device=None,
    rect: bool = False,
    augment: bool = False,
):
    """Build a *native* Ultralytics detection dataloader over a generated view.

    Uses the pinned source's own ``build_yolo_dataset`` + ``build_dataloader`` (or the validated
    ``YODataset`` collate), so the augmentation, letterbox and collate are exactly the ones the
    detector is trained/validated with.  ``rect``/``multi_scale`` stay off: the protocol is a
    fixed square letterbox.  ``device`` must be a real device (``torch.device("cpu")`` or a CUDA
    device) -- the native ``build_dataloader`` does not accept ``None``.
    """
    if device is None:
        device = torch.device("cpu")
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    # absolute on purpose: the dataset root must not depend on the current working directory
    data_yaml = os.path.abspath(os.path.expanduser(str(data_yaml)))
    args = get_cfg(
        overrides={
            "task": "detect",
            "mode": mode,
            "data": data_yaml,
            "imgsz": int(imgsz),
            "rect": bool(rect),
            "multi_scale": False,
            "workers": int(workers),
            "augment": bool(augment),
        }
    )
    data = check_det_dataset(data_yaml, split=mode)
    # ``check_det_dataset`` resolves the split entry against the yaml's own ``path``; passing the
    # yaml path again would make the dataset resolve ``data.yaml/images/val``.
    img_path = data[mode]
    dataset = build_yolo_dataset(args, img_path, int(batch), data, mode=mode, stride=int(stride))
    return build_dataloader(dataset, int(batch), int(workers), shuffle=False, rank=-1, device=device)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build an isolated native-YOLO view of SODA10M (COCO -> YOLO txt)")
    ap.add_argument("--yaml", default=None, help="path to soda10m.yaml (paths resolve relative to it)")
    ap.add_argument("--out", default=DEFAULT_VIEW_DIR, help="output view directory (inside this project)")
    ap.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="smoke mode: keep only the N lowest stable image ids per split (images AND labels)",
    )
    ap.add_argument("--force", action="store_true", help="remove an existing view directory first")
    ap.add_argument("--no-verify-image-sizes", action="store_true", help="skip the PIL size cross-check")
    ap.add_argument("--max-size-mismatches", type=int, default=0, help="allowed JSON/file size mismatches")
    ap.add_argument("--report", default=None, help="where to write the JSON report")
    ap.add_argument("--no-report", action="store_true", help="do not write a JSON report")
    args = ap.parse_args(argv)

    report = build_view(
        args.out,
        yaml_path=args.yaml,
        limit_per_split=args.limit_per_split,
        force=args.force,
        verify_image_sizes=not args.no_verify_image_sizes,
        max_size_mismatches=args.max_size_mismatches,
    )
    if not args.no_report:
        report_path = args.report or os.path.join(project_root(), "reports", "v2", "data_view.json")
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"data view report -> {report_path}")
    print(
        json.dumps(
            {
                "data_yaml": report["data_yaml"],
                "splits": {
                    k: {
                        "n_images": v["n_images"],
                        "n_visible_images": v["n_visible_images"],
                        "n_label_files": v["n_label_files"],
                        "stats": v["stats"],
                    }
                    for k, v in report["splits"].items()
                },
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
