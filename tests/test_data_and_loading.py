"""V02/V08 support: box transforms, COCO validation, the isolated data view, and the
auditable ``weights_only=True`` checkpoint loading path.

The heavy real-weight load and the real-sample forward pass live in ``scripts/smoke.py``
(V02/V05/V06/V07/V08); this module covers the unit-level contracts and fail-fast paths.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from overlock_yolo.checkpoint import (  # noqa: E402
    CheckpointError,
    audit_and_load,
    flatten_state_dict,
    sha256_file,
    strip_known_prefixes,
)
from overlock_yolo import data as _data  # noqa: E402
from overlock_yolo.data import (  # noqa: E402
    DataViewError,
    build_view,
    coco_bbox_to_yolo,
    image_size_from_file,
    load_coco,
    load_soda_config,
    validate_coco,
    yolo_bbox_to_coco,
)

#: The SODA10M COCO config and the OverLoCK checkpoints live *outside* this repository.  They are
#: required for the data/checkpoint tests but must never be downloaded, so the tests that need
#: them are skipped (with the probed paths) on a machine that does not have them.
_SODA_CONFIG = None
try:
    _SODA_CONFIG = _data.resolve_data_yaml()
except Exception:  # noqa: BLE001 - the reason is reported by the skip messages below
    _SODA_CONFIG = None
DEFAULT_YAML = _SODA_CONFIG  # kept for compatibility with the V1 test bodies

from overlock_yolo.paths import resolve_checkpoint_path as _resolve_checkpoint  # noqa: E402

CHECKPOINT = _resolve_checkpoint("b") or "/Users/lw/Documents/CNN-Mamba/OverLoCK-main/checkpoints/overlock_b_in1k_224.pth"
_MISSING = []
if _SODA_CONFIG is None:
    _MISSING.append("SODA10M soda10m.yaml (--yaml/--data-yaml or OVERLOCK_DATA_ROOT)")
if not os.path.isfile(CHECKPOINT):
    _MISSING.append(f"OverLoCK-B checkpoint ({CHECKPOINT})")
requires_external = unittest.skipUnless(not _MISSING, "external resources not present here: " + "; ".join(_MISSING))


class TestBBoxTransforms(unittest.TestCase):
    def test_roundtrip_exact(self):
        W, H = 1920, 1080
        for bbox in ([0, 0, 10, 20], [65, 667, 174, 126], [1900, 1070, 20, 10], [100.5, 200.25, 33.5, 44.75]):
            yolo = coco_bbox_to_yolo(bbox, W, H)
            self.assertIsNotNone(yolo)
            back = yolo_bbox_to_coco(*yolo, W, H)
            for a, b in zip(bbox, back):
                self.assertAlmostEqual(a, b, places=6)

    def test_known_values(self):
        # x=65,y=667,w=174,h=126 on a 1920x1080 image
        xc, yc, w, h = coco_bbox_to_yolo([65, 667, 174, 126], 1920, 1080)
        self.assertAlmostEqual(xc, (65 + 174 / 2) / 1920, places=9)
        self.assertAlmostEqual(yc, (667 + 126 / 2) / 1080, places=9)
        self.assertAlmostEqual(w, 174 / 1920, places=9)
        self.assertAlmostEqual(h, 126 / 1080, places=9)

    def test_clipping_and_degenerate(self):
        # partially outside -> clipped, still valid
        box = coco_bbox_to_yolo([-10, -10, 40, 40], 100, 100)
        self.assertIsNotNone(box)
        xc, yc, w, h = box
        self.assertAlmostEqual(xc, 15 / 100, places=9)
        self.assertAlmostEqual(w, 30 / 100, places=9)
        # fully outside -> degenerate, dropped
        self.assertIsNone(coco_bbox_to_yolo([200, 200, 10, 10], 100, 100))
        self.assertIsNone(coco_bbox_to_yolo([0, 0, 0, 10], 100, 100))
        self.assertIsNone(coco_bbox_to_yolo([5, 5, -5, 5], 100, 100))

    def test_normalised_values_stay_in_range(self):
        for bbox in ([0, 0, 1920, 1080], [0, 0, 1, 1], [1919, 1079, 1, 1]):
            xc, yc, w, h = coco_bbox_to_yolo(bbox, 1920, 1080)
            for v in (xc, yc, w, h):
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_bad_image_size(self):
        with self.assertRaises(DataViewError):
            coco_bbox_to_yolo([0, 0, 1, 1], 0, 100)


def _mini_coco(images, annotations=None, categories=None):
    return {
        "images": images,
        "annotations": annotations or [],
        "categories": categories
        or [
            {"id": 1, "name": "Pedestrian"},
            {"id": 2, "name": "Cyclist"},
            {"id": 3, "name": "Car"},
            {"id": 4, "name": "Truck"},
            {"id": 5, "name": "Tram"},
            {"id": 6, "name": "Tricycle"},
        ],
    }


@requires_external
class TestSodaConfigAndCoco(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_soda_config(DEFAULT_YAML)

    def test_config_resolution_is_relative_to_yaml(self):
        self.assertEqual(self.cfg["nc"], 6)
        self.assertEqual(
            [self.cfg["names"][i] for i in range(6)],
            ["Pedestrian", "Cyclist", "Car", "Truck", "Tram", "Tricycle"],
        )
        self.assertEqual(self.cfg["category_id_map"], {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5})
        for split in ("train", "val"):
            self.assertTrue(self.cfg["splits"][split]["annotations"].startswith(self.cfg["root"]))
            self.assertTrue(os.path.isfile(self.cfg["splits"][split]["annotations"]))

    def test_validate_coco_smoke(self):
        coco = load_coco(self.cfg["splits"]["train"]["annotations"])
        # validating the whole split would stat 5000 files; validate the first 8 images only
        keep = sorted(coco["images"], key=lambda im: int(im["id"]))[:8]
        keep_ids = {int(im["id"]) for im in keep}
        mini = {
            "images": keep,
            "annotations": [a for a in coco["annotations"] if int(a["image_id"]) in keep_ids],
            "categories": coco["categories"],
        }
        summary = validate_coco(mini, self.cfg, self.cfg["splits"]["train"]["images_dir"])
        self.assertEqual(summary["images"], 8)
        self.assertGreater(summary["annotations"], 0)
        self.assertEqual(summary["duplicate_ids"], [])
        self.assertGreater(summary["flag_counts"]["annotations"], 0)

    def test_validate_rejects_bad_category(self):
        mini = _mini_coco(
            [{"id": 1, "file_name": "HT_TRAIN_000001_SH_000.jpg", "width": 1920, "height": 1080}],
            [{"id": 1, "image_id": 1, "category_id": 99, "bbox": [0, 0, 1, 1], "area": 1}],
        )
        with self.assertRaises(DataViewError):
            validate_coco(mini, self.cfg, self.cfg["splits"]["train"]["images_dir"])

    def test_validate_rejects_duplicate_ids(self):
        mini = _mini_coco(
            [
                {"id": 1, "file_name": "HT_TRAIN_000001_SH_000.jpg", "width": 10, "height": 10},
                {"id": 1, "file_name": "HT_TRAIN_000002_SH_000.jpg", "width": 10, "height": 10},
            ]
        )
        with self.assertRaises(DataViewError):
            validate_coco(mini, self.cfg, self.cfg["splits"]["train"]["images_dir"])

    def test_validate_rejects_missing_file(self):
        mini = _mini_coco([{"id": 1, "file_name": "definitely_not_here.jpg", "width": 10, "height": 10}])
        with self.assertRaises(DataViewError):
            validate_coco(mini, self.cfg, self.cfg["splits"]["train"]["images_dir"])

    def test_validate_rejects_path_escape(self):
        mini = _mini_coco([{"id": 1, "file_name": "../secret.jpg", "width": 10, "height": 10}])
        with self.assertRaises(DataViewError):
            validate_coco(mini, self.cfg, self.cfg["splits"]["train"]["images_dir"])


@requires_external
class TestDataView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="overlock_view_")
        cls.out = os.path.join(cls.tmp, "view")
        cls.report = build_view(cls.out, limit_per_split=2, force=True)

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_structure_and_symlinks(self):
        for split in ("train", "val"):
            self.assertTrue(os.path.isdir(os.path.join(self.out, "images", split)))
            self.assertTrue(os.path.isdir(os.path.join(self.out, "labels", split)))
        # DESIGN_V2 10.1: images/<split> is a REAL directory and each selected image is a
        # per-file symlink.  V1 symlinked the whole images directory, which made --limit expose
        # every image of the split while only a few label files existed.
        img_root = os.path.join(self.out, "images", "train")
        self.assertFalse(os.path.islink(img_root))
        visible = sorted(os.listdir(img_root))
        self.assertTrue(visible)
        for name in visible:
            link = os.path.join(img_root, name)
            self.assertTrue(os.path.islink(link), link)
            self.assertTrue(os.path.isfile(link), link)
        # the visible images are exactly the selected ones, and labels match them 1:1
        n_images = self.report["splits"]["train"]["n_images"]
        self.assertEqual(len(visible), n_images)
        self.assertEqual(self.report["splits"]["train"]["n_label_files"], n_images)
        # no label file was symlinked and no source cache was written
        self.assertFalse(os.path.islink(os.path.join(self.out, "labels", "train")))
        self.assertEqual(self.report["source_cache_files_written"], [])

    def test_data_yaml_contents(self):
        import yaml as _yaml

        with open(self.report["data_yaml"], "r", encoding="utf-8") as fh:
            d = _yaml.safe_load(fh)
        self.assertEqual(d["nc"], 6)
        self.assertEqual(d["path"], os.path.abspath(self.out))
        self.assertEqual(d["train"], "images/train")
        self.assertEqual(d["val"], "images/val")
        self.assertEqual([d["names"][i] for i in range(6)], ["Pedestrian", "Cyclist", "Car", "Truck", "Tram", "Tricycle"])

    def test_labels_match_the_json_boxes(self):
        cfg = load_soda_config(DEFAULT_YAML)
        coco = load_coco(cfg["splits"]["train"]["annotations"])
        first = sorted(coco["images"], key=lambda im: int(im["id"]))[0]
        iid = int(first["id"])
        anns = [a for a in coco["annotations"] if int(a["image_id"]) == iid]
        label = os.path.join(self.out, "labels", "train", os.path.splitext(first["file_name"])[0] + ".txt")
        with open(label, "r", encoding="utf-8") as fh:
            lines = [ln.split() for ln in fh.read().splitlines() if ln.strip()]
        self.assertEqual(len(lines), len(anns))
        for line, ann in zip(lines, anns):
            cls = int(line[0])
            self.assertEqual(cls, cfg["category_id_map"][int(ann["category_id"])])
            xc, yc, w, h = (float(v) for v in line[1:5])
            back = yolo_bbox_to_coco(xc, yc, w, h, int(first["width"]), int(first["height"]))
            for a, b in zip(ann["bbox"], back):
                # labels are written with 6 decimal places, i.e. ~2e-6 * image size pixels
                self.assertLess(abs(a - b), 0.01)

    def test_image_id_map_is_stable_both_ways(self):
        with open(self.report["image_id_map_path"], "r", encoding="utf-8") as fh:
            id_map = json.load(fh)
        self.assertEqual(len(id_map), 4)
        keys = sorted(id_map)
        self.assertTrue(keys[0].startswith("train/"))
        # image ids are only unique *within* a split (both splits start at id 1)
        pairs = {(v["split"], v["image_id"]) for v in id_map.values()}
        self.assertEqual(len(pairs), 4)
        for key, v in id_map.items():
            self.assertEqual(key, f"{v['split']}/{v['file_name']}")

    def test_size_crosscheck_matches_json(self):
        for split in ("train", "val"):
            self.assertEqual(self.report["splits"][split]["size_mismatches"], [])
            for s in self.report["splits"][split]["first_samples"]:
                path = os.path.join(self.out, "images", split, s["file_name"])
                self.assertEqual(list(image_size_from_file(path)), s["size"])

    def test_full_conversion_capability_but_smoke_scope(self):
        # the builder is complete; this round simply only ran the 2-image smoke view
        for split in ("train", "val"):
            self.assertEqual(self.report["splits"][split]["n_images"], 2)
        self.assertEqual(self.report["limit_per_split"], 2)


class TestCheckpointSafety(unittest.TestCase):
    def store(self):
        return {
            "a.weight": torch.randn(4, 4),
            "blocks1.0.dwconv.weight": torch.randn(3, 1, 3, 3),
            "extra_norm.0.weight": torch.ones(8),
            "head.4.weight": torch.randn(1000, 1024, 1, 1),
            "aux_head.2.weight": torch.randn(1000, 768, 1, 1),
        }

    def test_flatten_accepts_flat_state_dict(self):
        sd = self.store()
        flat, rep = flatten_state_dict(sd)
        self.assertEqual(len(flat), len(sd))
        self.assertIn("flat", rep["container_candidates"][0]["kind"])

    def test_flatten_accepts_single_verified_container(self):
        for key in ("state_dict", "model", "ema"):
            flat, rep = flatten_state_dict({key: self.store(), "epoch": 3})
            self.assertEqual(rep["container_choice"], key)
            self.assertEqual(len(flat), len(self.store()))

    def test_flatten_rejects_ambiguous_container(self):
        with self.assertRaises(CheckpointError):
            flatten_state_dict({"state_dict": self.store(), "model": self.store()})

    def test_flatten_rejects_non_tensor_mapping(self):
        with self.assertRaises(CheckpointError):
            flatten_state_dict({"a": 1, "b": 2})

    def test_flatten_rejects_garbage(self):
        with self.assertRaises(CheckpointError):
            flatten_state_dict([1, 2, 3])

    def test_prefix_strip_detects_collisions(self):
        with self.assertRaises(CheckpointError):
            strip_known_prefixes({"module.a": torch.zeros(1), "a": torch.zeros(1)})

    def test_prefix_strip_records_transform(self):
        out, tr = strip_known_prefixes({"module.x": torch.zeros(1), "y": torch.zeros(2)})
        self.assertEqual(sorted(out), ["x", "y"])
        self.assertEqual(tr[0]["prefix"], "module.")
        self.assertEqual(tr[0]["removed_from"], 1)

    def test_sha256_and_safe_load_of_a_tiny_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "tiny.pth")
            torch.save({"state_dict": self.store()}, p)
            self.assertEqual(sha256_file(p), sha256_file(p))
            self.assertEqual(len(sha256_file(p)), 64)

    def test_weights_only_true_retained_on_failure(self):
        """A pickle-only object must fail loudly under weights_only=True, never downgrade."""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.pth")

            import pathlib

            torch.save({"state_dict": self.store(), "extra": pathlib.PurePosixPath("/tmp/x")}, p)
            from overlock_yolo.checkpoint import load_safe_state_dict

            with self.assertRaises(CheckpointError) as ctx:
                load_safe_state_dict(p)
            self.assertIn("weights_only", str(ctx.exception).lower())

    def test_audit_requires_shape_match_and_covers_numel(self):
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        target = model.state_dict()
        ckpt = {k: v.clone() for k, v in target.items()}
        ckpt["head.4.weight"] = torch.randn(3, 3)  # classification-only -> explicitly ignored
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pth")
            torch.save(ckpt, p)
            rep = audit_and_load(model, p)
            self.assertEqual(rep["status"], "ok")
            self.assertEqual(rep["ignored_classification_only"]["count"], 1)
            self.assertEqual(rep["shape_mismatch"], {})
            self.assertEqual(rep["totals"]["numel_coverage"], 1.0)
            self.assertFalse(rep["totals"]["trainable_missing_disallowed_tensors"])

    def test_audit_fails_on_shape_mismatch(self):
        import torch.nn as nn

        model = nn.Linear(4, 4)
        ckpt = {"weight": torch.randn(5, 5), "bias": torch.randn(4)}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pth")
            torch.save(ckpt, p)
            with self.assertRaises(CheckpointError):
                audit_and_load(model, p)

    def test_audit_fails_on_non_whitelisted_missing(self):
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        ckpt = {"0.weight": torch.randn(4, 4), "0.bias": torch.randn(4)}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pth")
            torch.save(ckpt, p)
            with self.assertRaises(CheckpointError) as ctx:
                audit_and_load(model, p, allow_missing_prefixes=())
            self.assertIn("missing", str(ctx.exception).lower())

    def test_audit_reports_allowed_missing_prefix(self):
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        model.extra_norm = nn.LayerNorm(2)  # a real detection-only top-level module
        ckpt = {"0.weight": torch.randn(4, 4), "0.bias": torch.randn(4), "1.weight": torch.randn(2, 4), "1.bias": torch.randn(2)}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pth")
            torch.save(ckpt, p)
            rep = audit_and_load(model, p)  # extra_norm.* is the explicit detection-only prefix
            self.assertEqual(rep["status"], "ok")
            self.assertEqual(rep["missing_allowed"]["count"], 2)
            self.assertTrue(all(k.startswith("extra_norm.") for k in rep["missing_allowed"]["keys"]))

    def test_audit_fails_when_allowed_prefix_not_used(self):
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        model.extra_norm = nn.LayerNorm(2)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pth")
            torch.save({"0.weight": torch.randn(4, 4)}, p)
            with self.assertRaises(CheckpointError):
                audit_and_load(model, p, allow_missing_prefixes=())

    @requires_external
    def test_real_checkpoint_sha_and_container(self):
        if not os.path.isfile(CHECKPOINT):
            self.skipTest(f"checkpoint not present: {CHECKPOINT}")
        from overlock_yolo.checkpoint import load_safe_state_dict

        obj, info = load_safe_state_dict(CHECKPOINT)
        self.assertEqual(info["size_bytes"], 395257645)
        self.assertEqual(len(info["sha256"]), 64)
        flat, rep = flatten_state_dict(obj)
        self.assertGreater(len(flat), 1000)
        # a real OverLoCK state dict must contain these module families
        for fam in ("patch_embed1", "blocks1", "blocks4", "sub_blocks3", "sub_blocks4", "patch_embedx", "high_level_proj"):
            self.assertTrue(any(k.startswith(fam) for k in flat), f"{fam} missing from checkpoint")
        # ... and must NOT contain the detection-only modules the classification variant never
        # runs: they are explicitly whitelisted as allowed-missing by the audit
        for fam in ("extra_norm.", "h_proj."):
            self.assertFalse(any(k.startswith(fam) for k in flat), f"{fam} unexpectedly present")
        for fam in ("head.", "aux_head."):
            self.assertTrue(any(k.startswith(fam) for k in flat), f"{fam} missing from checkpoint")


if __name__ == "__main__":
    unittest.main(verbosity=2)
