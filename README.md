# OverLoCK (XT/T/S/B) x YOLO11/YOLO26 (n/s/m/l/x) -- 640x640 detector

Isolated implementation of `DESIGN_V2.md` (copy of
`OverLoCK_YOLO11_YOLO26_V2_MultiVariant_640_Git_Design_20261003.md`).
**40 structure combinations** (`variant x family x scale`), the families' own native losses and
the **native Ultralytics** evaluation metrics, all at a fixed **640x640** equal-ratio
resize + letterbox protocol.

```
RGB -> native letterbox to exactly 640x640 -> /255              (native Ultralytics loader)
    -> ONE ImageNet mean/std normalisation (buffers on the stem)
    -> official OverLoCK detection backbone (xt | t | s | b)
    -> x1/x2/x3 at stride 8/16/32
    -> three 1x1 Conv-BN-SiLU adapters -> the native neck entry widths
    -> the native neck+head of that family's YAML at the requested scale (YOLO11 | YOLO26)
    -> that family's own criterion: v8DetectionLoss (11) | E2ELoss (26)
```

Default experiment: **OverLoCK-T + YOLO11s + 640x640**.

---

## Status (read this first)

| | |
|---|---|
| Implemented & CPU-verified locally | 40 combinations build; T/B real weights audited; native tails, losses, validator, data bridge, profiling, entry points |
| Measured on the target GPU | **nothing** — no RTX 4090 was rented, and no GPU result is claimed |
| Accuracy (AP) | **not measured** — the local validator runs are interface checks on a 2-image view with a randomly initialised neck/head |
| Local environment | conda `3dete`, CPU, FP32, `workers=0` (Python 3.10 / torch 2.11). This is **not** the server environment (Python 3.12 / torch 2.5.1+cu124) |
| `NATTEN` | **not installed locally**, so every local attention call used the bundled differentiable PyTorch reference and is reported as `torch_reference` |

Facts and numbers: `reports/v2/*.json` (JSON is the source of truth) and `reports/v2/HANDOFF.md`.

---

## Quick start

Every CLI resolves its paths through `overlock_yolo/paths.py`: explicit CLI flag → environment
variable → repository default.  Relative config paths are relative to the **project root**, never
to the current working directory.  No CLI needs to be run from a specific directory.

```bash
python -c "import sys; print(sys.executable)"     # use the pinned interpreter

# 1. inspect the resolved experiment (no training)
python scripts/train.py --config configs/overlock_t_yolo11s_soda.yaml --help

# 2. build a small data view (2 images per split, per-file symlinks, no source writes)
python scripts/prepare_data.py --out data/soda_smoke --limit-per-split 2 --force

# 3. native Ultralytics validation (interface check on that 2-image view)
python scripts/val.py --config configs/overlock_t_yolo11s_soda.yaml --view data/soda_smoke

# 4. cost report (unique parameters + explicitly partial MACs)
python scripts/profile_model.py --config configs/overlock_t_yolo11s_soda.yaml
python scripts/profile_model.py --matrix --verify-matrix

# 5. local CPU verification of the whole contract (M01-M11) + targeted regression tests
python scripts/smoke.py --out reports/v2/validation.json
python tests/run_v2_tests.py                       # pytest-compatible; the shim is only for the
                                                   # local env, which has no pytest installed
python -m unittest discover -s tests -p "test_*.py"   # V1 unit tests, still green
python scripts/report_environment.py               # environment.json + compatibility.json
```

Training a **full** dataset (server):

```bash
python scripts/prepare_data.py --out data/soda_full          # full view, no --limit-per-split
python scripts/train.py --config configs/overlock_t_yolo11s_soda.yaml \
    --view data/soda_full --device 0 --batch 16 --workers 8 --amp --epochs 100
```

`scripts/train.py` **refuses** to train on a view that carries the smoke marker unless
`--allow-smoke-view` is passed, so a rented GPU cannot silently train on two images.

---

## Variant / family / scale are three independent selectors

```yaml
backbone: {family: overlock, variant: t}   # xt | t | s | b
yolo:     {family: yolo11, scale: s}       # yolo11 | yolo26   x   n | s | m | l | x
train:    {imgsz: 640, rect: false, multi_scale: false, scale: 0.5}
```

* `yolo.scale` is the **architecture** spec; `train.scale` is the native geometric
  **augmentation** factor.  They are different namespaces, and a bare top-level `scale:` is
  rejected outright.
* Duplicate YAML keys are an error (a later value never silently wins), and unknown
  keys/variants/scales fail fast.
* CLI arguments override the file; a CLI value that was **not** passed never overrides the file.
* Adapters are derived from `backbone.feature_info` and the parsed neck entry widths, e.g.
  T+s → 128→256, 384→256, 640→512 and T+n → 128→128, 384→128, 640→256.

Configs: `configs/overlock_t_yolo11s_soda.yaml` (default), `..._t_yolo26s_...`,
`..._t_yolo11n_...`, `..._b_yolo11s_...`.

## Family-specific losses and heads (native, not re-implemented)

| | YOLO11 | YOLO26 |
|---|---|---|
| YAML | `11/yolo11.yaml` | `26/yolo26.yaml` |
| `reg_max` | 16 | 1 |
| branches | one-to-many | one-to-many + one-to-one (O2O features detached from the backbone, as upstream) |
| criterion | native `v8DetectionLoss` | native `E2ELoss` (O2M topk10, O2O topk7 + topk2=1, its own branch-weight schedule, L1 regression term) |
| native inference | dense + NMS | end-to-end top-k, no extra NMS |

`model(dict)` returns `(loss, loss_items)` exactly like the native trainer expects;
`model(tensor)` returns predictions.  The native epoch-end `criterion.update()` lifecycle and the
removal of the criterion from a saved payload are preserved.

## Evaluation protocol

* Primary metrics: the pinned native `DetectionValidator` + `DetMetrics`
  (`P`, `R`, `mAP50`, `mAP50-95`, per-class values, `results_dict`).  COCOeval is **not** the
  primary evaluator, and `save_json` stays off so the native COCO branch cannot replace it.
* `conf` defaults to the native detection default **0.001** (not the 0.25 display threshold).
* Input: equal-ratio resize + letterbox to **exactly** 640x640 (`auto=False`,
  `scale_fill=False`, pad 114, `scaleup=False` to match the Ultralytics validation default).
  `rect`/`multi_scale` are rejected, the validation `LetterBox` is pinned to a square target, and
  every batch shape is asserted and recorded in the validation report.
* 640 gives P3/P4/P5 = 80/40/20 and **A = 8400** anchors; YOLO11 eval output is `[B,10,8400]`,
  YOLO26 end-to-end output is `[B,K,6]` (correctly *not* forced to the YOLO11 shape).

## COCO data

`soda10m.yaml` is a COCO-format config.  `scripts/prepare_data.py` generates an isolated
native-YOLO view: **real** `images/{train,val}` directories holding one symlink per selected
image, labels next to them, `--limit-per-split N` restricting images *and* labels together, and
**nothing** written into the COCO source tree (no `labels.cache` next to the source annotations).
The source JSON stays the ground truth; category ids go through the explicit
`category_id_map` (SODA 1..6 → 0..5) and cannot be renumbered by appearance order.  crowd/ignore/
degenerate boxes are counted and reported, never silently turned into positives.

The old MMCV/MMDetection route is deliberately **not** used: `CocoDataset` +
`DefaultFormatBundle` emits MMCV `DataContainer` batches (`gt_bboxes`/`gt_labels`/`gt_masks`)
that an Ultralytics detector cannot consume, and installing that stack would create two
inconsistent augmentation/normalisation paths for the same JSON.  Only the COCO parsing/mapping
is reused; the loader, augmentation and collate are the native Ultralytics ones.

---

## Layout

```
overlock_yolo/          the package
  paths.py              path resolution + pinned-Ultralytics installation (no hardcoded paths)
  config.py             strict experiment config (duplicates/unknown keys fail fast)
  variants.py           the 4 x 2 x 5 tables, adapter mapping, family contract
  backbone.py           vendored OverLoCK detection source (provenance header) + variant factory
  attention_backend.py  NATTEN backend resolution + memory-bounded differentiable reference
  model.py              detector assembly (native tail, single stem registration, native criterion)
  trainer.py            native trainer integration, project checkpoint format
  validator.py          native validator with the pinned square letterbox
  data.py               COCO -> isolated native-YOLO view
  profile_model.py      unique parameters, partial MACs, analytic na2d_av MACs
  checkpoint.py         audited weights_only=True backbone loading
  cli.py                shared CLI bootstrap
scripts/                train, val, profile_model, prepare_data, smoke (M01-M11),
                        server_preflight, gpu_smoke
configs/                four ready-to-run experiment configs
vendor/ultralytics/     pinned Ultralytics 8.4.148 pure-source snapshot (AGPL-3.0)
reports/v2/             JSON reports + HANDOFF.md (source of truth)
tests/                  V1 unit tests (unittest) + V2 regression tests
  run_v2_tests.py       standalone runner (works without pytest)
  pytest_contract/      the pytest-style V2 regression suite
DESIGN_V2.md            the V2 contract
THIRD_PARTY_NOTICES.md  vendored sources and licenses
```

## Server workflow (nothing here is executed locally)

1. `git clone https://github.com/russliuw/overlock_yolo11.git overlock_yolo11 && cd overlock_yolo11`
   (branch `codex/overlock-yolo-portable`)
2. `python scripts/server_preflight.py --expect-gpu` (read-only; explains every failure)
3. install per `requirements-server.txt` + `constraints-server.txt` (never a blanket `-U`)
4. put COCO data and the OverLoCK checkpoints in place, point the config paths at them
5. `python scripts/prepare_data.py --out data/soda_full`
6. `python scripts/gpu_smoke.py --config configs/overlock_t_yolo11s_soda.yaml --device 0`
7. `python scripts/train.py --config ... --view data/soda_full --device 0`
8. `python scripts/val.py --config ... --view data/soda_full --device 0`

The remote is **https://github.com/russliuw/overlock_yolo11.git** (private) and branch `codex/overlock-yolo-portable` is pushed; cloning it
was verified end to end (paths resolve to the clone's own `vendor/ultralytics`, CLIs run).
Weights and datasets never travel through Git: `checkpoints/` and the data views are git-ignored,
and the repository only records their expected file names/paths.

## Licenses

`vendor/ultralytics/ultralytics` is Ultralytics YOLO **8.4.148**, **AGPL-3.0**
(https://ultralytics.com/license); it is redistributed unmodified except for two unused binary
demo images that were removed.  `overlock_yolo/backbone.py` is a vendored, provenance-annotated
copy of the OverLoCK detection backbone.  See `THIRD_PARTY_NOTICES.md` before redistributing.
