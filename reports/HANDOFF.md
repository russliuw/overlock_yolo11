# HANDOFF — OverLoCK-Base + native YOLO11s, SODA10M 6-class detection

Status: **`IMPLEMENTED / CPU_SMOKE_PASSED / GPU_AND_ACCURACY_UNVERIFIED`**
Generated from [`validation.json`](validation.json) (the single source of truth), plus
[`environment.json`](environment.json), [`source_manifest.json`](source_manifest.json),
[`checkpoint_load.json`](checkpoint_load.json) and [`data_view.json`](data_view.json).

Round scope: implement the design, run cheap CPU checks, leave reproducible evidence.
**No formal training, no AP evaluation, no GPU work, no deployment.** `natten`, `mmcv`,
`mmengine` and `mmdet` are absent from this machine and were not installed.

---

## 1. Files, interfaces, provenance

### 1.1 New files (all under `overlock_yolo11/`; nothing else on disk was modified)

| Path | Responsibility |
|---|---|
| `DESIGN.md` | byte-identical copy of the design contract (SHA256 in `source_manifest.json`) |
| `overlock_yolo/backbone.py` | isolated copy of the official OverLoCK **detection** backbone, with a provenance header listing every modification |
| `overlock_yolo/attention_backend.py` | `auto\|natten\|torch_reference` resolution + exact **differentiable** CPU `na2d_av` reference |
| `overlock_yolo/checkpoint.py` | `weights_only=True` loader, container identification, per-key + per-numel load audit |
| `overlock_yolo/model.py` | `OverLoCKYOLO11`: normalisation + adapters + native tail routing |
| `overlock_yolo/trainer.py` | `OverLoCKDetectionTrainer(DetectionTrainer)` + optimizer partition audit |
| `overlock_yolo/data.py` | SODA10M COCO → native YOLO view (symlinked images, labels/maps in the view) |
| `configs/overlock_b_yolo11s_soda.yaml` | template training recipe (explicitly not tuned / not run) |
| `scripts/prepare_data.py`, `scripts/smoke.py`, `scripts/train.py` | data view, V01–V10 validation driver, training CLI |
| `tests/test_attention_backend.py`, `tests/test_neck_contract.py`, `tests/test_data_and_loading.py` | 73 unittest cases |
| `reports/*` | environment, source manifest, checkpoint audit, data view, validation, this handoff |
| `data/soda_smoke/` | the 2+2 image smoke view (images are **symlinks**, labels are new files) |
| `artifacts/smoke.log` | full stdout of the validation run |

### 1.2 Interfaces

```python
build_detector(weights=..., attention_backend="auto", nc=6, seed=0,
               freeze_backbone=False, backbone_lr_mult=1.0,
               backbone_bn_eval=False, yolo_state_dict=None) -> OverLoCKYOLO11

OverLoCKYOLO11                       # BaseModel subclass, so it keeps the native surface
  .forward(tensor)                   # eval: (y, preds); train: {"boxes","scores","feats"}
  .forward(dict)                     # -> predict(batch["img"])
  .loss(batch, preds=None)           # native lazy criterion
  .init_criterion()                  # -> ultralytics v8DetectionLoss
  .predict / .stride / .nc / .names / .args / .model[-1] / .set_head_attr(...)
  ._predict_once(x)                  # in-place style; routes 4/6/10 through the adapters
  ._run_tail(feats)                  # native tail only, from 3 adapted entry levels
  ._adapt(x, detach_backbone=False)  # normalise -> backbone -> adapters
  .stem, .adapters, .tail_start      # explicit backbone / adapter / native-tail modules
  .backbone_parameters / .parameter_groups_report() / .attention_backend_report()
  .load_backbone_weights(path)       # audited weights_only=True load

OverLoCKDetectionTrainer(DetectionTrainer)   # .get_model() returns the hybrid detector
build_overlock_b(attention_backend=..., device=..., row_chunk=0)   # explicit backbone factory
na2d_av(attn, value, kernel_size, *, dilation=1, is_causal=False,
        backend=..., resolved=..., row_chunk=0)
resolve_backend(backend, device) -> ResolvedBackend(requested, resolved, reason, cuda_available)
audit_and_load(model, checkpoint_path, ...) -> report      # raises on any non-whitelisted gap
build_view(out_dir, yaml_path, limit_per_split, force, ...) -> report
```

### 1.3 Source provenance and official-implementation differences

| Item | Value |
|---|---|
| official detection source | `OverLoCK-main/detection/models/overlock.py`, SHA256 `9671034d…8aa61` (read-only, unmodified) |
| vendored adapter | `overlock_yolo/backbone.py`, SHA256 `34527d5f…73587` |
| native YOLO11 yaml | `ultralytics-main/ultralytics/cfg/models/11/yolo11.yaml`, SHA256 `43d8a7c8…784c` |
| local Ultralytics | `ultralytics-main` (version **8.4.148**); the conda site-packages copy is **8.0.196** and is explicitly bypassed |
| SODA config | `data/SODA10M/soda10m.yaml`, SHA256 `6d817ec6…e584` |

`overlock_b` in the detection and classification files take **identical** structural arguments
(`depth [8,8,10,4]`, `sub_depth [20,4]`, `embed_dim [80,160,384,576]`,
`kernel_size [17,15,13,7]`, `mlp_ratio [4,4,4,4]`, `sub_num_heads [6,9]`,
`sub_mlp_ratio [3,3]`), so the classification checkpoint is structurally compatible with the
detection backbone. Full modification list (7 items) is in `source_manifest.json` and in the
`backbone.py` header; summarised:

1. provenance docstring; 2. mmdet/mmcv/mmengine registry+logger+`load_checkpoint` imports
removed, explicit `build_overlock_b` factory replaces `@MODELS.register_module()` and the
`pretrained=<GitHub URL>` bodies; 3. `natten.functional.na2d_av` → explicit backend hook;
4. the classification `head`/`aux_head` are never constructed (upstream builds and then
unconditionally `del`s them); 5. `torch.utils.checkpoint` import made version tolerant;
6. added `forward_multiscale()` (== `forward_features` + an output-signature assertion) and
`attention_backend_report()`; 7. cosmetic import inlining (`to_2tuple`, `DropPath`).

**Unchanged**: depth, kernels, mlp ratios, heads, `smk_size`, `deploy`, `use_gemm`, drop rates,
`use_checkpoint`, the overview/focus branches, `high_level_proj`, `patch_embedx`, `h_proj`, the
relative position bias (`apply_rpb`), the two softmaxes, the small-feature
interpolate-and-restore path, `extra_norm` placement, `_init_weights`, and every module or
parameter name.

Note on `use_checkpoint`: upstream passes a boolean and stores it as `self.use_checkpoint`, so
`use_checkpoint: [0,0,0,0]` disables checkpointing at every depth. The vendored copy keeps that
behaviour (`i < use_checkpoint[k]`).

### 1.4 Parameter accounting (measured)

| Bucket | Tensors | Parameters | Share |
|---|---:|---:|---:|
| OverLoCK-Base backbone (`stem.stem.backbone`) | 3,085 | 96,091,496 | 95.49 % |
| adapters (1×1 Conv-BN-SiLU ×3) | 9 | 546,816 | 0.54 % |
| native YOLO11s neck + Detect head (layers 11–23) | 136 | 3,988,130 | 3.96 % |
| **detector total** | **3,230** | **100,626,442** | 100 % |

All 3,229 trainable parameters are assigned exactly once across the native three optimizer
groups; the single frozen parameter is the native `Detect.dfl.conv.weight` (the fixed DFL
projection), which the audit recognises by name. This model is ~10.6× the parameter count of a
YOLO11s detector — the official OverLoCK-Base checkpoint size cannot be used as a proxy for the
detector's size, and no parameter-count comparison with YOLO11s is claimed.

Random vs pretrained:

* **pretrained (audited)**: everything the classification checkpoint contains, i.e. the whole
  OverLoCK backbone except the 14 keys below — per-module numel coverage is **1.000** for
  `blocks1..4`, `sub_blocks3/4`, and 0.923 for the `other` bucket (`high_level_proj`,
  `patch_embed*`), whose only gap is `h_proj.*`.
* **random (seed 0, torch default init)**: the three adapters, the whole native YOLO11s
  neck/Detect head, and `extra_norm.*` + `h_proj.*` (absent from the classification checkpoint).

Input normalisation is a **single** ImageNet mean/std step implemented as registered buffers in
`OverLoCKBackboneAdapter`; the loader/native preprocess only does RGB→FP32→/255.

---

## 2. Checkpoint loading (`reports/checkpoint_load.json`)

| Field | Value |
|---|---|
| path | `/Users/lw/Documents/CNN-Mamba/OverLoCK-main/checkpoints/overlock_b_in1k_224.pth` |
| size | 395,257,645 bytes (matches the design's recorded size) |
| SHA256 | `558b8b79c205c10f13e42d0ee22a8309e248fd6ab76a7ef40b4e0141c71481e8` |
| load call | `torch.load(path, map_location='cpu', weights_only=True)` — status **ok** |
| container | top-level flat tensor `state_dict` (4,793 tensors); no `state_dict`/`model`/`ema` wrapper, no prefix transforms needed |
| matched | 4,778 tensors / 96,052,305 numel |
| ignored (classification-only) | 15 tensors: `head.*` (8), `aux_head.*` (7) |
| allowed missing (detection-only) | 14 keys / 421,008 numel: `extra_norm.{0..4}.{weight,bias}` (10), `h_proj.{0,1}.{weight,bias}` (4) |
| non-whitelisted missing | **0** |
| shape mismatches | **0** |
| unexpected keys | **0** |
| total numel coverage | **0.995636** (99.56 %) — the only gap is the 14 allowed-missing detection-only keys |
| trainable params not covered | 0 after removing the explicit whitelist |

`extra_norm.*` and `h_proj.*` are genuinely absent: the classification variant defines both but
its forward never calls them, so they were never trained and never saved. They keep the
upstream initialisation (`trunc_normal_(std=0.02)` / LayerScale `1e-5`). They were discovered by
measuring the real file, not assumed.

Representative parameters re-read from disk and compared exactly (`torch.equal`): 
`patch_embed1.0.weight` ✓, `blocks3.0.dwconv.weight` ✓, `blocks3.0.proj.1.lk_origin.weight` ✓,
`blocks4.3.proj.1.dil_conv_k3_3.weight` ✓, `sub_blocks3.0.weight_proj.weight` ✓,
`sub_blocks3.0.rpb1` ✓, `sub_blocks4.3.proj.2.weight` ✓, `patch_embedx.h_proj.0.weight` ✓,
`high_level_proj.weight` ✓, `h_proj.0.weight` → `absent_in_checkpoint` (expected),
`extra_norm.4.weight` → `absent_in_checkpoint` (expected).

A `weights_only=True` failure is retained as a hard error: `overlock_yolo.checkpoint` never
retries with `weights_only=False` and never registers extra globals (covered by
`test_weights_only_true_retained_on_failure`). The official `pretrained=True` route was never
used, so no download URL was ever resolved.

---

## 3. Environment and attention backend

`reports/environment.json` records the full probe. Summary:

| Item | Value |
|---|---|
| interpreter | `/opt/miniconda3/envs/3dete/bin/python`, Python 3.10.20 |
| torch / torchvision / timm / einops | 2.11.0 / 0.26.0 / 1.0.27 / 0.8.2 |
| Ultralytics (resolved) | `/Users/lw/Documents/CNN-Mamba/ultralytics-main/ultralytics/__init__.py` (8.4.148) |
| natten / mmcv / mmengine / mmdet | **not importable** (not installed, not installed by this round) |
| device / dtype / amp / workers | cpu / float32 / False / 0 |
| CUDA available | False |

Attention backend: **requested `auto` → resolved `torch_reference`** (reason: *NATTEN not
importable*), also exercised as an explicit `torch_reference` request. Requesting `natten`
raises `AttentionBackendError`; there is no silent fallback and no claim that NATTEN ran.
`na2d_av` is never replaced by average pooling, zeros, a plain convolution or identity.

---

## 4. Acceptable-check results (V01–V10 + V07b)

Commands (all from `/Users/lw/Documents/CNN-Mamba/overlock_yolo11`):

```bash
export YOLO_CONFIG_DIR=$PWD/cache/ultralytics MPLCONFIGDIR=$PWD/cache/mpl OMP_NUM_THREADS=2
PY=/opt/miniconda3/envs/3dete/bin/python
$PY scripts/prepare_data.py --out data/soda_smoke --limit-per-split 2 --force
$PY -u scripts/smoke.py --square 224 --rect 128x160 --full-backward 96 \
    --out reports/validation.json --environment-out reports/environment.json     # exit 0
$PY -m unittest discover -s tests -p "test_*.py"                                 # 73 tests, OK
```

`reports/validation.json` → `summary = {total 11, pass 11, fail 0, skipped 0, partial 0}`.
Final run wall time **72.2 s** (sum of the individual check times), peak RSS **2.36 GB**, 2 CPU
threads, batch 1, `num_workers=0`, FP32, no AMP. Per-check times are in the table below.

| ID | Check | Status | Evidence (compact, actual numbers) |
|---|---|---|---|
| V01 | imports & provenance | **pass** | local Ultralytics resolved and asserted inside the pinned root; torch/torchvision/timm/einops importable; natten/mmcv/mmengine/mmdet absent; importing this package pulls in none of them |
| V02 | checkpoint load audit | **pass** | SHA256 `558b8b79…81e8`, `weights_only=True`, container = top-level state dict, 4,778/4,792 matched tensors, 0 disallowed missing, 0 shape mismatch, numel coverage **0.995636**, 11 representative params `torch.equal` to disk |
| V03 | reference `na2d_av` | **pass** | 6 oracle cases (including K=1, H=W=K, H=1, rectangular 9×4) worst max-abs-diff **9.5e-7**; attn/value grads finite (value grad absmax 10.33); row-chunk invariant; 23 unit cases incl. `gradcheck` on both operands and 9 fail-fast cases |
| V04 | native neck routing equivalence | **pass** | captured real native features at layers 4/6/10 = 256/256/512 ch; replaying the **same native tail objects** gives **max abs diff 0.0, bitwise equal**; tail indices 11–23, external refs `{4,6,10}`; Detect stride `[8,16,32]`, nc 6, reg_max 16, end2end False; Detect input channels `[128,256,512]` ≠ neck entries `[256,256,512]` |
| V05 | full Base square forward | **pass** | 224×224, real weights, eval + `inference_mode`: 2.54 s; head features 128/256/512 at 28×28/14×14/7×7; `y` `[1,10,1029]`, boxes `[1,64,1029]`, scores `[1,6,1029]`; all finite; peak RSS 1.50 GB |
| V06 | full Base rectangular forward | **pass** | 160×128 (30 s of design-mandated small-feature coverage): 1.77 s; head features 128/256/512 at 20×16/10×8/**5×4**; `y` `[1,10,420]`; all finite; the 2×3 backbone level exercises the official interpolate-and-restore branch (`min(H,W)=2 < kernel_size`) |
| V07 | native loss + backward, backbone detached | **pass** | detach at the **raw backbone outputs** so adapters stay in the graph: backbone grad tensors **0**, adapters **9/9 non-zero** (abs sum 1.55e5), neck+head **135 tensors, 127 non-zero** (abs sum 7.00e5); non-empty and empty-label losses both finite; detached predictions match the full forward (`boxes` diff 0.0, `scores` diff 7.2e-7, allclose) |
| V08 | real SODA samples | **pass** | 2 real samples (`images/train/HT_TRAIN_000001_SH_000.jpg`, `images/val/HT_VAL_000001_SH_001.jpg`) through the generated view + letterbox to 384×640: 25 boxes / classes {1,2,3,4} and 27 boxes / classes {0,1,2,3,4}; `y` and pred shapes correct; normalised input range [-1.96, 2.64] and [-2.12, 2.64]; box round-trip versus the JSON verified; `v8DetectionLoss` vector finite — (6.24, 6059.00, 4.33) and (6.47, 4100.40, 4.26) — with a successful backward |
| V09 | config & trainer entry | **pass** | `OverLoCKDetectionTrainer` returns `OverLoCKYOLO11` with nc 6, SODA names, `model[-1]` = native `Detect`, criterion `v8DetectionLoss`, `loss`/`predict`/`stride` present; native AdamW builds 3 groups (1,277 weight+decay / 696 norm / 1,257 bias) covering **all 3,229 trainable params exactly once** (only the frozen native `dfl.conv.weight` extra); `backbone_lr_mult` default 1.0, `freeze_backbone` default **False** |
| V10 | state/artefact accounting | **pass** | 96,091,496 backbone + 546,816 adapter + 3,988,130 neck/head = 100,626,442; no duplicate module registration; resolved backend `torch_reference`; dtype float32; device cpu; only this new directory was written (git status entries under `overlock_yolo11/` only) |
| V07b | full-Base backward (optional) | **pass** | 96×96, two stages: (A) OverLoCK-Base fwd+bwd 15.8 s, **3,085/3,085 backbone parameter tensors receive finite grads**, input grad finite (absmax 2.0e-6); (B) adapter→neck→head through `v8DetectionLoss` fwd+bwd 4.1 s, 9 adapter + 135 head grad tensors, all param grads finite; peak RSS 2.36 GB; inside the 180 s budget |

`scripts/train.py --help` runs and lists all arguments. `tests/`: **73 passed**, 0 failed
(`unittest discover -s tests`).

---

## 5. Data view

`reports/data_view.json`; the smoke view is `data/soda_smoke/`:

* `images/{train,val}` are **symlinks** to `data/SODA10M/{train,val}` — no image was copied.
* `labels/{train,val}` hold newly written YOLO txt files (37 train + 29 val boxes for the 2+2
  selected images; 0 degenerate boxes dropped, 0 empty label files among the selected ones).
* `data.yaml` (native `path/train/val/nc/names`) and `image_id_map.json` (stable
  `split/file_name → image id` map) are inside the view.
* Selected images are the **2 lowest stable image ids** per split; the full-conversion capability
  is implemented and documented but deliberately **not run** this round.
* Every box is verified against the JSON: JSON pixel `xywh` → normalised YOLO `xc yc w h` with
  clipping, and the reverse transform is unit-checked (`test_roundtrip_exact`).
* Validated fail-fast: unknown category, non-contiguous `category_id_map`, category-name
  mismatch, duplicate image id, missing file, path escape. `Tram` is not renamed and `Tricycle`
  is not merged into `Cyclist`.
* Image sizes from the JSON are cross-checked against the actually decoded files
  (0 mismatches).
* `iscrowd`/`ignore`: measured over the whole split — **train 41,110 annotations, 0 with
  `iscrowd=1`, 0 with `ignore=1`; val 37,129 annotations, 0 and 0**. SODA10M therefore has no
  ignore-region semantics to lose, and converting every instance to a normal YOLO box is exact
  for this dataset. (Also 0 boxes with non-positive `w`/`h`.) The validator counts and reports
  these flags for every split anyway, so the check would surface immediately if the source JSON
  ever changed.
* Nothing was written into `data/SODA10M` (no `.cache`, no new files).

---

## 6. Not executed / not verified (explicit)

| Item | Status |
|---|---|
| GPU native NATTEN parity for `na2d_av` | **not verified** — no GPU, no NATTEN. The CPU reference passed an independent oracle; the bounded claim is *“CPU reference implementation passes against the window semantics oracle”*. |
| Full-detector **single-pass** forward+backward | **not performed**. V07b verified the two chains separately (backbone fwd+bwd; adapter→neck→head fwd+bwd through `v8DetectionLoss`); V07 verified the detached path end to end. A single `loss.backward()` through OverLoCK→adapter→neck→head on one graph was not run. |
| Formal training (any epoch) | **not run**. Only trainer construction, optimizer partitioning and one loss/backward smoke. |
| AP / AP50 / AP50:95 / per-class AP / scale buckets | **not computed**. No mAP of any kind is reported — the head is randomly initialised. |
| Validation-set evaluation, `maxDets`/NMS thresholds, COCO↔native protocol alignment | **not done** |
| DDP, resume, EMA final-eval, checkpoint reload round-trip | **not verified** |
| `compile`, `fuse`, export (ONNX/TensorRT/…), deployment, latency | **not run** |
| FP16/AMP behaviour, `Detect.stride/anchors/strides` after `.half()`/`.to()` | **not verified** (the design defers FP16 acceptance) |
| YOLO neck/head from real YOLO weights | **not done** — no YOLO checkpoint was scanned for or downloaded; the neck/head are random and flagged as such |
| COCO 80→SODA 6 class-output remapping | **not attempted** (correctly out of scope) |
| P2 head, CNN side branch, extra attention, BiFPN, gating, distillation, new loss/assigner | **not implemented** (V1 scope) |

Historical context only, **not reproduced here**: model A COCO AP50 52.21 %, YOLO11s reference
53.92 %. Those numbers are from a different protocol and a different parameter count; no
comparison is claimed, and the evaluation protocol must be re-established before any comparison
is attempted.

---

## 7. Environment caveats worth carrying to the GPU machine

1. Run with `YOLO_CONFIG_DIR` (and `MPLCONFIGDIR`) pointed inside the task tree. Without it,
   Ultralytics tries to write `~/Library/Application Support/Ultralytics/settings.json`; on this
   machine those writes are denied by the sandbox (the file was **not** modified — the write
   failed). All results here used the in-tree config dir.
2. `overlock_yolo.model.ensure_local_ultralytics()` purges and re-imports `ultralytics` from the
   pinned local root. Import this package *before* any other `ultralytics` import if the conda
   copy (8.0.196) might otherwise be picked up.
3. `OSError: Killed: 9` (exit 137) was observed while developing the smoke script when a
   full-detector autograd graph was kept alive at high resolution. V07 now detaches at the raw
   backbone outputs and V08 scales real samples; the final run peaks at 2.36 GB. Budget more
   memory if you raise `--full-backward`.
4. The MPS/CPU conv kernels are not run-to-run bitwise reproducible (identical weights/shapes
   differ by ~1e-7 across `inference_mode` vs `no_grad` paths), so cross-run comparisons use a
   tight `allclose`, not `torch.equal`.

---

## 8. How to continue on a GPU machine

```bash
# 0) environment: install NATTEN for the native attention path (this round did not install it)
#    and keep Ultralytics caches inside the task tree
export YOLO_CONFIG_DIR=$PWD/cache/ultralytics MPLCONFIGDIR=$PWD/cache/mpl

# 1) full data view (~10k label files, symlinked images; a few minutes, CPU is fine)
python scripts/prepare_data.py --out data/soda_full

# 2) first GPU check: NATTEN parity + backend wiring, report the RESOLVED backend
python - <<'PY'
from overlock_yolo.attention_backend import resolve_backend, na2d_av, na2d_av_reference
import torch
d = torch.device("cuda")
print(resolve_backend("auto", d).as_dict())
a = torch.rand(1, 4, 64, 64, 25, device=d); v = torch.randn(1, 4, 64, 64, 8, device=d)
print((na2d_av(a, v, 5, backend="natten") - na2d_av_reference(a, v, 5)).abs().max())
PY

# 3) R1 training (template recipe; NOT tuned, NOT a controlled comparison)
python scripts/train.py --data data/soda_full/data.yaml \
  --backbone-weights /Users/lw/Documents/CNN-Mamba/OverLoCK-main/checkpoints/overlock_b_in1k_224.pth \
  --imgsz 640 --batch 16 --epochs 100 --device 0 --workers 8 \
  --optimizer auto --project runs/overlock_yolo11 --name r1_base_yolo11s

# 4) R0 reference: reproduce YOLO11s under the same data split / resolution / evaluator / budget
#    before attributing any change to the backbone. Unknowns you must fix yourself:
#    the historical run's exact split, imgsz, augmentation recipe, epoch budget and evaluator.
```

Before an R0/R1 comparison is meaningful, align: identical data split and view, identical
`imgsz`/letterbox, identical augmentation, identical epoch/iteration budget, identical
evaluator (same `maxDets`, score/NMS thresholds, image-id mapping, resize and inverse
transform), and shared neck/head initialisation. Report COCO AP50, AP50:95, per-class AP,
AP50_S/M/L with stated definitions, pre-NMS candidate coverage, and full-detector parameters,
memory and same-device latency.

CPU reference timing from this round is **not** deployment performance and must not be quoted
as such.
