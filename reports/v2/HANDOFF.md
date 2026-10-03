# HANDOFF — OverLoCK (XT/T/S/B) × YOLO11/YOLO26 × n/s/m/l/x @ 640

**Round**: V2 (incremental fix + extension of the V1 `hub-15-murw34bt` delivery).
**Design contract**: `DESIGN_V2.md` (copy of
`OverLoCK_YOLO11_YOLO26_V2_MultiVariant_640_Git_Design_20261003.md`; sections 10/11 are the
newest user additions and take precedence over the V1 document).
**Verdict**: `IMPLEMENTED / CPU_CHECKS_PASSED / GPU_AND_ACCURACY_UNVERIFIED`.
JSON reports in this directory are the source of truth; this file is the summary.

---

## 1. What was delivered

| Deliverable | Where |
|---|---|
| 40 structure combinations (`variant × family × scale`), default **T+YOLO11s+640** | `overlock_yolo/{variants,model,config}.py`, `configs/` |
| Strict experiment config (duplicate/unknown keys fail fast; CLI never overrides with defaults) | `overlock_yolo/config.py` |
| Central path resolution, no machine-specific path at import time | `overlock_yolo/paths.py`, `overlock_yolo/cli.py` |
| Four OverLoCK variants from the official factory bodies | `overlock_yolo/{variants,backbone}.py` |
| Lazy per-device NATTEN backend + memory-bounded differentiable reference | `overlock_yolo/attention_backend.py` |
| Native neck/head from a real `DetectionModel(dict)`-equivalent build, `bias_init`/stride included | `overlock_yolo/model.py` |
| Native losses: `v8DetectionLoss` (11) / `E2ELoss` (26), native trainer lifecycle | `overlock_yolo/{model,trainer}.py` |
| Native Ultralytics validation at a pinned 640×640 square protocol | `overlock_yolo/validator.py` |
| COCO → isolated native-YOLO view (real dirs, per-file symlinks, no source writes) | `overlock_yolo/data.py` |
| Parameters / partial MACs / analytic `na2d_av` MACs | `overlock_yolo/profile_model.py` |
| Server constraints, preflight, GPU smoke | `requirements-server.txt`, `constraints-server.txt`, `scripts/{server_preflight,gpu_smoke}.py` |
| Pinned Ultralytics source snapshot (AGPL-3.0) | `vendor/ultralytics/`, `THIRD_PARTY_NOTICES.md` |
| Independent local Git repository | this directory (branch `codex/overlock-yolo-portable`) |

## 2. V1 defects fixed, with evidence (DESIGN_V2 §7)

| # | Defect | Fix | Evidence |
|---|---|---|---|
| 1 | `forward(dict)` returned **predictions**, not loss | `forward(dict)` → `self.loss(batch)`; `loss()` mirrors `BaseModel.loss` | M07; `test_forward_dict_returns_loss_not_predictions[yolo11/yolo26]`, `test_loss_is_not_recursive_via_forward` |
| 2 | CPU-built model cached the **reference backend** after `.to(cuda)` | device-keyed `BackendCache` resolved lazily from the incoming tensor's device; explicit `natten` on CPU fails loudly | M08; `test_backend_resolution_is_lazy_and_per_device`, `test_stale_backend_cache_would_be_detected` |
| 3 | `parse_model` bypassed head init; the "rebuild then swap modules" helper dropped stride/bias | native path replicated *on the Detect head only*: real stride probe + `bias_init()` (both branches) + `initialize_weights`; backbone weights load **after** and never re-init the head | M03 (bias non-zero, stride 8/16/32, reg_max, one2one); `test_detect_head_is_initialised_and_stride_correct`, `test_backbone_load_does_not_reinitialise_the_head` |
| 4 | stem **and** adapters registered twice → duplicate `state_dict` paths | the stem lives only at `model[tail_start-1]`; `stem`/`adapters`/`tail` are properties | M04 (`duplicate_registration: []`); `test_single_registration_no_state_dict_aliases` |
| 5 | reference built the full `[B,heads,D,H·W,K²]` patch tensor before chunking; the circular-pad rationale was wrong | gather/patch construction moved **inside** the row-block loop; circular padding replaced by a vectorised modulo index map; comments corrected | M08 (values/gradients vs loop oracle, chunk invariance); `test_reference_never_materialises_the_full_patch_tensor` (peak tensor drops >10×), `test_reference_matches_naive_loop_oracle_on_edges` |
| 6 | one YAML mixed structural `scale: s` with augmentation `scale: 0.5`; Base/s and macOS paths hardcoded | separate `yolo.scale` / `train.scale` namespaces, top-level `scale:` rejected, strict schema, CLI→config mapping with "explicit only" semantics, `--project-root`/`--ultralytics-root` | M01; `test_structure_scale_and_augmentation_scale_are_separate`, `test_duplicate_and_unknown_config_keys_fail`, `test_cli_overrides_do_not_apply_defaults`, `test_no_hardcoded_machine_path_is_used_at_runtime`, `test_paths_resolve_from_another_cwd` |
| 7 | `--resume` was demanded but always refused; misleading required `--backbone-weights` | project checkpoint format (`overlock-yolo-state-v1`) with real resume of model/optimizer/epoch/`criterion.updates`; stock Ultralytics checkpoints explicitly rejected; `--init-checkpoint` separated from `--resume` | M09 round trip; `test_project_checkpoint_round_trip_and_structure_gate`; `trainer.check_resume`/`resume_training` |
| + | V1's "unused by forward ⇒ absent from state_dict" explanation for the missing keys was wrong | corrected to: *this classification checkpoint does not contain those detection-only parameters* (`extra_norm.*`, `h_proj.*`), so they keep their initialisation | `checkpoint_t.json` / `checkpoint_b.json` (`whitelist_correction`) |
| + | `--limit` exposed the whole image directory while only generating a few labels | `images/<split>` is a real directory with one symlink per selected image; `--limit` restricts images **and** labels; smoke views carry a marker that training refuses without `--allow-smoke-view` | M11; `test_structure_and_symlinks` (V1 test updated for the new contract) |

## 3. `reports/v2` contents

| File | Contents |
|---|---|
| `environment.json` | local interpreter/torch/threads, the CPU protocol actually used, differences from the target server, explicit "not verified" list |
| `compatibility.json` | 40 combinations, per-variant factory bodies + feature info, 10 native tail measurements, adapter maps, family contracts, vendored-source digest, target stack + wheel-availability evidence, loss/metric/protocol/resume facts |
| `checkpoint_t.json`, `checkpoint_b.json` | full T/B audit: container, prefix transforms, per-key match, ignored classification-only keys, whitelisted detection-only keys, per-module numel coverage, representative-parameter equality, cross-variant rejection |
| `checkpoint_manifest.json` | what checkpoints are expected, where, their hashes/sizes, and the rule that XT/S are never downloaded |
| `variant_matrix.json` | 40 cells (backbone + adapter + tail = total), plus 5 full-assembly cross-checks — all `match: true` |
| `profile_t_yolo11s.json/.md`, `profile_t_yolo26s.*`, `profile_t_yolo11n.*`, `profile_b_yolo11s.*` | unique parameter accounting, 224↔640 invariance, **partial** MACs, analytic `na2d_av` MACs, profiler versions, explicit `complete_flops.available=false` |
| `validation.json` | M01–M11 evidence (11/11 pass) |
| `regression_tests.json` | 25/25 targeted regression tests pass |
| `data_view.json` | the generated smoke view: real dirs, 1:1 image/label counts, per-class counts, no source writes |
| `git_verification.json` | repository/clone identity, tracked-file counts, sizes, vendor digest equality and the checks run inside the clone |

## 4. Measured facts (CPU, FP32, batch 1, `workers=0`)

**Structure** — 40/40 combinations resolve; every one of the 5 matrix cross-checks equals a fully
assembled model to the parameter:

| variant | backbone params | P3/P4/P5 channels |
|---|---|---|
| xt | 15,657,380 | 112 / 340 / 420 |
| t | 33,127,096 | 128 / 384 / 640 |
| s | 56,446,524 | 128 / 448 / 640 |
| b | 96,091,496 | 160 / 528 / 720 |

**640 protocol** (real T weights, batch 1):

| combination | adapted/detect shapes | anchors | eval output | forward |
|---|---|---|---|---|
| t+yolo11s | 80×80 / 40×40 / 20×20, 128/256/512 ch | 8400 | `[1, 10, 8400]` (NMS path) | 3.0 s |
| t+yolo26s | 80×80 / 40×40 / 20×20, 128/256/512 ch | 8400 | `[1, 300, 6]` (end-to-end top-k) | 3.2 s |

Letterbox on a real SODA sample (1920×1080): `ratio (0.3333, 0.3333)`, `pad (0, 140)`,
network `640×640` — equal-ratio resize + padding, not a stretch.

**Training step** (96×96, synthetic batch, both families): `model(batch)` → loss vector → single
backward → optimizer step. Backbone gradients 1759 tensors, adapters 9, head 135 (11) / 246 (26),
all finite. YOLO11 logs `box/cls/dfl`; YOLO26 logs `box/cls/l1` and its `E2ELoss.update()`
advances the branch schedule. A one-to-one-only backward produces **zero** backbone gradients
(the native detach), a one-to-many-only backward does not.

**Native validator** (2-image view, random neck/head — an interface check, not accuracy):
metric keys `metrics/{precision,recall,mAP50,mAP50-95}(B)`; `conf=0.001`; `save_json=False`;
every batch square at 640 with no protocol violations. YOLO11 used the NMS path
(`end2end_effective=False`, `reg_max=16`, no one2one); YOLO26 used the end-to-end path
(`nms=False`, `end2end_effective=True`, `reg_max=1`, one2one present) with no extra NMS.

**Costs** (unique objects, 1 MAC = 1 multiply-accumulate; GFLOPs would use 1 MAC = 2 FLOPs):

| combination | parameters | partial MACs (conv/linear/BN) | analytic `na2d_av` MACs |
|---|---|---|---|
| t+yolo11n | 34,583,034 | 49.84 G | 0.492 G |
| t+yolo11s | 37,576,026 | 53.15 G | 0.492 G |
| t+yolo26s | 38,098,420 | 53.63 G | 0.492 G |
| b+yolo11s | 100,626,442 | 149.66 G | 1.226 G |

`complete_flops.available = false` everywhere: no complete profiler covers `na2d_av`, the
dynamic-kernel `einsum`, softmax or the interpolations. Parameter counts are identical at 224 and
640, as they must be.

**Checkpoints** — `weights_only=True` load succeeds for both local files
(T `141,560,021 B`, `40c7ef3e…ca13`; B `395,257,645 B`, `558b8b79…81e8`). Every non-whitelisted
trainable key matches (`missing_disallowed = 0`); the 14 whitelisted keys are exactly
`extra_norm.*` + `h_proj.*` (333,056 / 421,008 numel). Loading T weights into a B backbone (and
vice versa) raises `CheckpointError`. XT/S have **no** checkpoint locally and were only built as
explicitly random structures.

## 4b. Independent local Git repository

| | |
|---|---|
| Path | `/Users/lw/Documents/CNN-Mamba/overlock_yolo11` (its own `.git`; the parent `CNN-Mamba` repo tracks nothing here and its index was not modified) |
| Branch | `codex/overlock-yolo-portable` |
| HEAD | `0b7c3e8` (`0b7c3e86c5645ee1e60aa25f3be95d367c01bd8c`), 3 commits |
| Tracked files | 424 (62 project files + 362 vendored Ultralytics files) |
| `.git` size | 3.1 MB; largest tracked blob 206,847 B; **no file > 1 MB** |
| Remote | **none configured** — nothing was pushed or published |
| Not committed | all `*.pth/*.pt/*.ckpt/*.onnx/*.engine`, data views, `*.cache`, `logs`, `runs`, `artifacts`, run products, machine-local config |
| Clone verification | local `git clone` → same commit, 424 files, no forbidden content; every CLI `--help` worked from `cwd=/tmp`; `project_root` and the Ultralytics root resolved to the clone's own `vendor/ultralytics` (origin asserted inside it, version 8.4.148); a random `t+yolo26n` model built; missing data/checkpoints reported with an actionable error and **nothing downloaded**; the vendored tree is byte-identical to the local snapshot (digest `72cc2311…94a2`, equal to the value recorded in `compatibility.json`); V1 unittest suite 55 OK (13 skipped without the external data) and V2 suite 25/25 |

Future workflow (nothing of this was executed): create your remote → `git remote add origin <your-url>`
→ `git push -u origin codex/overlock-yolo-portable` → clone it on the server → install the locked
requirements → place COCO + checkpoints → preflight → GPU smoke → train → native val. **No remote
URL is configured or claimed**, and datasets/checkpoints never travel through Git.

## 5. Not verified (do not read these as done)

* **Target GPU**: no RTX 4090 was rented. No latency, memory, driver or NATTEN-kernel number is
  claimed. `scripts/gpu_smoke.py` is generated and unexecuted.
* **NATTEN native kernel**: not installed locally; every local attention call is reported as
  `torch_reference`. The CUDA path is asserted only by `BackendCache` scheduling unit tests.
* **AMP**: never enabled; a separate `--amp` check exists in `gpu_smoke.py`.
* **Accuracy / AP**: never measured. The validator numbers above come from randomly initialised
  neck/head on two images.
* **Full training**: not started; `train.epochs`/batch templates are untested schedules.
* **Resume across a real multi-epoch run**: the save/load/reload path and the restored
  `criterion.updates` are unit-tested, but no multi-epoch run was executed.
* **DDP / multi-GPU, TensorRT/ONNX deployment, `deploy=True` reparameterisation**: out of scope.
* **COCO↔YOLO full conversion**: only 1–2 image views were generated (by design).

## 6. Minimal next steps on the server

1. `git clone <your-remote-url> overlock_yolo11 && cd overlock_yolo11`
2. `python scripts/server_preflight.py --expect-gpu` — it checks the real `torch.version.cuda`,
   runs an actual CUDA kernel and the NATTEN kernel, and explains each failure with its fix.
3. Install per `requirements-server.txt` with `-c constraints-server.txt`.
4. Place SODA10M and `overlock_{t,b}_in1k_224.pth` (see `reports/v2/checkpoint_manifest.json`);
   adjust `configs/*.yaml` paths; `python scripts/prepare_data.py --out data/soda_full`.
5. `python scripts/gpu_smoke.py --config configs/overlock_t_yolo11s_soda.yaml --device 0`
   (add `--probe-batch 1,2,4` to size the batch; AMP stays a separate `--amp` run).
6. `python scripts/train.py --config configs/overlock_t_yolo11s_soda.yaml --view data/soda_full
   --device 0 --batch 16 --workers 8 --amp`.
7. `python scripts/val.py --config ... --view data/soda_full --device 0` for the native metric,
   and re-run the T/B audit (`scripts/smoke.py --only M05`) to confirm the server copies hash the
   same.

## 7. Boundaries and cautions

* The local environment is conda `3dete` (Python 3.10 / torch 2.11 CPU). It is **not** the
  target stack; nothing here validates Python 3.12 / torch 2.5.1+cu124.
* `overlock_yolo/backbone.py` is a vendored official source file; the original repository is
  untouched. `vendor/ultralytics/ultralytics` is Ultralytics 8.4.148 (**AGPL-3.0**) with two
  unused demo images removed. Read `THIRD_PARTY_NOTICES.md` before redistributing.
* `reports/*.json` (V1) are kept as history and describe the V1 code, not this one.
* No remote is configured and nothing was pushed; the repository exists only locally.
