# Third-party sources and licenses

This repository vendors two third-party source trees so it can be cloned and run without
depending on a sibling checkout on the author's machine.  Neither is modified beyond what is
described below, and both keep their original copyright/license headers.

---

## 1. `vendor/ultralytics/ultralytics` — Ultralytics YOLO

| | |
|---|---|
| Upstream project | Ultralytics YOLO — https://github.com/ultralytics/ultralytics |
| Upstream package | https://pypi.org/project/ultralytics/ |
| Version | **8.4.148** (`ultralytics/__init__.py`) |
| License | **AGPL-3.0** — https://ultralytics.com/license |
| Vendored files | 362 (230 Python modules + `cfg/**/*.yaml`) |
| Combined SHA256 | `a093021414bff442728cfcd1473f8f56182baed8309cfe7937b3a6238fe17d20` (sorted `relpath sha256` lines) |
| Bytes | 3,873,259 |

**Provenance.** Copied file-by-file from the local checkout
`<sibling>/ultralytics-main/ultralytics` (a plain source snapshot of the released 8.4.148
package, no `.git` metadata present), which is the exact source tree all local V1/V2 checks were
run against.  This is a *fixed source snapshot*, not an upstream commit hash: the local snapshot
carries no Git metadata, so no commit is claimed.

**Local differences.** Exactly two files were dropped and nothing was edited: the demo image
assets `assets/bus.jpg` and `assets/zidane.jpg`.  They are binary sample images used only by
Ultralytics' own `YOLO("...")` smoke paths, are not referenced anywhere in this project, and
would otherwise add ~188 KB of binaries to the repository.  Every remaining file is byte-identical
to the source snapshot (verified by per-file SHA256 comparison).

**License obligations.** This project links against and redistributes Ultralytics code, which is
AGPL-3.0.  Every vendored `.py` file keeps its `# Ultralytics 🚀 AGPL-3.0 License` header.  A
`LICENSE` file was **not present** in the local snapshot (only the per-file headers), so none is
shipped here; the authoritative license text is at https://ultralytics.com/license.  Do not
relicense this directory.  If you redistribute this repository, keep the headers intact and ship
the AGPL-3.0 text alongside it.

---

## 2. `overlock_yolo/backbone.py` — OverLoCK detection backbone (vendored *file*, not a tree)

| | |
|---|---|
| Upstream project | OverLoCK — https://github.com/LMMMEng/OverLoCK |
| Paper | https://arxiv.org/abs/2502.20087 |
| Source file | `detection/models/overlock.py` of the local `<sibling>/OverLoCK-main` checkout |
| License | license of the original OverLoCK repository (see that repository) — unchanged here |

The file is copied into the package as an isolated, provenance-annotated adapter because the
upstream file imports `mmdet` / `mmcv` / `mmengine` / `natten` at module level.  The exact,
enumerated list of modifications is in the module docstring of
`overlock_yolo/backbone.py` (provenance header).  The original repository is never modified.

Upstream credits retained in the header: `apply_rpb` is borrowed by OverLoCK from
https://tinyurl.com/mrbub4t3, and `DilatedReparamBlock` follows UniRepLKNet
(https://github.com/AILab-CVC/UniRepLKNet).

---

## 3. Runtime dependencies (not vendored)

`torch`, `torchvision`, `natten`, `timm`, `einops`, `PyYAML`, `Pillow`, `numpy`, `scipy` and
`thop` are installed from wheels on the target machine; see `requirements-server.txt` and
`constraints-server.txt`.  None of their source is redistributed here.  NATTEN (SHI-Labs) is
used through its published wheel only.
