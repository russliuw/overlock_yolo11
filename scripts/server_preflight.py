#!/usr/bin/env python
"""Read-only server preflight (DESIGN_V2.md 3.1).  Never installs, upgrades or downloads anything.

Checks the Python version, CPU architecture, torch/torchvision and their CUDA build, NATTEN /
timm / einops / PyYAML, the torchvision NMS operator, the pinned Ultralytics source, the GPU
name / memory / compute capability, the driver, a *real* CUDA kernel execution and the actual
``torch.version.cuda``.

Two things this deliberately does NOT do:

* it never treats ``nvidia-smi``'s "CUDA Version" (a driver *capability* ceiling) as the CUDA
  version torch was compiled against;
* it never reports GPU success from a wheel merely existing -- the CUDA kernel test must run.

Exit code 0 when every *required* check passes, 1 otherwise (with the minimal fix per failure).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

#: Version targets from DESIGN_V2.md 3.1.  Report-only: a mismatch is a warning, not a failure,
#: except for the CUDA kernel test and the pinned Ultralytics source.
TARGET = {
    "python": "3.12.x",
    "torch": "2.5.1+cu124",
    "torchvision": "0.20.1",
    "natten": "0.17.4+torch250cu124",
    "timm": "0.6.13 (minimal fix candidate) or the project-adapted modern interface",
}

REQUIRED_WHEEL_INDEX = "https://download.pytorch.org/whl/cu124"
NATTEN_INDEX = "https://whl.natten.org/old/"


class Report:
    def __init__(self):
        self.checks = []

    def add(self, name, status, detail, fix=None, required=True):
        assert status in ("pass", "warn", "fail", "info"), status
        self.checks.append({"check": name, "status": status, "detail": detail, "fix": fix, "required": required})
        tag = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "info": "INFO"}[status]
        print(f"[{tag}] {name}: {detail}" + (f"\n        fix: {fix}" if fix and status in ("fail", "warn") else ""))

    def ok(self) -> bool:
        return not any(c["status"] == "fail" and c["required"] for c in self.checks)

    def as_dict(self):
        counts = {k: 0 for k in ("pass", "warn", "fail", "info")}
        for c in self.checks:
            counts[c["status"]] += 1
        return {
            "target": TARGET,
            "summary": {**counts, "required_failures": sum(1 for c in self.checks if c["status"] == "fail" and c["required"])},
            "checks": self.checks,
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Read-only environment preflight (no installs, no downloads)")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "v2", "server_preflight.json"))
    ap.add_argument("--ultralytics-root", default=None)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--expect-gpu", action="store_true", help="mark the CUDA checks as required")
    ap.add_argument("--skip-cuda-kernel", action="store_true", help="skip the actual CUDA kernel test (not recommended)")
    args = ap.parse_args(argv)

    rep = Report()

    # --- python / platform ------------------------------------------------------------
    py = ".".join(str(v) for v in sys.version_info[:3])
    rep.add("python_version", "pass" if sys.version_info[:2] == (3, 12) else "warn", f"{py} (target {TARGET['python']})",
            fix="create a Python 3.12 environment, or fall back to a separate Python 3.10 env if a dependency blocks 3.12")
    rep.add("platform", "info", f"{platform.platform()} / {platform.machine()}")
    if platform.system() != "Linux":
        rep.add("os", "warn", f"{platform.system()} is not the target Ubuntu 22.04", fix="run this on the target server")

    # --- torch / torchvision ----------------------------------------------------------
    try:
        import torch

        rep.add("torch", "pass", torch.__version__ + (" (target 2.5.1+cu124)" if not torch.__version__.startswith("2.5.1") else ""))
        rep.add("torch.version.cuda", "pass" if torch.version.cuda else "fail",
                str(torch.version.cuda),
                fix=f"install the CUDA 12.4 wheel: python -m pip install torch==2.5.1 torchvision==0.20.1 "
                    f"--index-url {REQUIRED_WHEEL_INDEX}",
                required=args.expect_gpu)
        rep.add("torch.cuda.is_available", "pass" if torch.cuda.is_available() else ("fail" if args.expect_gpu else "warn"),
                str(torch.cuda.is_available()), fix="check the NVIDIA driver and that the CUDA build of torch is installed")
    except Exception as exc:
        rep.add("torch", "fail", f"{type(exc).__name__}: {exc}", fix="install the locked requirements", required=True)
        torch = None

    try:
        import torchvision

        rep.add("torchvision", "pass", torchvision.__version__)
        has_nms = hasattr(torchvision.ops, "nms")
        rep.add("torchvision.ops.nms", "pass" if has_nms else "fail", str(has_nms),
                fix="torchvision is required for the native NMS path")
    except Exception as exc:
        rep.add("torchvision", "fail", f"{type(exc).__name__}: {exc}", fix="install torchvision==0.20.1")

    # --- other dependencies -----------------------------------------------------------
    for name, required in (("natten", True), ("timm", True), ("einops", True), ("yaml", True), ("PIL", True)):
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "unknown")
            status = "pass"
            fix = None
            if name == "natten" and "0.17.4" not in str(version):
                status, fix = "warn", f"install the exact wheel: python -m pip install "
                fix = (f"python -m pip install 'natten==0.17.4+torch250cu124' --only-binary=:all: --no-deps "
                       f"-f {NATTEN_INDEX}")
            rep.add(name, status, str(version), fix=fix, required=required)
        except Exception as exc:
            fix = None
            if name == "natten":
                fix = (f"python -m pip install 'natten==0.17.4+torch250cu124' --only-binary=:all: --no-deps "
                       f"-f {NATTEN_INDEX}")
            rep.add(name, "fail" if required else "warn", f"{type(exc).__name__}: {exc}", fix=fix, required=required)

    # --- pinned ultralytics source ----------------------------------------------------
    try:
        from overlock_yolo.paths import install_ultralytics_root, prepare_ultralytics_env, project_root, resolve_ultralytics_root
        from overlock_yolo.variants import YOLO_YAML_RELPATH

        prepare_ultralytics_env()
        root = resolve_ultralytics_root(args.ultralytics_root, project_root_=args.project_root)
        origin = install_ultralytics_root(root)
        import ultralytics

        rep.add("ultralytics_source", "pass", f"{origin} (version {ultralytics.__version__})")
        missing = [os.path.join(root, *p) for p in YOLO_YAML_RELPATH.values() if not os.path.isfile(os.path.join(root, *p))]
        rep.add("ultralytics_model_yamls", "pass" if not missing else "fail",
                "yolo11.yaml + yolo26.yaml present" if not missing else f"missing {missing}",
                fix="vendor the pinned Ultralytics source snapshot (vendor/ultralytics/ultralytics)")
    except Exception as exc:
        rep.add("ultralytics_source", "fail", f"{type(exc).__name__}: {exc}",
                fix="pass --ultralytics-root or vendor the pinned source snapshot", required=True)

    # --- GPU details + a real CUDA kernel ---------------------------------------------
    smi = _nvidia_smi()
    if smi.get("available"):
        rep.add("nvidia_smi_gpus", "info", json.dumps(smi["gpus"]))
        rep.add(
            "driver_cuda_ceiling",
            "info",
            f"nvidia-smi reports CUDA Version {smi.get('cuda_version')} (driver capability ceiling, NOT torch.version.cuda)",
        )
    else:
        rep.add("nvidia_smi", "fail" if args.expect_gpu else "warn", smi.get("error", "not available"),
                fix="install/repair the NVIDIA driver on the server")

    if torch is not None and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        rep.add("gpu_device", "pass",
                f"{props.name}, {props.total_memory / 1e9:.1f} GB, compute capability {props.major}.{props.minor}")
        if "4090" in props.name:
            rep.add("gpu_target_match", "pass", "RTX 4090 detected")
        else:
            rep.add("gpu_target_match", "warn", f"{props.name} is not the planned RTX 4090")
        if not args.skip_cuda_kernel:
            ok, detail = _cuda_kernel_test(torch)
            rep.add("cuda_kernel_execution", "pass" if ok else "fail", detail,
                    fix="the driver/CUDA runtime cannot execute kernels; reinstall the driver or the matching torch build")
            if ok:
                ok_n, detail_n = _natten_kernel_test()
                rep.add("natten_kernel_execution", "pass" if ok_n else "fail", detail_n,
                        fix="install the cp312 cu124 NATTEN wheel; without it GPU attention silently uses the "
                            "reference implementation, which is much slower")
        else:
            rep.add("cuda_kernel_execution", "info", "skipped by --skip-cuda-kernel", required=False)
    else:
        rep.add("cuda_kernel_execution", "fail" if args.expect_gpu else "warn", "no CUDA device available",
                fix="this is expected on the local CPU machine; run the preflight on the server")

    payload = rep.as_dict()
    payload["env"] = {
        "python": sys.version,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "project_root": ROOT,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\npreflight -> {args.out}")
    print(json.dumps(payload["summary"], indent=2))
    return 0 if rep.ok() else 1


def _nvidia_smi() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if out.returncode != 0:
        return {"available": False, "error": (out.stderr or "nvidia-smi failed").strip()[:200]}
    gpus = [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]
    version = None
    try:
        head = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
        for line in head.splitlines():
            if "CUDA Version" in line:
                version = line.split("CUDA Version:")[1].split()[0]
                break
    except Exception:  # pragma: no cover
        pass
    return {"available": bool(gpus), "gpus": gpus, "cuda_version": version}


def _cuda_kernel_test(torch) -> tuple:
    try:
        a = torch.rand(64, 64, device="cuda")
        b = torch.rand(64, 64, device="cuda")
        c = (a @ b).sum().item()
        torch.cuda.synchronize()
        return True, f"matmul executed on {torch.cuda.get_device_name(0)} (checksum finite: {c == c})"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _natten_kernel_test() -> tuple:
    try:
        import torch
        from natten.functional import na2d_av

        attn = torch.rand(1, 2, 8, 8, 25, device="cuda")
        value = torch.rand(1, 2, 8, 8, 16, device="cuda")
        out = na2d_av(attn, value, 5)
        torch.cuda.synchronize()
        return True, f"native na2d_av produced {tuple(out.shape)} on CUDA"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


if __name__ == "__main__":
    raise SystemExit(main())
