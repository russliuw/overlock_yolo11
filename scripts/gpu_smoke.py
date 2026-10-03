#!/usr/bin/env python
"""GPU smoke / memory probing -- generated for the server, NOT executed on this CPU machine.

Sequence (DESIGN_V2.md 8.2)::

    1. NATTEN vs reference numeric + gradient comparison on small tensors
       (real K = 5/7/13, boundary and H==K shapes, the real head_dim)
    2. T+s 640x640 forward with real weights
    3. the same image through the full native loss + backward + one optimizer step
    4. peak memory / latency, excluding vs including pre/post-processing
    5. checkpoint save + rebuild + reload

FP32 first.  AMP is a separate, explicitly requested check (``--amp``) and a FP32 success is
never reported as AMP-verified.  Batch sizing starts at 1 and is only probed when asked; an OOM
is recorded as-is (no unbounded retry).  Single-GPU only: DDP is not claimed.

Examples::

    python scripts/gpu_smoke.py --config configs/overlock_t_yolo11s_soda.yaml --device 0
    python scripts/gpu_smoke.py --config configs/overlock_t_yolo11s_soda.yaml --probe-batch 1,2,4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from overlock_yolo.cli import cli_guard, bootstrap, data_or_view, emit_summary, load_config_for_args  # noqa: E402
from overlock_yolo.config import build_arg_parser  # noqa: E402
from overlock_yolo.paths import project_root  # noqa: E402


@cli_guard
def main(argv=None) -> int:
    ap = build_arg_parser("GPU smoke / memory probe (server only; not run on the CPU machine)")
    ap.add_argument("--out", default=os.path.join(project_root(), "reports", "v2", "gpu_smoke.json"))
    ap.add_argument("--probe-batch", default=None, help="comma-separated batch sizes to probe, e.g. 1,2,4")
    # NOTE: --amp already exists in the shared experiment argument group.
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--data", default=None)
    args = ap.parse_args(argv)

    boot = bootstrap(args)
    cfg = load_config_for_args(args, require_weights_file=False)
    # --amp is a *separate* check: it never turns the FP32 result into an AMP result.
    emit_summary(cfg)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device: scripts/gpu_smoke.py is a server check. The local CPU run must not be "
            "reported as a GPU verification."
        )
    device = torch.device(f"cuda:{args.device or '0'}".replace("cuda:cuda:", "cuda:"))
    torch.cuda.set_device(device)

    from overlock_yolo.attention_backend import natten_available, na2d_av_reference
    from overlock_yolo.cli import build_from_config
    from overlock_yolo.trainer import load_project_checkpoint, save_project_checkpoint

    result = {
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_gb": torch.cuda.get_device_properties(device).total_memory / 1e9,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "natten_available": natten_available(),
        "combination": cfg["_resolved"]["combination"],
        "steps": {},
    }

    # --- 1. NATTEN vs reference -------------------------------------------------------
    if natten_available():
        from natten.functional import na2d_av as na2d_native

        rows = []
        for (b, heads, h, w, d, k) in ((1, 2, 7, 7, 16, 5), (1, 2, 5, 5, 16, 5), (1, 2, 9, 9, 16, 7), (1, 1, 13, 13, 16, 13), (1, 2, 4, 6, 32, 5)):
            a32 = torch.rand(b, heads, h, w, k * k, device=device)
            v32 = torch.rand(b, heads, h, w, d, device=device)
            native = na2d_native(a32.contiguous(), v32.contiguous(), k)
            ref = na2d_av_reference(a32, v32, k)
            fwd_err = float((native - ref).abs().max())
            a1 = a32.detach().clone().requires_grad_(True)
            v1 = v32.detach().clone().requires_grad_(True)
            na2d_native(a1, v1, k).sum().backward()
            a2 = a32.detach().clone().requires_grad_(True)
            v2 = v32.detach().clone().requires_grad_(True)
            na2d_av_reference(a2, v2, k).sum().backward()
            bwd_err = max(float((a1.grad - a2.grad).abs().max()), float((v1.grad - v2.grad).abs().max()))
            rows.append({"case": [b, heads, h, w, d, k], "forward_max_abs_err": fwd_err, "backward_max_abs_err": bwd_err,
                         "forward_within_1e-4": fwd_err < 1e-4, "backward_within_1e-3": bwd_err < 1e-3})
        result["steps"]["natten_vs_reference"] = rows
        print("[gpu_smoke] NATTEN vs reference:", json.dumps(rows, indent=2))
    else:
        result["steps"]["natten_vs_reference"] = {
            "status": "skipped",
            "reason": "NATTEN is not importable; 'auto' will silently use the CPU reference which is far slower",
        }

    # --- 2/3/4/5. model forward, loss/backward, memory, checkpoint --------------------
    from overlock_yolo.paths import resolve_checkpoint_path

    weights = resolve_checkpoint_path(cfg["_resolved"]["backbone_variant"])
    result["backbone_weights"] = weights
    model = build_from_config(cfg, ultra_root=boot.ultralytics_root, weights=weights).to(device).float()
    model.train()
    data = data_or_view(cfg, args.data)
    batch_plan = [int(x) for x in (args.probe_batch.split(",") if args.probe_batch else ["1"])]

    from overlock_yolo.data import build_native_dataloader

    for batch_size in batch_plan:
        entry = {"batch": batch_size}
        try:
            torch.cuda.reset_peak_memory_stats(device)
            if data:
                loader = build_native_dataloader(data, int(cfg["_resolved"]["imgsz"]), mode="val", batch=batch_size, workers=0, device=device)
                batch = next(iter(loader))
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                batch["img"] = batch["img"].float() / 255
            else:
                g = torch.Generator().manual_seed(0)
                batch = {
                    "img": torch.rand(batch_size, 3, int(cfg["_resolved"]["imgsz"]), int(cfg["_resolved"]["imgsz"]), generator=g, device=device),
                    "batch_idx": torch.zeros(4, device=device),
                    "cls": torch.randint(0, int(cfg["_resolved"]["nc"]), (4,), generator=g).float().to(device),
                    "bboxes": (torch.rand(4, 4, generator=g) * 0.3 + 0.2).to(device),
                }
                entry["data_source"] = "synthetic (no view given)"
            t0 = time.time()
            loss, items = model(batch)
            loss.sum().backward()
            opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-5)
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            entry["loss_seconds"] = round(time.time() - t0, 3)
            entry["loss_items"] = {k: float(v) for k, v in items.items()}
            entry["peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            entry["status"] = "pass"
        except RuntimeError as exc:
            entry["status"] = "OOM" if "out of memory" in str(exc).lower() else "error"
            entry["error"] = str(exc)[:300]
            torch.cuda.empty_cache()
        result["steps"].setdefault("train_step_by_batch", []).append(entry)
        print(f"[gpu_smoke] batch={batch_size}: {json.dumps(entry)[:300]}")

    # --- inference timing (eval, no grad) --------------------------------------------
    model.eval()
    imgsz = int(cfg["_resolved"]["imgsz"])
    x = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.repeats):
            model(x)
        torch.cuda.synchronize()
        forward_ms = (time.time() - t0) / args.repeats * 1e3
    result["steps"]["forward_latency_fp32"] = {
        "imgsz": imgsz,
        "batch": 1,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "model_forward_ms_excluding_prepost": round(forward_ms, 2),
        "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "note": "model forward only: resize/letterbox, decode, NMS and post-processing are excluded",
    }

    # --- 5. checkpoint round trip -----------------------------------------------------
    ckpt_path = os.path.join(project_root(), "artifacts", "v2", "gpu_smoke_state.pt")
    manifest = save_project_checkpoint(model, ckpt_path)
    rebuilt = build_from_config(cfg, ultra_root=boot.ultralytics_root, weights=None).to(device).float()
    reload = load_project_checkpoint(rebuilt, ckpt_path, strict=True)
    result["steps"]["checkpoint_round_trip"] = {
        "manifest": {k: v for k, v in manifest.items() if k != "structure"},
        "loaded_tensors": reload["loaded_tensors"],
        "missing_keys": reload["missing_keys"],
        "structure": reload["structure"],
    }

    if args.amp:
        result["steps"]["amp_check"] = _amp_check(model, x)
        result["amp_is_a_separate_check"] = True

    result["limitations"] = [
        "single GPU only (DDP is not claimed)",
        "this is a smoke check, not a training or accuracy (AP) result",
        "the FP32 numbers above are valid only for this exact driver/torch/NATTEN combination",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"[gpu_smoke] report -> {args.out}")
    return 0


def _amp_check(model, x) -> dict:
    import torch

    try:
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
        pred = out[0] if isinstance(out, tuple) else out
        ok = bool(torch.isfinite(pred.float()).all())
        return {"status": "pass" if ok else "fail", "finite": ok, "dtype": str(pred.dtype),
                "note": "AMP is an independent check; FP32 success does not imply AMP correctness"}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    raise SystemExit(main())
