"""V2 regression tests for the interface defects listed in DESIGN_V2.md 7.

These are the *targeted* tests: each one fails on the V1 code and passes on this one.  They are
fast (no 640 forward, no real checkpoint) so they can run on every change::

    OMP_NUM_THREADS=2 python -m pytest tests/test_v2_contract.py -q

The slower, end-to-end evidence lives in ``scripts/smoke.py`` (M01-M11) and
``reports/v2/validation.json``.

Locally pytest is not installed, so `tests/run_v2_tests.py` runs this module with a small shim;
both produce the same verdicts.  The file lives in ``tests/pytest_contract/`` so that
``unittest discover -s tests`` does not try to import it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from overlock_yolo.paths import install_ultralytics_root, prepare_ultralytics_env, resolve_ultralytics_root  # noqa: E402

prepare_ultralytics_env()
ULTRA_ROOT = resolve_ultralytics_root()
install_ultralytics_root(ULTRA_ROOT)

from overlock_yolo.cli import build_from_config  # noqa: E402
from overlock_yolo.config import ConfigError, load_experiment  # noqa: E402
from overlock_yolo.model import build_native_tail  # noqa: E402

SMALL = 96


def cfg(**overrides):
    return load_experiment(None, cli_overrides=overrides)


def build(**overrides):
    return build_from_config(cfg(**overrides), ultra_root=ULTRA_ROOT, weights=None)


def mini_batch(size: int = SMALL, n: int = 3, nc: int = 6):
    g = torch.Generator().manual_seed(0)
    return {
        "img": torch.rand(1, 3, size, size, generator=g),
        "batch_idx": torch.zeros(n),
        "cls": torch.randint(0, nc, (n,), generator=g).float(),
        "bboxes": torch.rand(n, 4, generator=g) * 0.3 + 0.2,
    }


# ---------------------------------------------------------------------------- (7.1) dict forward
@pytest.mark.parametrize("family", ["yolo11", "yolo26"])
def test_forward_dict_returns_loss_not_predictions(family):
    """DESIGN_V2 7.1: the native trainer calls ``loss, loss_items = self.model(batch)``."""
    model = build(variant="t", yolo_family=family, yolo_scale="n")
    model.train()
    out = model(mini_batch())
    assert isinstance(out, tuple) and len(out) == 2, type(out)
    loss, items = out
    assert torch.is_tensor(loss) and loss.numel() >= 1
    assert isinstance(items, dict) and items, items
    # the loss must be sumable and differentiable exactly as the trainer uses it
    loss.sum().backward()
    finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert finite

    # tensor input still returns predictions
    model.eval()
    with torch.no_grad():
        preds = model(torch.zeros(1, 3, SMALL, SMALL))
    assert preds is not None


def test_loss_is_not_recursive_via_forward():
    """``loss()`` must call ``forward`` on the image, not re-enter ``forward(dict)``."""
    model = build(variant="t", yolo_family="yolo11", yolo_scale="n")
    model.train()
    batch = mini_batch()
    preds = model(batch["img"])
    loss_direct, items_direct = model.loss(batch, preds)
    loss_via_forward, items_via_forward = model(batch)
    assert torch.allclose(loss_direct.detach(), loss_via_forward.detach())
    assert set(items_direct) == set(items_via_forward)


# ------------------------------------------------------------------- (7.2) device-bound backend
def test_backend_resolution_is_lazy_and_per_device():
    """Construction on CPU must not freeze the backend for a later CUDA move."""
    from overlock_yolo.attention_backend import BackendCache, AttentionBackendError

    model = build(variant="t", yolo_family="yolo11", yolo_scale="n")
    report = model.attention_backend_report()
    assert report["requested"] == ["auto"]
    # CPU construction alone resolves nothing: resolution happens at forward time
    assert report["resolved_by_device"] == {}
    with torch.no_grad():
        model(torch.zeros(1, 3, SMALL, SMALL))
    resolved = model.attention_backend_report()
    assert "cpu" in resolved["resolutions"]
    assert resolved["resolved_by_device"][list(resolved["resolved_by_device"])[0]]["resolved"] == "torch_reference"

    # An explicit 'natten' request must fail loudly on CPU -- here because NATTEN is absent.
    cache = BackendCache("natten")
    with pytest.raises(AttentionBackendError):
        cache.resolve(torch.device("cpu"))
    if torch.cuda.is_available():  # pragma: no cover - CPU-only local runs
        assert cache.resolve(torch.device("cuda:0")).resolved == "natten"
        # a device-keyed cache must hold BOTH answers at once, which is the whole point
        assert set(cache.report()["resolved_by_device"]) == {"cpu:None", "cuda:0"}


def test_stale_backend_cache_would_be_detected():
    """A CUDA request on a machine without NATTEN must fail loudly, never fall back silently."""
    from overlock_yolo.attention_backend import AttentionBackendError, resolve_backend

    if torch.cuda.is_available():  # pragma: no cover
        pytest.skip("CUDA present; the no-CUDA branch is what this asserts")
    with pytest.raises(AttentionBackendError):
        resolve_backend("natten", device=torch.device("cuda:0"))


# ---------------------------------------------------------------------- (7.3) head initialisation
@pytest.mark.parametrize("family,reg_max", [("yolo11", 16), ("yolo26", 1)])
def test_detect_head_is_initialised_and_stride_correct(family, reg_max):
    modules, _save, _yaml, report = build_native_tail(family, "s", 6, ULTRA_ROOT)
    head = modules[-1]
    assert int(head.reg_max) == reg_max
    assert [float(s) for s in head.stride.tolist()] == [8.0, 16.0, 32.0]
    assert float(head.cv3[0][-1].bias.abs().max()) > 0, "bias_init never ran"
    assert torch.allclose(head.cv2[0][-1].bias.detach(), torch.full_like(head.cv2[0][-1].bias, 2.0))
    if family == "yolo26":
        assert getattr(head, "one2one_cv2", None) is not None
        assert float(head.one2one_cv3[0][-1].bias.abs().max()) > 0, "one2one bias_init never ran"
    else:
        assert getattr(head, "one2one_cv2", None) is None


def test_backbone_load_does_not_reinitialise_the_head():
    """Loading backbone weights must leave the neck/head initialisation untouched."""
    model = build(variant="t", yolo_family="yolo11", yolo_scale="s")
    before = {k: v.clone() for k, v in model.tail.state_dict().items()}
    head_bias = model.model[-1].cv3[0][-1].bias.detach().clone()
    # loading a *non-existent* file must raise before touching anything
    with pytest.raises(Exception):
        model.load_backbone_weights(os.path.join(ROOT, "does-not-exist.pth"))
    after = model.tail.state_dict()
    assert all(torch.equal(v, after[k]) for k, v in before.items())
    assert torch.equal(head_bias, model.model[-1].cv3[0][-1].bias.detach())


# ----------------------------------------------------------------------- (7.4) registration paths
def test_single_registration_no_state_dict_aliases():
    model = build(variant="t", yolo_family="yolo11", yolo_scale="s")
    rep = model.parameter_groups_report()
    assert rep["total"]["duplicate_registration"] == []
    assert rep["total"]["buckets_cover_all_parameters"]
    sd = model.state_dict()
    assert len(sd) == len(set(sd))
    # the stem is reachable through exactly one canonical path
    assert model.stem is model.model[model.tail_start - 1]
    stem_keys = [k for k in sd if k.endswith("adapter1.conv.weight")]
    assert len(stem_keys) == 1, stem_keys
    assert stem_keys[0] == f"model.{model.tail_start - 1}.adapter1.conv.weight", stem_keys
    assert not any(k.startswith("stem_module.") for k in sd), "a second stem registration path exists"
    assert model.model[-1].__class__.__name__ == "Detect"
    # the adapters property is a *view*, not a second registration
    assert len(list(model.adapters)) == 3
    assert len(model.state_dict()) == len(sd)


# ------------------------------------------------------------------ (7.5) reference memory bound
def test_reference_never_materialises_the_full_patch_tensor():
    """The chunked reference must gather inside the loop (DESIGN_V2 7.5)."""
    from overlock_yolo.attention_backend import na2d_av_reference

    b, heads, h, w, d, k = 1, 2, 32, 32, 16, 5
    attn = torch.rand(b, heads, h, w, k * k, dtype=torch.float64, requires_grad=True)
    value = torch.rand(b, heads, h, w, d, dtype=torch.float64, requires_grad=True)
    full = na2d_av_reference(attn, value, k, row_chunk=0)
    chunked = na2d_av_reference(attn, value, k, row_chunk=97)
    assert torch.equal(full, chunked), "chunking must not change the maths"

    # Forward-only peak allocation: a small row chunk must stay well below the
    # [B, heads, D, H*W, K*K] intermediate the unchunked version builds.
    import torch.utils._python_dispatch as _pd

    class PeakTracker(_pd.TorchDispatchMode):
        """Track the largest single tensor produced during the forward."""

        def __init__(self):
            self.peak = 0
            self.n = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in out if isinstance(out, (tuple, list)) else [out]:
                if torch.is_tensor(t):
                    self.peak = max(self.peak, t.numel())
                    self.n += 1
            return out

    def forward_peak(chunk):
        a = attn.detach()
        v = value.detach()
        tracker = PeakTracker()
        with torch.no_grad(), tracker:
            na2d_av_reference(a, v, k, row_chunk=chunk)
        return tracker.peak

    full_peak = forward_peak(0)
    chunked_peak = forward_peak(h)  # one output row per block
    theoretical = b * heads * d * h * w * k * k
    assert chunked_peak < full_peak / 10, (chunked_peak, full_peak)
    assert full_peak >= theoretical, (full_peak, theoretical)


def test_reference_matches_naive_loop_oracle_on_edges():
    import itertools

    from overlock_yolo.attention_backend import na2d_av_reference

    def oracle(attn, value, k):
        b, hd, h, w, _ = attn.shape
        d = value.shape[-1]
        out = torch.zeros(b, hd, h, w, d, dtype=value.dtype)
        for bi, hi, y, x in itertools.product(range(b), range(hd), range(h), range(w)):
            sy = min(max(y - k // 2, 0), h - k)
            sx = min(max(x - k // 2, 0), w - k)
            acc = torch.zeros(d, dtype=value.dtype)
            for i, j in itertools.product(range(k), range(k)):
                acc += attn[bi, hi, y, x, i * k + j] * value[bi, hi, sy + i, sx + j]
            out[bi, hi, y, x] = acc
        return out

    for (h, w, k) in ((5, 5, 5), (7, 5, 5), (4, 7, 3), (13, 13, 13), (6, 6, 3), (8, 5, 5), (5, 8, 5)):
        a = torch.randn(1, 2, h, w, k * k, dtype=torch.float64)
        v = torch.randn(1, 2, h, w, 4, dtype=torch.float64)
        assert torch.allclose(na2d_av_reference(a, v, k, row_chunk=2), oracle(a, v, k), atol=1e-12)


# ------------------------------------------------------------------- (7.6) config / hardcoding
def test_no_hardcoded_machine_path_is_used_at_runtime():
    """Machine-specific paths may appear as *documented provenance*, never as code literals."""
    import ast

    bad = []
    pkg = os.path.join(ROOT, "overlock_yolo")
    for name in sorted(os.listdir(pkg)):
        if not name.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(pkg, name), encoding="utf-8").read())
        # A module/class/function docstring is documentation, not a used value: locate the
        # Constant nodes that *are* docstrings and exclude them from the scan.
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", [])
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                    docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and "/Users/" in node.value
                and id(node) not in docstrings
            ):
                bad.append((name, node.lineno, node.value[:60]))
    assert bad == [], f"machine-specific path literal(s) in executable code: {bad}"


def test_repository_defaults_contain_no_absolute_paths():
    """The shipped configs and the path resolver must not name a user's machine."""
    import yaml

    for name in sorted(os.listdir(os.path.join(ROOT, "configs"))):
        text = open(os.path.join(ROOT, "configs", name), encoding="utf-8").read()
        assert "/Users/" not in text, name
        yaml.safe_load(text)  # must stay parseable
    src = open(os.path.join(ROOT, "overlock_yolo", "paths.py"), encoding="utf-8").read()
    assert '"/Users' not in src and "'/Users" not in src


def test_paths_resolve_from_another_cwd():
    probe = (
        "import sys; sys.path.insert(0, %r);"
        "from overlock_yolo.paths import resolve_project_root, resolve_ultralytics_root;"
        "print(resolve_project_root()); print(resolve_ultralytics_root())" % ROOT
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd="/")
    assert out.returncode == 0, out.stderr[-400:]
    root, ultra = out.stdout.strip().splitlines()
    assert root == ROOT
    assert os.path.isfile(os.path.join(ultra, "ultralytics", "__init__.py"))


def test_duplicate_and_unknown_config_keys_fail():
    with tempfile.TemporaryDirectory() as td:
        dup = os.path.join(td, "d.yaml")
        open(dup, "w").write("yolo:\n  scale: s\n  scale: m\n")
        with pytest.raises(ConfigError):
            load_experiment(dup)
        unknown = os.path.join(td, "u.yaml")
        open(unknown, "w").write("yolo:\n  scael: s\n")
        with pytest.raises(ConfigError):
            load_experiment(unknown)
        amb = os.path.join(td, "a.yaml")
        open(amb, "w").write("scale: 0.5\n")
        with pytest.raises(ConfigError):
            load_experiment(amb)


def test_structure_scale_and_augmentation_scale_are_separate():
    c = cfg(variant="t", yolo_scale="s", aug_scale=0.25)
    assert c["_resolved"]["yolo_scale"] == "s"
    assert c["_resolved"]["aug_scale"] == 0.25
    assert c["yolo"]["scale"] == "s" and c["train"]["scale"] == 0.25


def test_rect_and_multiscale_are_rejected():
    with tempfile.TemporaryDirectory() as td:
        for body in ("train:\n  rect: true\n", "train:\n  multi_scale: true\n"):
            path = os.path.join(td, "x.yaml")
            open(path, "w").write(body)
            with pytest.raises(ConfigError):
                load_experiment(path)


def test_cli_overrides_do_not_apply_defaults():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "e.yaml")
        open(path, "w").write("backbone:\n  variant: b\nyolo:\n  family: yolo26\n  scale: m\ntrain:\n  scale: 0.3\n")
        c = load_experiment(path, cli_overrides={"variant": "t"})
        assert c["_resolved"]["combination"] == "t+yolo26m"
        assert c["_resolved"]["aug_scale"] == 0.3  # not reset to the default 0.5


# ------------------------------------------------------------------------- (7.7) resume / save
def test_project_checkpoint_round_trip_and_structure_gate():
    from overlock_yolo.trainer import load_project_checkpoint, save_project_checkpoint

    model = build(variant="t", yolo_family="yolo26", yolo_scale="n")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "s.pt")
        save_project_checkpoint(model, path)
        rebuilt = build(variant="t", yolo_family="yolo26", yolo_scale="n", seed=99)
        report = load_project_checkpoint(rebuilt, path, strict=True)
        assert report["missing_keys"] == [] and report["unexpected_keys"] == []
        assert all(torch.equal(v, rebuilt.state_dict()[k]) for k, v in model.state_dict().items())
        wrong = build(variant="b", yolo_family="yolo11", yolo_scale="x")
        with pytest.raises(ConfigError):
            load_project_checkpoint(wrong, path, strict=False)


def test_thop_artifacts_never_reach_a_checkpoint():
    from overlock_yolo.profile_model import count_partial_macs, strip_thop_artifacts
    from overlock_yolo.trainer import load_project_checkpoint, save_project_checkpoint

    model = build(variant="t", yolo_family="yolo11", yolo_scale="n")
    count_partial_macs(model, imgsz=64)
    # the profiler must not contaminate the caller's model (it profiles a deep copy)
    assert not any(k.endswith(("total_ops", "total_params")) for k in model.state_dict())

    # ... and if such buffers are present anyway (e.g. a hand-run thop.profile), they are removed
    import thop

    thop.profile(model, inputs=[torch.zeros(1, 3, 64, 64)], verbose=False)
    assert any(k.endswith("total_ops") for k in model.state_dict())
    assert strip_thop_artifacts(model) > 0
    assert not any(k.endswith(("total_ops", "total_params")) for k in model.state_dict())

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "s.pt")
        save_project_checkpoint(model, path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert not any(k.endswith(("total_ops", "total_params")) for k in payload["model"])
        rebuilt = build(variant="t", yolo_family="yolo11", yolo_scale="n")
        load_project_checkpoint(rebuilt, path, strict=True)


# --------------------------------------------------------------- native family contracts (10.2)
def test_native_criterion_selection_per_family():
    m11 = build(variant="t", yolo_family="yolo11", yolo_scale="n")
    m26 = build(variant="t", yolo_family="yolo26", yolo_scale="n")
    c11, c26 = m11.init_criterion(), m26.init_criterion()
    assert type(c11).__name__ == "v8DetectionLoss"
    assert type(c26).__name__ == "E2ELoss"
    # the native O2M/O2O assignment knobs (this checkout keeps them on the assigner)
    assert c26.one2many.assigner.topk == 10
    assert c26.one2one.assigner.topk == 7
    assert c26.one2one.assigner.topk2 == 1
    assert abs(c26.o2m - 0.8) < 1e-12 and abs(c26.o2o - 0.2) < 1e-12
    assert c26.total == 1.0 and c26.final_o2m == 0.1
    assert c26.o2m_copy == c26.o2m
    # yolo11 must NOT use the dual-branch criterion
    assert not hasattr(c11, "one2one")
    assert not hasattr(c11, "o2m") and not hasattr(c11, "o2o")
    assert c11.assigner.topk == 10
    c26.update()
    assert c26.updates == 1 and 0.0 <= c26.o2m <= 0.8


def test_yolo26_one_to_one_features_are_detached():
    model = build(variant="t", yolo_family="yolo26", yolo_scale="n")
    model.train()
    preds = model(torch.rand(1, 3, SMALL, SMALL))
    assert set(preds) == {"one2many", "one2one"}
    loss = sum(v.float().pow(2).mean() for v in preds["one2one"]["boxes"]) + sum(
        v.float().pow(2).mean() for v in preds["one2one"]["scores"]
    )
    loss.backward()
    assert not any(p.grad is not None for p in model.stem.backbone.parameters())
    model.zero_grad(set_to_none=True)
    loss = sum(v.float().pow(2).mean() for v in preds["one2many"]["boxes"]) + sum(
        v.float().pow(2).mean() for v in preds["one2many"]["scores"]
    )
    loss.backward()
    assert any(p.grad is not None for p in model.stem.backbone.parameters())


def test_end2end_property_is_preserved():
    model = build(variant="t", yolo_family="yolo26", yolo_scale="n")
    assert model.end2end is False  # declared True, but inference mode defaults to one2many
    model.end2end = True
    assert model.end2end is True
    model.end2end = False
    assert model.end2end is False
    model.set_head_attr(max_det=42)
    assert model.model[-1].max_det == 42


# ------------------------------------------------------------------------ variant independence
def test_three_selectors_are_independent():
    a = build(variant="xt", yolo_family="yolo11", yolo_scale="n")
    b = build(variant="xt", yolo_family="yolo26", yolo_scale="x")
    assert a.backbone_variant == "xt" and a.yolo_family == "yolo11" and a.yolo_scale == "n"
    assert b.backbone_variant == "xt" and b.yolo_family == "yolo26" and b.yolo_scale == "x"
    assert a.combination == "xt+yolo11n" and b.combination == "xt+yolo26x"
    # adapters follow the *yolo* scale, not a letter shared with the backbone
    assert a.stem.adapter_channels == (128, 128, 256)
    assert b.stem.adapter_channels == (768, 768, 768)


def test_adapter_channels_match_the_native_neck_entries():
    for variant, family, scale in (("t", "yolo11", "s"), ("b", "yolo26", "m"), ("xt", "yolo11", "l")):
        model = build(variant=variant, yolo_family=family, yolo_scale=scale)
        for i, entry in enumerate(model.adapter_map):
            conv = getattr(model.stem, f"adapter{i + 1}").conv
            assert int(conv.in_channels) == entry["backbone_channels"]
            assert int(conv.out_channels) == entry["adapter_channels"]
