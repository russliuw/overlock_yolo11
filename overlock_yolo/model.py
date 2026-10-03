"""OverLoCK (xt/t/s/b) + native YOLO11/YOLO26 (n/s/m/l/x) detector assembly.

Contract (DESIGN_V2.md 4, 7, 10.2)::

    RGB -> native letterbox to exactly 640x640 -> float/255            (native dataloader)
        -> ONE ImageNet mean/std normalisation (buffers on the stem)
        -> isolated official OverLoCK *detection* backbone (variant)
        -> x1/x2/x3 at stride 8/16/32
        -> three 1x1 Conv-BN-SiLU adapters -> the native neck entry widths
        -> the *native* neck+head taken from a real ``DetectionModel(dict)`` build of the
           family YAML with the requested scale (YOLO11 or YOLO26, C3k/repeats/reg_max and the
           one-to-one branch all come from that build)
        -> the family's own native criterion: ``v8DetectionLoss`` (11) / ``E2ELoss`` (26)

The detector is a ``BaseModel`` subclass, so ``forward(tensor)`` returns predictions,
``forward(dict)`` returns ``(loss, loss_items)`` like the native trainer expects, and
``model[-1]`` stays the real, correctly initialised ``Detect`` head.

Fixes carried over from the V1 review (DESIGN_V2.md 7):

* ``forward(dict)`` computes the loss (V1 returned predictions here).
* backends are resolved lazily per device, so ``.to("cuda")`` after a CPU build is honoured.
* the head comes from a real native build with ``bias_init``/stride, never from a hand-rolled
  ``parse_model`` call plus a post-hoc module swap.
* the stem/adapters have exactly ONE registration path (``self.model[tail_start-1]``); the
  ``stem``/``adapters`` attributes are properties, so ``state_dict()`` has no duplicates.
* no macOS path is imported at module import time; the Ultralytics root is installed by the CLI.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .attention_backend import BackendCache, backend_cache_report
from .backbone import OverLoCK, build_overlock
from .checkpoint import audit_and_load, verify_loaded_parameters
from .variants import (
    FEATURE_INFO,
    OVERLOCK_VARIANTS,
    P3P4P5_CHANNELS,
    adapter_mapping_report,
    expected_channels,
    expected_detect_entry_channels,
    expected_neck_entry_channels,
    normalize_family,
    normalize_scale,
    normalize_variant,
    yolo_yaml_relpath,
)

__all__ = [
    "DetectorImportError",
    "ultralytics_namespace",
    "load_yolo_yaml_dict",
    "build_native_tail",
    "OverLoCKStem",
    "OverLoCKYOLODetector",
    "build_detector",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "DETECTOR_ARG_KEYS",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Detector-specific override keys popped from ``args`` before the native trainer validation.
DETECTOR_ARG_KEYS = (
    "backbone_weights",
    "backbone_lr_mult",
    "freeze_backbone",
    "backbone_bn_eval",
    "backbone_variant",
    "yolo_family",
    "yolo_scale",
    "overlock_config",
    "attention_backend_override",
)


class DetectorImportError(RuntimeError):
    """Raised when the pinned Ultralytics source has not been installed on ``sys.path``."""


# --------------------------------------------------------------------------------------
# pinned Ultralytics access (lazy: the CLI installs the root first)
# --------------------------------------------------------------------------------------
def ultralytics_namespace() -> dict:
    """Return the native classes this module needs, or fail with an actionable message.

    Nothing here imports ``ultralytics`` at module import time: the CLI resolves the pinned
    source root and calls :func:`overlock_yolo.paths.install_ultralytics_root` first.  This keeps
    a hardcoded machine path out of the package while still guaranteeing that exactly one
    Ultralytics source is used.
    """
    try:
        import ultralytics  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        raise DetectorImportError(
            "ultralytics is not importable. Resolve the pinned source root and call "
            "overlock_yolo.paths.install_ultralytics_root() before building a detector."
        ) from exc
    from ultralytics.nn.modules.head import Detect
    from ultralytics.nn.tasks import BaseModel, DetectionModel, parse_model
    from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
    from ultralytics.utils.torch_utils import initialize_weights

    return {
        "ultralytics": ultralytics,
        "Detect": Detect,
        "BaseModel": BaseModel,
        "DetectionModel": DetectionModel,
        "parse_model": parse_model,
        "E2ELoss": E2ELoss,
        "v8DetectionLoss": v8DetectionLoss,
        "initialize_weights": initialize_weights,
    }


# --------------------------------------------------------------------------------------
# native family YAML -> dict (task/scale forced; the filename must not guess for us)
# --------------------------------------------------------------------------------------
def load_yolo_yaml_dict(family: str, scale: str, ultra_root: str) -> Tuple[dict, dict]:
    """Load the family YAML from the pinned source and set ``scale``/``task``/``nc`` explicitly.

    ``yaml_model_load`` derives the scale from the *filename* via ``guess_model_scale``, which is
    empty for ``yolo11.yaml``/``yolo26.yaml`` -- both V1 and a naive port silently fall back to
    ``n``.  The request is therefore applied after loading and then *verified* against the YAML's
    own ``scales`` table.
    """
    from ultralytics.utils import YAML as UltraYAML

    family = normalize_family(family)
    scale = normalize_scale(scale)
    relpath = yolo_yaml_relpath(family)
    yaml_path = os.path.join(ultra_root, *relpath)
    if not os.path.isfile(yaml_path):
        raise DetectorImportError(f"native YAML for {family} not found at {yaml_path}")

    data = UltraYAML.load(yaml_path)
    scales = data.get("scales") or {}
    if scale not in scales:
        raise DetectorImportError(f"{yaml_path} has no scale {scale!r}; available {sorted(scales)}")
    row = list(scales[scale])
    if len(row) != 3:
        raise DetectorImportError(f"{yaml_path} scale {scale!r} must be [depth, width, max_channels], got {row}")
    data["scale"] = scale
    data["task"] = "detect"
    if not data.get("nc"):
        data["nc"] = 80
    meta = {
        "family": family,
        "scale": scale,
        "yaml_path": yaml_path,
        "scales_row": [float(row[0]), float(row[1]), int(row[2])],
        "yaml_nc": int(data.get("nc", 80)),
        "end2end_declared": bool(data.get("end2end", False)),
        "reg_max_declared": data.get("reg_max"),
        "backbone_rows": len(data.get("backbone", [])),
        "head_rows": len(data.get("head", [])),
    }
    return data, meta


def build_native_tail(
    family: str,
    scale: str,
    nc: int,
    ultra_root: str,
    *,
    ch: int = 3,
    verbose: bool = False,
):
    """Build the native neck+head from a real ``DetectionModel(dict)``-equivalent construction.

    Returns ``(modules, save, yaml_dict, report)`` where ``modules`` is the pristine native module
    list for the whole graph (backbone rows included, so ``m.f`` indices stay native), and
    ``report`` records the split point, the entry layers produced by the backbone rows, the
    Detect head state (stride, reg_max, one-to-one branch) and the parameter count.

    Two V1 defects are fixed here:

    * the Detect head is initialised by the *native* path -- ``stride`` from a real head probe,
      ``bias_init()`` for both branches, and ``initialize_weights`` for the neck/head -- instead
      of being created by a bare ``parse_model`` call with no ``bias_init``;
    * the scale is taken from the requested value and cross-checked against the YAML table,
      instead of being read off the filename.
    """
    ns = ultralytics_namespace()
    parse_model = ns["parse_model"]
    Detect = ns["Detect"]
    initialize_weights = ns["initialize_weights"]

    yaml_dict, yaml_meta = load_yolo_yaml_dict(family, scale, ultra_root)
    yaml_dict = deepcopy(yaml_dict)
    yaml_dict["nc"] = int(nc)
    yaml_dict["channels"] = int(ch)
    modules, save = parse_model(deepcopy(yaml_dict), ch=int(ch), verbose=verbose)

    detect = modules[-1]
    if not isinstance(detect, Detect):
        raise AssertionError(f"last parsed module is {type(detect).__name__}, expected the native Detect")

    n_backbone = len(yaml_dict["backbone"])
    n_head = len(yaml_dict["head"])
    tail_start = n_backbone
    if n_backbone + n_head != len(modules):
        raise AssertionError(
            f"parsed module count {len(modules)} != backbone+head rows {n_backbone + n_head}; a module "
            "with n>1 would break the native index mapping"
        )
    entry_layers = _external_entry_layers(modules, tail_start)
    expected_entries = (4, 6, 10)
    if entry_layers != expected_entries:
        raise AssertionError(
            f"neck consumes backbone layers {entry_layers}, expected the native {expected_entries}"
        )

    # --- native stride derivation (exactly what DetectionModel.__init__ does, on the head only)
    detect.inplace = yaml_dict.get("inplace", True)
    probe = 256  # 2x the minimum stride, same probe size as the native builder
    input_strides = (8, 16, 32)
    with torch.no_grad():
        detect_stride = torch.tensor(
            [
                probe / branch(torch.zeros(1, _first_conv(branch).in_channels, probe // s, probe // s)).shape[-2]
                for branch, s in zip(detect.cv2, input_strides)
            ]
        )
    detect.stride = detect_stride
    detect.bias_init()  # native bias init for one2many (and one2one when present)
    initialize_weights(detect)  # native init of the freshly built head

    params = sum(p.numel() for p in modules.parameters()) - sum(
        p.numel() for i, m in enumerate(modules) if i < tail_start for p in m.parameters()
    )
    report = {
        "family": yaml_meta["family"],
        "scale": yaml_meta["scale"],
        "yaml": yaml_meta,
        "tail_start": tail_start,
        "entry_layers": list(entry_layers),
        "neck_entry_channels": list(expected_neck_entry_channels(family, scale)),
        "detect_entry_channels": list(expected_detect_entry_channels(family, scale)),
        "tail_parameters": int(params),
        "detect": {
            "class": type(detect).__name__,
            "nc": int(detect.nc),
            "reg_max": int(detect.reg_max),
            "stride": [float(s) for s in detect.stride.tolist()],
            "has_one2one": getattr(detect, "one2one_cv2", None) is not None,
            "end2end_flag": bool(getattr(detect, "end2end", False)),
            "nl": int(detect.nl),
        },
        "tail_modules": [type(m).__name__ for m in modules[tail_start:]],
        "c3k_flags": _c3k_flags(modules[tail_start:]),
    }
    return modules, save, yaml_dict, report


def _external_entry_layers(modules, tail_start: int) -> tuple:
    """Backbone layers consumed by the tail, in first-use order."""
    out: List[int] = []
    for m in modules[tail_start:]:
        for j in _depends_on(modules, int(m.i)):
            if j < tail_start and j not in out:
                out.append(j)
    return tuple(sorted(out))


def _refs(m) -> tuple:
    return tuple(int(i) for i in m.f) if isinstance(m.f, (list, tuple)) else (int(m.f),)


def _depends_on(modules, i: int) -> set:
    """Layers whose output layer ``i`` consumes (``f == -1`` means the previous layer)."""
    out = set()
    for j in _refs(modules[i]):
        out.add(i - 1 if j == -1 else j)
    return {j for j in out if 0 <= j < len(modules)}


def _c3k_flags(tail_modules) -> List[dict]:
    """``C3k2`` block kind per tail layer.

    ``C3k2.__init__`` does not store the ``c3k`` flag, so it is read back from the built module:
    a ``c3k=True`` C3k2 fills ``m.m`` with ``C3k`` blocks (and ``attn=True`` with ``PSABlock``),
    otherwise with ``Bottleneck``.  m/l/x force ``c3k=True`` on the YOLO11 side while YOLO26's
    neck sets it per layer, so this is a real structural discriminator between the two families.
    """
    flags = []
    for m in tail_modules:
        if type(m).__name__ != "C3k2":
            continue
        kinds = set()
        for block in m.m:
            kinds.add(type(block).__name__)
            if isinstance(block, nn.Sequential):
                kinds.update(type(child).__name__ for child in block)
        kinds = sorted(kinds)
        leaf = None
        for attr in ("cv2", "cv3"):
            found = _leaf_conv(getattr(m, attr, None), first=False)
            if found is not None:
                leaf = int(found.out_channels)
                break
        flags.append(
            {
                "i": int(m.i),
                "block_kinds": kinds,
                "c3k": "C3k" in kinds,
                "attn": "PSABlock" in kinds,
                "c2": leaf,
            }
        )
    return flags


# --------------------------------------------------------------------------------------
# stem: normalisation + OverLoCK + three 1x1 adapters
# --------------------------------------------------------------------------------------
class OverLoCKStem(nn.Module):
    """Single ImageNet normalisation + official OverLoCK detection backbone + 1x1 adapters.

    Registered exactly once inside ``OverLoCKYOLODetector.model[tail_start - 1]``.  Native
    bookkeeping attributes (``i``/``f``) are set so the module could also be driven by
    ``BaseModel._predict_once``.
    """

    def __init__(
        self,
        variant: str = "t",
        conv_cls=None,
        adapter_channels=(256, 256, 512),
        attention_backend: str = "auto",
        row_chunk: int = 0,
        strict_shape: bool = True,
    ):
        super().__init__()
        variant = normalize_variant(variant)
        self.variant = variant
        self.backbone = build_overlock(variant, attention_backend=attention_backend, row_chunk=row_chunk)
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=True)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=True)
        self.strict_shape = bool(strict_shape)

        if conv_cls is None:
            from ultralytics.nn.modules.conv import Conv

            conv_cls = Conv
        src = expected_channels(variant)
        if len(adapter_channels) != 3:
            raise ValueError(f"adapter_channels must have 3 entries, got {tuple(adapter_channels)}")
        self.adapter1 = conv_cls(int(src[0]), int(adapter_channels[0]), 1, 1)
        self.adapter2 = conv_cls(int(src[1]), int(adapter_channels[1]), 1, 1)
        self.adapter3 = conv_cls(int(src[2]), int(adapter_channels[2]), 1, 1)
        self.adapter_channels = tuple(int(c) for c in adapter_channels)
        self.out_channels = tuple(expected_channels(variant))
        self.out_strides = (8, 16, 32)

        self.i = None
        self.f = -1
        self.type = type(self).__name__

    # ------------------------------------------------------------------ normalisation
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """The single ImageNet mean/std normalisation applied to the [0,1] letterboxed input."""
        return (x - self.pixel_mean) / self.pixel_std

    # ------------------------------------------------------------------ forward
    def backbone_features(self, x: torch.Tensor, already_normalized: bool = False) -> Tuple[torch.Tensor, ...]:
        """Raw OverLoCK x0/x1/x2/x3 (no adapters)."""
        x = x.to(dtype=self.pixel_mean.dtype, device=self.pixel_mean.device)
        if not already_normalized:
            x = self.normalize(x)
        return self.backbone.forward_multiscale(x, strict=self.strict_shape)

    def adapt(self, feats) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the three 1x1 adapters to raw backbone levels (x0 is computed but unused)."""
        _x0, x1, x2, x3 = feats
        return self.adapter1(x1), self.adapter2(x2), self.adapter3(x3)

    def forward(self, x: torch.Tensor):
        """Return ``(a0, a1, a2, x0)`` -- the three adapted levels plus the untouched x0."""
        feats = self.backbone_features(x)
        a1, a2, a3 = self.adapt(feats)
        return a1, a2, a3, feats[0]

    def attention_backend_report(self) -> dict:
        return self.backbone.attention_backend_report()


# --------------------------------------------------------------------------------------
# detector
# --------------------------------------------------------------------------------------
class OverLoCKYOLODetector:
    """Base class marker: the concrete class is created lazily against the pinned source.

    Creating the real class requires ``BaseModel`` from the pinned Ultralytics source, which is
    only importable after the CLI has installed that root.  :func:`build_detector` returns an
    instance of the concrete class; ``isinstance(model, nn.Module)`` and ``BaseModel`` both hold.
    """


def _detector_class():
    """Build (once) the ``BaseModel`` subclass bound to the currently installed Ultralytics."""
    ns = ultralytics_namespace()
    BaseModel = ns["BaseModel"]

    class _OverLoCKYOLODetector(BaseModel):
        """Trainable detector: OverLoCK variant + native YOLO11/YOLO26 neck/head."""

        #: filled in by ``__init__``; declared here so the class is introspectable
        tail_start: int

        def __init__(
            self,
            variant: str = "t",
            family: str = "yolo11",
            scale: str = "s",
            nc: int = 6,
            names: Optional[dict] = None,
            ultra_root: Optional[str] = None,
            yaml_dict: Optional[dict] = None,
            weights: Optional[str] = None,
            pretrained: bool = False,
            attention_backend: str = "auto",
            seed: int = 0,
            row_chunk: int = 0,
            freeze: bool = False,
            lr_mult: float = 1.0,
            bn_eval: bool = False,
            strict_shape: bool = True,
            strict_weights: bool = True,
            verbose: bool = False,
        ):
            super().__init__()
            from .paths import resolve_ultralytics_root

            variant = normalize_variant(variant)
            family = normalize_family(family)
            scale = normalize_scale(scale)
            self.backbone_variant = variant
            self.yolo_family = family
            self.yolo_scale = scale
            self.combination = f"{variant}+{family}{scale}"
            self.attention_backend_requested = attention_backend
            self.backbone_lr_mult = float(lr_mult)
            self.freeze_backbone = bool(freeze)
            self._backbone_bn_eval = bool(bn_eval)
            self.seed = int(seed)
            self.pretrained_backbone_report = None
            self.backbone_weights_path = weights
            self.pretrained_requested = bool(pretrained)
            self.yolo_init_source = "random (no YOLO neck/head state_dict provided)"
            self.ultra_root = ultra_root or resolve_ultralytics_root()

            # 1) native neck+head from the family YAML with the requested scale
            modules, save, native_yaml, tail_report = build_native_tail(
                family, scale, nc, self.ultra_root, verbose=verbose
            )
            self.tail_start = int(tail_report["tail_start"])
            self.native_tail_report = tail_report
            self.yaml = native_yaml
            self.native_entry_layers = tuple(tail_report["entry_layers"])

            detect = modules[-1]
            expected_neck = tuple(tail_report["neck_entry_channels"])
            produced = tuple(int(_last_conv(modules[layer]).out_channels) for layer in self.native_entry_layers)
            if produced != expected_neck:
                raise AssertionError(
                    f"native neck entry layers {self.native_entry_layers} produce {produced}, expected "
                    f"{expected_neck} for {family}{scale}"
                )
            self.adapter_map = adapter_mapping_report(variant, family, scale)
            self.entry_map = {int(layer): k for k, layer in enumerate(self.native_entry_layers)}

            # 2) stem (normalisation + OverLoCK + adapters) built with torch's deterministic init
            torch.manual_seed(self.seed)
            stem = OverLoCKStem(
                variant=variant,
                conv_cls=type(modules[0]),
                adapter_channels=expected_neck,
                attention_backend=attention_backend,
                row_chunk=row_chunk,
                strict_shape=strict_shape,
            )
            stem.i = self.tail_start - 1

            # 3) assemble: self.model keeps native indices; the stem owns backbone+adapters once.
            # ``stem`` is a LOCAL name, never stored as an attribute: ``self.stem`` is a property
            # that returns ``self.model[tail_start - 1]``, so there is exactly one registration
            # path and ``state_dict()`` has no aliases (V1 registered it under both
            # ``stem_module.*`` and ``model.<i>.*``).
            model = nn.Sequential()
            for i, m in enumerate(modules):
                if i < self.tail_start:
                    model.add_module(str(i), stem if i == self.tail_start - 1 else HeadPlaceholder(i))
                else:
                    model.add_module(str(i), m)
            self.model = model
            self.save = save
            self.nc = int(nc)
            self.names = dict(names) if names is not None else {i: f"{i}" for i in range(nc)}
            self.yaml = dict(native_yaml)
            self.yaml["names"] = dict(self.names)
            self.inplace = native_yaml.get("inplace", True)
            self.stride = detect.stride.clone()
            self._end2end = bool(getattr(detect, "end2end", False))

            #: training hyper-parameters (the native trainer overwrites this through
            #: ``set_model_attributes``); a default is installed so ``init_criterion`` and
            #: ``loss`` work standalone, exactly like a natively built DetectionModel.
            self.args = self._default_args()

            # 4) optional pretrained backbone weights (audited, safe, never re-initialised after)
            if weights is not None:
                self.load_backbone_weights(weights, strict=strict_weights)

            self.apply_backbone_flags()

        @staticmethod
        def _default_args():
            """Default native training args (overwritten by the trainer when training)."""
            from ultralytics.cfg import get_cfg

            return get_cfg(overrides={"task": "detect", "mode": "train"})

        # -------------------------------------------------------------- single registration
        @property
        def stem(self) -> OverLoCKStem:
            """The one and only stem object (registered at ``model[tail_start - 1]``)."""
            return self.model[self.tail_start - 1]

        @property
        def adapters(self) -> nn.ModuleList:
            """View over the three adapters.  Not a second registration path: ``ModuleList`` is
            transient and is never assigned as an attribute, so ``state_dict()`` has no aliases."""
            return nn.ModuleList([self.stem.adapter1, self.stem.adapter2, self.stem.adapter3])

        @property
        def backbone(self) -> OverLoCK:
            return self.stem.backbone

        @property
        def backbone_parameters(self):
            return list(self.stem.backbone.parameters())

        @property
        def backbone_named_parameters(self):
            prefix = f"model.{self.tail_start - 1}.backbone."
            return [(prefix + k, v) for k, v in self.stem.backbone.named_parameters()]

        @property
        def tail(self) -> nn.Module:
            """The native neck+head slice (``model[tail_start:]``)."""
            return self.model[self.tail_start :]

        # -------------------------------------------------------------- native surface
        @property
        def end2end(self) -> bool:
            """Whether inference uses the one-to-one (NMS-free) path."""
            return bool(getattr(self.model[-1], "end2end", False))

        @end2end.setter
        def end2end(self, value) -> None:
            self.model[-1].end2end = bool(value)

        def set_head_attr(self, **kwargs) -> None:
            """Forward head attributes (native ``DetectionModel.set_head_attr``)."""
            head = self.model[-1]
            for k, v in kwargs.items():
                if not hasattr(head, k):
                    raise AttributeError(f"Detect head has no attribute {k!r}")
                setattr(head, k, v)

        def init_criterion(self):
            """The *family's own* native criterion (DESIGN_V2.md 10.2).

            YOLO26 builds a Detect head with a one-to-one branch, so the native
            ``DetectionModel.init_criterion`` selects ``E2ELoss`` -- with its own O2M topk10 /
            O2O topk7+topk2=1 assignment, its branch-weight schedule and its L1 regression term.
            YOLO11 selects ``v8DetectionLoss``.  Nothing is re-implemented here.
            """
            ns = ultralytics_namespace()
            if getattr(self.model[-1], "one2one_cv2", None) is not None:
                return ns["E2ELoss"](self)
            return ns["v8DetectionLoss"](self)

        def loss(self, batch: dict, preds=None):
            """Native ``BaseModel.loss`` contract: returns ``(loss_vector, loss_items)``."""
            if getattr(self, "criterion", None) is None:
                self.criterion = self.init_criterion()
            if preds is None:
                preds = self.forward(batch["img"])
            return self.criterion(preds, batch)

        def forward(self, x, *args, **kwargs):
            """``dict -> (loss, loss_items)`` (training); ``tensor -> predictions`` (inference)."""
            if isinstance(x, dict):
                return self.loss(x, *args, **kwargs)
            return self.predict(x, *args, **kwargs)

        # -------------------------------------------------------------- forward body
        def _adapt(self, x: torch.Tensor) -> List[torch.Tensor]:
            """Normalise -> OverLoCK -> adapters; returns the three neck entry features."""
            return list(self.stem.adapt(self.stem.backbone_features(x)))

        def _run_tail(self, feats: List[torch.Tensor], profile: bool = False):
            """Run only the native tail (``tail_start..end``) from the adapted entry features."""
            if len(feats) != 3:
                raise AssertionError(f"expected 3 adapted features, got {len(feats)}")
            x = feats[-1]
            y = {layer: feats[k] for layer, k in self.entry_map.items()}
            for m in self.model[self.tail_start :]:
                i = int(m.i)
                if m.f != -1:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                x = m(x)
                y[i] = x
            return x

        def _predict_once(self, x, profile: bool = False, embed=None):
            if embed not in (None, frozenset({-1}), {-1}):
                raise NotImplementedError("intermediate embedding extraction is not supported")
            return self._run_tail(self._adapt(x), profile=profile)

        def _predict_augment(self, x):  # pragma: no cover - out of scope
            raise NotImplementedError("augmented (TTA) inference is not implemented")

        # -------------------------------------------------------------- flags / reports
        def set_backbone_bn_eval(self, enabled: bool) -> None:
            self._backbone_bn_eval = bool(enabled)
            self.apply_backbone_flags()

        def apply_backbone_flags(self) -> None:
            for p in self.stem.backbone.parameters():
                p.requires_grad = not self.freeze_backbone
            if self.freeze_backbone:
                self.stem.backbone.eval()
            elif self._backbone_bn_eval:
                for m in self.stem.backbone.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        m.eval()

        def train(self, mode: bool = True):
            super().train(mode)
            self.apply_backbone_flags()
            return self

        def load_backbone_weights(self, path: str, strict: bool = True, container_key: Optional[str] = None) -> dict:
            """Audited ``weights_only=True`` load of the OverLoCK backbone for this variant."""
            report = audit_and_load(self.stem.backbone, path, container_key=container_key)
            report["variant"] = self.backbone_variant
            report["expected_variant_file"] = f"overlock_{self.backbone_variant}_in1k_224.pth"
            report["representative_param_check"] = verify_loaded_parameters(
                self.stem.backbone,
                path,
                keys=_representative_keys(self.backbone_variant),
            )
            report["attention_backend"] = self.attention_backend_report()
            self.pretrained_backbone_report = report
            self.yaml["backbone_variant"] = self.backbone_variant
            return report

        def attention_backend_report(self) -> dict:
            return self.stem.attention_backend_report()

        def adapter_forward_report(self) -> List[dict]:
            return list(self.adapter_map)

        def parameter_groups_report(self) -> dict:
            """Per-bucket unique-parameter accounting (never double counts an alias)."""
            buckets = {
                "backbone_overlock": self.stem.backbone,
                "adapters": nn.ModuleList([self.stem.adapter1, self.stem.adapter2, self.stem.adapter3]),
                "neck_head_native": self.tail,
            }
            seen: Dict[int, str] = {}
            duplicate = []
            out = {"buckets": {}, "total": {}}
            for name, mod in buckets.items():
                tensors = list(mod.parameters())
                numel = int(sum(v.numel() for v in tensors))
                for v in tensors:
                    if id(v) in seen:
                        duplicate.append([seen[id(v)], name])
                    seen[id(v)] = name
                out["buckets"][name] = {
                    "tensors": len(tensors),
                    "numel": numel,
                    "trainable_numel": int(sum(v.numel() for v in tensors if v.requires_grad)),
                }
            all_params = list(self.parameters())
            all_named = dict(self.named_parameters())
            out["total"] = {
                "numel_unique": int(sum(v.numel() for v in all_params)),
                "numel_sum_of_buckets": int(sum(b["numel"] for b in out["buckets"].values())),
                "tensors_unique": len(all_params),
                "tensors_named": len(all_named),
                "trainable_numel": int(sum(v.numel() for v in all_params if v.requires_grad)),
                "buffers_numel": int(sum(b.numel() for b in self.buffers())),
                "duplicate_registration": duplicate,
                "buckets_cover_all_parameters": sum(b["numel"] for b in out["buckets"].values())
                == int(sum(v.numel() for v in all_params)),
            }
            head = self.model[-1]
            out["head"] = {
                "class": type(head).__name__,
                "nc": int(head.nc),
                "reg_max": int(head.reg_max),
                "stride": [float(s) for s in head.stride.tolist()],
                "has_one2one": getattr(head, "one2one_cv2", None) is not None,
                "end2end": bool(self.end2end),
            }
            out["combination"] = self.combination
            return out

        def state_metadata(self) -> dict:
            """Everything needed to rebuild this exact structure (used by checkpoint save/load)."""
            return {
                "format": "overlock-yolo-state-v1",
                "backbone_variant": self.backbone_variant,
                "yolo_family": self.yolo_family,
                "yolo_scale": self.yolo_scale,
                "nc": int(self.nc),
                "names": dict(self.names),
                "stride": [float(s) for s in self.stride.tolist()],
            }

    _OverLoCKYOLODetector.__name__ = "OverLoCKYOLODetector"
    _OverLoCKYOLODetector.__qualname__ = "OverLoCKYOLODetector"
    _OverLoCKYOLODetector.__module__ = __name__
    return _OverLoCKYOLODetector


_DETECTOR_CLASS_CACHE: Dict[str, Any] = {}


def detector_class():
    """Cached concrete detector class for the currently installed Ultralytics source."""
    import ultralytics

    key = os.path.abspath(ultralytics.__file__)
    cls = _DETECTOR_CLASS_CACHE.get(key)
    if cls is None:
        cls = _detector_class()
        _DETECTOR_CLASS_CACHE[key] = cls
    return cls


class HeadPlaceholder(nn.Module):
    """Inert stand-in keeping the native layer-index metadata of the replaced backbone slots.

    The detector executes the stem explicitly and then ``model[tail_start:]``, so slots
    ``0..tail_start-1`` never run.  They exist so ``self.model`` keeps native indices,
    ``self.model[-1]`` stays the real Detect, and ``state_dict`` keys line up with the native
    layout.  Defined at module level (not inside ``_detector_class``) so pickling/deepcopy work.
    """

    def __init__(self, index: int):
        super().__init__()
        self.i = index
        self.f = -1

    def forward(self, x):
        return x


def _first_conv(module) -> nn.Conv2d:
    """The ``nn.Conv2d`` carrying the input width of a native YOLO layer."""
    leaf = _leaf_conv(module, first=True)
    if leaf is not None:
        return leaf
    for attr in ("cv1", "cv2", "cv3", "conv"):
        leaf = _leaf_conv(getattr(module, attr, None), first=True)
        if leaf is not None:
            return leaf
    raise AttributeError(f"cannot locate an input conv on {type(module).__name__}")


def _leaf_conv(m: Optional[nn.Module], first: bool = True) -> Optional[nn.Conv2d]:
    for _ in range(8):
        if m is None:
            return None
        if isinstance(m, nn.Conv2d):
            return m
        if isinstance(m, (nn.Sequential, nn.ModuleList)):
            if not len(m):
                return None
            m = m[0] if first else m[-1]
            continue
        m = getattr(m, "conv", None)
    return None


def _last_conv(module) -> nn.Conv2d:
    """The ``nn.Conv2d`` carrying the output width of a native YOLO layer (``cv2`` for CSP)."""
    for attr in ("cv2", "cv3", "cv1", "conv"):
        leaf = _leaf_conv(getattr(module, attr, None), first=False)
        if leaf is not None:
            return leaf
    leaf = _leaf_conv(module, first=False)
    if leaf is not None:
        return leaf
    raise AttributeError(f"cannot locate an output conv on {type(module).__name__}")


def _representative_keys(variant: str) -> Tuple[str, ...]:
    """Shared keys present in every OverLoCK variant checkpoint (verified per variant)."""
    del variant
    return (
        "patch_embed1.0.weight",
        "blocks3.0.dwconv.weight",
        "blocks3.0.proj.1.lk_origin.weight",
        "blocks4.3.proj.1.dil_conv_k3_3.weight",
        "sub_blocks3.0.weight_proj.weight",
        "sub_blocks3.0.rpb1",
        "sub_blocks4.3.proj.2.weight",
        "patch_embedx.h_proj.0.weight",
        "h_proj.0.weight",
        "high_level_proj.weight",
        "extra_norm.4.weight",
    )


# --------------------------------------------------------------------------------------
# convenience factory
# --------------------------------------------------------------------------------------
def build_detector(
    variant: str = "t",
    family: str = "yolo11",
    scale: str = "s",
    nc: int = 6,
    names: Optional[dict] = None,
    weights: Optional[str] = None,
    pretrained: bool = False,
    attention_backend: str = "auto",
    ultra_root: Optional[str] = None,
    seed: int = 0,
    row_chunk: int = 0,
    freeze: bool = False,
    lr_mult: float = 1.0,
    bn_eval: bool = False,
    strict_shape: bool = True,
    verbose: bool = False,
):
    """Build the hybrid detector for one ``variant x family x scale`` combination.

    ``pretrained=True`` requires ``weights`` to resolve to a readable file; nothing is ever
    downloaded.  Without weights the backbone is random, and the caller is responsible for
    reporting that honestly.
    """
    if pretrained and not (weights and os.path.isfile(weights)):
        raise FileNotFoundError(
            f"pretrained=True but no checkpoint file at {weights!r} for variant {variant!r}; "
            "this project never downloads weights"
        )
    cls = detector_class()
    model = cls(
        variant=variant,
        family=family,
        scale=scale,
        nc=nc,
        names=names,
        ultra_root=ultra_root,
        weights=weights,
        pretrained=pretrained,
        attention_backend=attention_backend,
        seed=seed,
        row_chunk=row_chunk,
        freeze=freeze,
        lr_mult=lr_mult,
        bn_eval=bn_eval,
        strict_shape=strict_shape,
        verbose=verbose,
    )
    return model


def combination_report(variant: str, family: str, scale: str) -> dict:
    """Static (no-build) report of one combination: channels, routing, native YAML metadata."""
    variant = normalize_variant(variant)
    family = normalize_family(family)
    scale = normalize_scale(scale)
    return {
        "combination": f"{variant}+{family}{scale}",
        "backbone_variant": variant,
        "yolo_family": family,
        "yolo_scale": scale,
        "backbone_feature_info": [list(x) for x in FEATURE_INFO[variant]],
        "backbone_p3p4p5": list(P3P4P5_CHANNELS[variant]),
        "adapter_map": adapter_mapping_report(variant, family, scale),
        "overlock_factory": {k: (list(v) if isinstance(v, list) else v) for k, v in OVERLOCK_VARIANTS[variant].items()},
    }
