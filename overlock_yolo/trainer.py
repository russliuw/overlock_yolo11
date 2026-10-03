"""Native-trainer integration for the OverLoCK + YOLO detector.

Contract (DESIGN_V2.md 5, 7.7, 10.2)
------------------------------------
* ``OverLoCKDetectionTrainer`` is a real ``DetectionTrainer`` subclass; ``get_model()`` returns
  the hybrid detector for the resolved ``variant x family x scale`` combination.
* The criterion is the family's own native one: YOLO11 -> ``v8DetectionLoss``, YOLO26 ->
  ``E2ELoss`` (O2M topk10 / O2O topk7+topk2=1, its own branch-weight schedule and L1 term).
  The native trainer's lifecycle is preserved: ``criterion.update()`` at every completed epoch,
  the criterion dropped from the EMA/checkpoint payload, and the resume path restoring the
  update counter.
* ``rect``/``multi_scale`` are forced off so every batch is exactly ``imgsz x imgsz``.
* Saving writes this project's own checkpoint format (plain tensor ``state_dict`` + the
  structure metadata needed to rebuild).  Loading such a checkpoint for a *new* run's backbone
  initialisation and resuming a run are both supported; a stock Ultralytics checkpoint is
  explicitly rejected instead of being half-understood.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

from .config import ConfigError
from .model import build_detector, detector_class, ultralytics_namespace

__all__ = [
    "OverLoCKDetectionTrainer",
    "build_trainer",
    "train_main",
    "move_optimizer_to",
    "save_project_checkpoint",
    "load_project_checkpoint",
    "CHECKPOINT_FORMAT",
]

#: Marker stored in our own checkpoints; anything else is rejected by the resume path.
CHECKPOINT_FORMAT = "overlock-yolo-state-v1"

#: THOP's ``profile()`` attaches these two *buffers* to every module it walks.  They are
#: profiling artefacts, not model state: they must never enter ``state_dict()`` (which would
#: break ``load_state_dict``) and must be dropped from a checkpoint that accidentally has them.
_THOP_ARTIFACT_SUFFIXES = (".total_ops", ".total_params")


def strip_thop_artifacts(model) -> int:
    """Remove THOP's ``total_ops``/``total_params`` buffers (see profile_model)."""
    from .profile_model import strip_thop_artifacts as _strip

    return _strip(model)


def clean_state_dict(sd: dict) -> dict:
    """Drop THOP profiling artefacts from a ``state_dict``-shaped mapping."""
    return {k: v for k, v in sd.items() if not k.endswith(_THOP_ARTIFACT_SUFFIXES)}


def _trainer_base():
    from ultralytics.models.yolo.detect import DetectionTrainer

    return DetectionTrainer


def _trainer_class():
    DetectionTrainer = _trainer_base()

    class OverLoCKDetectionTrainer(DetectionTrainer):
        """``DetectionTrainer`` whose ``get_model()`` returns the hybrid detector.

        ``detector_config`` is the fully resolved experiment configuration
        (:func:`overlock_yolo.config.load_experiment`); the trainer never re-reads YAML itself.
        """

        def __init__(self, cfg=None, overrides: Optional[dict] = None, _callbacks=None, detector_config: Optional[dict] = None):
            from ultralytics.cfg import DEFAULT_CFG

            overrides = dict(overrides or {})
            # Detector-specific keys must be popped before BaseTrainer validates the arg namespace.
            self.detector_overrides = {k: overrides.pop(k) for k in list(overrides) if k in _DETECTOR_KEYS}
            self.detector_config = detector_config
            super().__init__(cfg=cfg or DEFAULT_CFG, overrides=overrides, _callbacks=_callbacks)
            self.logged_backbone_report = None
            self.protocol_records = []

        # ------------------------------------------------------------------ model
        def get_model(self, cfg=None, weights=None, verbose: bool = True):
            """Return the hybrid detector for the resolved combination."""
            conf = self.detector_config
            res = conf["_resolved"]
            backbone_weights = res["backbone_weights"]
            resume_state = self.detector_overrides.get("resume_state")
            model = build_detector(
                variant=res["backbone_variant"],
                family=res["yolo_family"],
                scale=res["yolo_scale"],
                nc=int(self.data["nc"]) if self.data.get("nc") else int(res["nc"]),
                names=self.data.get("names") or res["names"],
                weights=None if resume_state is not None else backbone_weights,
                pretrained=False,
                attention_backend=conf["backbone"]["attention_backend"],
                seed=int(getattr(self.args, "seed", 0)),
                row_chunk=int(conf["backbone"]["row_chunk"]),
                freeze=bool(conf["backbone"]["freeze"]),
                lr_mult=float(conf["backbone"]["lr_mult"]),
                bn_eval=bool(conf["backbone"]["bn_eval"]),
                verbose=False,
            )
            if resume_state is not None:
                model.load_state_dict(resume_state, strict=True)
                print(f"[overlock] resumed model weights from {self.detector_overrides.get('resume_path')}")
            elif backbone_weights:
                self.logged_backbone_report = model.pretrained_backbone_report
                totals = model.pretrained_backbone_report["totals"]
                print(
                    f"[overlock] backbone {res['backbone_variant']} weights {backbone_weights}: "
                    f"numel coverage {totals['numel_coverage']:.6f}, matched "
                    f"{totals['matched_tensors']}/{totals['target_tensors']} tensors, non-whitelisted "
                    f"missing {totals['trainable_missing_disallowed_tensors']}"
                )
            else:
                print(
                    "[overlock] WARNING: no backbone weights -> the OverLoCK backbone is RANDOM "
                    "(this run cannot be reported as using pretrained weights)"
                )
            print("[overlock] NOTE: the native YOLO neck/head is RANDOMLY initialised (no detection pretraining)")
            return model

        # ------------------------------------------------------------------ protocol
        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            img = batch["img"]
            h, w = int(img.shape[-2]), int(img.shape[-1])
            if h != w or h != int(self.args.imgsz):
                raise AssertionError(
                    f"training batch is {h}x{w}; the protocol requires exactly "
                    f"{int(self.args.imgsz)}x{int(self.args.imgsz)} (rect/multi_scale must stay off)"
                )
            self.protocol_records.append({"shape": [int(v) for v in img.shape], "dtype": str(img.dtype)})
            return batch

        # ------------------------------------------------------------------ optimizer
        def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
            optimizer = super().build_optimizer(
                model, name=name, lr=lr, momentum=momentum, decay=decay, iterations=iterations
            )
            mult = float(self.detector_config["backbone"]["lr_mult"])
            if mult != 1.0:
                backbone_ids = {id(p) for p in model.backbone_parameters}
                for group in optimizer.param_groups:
                    params = list(group["params"])
                    n_backbone = sum(1 for p in params if id(p) in backbone_ids)
                    if n_backbone == 0:
                        continue
                    if n_backbone == len(params):
                        group["lr"] = group["lr"] * mult
                        group["backbone_lr_mult"] = mult
                    else:
                        raise RuntimeError(
                            "backbone lr_mult requires the backbone parameters to be separable inside a "
                            "native parameter group; refusing to silently rescale a mixed group"
                        )
                print(f"[overlock] backbone lr_mult={mult} applied to backbone-only parameter groups")
            self._optimizer_group_audit(model, optimizer)
            return optimizer

        @staticmethod
        def _optimizer_group_audit(model, optimizer) -> dict:
            """Every trainable parameter in exactly one group; the native frozen DFL is the only extra."""
            named = dict(model.named_parameters())
            assigned = []
            for group in optimizer.param_groups:
                assigned.extend(id(p) for p in group["params"])
            trainable_ids = {id(p) for p in named.values() if p.requires_grad}
            extra = sorted(n for n, p in named.items() if id(p) in set(assigned) - trainable_ids)
            native_frozen = {n for n, p in named.items() if not p.requires_grad}
            audit = {
                "n_groups": len(optimizer.param_groups),
                "n_assigned": len(assigned),
                "n_unique_assigned": len(set(assigned)),
                "n_trainable": len(trainable_ids),
                "missing": sorted(n for n, p in named.items() if p.requires_grad and id(p) not in set(assigned)),
                "duplicated": len(assigned) != len(set(assigned)),
                "extra_frozen_params": extra,
                "frozen_params": sorted(native_frozen),
            }
            if audit["missing"] or audit["duplicated"]:
                raise RuntimeError(f"optimizer parameter-group partition is not exact: {audit}")
            print(
                f"[overlock] optimizer groups={audit['n_groups']} params={audit['n_unique_assigned']} "
                f"(all {audit['n_trainable']} trainable parameters assigned exactly once; frozen "
                f"native: {audit['frozen_params']})"
            )
            return audit

        # ------------------------------------------------------------------ save / resume
        def _checkpoint_payload(self) -> dict:
            """This project's checkpoint payload: plain tensors + structure metadata.

            The criterion is stored **separately** from ``model`` (it lives in the model's
            ``__dict__``, so ``model.state_dict()`` does not contain it) -- the native flow drops
            the criterion before saving and restores its ``updates`` counter on resume.
            """
            criterion = getattr(self.model, "criterion", None)
            return {
                "format": CHECKPOINT_FORMAT,
                "structure": self.model.state_metadata(),
                "model": clean_state_dict({k: v.detach().cpu() for k, v in self.model.state_dict().items()}),
                "optimizer": self.optimizer.state_dict() if getattr(self, "optimizer", None) else None,
                "epoch": int(getattr(self, "epoch", -1)),
                "best_fitness": getattr(self, "best_fitness", None),
                "criterion_class": None if criterion is None else type(criterion).__name__,
                "criterion_updates": int(getattr(criterion, "updates", 0) or 0),
                "criterion_branch_weights": None
                if criterion is None or not hasattr(criterion, "o2m")
                else {"o2m": float(criterion.o2m), "o2o": float(criterion.o2o)},
                "train_args": {
                    k: v
                    for k, v in vars(self.args).items()
                    if isinstance(v, (int, float, str, bool, type(None)))
                },
                "config": {k: v for k, v in self.detector_config.items() if not k.startswith("_")},
                "resolved": self.detector_config.get("_resolved", {}),
                "config_path": self.detector_config.get("_meta", {}).get("config_path"),
            }

        def save_model(self):
            """Write ``last.pt`` (and ``best.pt`` when fitness improved) in this project's format."""
            from pathlib import Path

            from ultralytics.utils.torch_utils import de_parallel

            epoch = int(getattr(self, "epoch", -1)) + 1
            payload = self._checkpoint_payload()
            w = Path(self.wdir) if getattr(self, "wdir", None) else Path(self.save_dir) / "weights"
            w.mkdir(parents=True, exist_ok=True)
            last = w / "last.pt"
            torch.save(payload, last)  # default protocol: weights_only-loadable everywhere
            self.last, self.wdir = str(last), str(w)
            self.ckpt = payload
            fitness = getattr(self, "fitness", None)
            if fitness is None or (self.best_fitness is not None and fitness > self.best_fitness):
                self.best_fitness = fitness
                torch.save(payload, w / "best.pt")
                print(f"[overlock] saved {w / 'best.pt'} (epoch {epoch})")
            print(f"[overlock] saved {last} (epoch {epoch}, {len(payload['model'])} tensors)")
            del de_parallel  # imported for parity with the native save path only
            return str(last)

        def check_resume(self, overrides=None):
            """Only this project's own checkpoints may be resumed."""
            if not getattr(self.args, "resume", False):
                return
            from ultralytics.utils.patches import torch_load

            path = str(self.args.resume)
            if not os.path.isfile(path):
                raise ConfigError(f"--resume checkpoint not found: {path}")
            state = torch_load(path, map_location="cpu")
            if not (isinstance(state, dict) and state.get("format") == CHECKPOINT_FORMAT):
                raise ConfigError(
                    f"--resume {path} is not an {CHECKPOINT_FORMAT} checkpoint written by this project. "
                    "Resuming a stock Ultralytics checkpoint is not supported here; start a fresh run "
                    "or load its tensors explicitly with --init-checkpoint."
                )
            self.detector_overrides["resume_state"] = state["model"]
            self.detector_overrides["resume_path"] = path
            self.resume_checkpoint = state
            self.last = path
            self.start_epoch = int(state.get("epoch", -1)) + 1
            if self.start_epoch >= int(self.args.epochs):
                raise ConfigError(
                    f"the checkpoint is already at epoch {self.start_epoch} but --epochs is "
                    f"{int(self.args.epochs)}; raise --epochs to continue training"
                )

        def resume_training(self, ckpt=None):
            """Restore optimizer + criterion progress on top of the already loaded model weights."""
            state = getattr(self, "resume_checkpoint", None)
            if state is None:
                raise ConfigError("resume_training called without a validated project checkpoint")
            if state.get("optimizer") and getattr(self, "optimizer", None) is not None:
                self.optimizer.load_state_dict(state["optimizer"])
                move_optimizer_to(self.optimizer, self.device)
            criterion = getattr(self.model, "criterion", None)
            if criterion is None:
                criterion = self.model.init_criterion()
                self.model.criterion = criterion
            if hasattr(criterion, "updates"):
                criterion.updates = int(state.get("criterion_updates", 0))
                if hasattr(criterion, "decay"):
                    criterion.o2m = criterion.decay(criterion.updates)
                    criterion.o2o = max(criterion.total - criterion.o2m, 0)
            print(
                f"[overlock] resumed from {self.detector_overrides.get('resume_path')}: "
                f"epoch {self.start_epoch}, criterion.updates={getattr(criterion, 'updates', None)}"
            )

    OverLoCKDetectionTrainer.__name__ = "OverLoCKDetectionTrainer"
    OverLoCKDetectionTrainer.__qualname__ = "OverLoCKDetectionTrainer"
    OverLoCKDetectionTrainer.__module__ = __name__
    return OverLoCKDetectionTrainer


_TRAINER_CLASS = None

#: Keys consumed by this trainer and removed from the native ``args`` namespace.
_DETECTOR_KEYS = (
    "backbone_weights",
    "backbone_variant",
    "resume_state",
    "resume_path",
)


def trainer_class():
    global _TRAINER_CLASS
    if _TRAINER_CLASS is None:
        _TRAINER_CLASS = _trainer_class()
    return _TRAINER_CLASS


def move_optimizer_to(optimizer, device) -> None:
    """Move optimizer state (only present after a real step) to ``device``."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


# --------------------------------------------------------------------------------------
# checkpoint helpers (plain tensor state_dict + structure metadata)
# --------------------------------------------------------------------------------------
def save_project_checkpoint(model, path: str, extra: Optional[dict] = None) -> dict:
    """Save a safe, rebuildable checkpoint and return its manifest.

    THOP profiling artefacts are stripped from the model first, so a profile run before the save
    cannot put ``total_ops``/``total_params`` into the payload.
    """
    from .checkpoint import sha256_file

    strip_thop_artifacts(model)

    payload = {
        "format": CHECKPOINT_FORMAT,
        "structure": model.state_metadata(),
        "model": clean_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()}),
        "extra": extra or {},
    }
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)  # default protocol: weights_only-loadable everywhere
    return {
        "path": path,
        "format": CHECKPOINT_FORMAT,
        "size_bytes": os.path.getsize(path),
        "sha256": sha256_file(path),
        "n_tensors": len(payload["model"]),
        "structure": payload["structure"],
    }


def load_project_checkpoint(model, path: str, strict: bool = True) -> dict:
    """Rebuild-and-reload consistency check for :func:`save_project_checkpoint` output."""
    from ultralytics.utils.patches import torch_load

    state = torch_load(path, map_location="cpu")
    if not (isinstance(state, dict) and state.get("format") == CHECKPOINT_FORMAT):
        raise ConfigError(f"{path} is not an {CHECKPOINT_FORMAT} checkpoint")
    meta = state["structure"]
    mismatches = {
        k: (meta.get(k), getattr(model, k, None))
        for k in ("backbone_variant", "yolo_family", "yolo_scale", "nc")
        if meta.get(k) != getattr(model, k, None)
    }
    if mismatches:
        raise ConfigError(f"checkpoint structure does not match the model: {mismatches}")
    payload = clean_state_dict(state["model"])
    strip_thop_artifacts(model)  # so a profiling run cannot make the reload "unexpected keys"
    incompatible = model.load_state_dict(payload, strict=strict)
    return {
        "loaded_tensors": len(payload),
        "missing_keys": list(getattr(incompatible, "missing_keys", [])),
        "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
        "structure": meta,
        "extra": state.get("extra", {}),
    }


# --------------------------------------------------------------------------------------
# legacy/CLI helpers
# --------------------------------------------------------------------------------------
def build_trainer(cfg: dict, overrides: Optional[dict] = None, cfg_obj=None):
    """Construct the trainer from a resolved experiment config (no training loop started)."""
    cls = trainer_class()
    return cls(cfg=cfg_obj, overrides=overrides or {}, detector_config=cfg)


def train_main(argv: Optional[list] = None) -> int:  # pragma: no cover - thin wrapper
    """Delegate to ``scripts/train.py`` without requiring ``scripts`` to be a package."""
    import importlib.util

    from .paths import project_root

    script = os.path.join(project_root(), "scripts", "train.py")
    spec = importlib.util.spec_from_file_location("overlock_scripts_train", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(argv)
