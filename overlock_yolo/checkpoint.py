"""Auditable, safety-restricted checkpoint loading for the OverLoCK-Base backbone.

Design contract (DESIGN.md 6)
-----------------------------
1. Load the confirmed local ``.pth`` with ``map_location='cpu', weights_only=True``.
   The official factory would rewrite ``pretrained`` into a GitHub download URL, so it is
   never used.  A ``weights_only=True`` failure is *retained*: this module never retries
   with ``weights_only=False`` and never registers unknown globals.
2. The actual container is identified; only a plain tensor ``state_dict`` or a *verified*
   ``state_dict``/``model``/``ema`` tensor mapping is accepted.  Ambiguous multi-candidate
   containers are reported, not guessed.
3. Only the necessary ``module.`` prefix removal happens, and re-name collisions are
   detected.
4. ``head.*`` / ``aux_head.*`` (classification-only) may be excluded explicitly, per key
   list.  New detection parameters (``extra_norm.*``) may stay at their initialisation.
5. Every other shared trainable backbone parameter must be covered, and coverage is
   reported by ``numel`` per module, not just by tensor count.
6. SHA256, container choice, prefix transforms, matched/missing/unexpected/shape-mismatch,
   allowed exceptions and total coverage are recorded.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from typing import Iterable, Optional

import torch

__all__ = [
    "sha256_file",
    "load_safe_state_dict",
    "flatten_state_dict",
    "audit_and_load",
    "CheckpointError",
    "CLASSIFICATION_ONLY_PREFIXES",
]

#: Classification-only modules of the upstream OverLoCK that are not part of the detector.
CLASSIFICATION_ONLY_PREFIXES = ("head.", "aux_head.")

#: Target parameters that are allowed to be absent from the classification checkpoint and
#: keep their initialisation, verified against the real file (14 keys / 421,008 numel):
#:
#: * ``extra_norm.*`` (5 LayerNorm2d = 10 keys) -- the classification variant defines
#:   ``extra_norm`` but never calls it, so it was never trained and is absent from the file.
#: * ``h_proj.*`` (4 keys) -- likewise unused by the classification forward; the detection
#:   variant applies it in ``forward_sub_features``.
#:
#: Both are 1x1 projections / normalisations whose initialisation (``trunc_normal_(std=0.02)``
#: for the conv, LayerScale ``1e-5`` for the scale) is the upstream default, and neither is a
#: pretrained weight being silently dropped.  The list is explicit and per-prefix; no generic
#: prefix is whitelisted and every other missing key fails the audit.
DETECTION_ONLY_PREFIXES = ("extra_norm.", "h_proj.")


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be loaded or audited safely."""


# --------------------------------------------------------------------------------------
# hashing / container discovery
# --------------------------------------------------------------------------------------
def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Streaming SHA256 + size for a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_safe_state_dict(path: str) -> tuple:
    """``torch.load`` restricted to ``weights_only=True``.  Returns ``(obj, info)``.

    ``info`` records the exact call, size and hash so the report is reproducible.
    A failure propagates as :class:`CheckpointError` -- no downgrade to
    ``weights_only=False``.
    """
    if not os.path.isfile(path):
        raise CheckpointError(f"checkpoint not found: {path}")
    size = os.path.getsize(path)
    digest = sha256_file(path)
    info = {
        "path": os.path.abspath(path),
        "size_bytes": size,
        "sha256": digest,
        "load_call": "torch.load(path, map_location='cpu', weights_only=True)",
        "weights_only": True,
    }
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # retained, never downgraded
        info["status"] = "failed"
        info["error"] = f"{type(exc).__name__}: {exc}"
        raise CheckpointError(
            f"weights_only=True load failed for {path}: {exc}. "
            "Refusing to retry with weights_only=False or to add unknown globals."
        ) from exc
    info["status"] = "ok"
    info["top_level_type"] = type(obj).__name__
    return obj, info


def _to_flat_tensor_dict(obj) -> Optional[OrderedDict]:
    """Return a flat ``{str: Tensor}`` mapping, or ``None`` when ``obj`` is not one."""
    if isinstance(obj, OrderedDict):
        obj = dict(obj)
    if not isinstance(obj, dict):
        return None
    if not obj:
        return None
    if not all(isinstance(k, str) for k in obj):
        return None
    if not all(torch.is_tensor(v) for v in obj.values()):
        return None
    return OrderedDict(obj)


def flatten_state_dict(obj, container_key: Optional[str] = None) -> tuple:
    """Identify the real container and return ``(flat_state_dict, container_report)``.

    Accepted, in order:

    * a plain tensor state dict;
    * a dict whose tensor values are nested one level under a single unambiguous
      ``state_dict`` / ``model`` / ``ema`` key (verified: all sibling values are
      non-tensor metadata and the nested mapping is a flat tensor mapping).

    Anything else -- several flat tensor candidates, no tensors at all, non-tensor values
    mixed into the mapping -- raises :class:`CheckpointError` rather than guessing.
    """
    report = {"container_choice": None, "container_candidates": [], "prefix_transforms": []}

    flat = _to_flat_tensor_dict(obj)
    if flat is not None:
        report["container_choice"] = container_key or "<top-level state_dict>"
        report["container_candidates"].append(
            {"key": container_key or "<top-level>", "kind": "flat_tensor_state_dict", "n_tensors": len(flat)}
        )
        report["top_level_keys"] = None
        return flat, report

    if not isinstance(obj, dict):
        raise CheckpointError(f"unsupported checkpoint container type: {type(obj).__name__}")

    report["top_level_keys"] = [str(k) for k in obj.keys()]
    candidates = []
    for key in ("state_dict", "model", "ema"):
        if key in obj:
            nested = _to_flat_tensor_dict(obj[key])
            sibling_tensor_keys = [str(k) for k, v in obj.items() if k != key and torch.is_tensor(v)]
            candidates.append(
                {
                    "key": key,
                    "is_flat_tensor_mapping": nested is not None,
                    "n_tensors": 0 if nested is None else len(nested),
                    "sibling_tensor_keys": sibling_tensor_keys,
                }
            )
    report["container_candidates"] = candidates

    if container_key is not None:
        if container_key not in obj:
            raise CheckpointError(f"requested container key {container_key!r} not present in checkpoint")
        nested = _to_flat_tensor_dict(obj[container_key])
        if nested is None:
            raise CheckpointError(f"container {container_key!r} is not a flat tensor mapping")
        report["container_choice"] = container_key
        return nested, report

    usable = [c for c in candidates if c["is_flat_tensor_mapping"] and not c["sibling_tensor_keys"]]
    if len(usable) == 1:
        key = usable[0]["key"]
        report["container_choice"] = key
        return _to_flat_tensor_dict(obj[key]), report
    if len(usable) > 1:
        raise CheckpointError(
            "ambiguous checkpoint: several usable tensor containers "
            f"{[c['key'] for c in usable]}; pass container_key explicitly after inspection"
        )
    raise CheckpointError(
        "checkpoint is not a recognised tensor state dict and no verified "
        f"state_dict/model/ema container was found; candidates={candidates}"
    )


def strip_known_prefixes(sd: dict, prefixes: Iterable[str] = ("module.",)) -> tuple:
    """Remove only the given prefixes, detecting resulting name collisions."""
    prefixes = tuple(prefixes)
    out = OrderedDict()
    transforms = []
    collisions = []
    counts = {p: 0 for p in prefixes}
    for k, v in sd.items():
        new = k
        for p in prefixes:
            if new.startswith(p):
                new = new[len(p) :]
                counts[p] += 1
                break
        if new in out:
            collisions.append(new)
        out[new] = v
    for p, n in counts.items():
        if n:
            transforms.append({"prefix": p, "removed_from": n})
    if collisions:
        raise CheckpointError(f"prefix removal produced duplicate keys: {sorted(set(collisions))[:20]}")
    return out, transforms


# --------------------------------------------------------------------------------------
# audit + load
# --------------------------------------------------------------------------------------
def _numel_sum(state: dict) -> int:
    return int(sum(v.numel() for v in state.values() if torch.is_tensor(v)))


def audit_and_load(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    container_key: Optional[str] = None,
    allow_missing_keys: Iterable[str] = (),
    allow_missing_prefixes: Iterable[str] = DETECTION_ONLY_PREFIXES,
    ignore_prefixes: Iterable[str] = CLASSIFICATION_ONLY_PREFIXES,
    strict_shapes: bool = True,
) -> dict:
    """Load ``checkpoint_path`` into ``model`` and return a full audit report.

    ``allow_missing_keys`` is an explicit *per-key* exception list (recorded verbatim).
    ``allow_missing_prefixes`` is the short, explicit detection-only prefix list.
    ``ignore_prefixes`` are checkpoint keys the *model does not have* (classification
    head/aux head) which are dropped instead of being counted as "unexpected".
    """
    allow_missing_keys = tuple(allow_missing_keys)
    allow_missing_prefixes = tuple(allow_missing_prefixes)
    ignore_prefixes = tuple(ignore_prefixes)

    obj, load_info = load_safe_state_dict(checkpoint_path)
    flat, container_report = flatten_state_dict(obj, container_key=container_key)
    flat, transforms = strip_known_prefixes(flat)

    target = model.state_dict()
    report = {
        "checkpoint": load_info,
        "container": container_report,
        "prefix_transforms": transforms,
        "n_checkpoint_tensors": len(flat),
        "n_target_tensors": len(target),
        "allow_missing_keys": list(allow_missing_keys),
        "allow_missing_prefixes": list(allow_missing_prefixes),
        "ignore_prefixes": list(ignore_prefixes),
        "checkpoint_module_breakdown": {},
    }

    # --- container sanity: the checkpoint must look like an OverLoCK backbone --------
    top_modules = {}
    for k in flat:
        top_modules[k.split(".")[0]] = top_modules.get(k.split(".")[0], 0) + 1
    report["checkpoint_module_breakdown"] = dict(sorted(top_modules.items(), key=lambda kv: -kv[1]))

    ignored = {k: v for k, v in flat.items() if k.startswith(ignore_prefixes)}
    unexpected = {k: v for k, v in flat.items() if k not in target and k not in ignored}
    matched = {}
    shape_mismatch = {}
    for k, v in flat.items():
        if k in ignored or k not in target:
            continue
        if tuple(target[k].shape) != tuple(v.shape):
            shape_mismatch[k] = {"checkpoint": list(v.shape), "target": list(target[k].shape)}
        else:
            matched[k] = v

    missing = {}
    for k, v in target.items():
        if k in matched or k in shape_mismatch:
            continue
        missing[k] = v

    allowed_missing = {
        k: v
        for k, v in missing.items()
        if k in allow_missing_keys or k.startswith(allow_missing_prefixes)
    }
    disallowed_missing = {k: v for k, v in missing.items() if k not in allowed_missing}

    report["matched"] = {"count": len(matched), "numel": _numel_sum(matched)}
    report["ignored_classification_only"] = {
        "count": len(ignored),
        "numel": _numel_sum(ignored),
        "keys": sorted(ignored)[:200],
    }
    report["unexpected"] = {"count": len(unexpected), "keys": sorted(unexpected)[:200]}
    report["shape_mismatch"] = shape_mismatch
    report["missing_allowed"] = {
        "count": len(allowed_missing),
        "numel": _numel_sum(allowed_missing),
        "keys": sorted(allowed_missing),
    }
    report["missing_disallowed"] = {
        "count": len(disallowed_missing),
        "numel": _numel_sum(disallowed_missing),
        "keys": sorted(disallowed_missing),
    }

    # --- coverage by numel, per module bucket ----------------------------------------
    buckets = ("blocks1", "blocks2", "blocks3", "blocks4", "sub_blocks3", "sub_blocks4", "extra_norm", "other")
    coverage = {
        b: {"target_numel": 0, "matched_numel": 0, "missing_numel": 0, "target_tensors": 0, "matched_tensors": 0}
        for b in buckets
    }

    def bucket_of(key: str) -> str:
        head = key.split(".")[0]
        return head if head in buckets else "other"

    for k, v in target.items():
        b = bucket_of(k)
        coverage[b]["target_numel"] += int(v.numel())
        coverage[b]["target_tensors"] += 1
        if k in matched:
            coverage[b]["matched_numel"] += int(v.numel())
            coverage[b]["matched_tensors"] += 1
        elif k in missing:
            coverage[b]["missing_numel"] += int(v.numel())
    for b in coverage:
        t = coverage[b]["target_numel"]
        coverage[b]["numel_coverage"] = round(coverage[b]["matched_numel"] / t, 6) if t else None
    report["coverage_by_module"] = coverage
    allowed_missing_by_bucket = {b: 0 for b in buckets}
    for k, v in allowed_missing.items():
        allowed_missing_by_bucket[bucket_of(k)] += int(v.numel())
    report["coverage_by_module_excluding_allowed_missing"] = {
        b: (
            round(
                coverage[b]["matched_numel"]
                / (coverage[b]["target_numel"] - allowed_missing_by_bucket[b]),
                6,
            )
            if (coverage[b]["target_numel"] - allowed_missing_by_bucket[b]) > 0
            else None
        )
        for b in buckets
    }

    total_target = _numel_sum(target)
    trainable_target = {k: v for k, v in model.named_parameters() if v.requires_grad}
    trainable_missing = {k: v for k, v in trainable_target.items() if k in missing}
    trainable_missing_disallowed = {k: v for k, v in trainable_missing.items() if k not in allowed_missing}
    report["totals"] = {
        "target_tensors": len(target),
        "target_numel": total_target,
        "matched_tensors": len(matched),
        "matched_numel": _numel_sum(matched),
        "numel_coverage": round(_numel_sum(matched) / total_target, 6) if total_target else None,
        "trainable_tensors": len(trainable_target),
        "trainable_numel": _numel_sum(trainable_target),
        "trainable_missing_tensors": len(trainable_missing),
        "trainable_missing_numel": _numel_sum(trainable_missing),
        "trainable_missing_disallowed_tensors": len(trainable_missing_disallowed),
        "trainable_missing_disallowed_numel": _numel_sum(trainable_missing_disallowed),
    }

    # --- hard failure conditions ------------------------------------------------------
    problems = []
    if shape_mismatch:
        problems.append(f"{len(shape_mismatch)} shape mismatches (strict_shapes={strict_shapes})")
    if disallowed_missing:
        problems.append(
            f"{len(disallowed_missing)} non-whitelisted target keys missing, e.g. {sorted(disallowed_missing)[:5]}"
        )
    if unexpected:
        problems.append(f"{len(unexpected)} unexpected checkpoint keys, e.g. {sorted(unexpected)[:5]}")
    if problems:
        report["status"] = "failed"
        report["problems"] = problems
        raise CheckpointError("checkpoint audit failed: " + "; ".join(problems))

    # --- actual load (no blanket strict=False masking) --------------------------------
    incompatible = model.load_state_dict(matched, strict=False)
    unloaded = list(incompatible.unexpected_keys)
    if unloaded:
        report["status"] = "failed"
        report["problems"] = [f"load_state_dict reported unexpected keys: {unloaded[:10]}"]
        raise CheckpointError(report["problems"][0])

    report["status"] = "ok"
    report["load_state_dict_call"] = "model.load_state_dict(matched_subset, strict=False) (subset already audited)"
    return report


def verify_loaded_parameters(model: torch.nn.Module, checkpoint_path: str, keys: Iterable[str]) -> dict:
    """Re-read selected checkpoint tensors and assert exact equality with model params."""
    obj, _ = load_safe_state_dict(checkpoint_path)
    flat, _ = flatten_state_dict(obj)
    flat, _ = strip_known_prefixes(flat)
    out = {}
    params = dict(model.named_parameters())
    for k in keys:
        if k not in flat:
            out[k] = {"status": "absent_in_checkpoint"}
            continue
        if k not in params:
            out[k] = {"status": "absent_in_model"}
            continue
        same = bool(torch.equal(params[k].detach().cpu(), flat[k].cpu()))
        out[k] = {
            "status": "equal" if same else "not_equal",
            "shape": list(flat[k].shape),
            "numel": int(flat[k].numel()),
            "checkpoint_absmax": float(flat[k].abs().max()),
        }
    out["_all_equal"] = all(v.get("status") == "equal" for v in out.values())
    out["_n_checked"] = sum(1 for v in out.values() if isinstance(v, dict))
    return out


def dump_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, default=str)
