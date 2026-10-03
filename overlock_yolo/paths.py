"""Central path resolution (DESIGN_V2.md 3.2) -- no machine-specific path is hardcoded here.

Precedence for every path, highest first::

    1. explicit CLI / config value
    2. environment variable (see ``ENV_*``)
    3. repository default, relative to the *project root* (never to the current cwd)

The project root is discovered by walking up from this file until a directory containing
``overlock_yolo/paths.py`` and either ``DESIGN_V2.md`` or ``pyproject.toml`` is found, so the
package keeps working after a ``git clone`` to an arbitrary location.

Nothing in this module imports ``ultralytics``; :func:`install_ultralytics_root` is called by
the CLI entry points *before* any Ultralytics import happens.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

__all__ = [
    "PathResolutionError",
    "project_root",
    "resolve_project_root",
    "resolve_ultralytics_root",
    "install_ultralytics_root",
    "prepare_ultralytics_env",
    "expand_path_vars",
    "ultralytics_import_origin",
    "resolve_sibling_overlock_root",
    "resolve_checkpoint_path",
    "resolve_data_yaml",
    "default_out_dir",
    "describe_resolution",
]

ENV_PROJECT_ROOT = "OVERLOCK_PROJECT_ROOT"
ENV_ULTRALYTICS_ROOT = "OVERLOCK_ULTRALYTICS_ROOT"
ENV_OVERLOCK_ROOT = "OVERLOCK_SOURCE_ROOT"  # official OverLoCK-main checkout (checkpoints)
ENV_DATA_ROOT = "OVERLOCK_DATA_ROOT"  # directory holding SODA10M/

_MARKERS = ("DESIGN_V2.md", "DESIGN.md", "pyproject.toml", "setup.py")


class PathResolutionError(RuntimeError):
    """Raised when a required root cannot be resolved unambiguously."""


# --------------------------------------------------------------------------------------
# project root
# --------------------------------------------------------------------------------------
def expand_path_vars(value: str, root: Optional[str] = None) -> str:
    """Expand ``${PROJECT_ROOT}``, ``~`` and environment variables in a configured path."""
    if not isinstance(value, str):
        return value
    root = root or _PROJECT_ROOT
    return os.path.expanduser(os.path.expandvars(value.replace("${PROJECT_ROOT}", root).replace("$PROJECT_ROOT", root)))


def resolve_project_root(explicit: Optional[str] = None) -> str:
    """Absolute project root. ``explicit`` (CLI ``--project-root``) always wins."""
    if explicit:
        root = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.isdir(os.path.join(root, "overlock_yolo")):
            raise PathResolutionError(f"--project-root {root} does not contain an 'overlock_yolo' package directory")
        return root

    env = os.environ.get(ENV_PROJECT_ROOT)
    if env:
        return resolve_project_root(env)

    start = os.path.dirname(os.path.abspath(__file__))
    node = start
    for _ in range(8):
        if os.path.isdir(os.path.join(node, "overlock_yolo")) and any(
            os.path.exists(os.path.join(node, m)) for m in _MARKERS
        ):
            return node
        parent = os.path.dirname(node)
        if parent == node:
            break
        node = parent
    # last resort: the directory containing the package
    return os.path.dirname(start)


_PROJECT_ROOT = resolve_project_root()


def project_root(explicit: Optional[str] = None) -> str:
    """The project root: ``explicit`` when given, otherwise the auto-detected one."""
    return _PROJECT_ROOT if explicit is None else resolve_project_root(explicit)


def prepare_ultralytics_env(root: Optional[str] = None) -> dict:
    """Point Ultralytics' settings/cache directories inside the project and disable analytics.

    Called by every CLI *before* importing Ultralytics so nothing is written into the user's home
    directory (and a git-cloned copy behaves identically on a fresh machine).  Respects an
    already-set ``YOLO_CONFIG_DIR``.
    """
    root = root or _PROJECT_ROOT
    config_dir = os.environ.get("YOLO_CONFIG_DIR") or os.path.join(root, "cache", "ultralytics")
    os.makedirs(config_dir, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = config_dir
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(root, "cache", "mpl"))
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    return {"YOLO_CONFIG_DIR": config_dir, "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"]}


# --------------------------------------------------------------------------------------
# ultralytics source root
# --------------------------------------------------------------------------------------
def _is_ultralytics_source(root: str) -> bool:
    return bool(root) and os.path.isfile(os.path.join(root, "ultralytics", "__init__.py"))


def resolve_ultralytics_root(
    explicit: Optional[str] = None,
    *,
    project_root_: Optional[str] = None,
    allow_sibling: bool = True,
) -> str:
    """Locate the pinned Ultralytics *source* root (the directory holding ``ultralytics/``).

    Order:

    1. ``explicit`` (``--ultralytics-root``)
    2. ``$OVERLOCK_ULTRALYTICS_ROOT``
    3. ``<project>/vendor/ultralytics``  -- the pinned snapshot shipped with this repository
    4. ``<project>/../ultralytics-main`` -- the author's sibling checkout (development only)

    Raises :class:`PathResolutionError` with all probed locations when none of them holds a
    real source tree, so a missing dependency is never silently replaced by a pip-installed
    ``ultralytics`` of unknown version.
    """
    root = project_root_ or _PROJECT_ROOT
    candidates: List[tuple] = []
    if explicit:
        candidates.append(("cli", os.path.abspath(expand_path_vars(explicit, root))))
    env = os.environ.get(ENV_ULTRALYTICS_ROOT)
    if env:
        candidates.append(("env:" + ENV_ULTRALYTICS_ROOT, os.path.abspath(expand_path_vars(env, root))))
    candidates.append(("vendor", os.path.join(root, "vendor", "ultralytics")))
    if allow_sibling:
        candidates.append(("sibling-dev", os.path.normpath(os.path.join(root, os.pardir, "ultralytics-main"))))

    probed = [{"source": s, "path": p, "has_source": _is_ultralytics_source(p)} for s, p in candidates]
    for _source, path in candidates:
        if _is_ultralytics_source(path):
            return path
    raise PathResolutionError(
        "no Ultralytics source root found. Pass --ultralytics-root, set "
        f"{ENV_ULTRALYTICS_ROOT}, or place the pinned snapshot at "
        f"{os.path.join(root, 'vendor', 'ultralytics')}. Probed: {probed}"
    )


def install_ultralytics_root(root: str) -> str:
    """Put ``root`` first on ``sys.path`` and assert the import resolves inside it.

    Every CLI calls this **before** importing anything that imports ``ultralytics``.  If an
    ``ultralytics`` module was already imported from somewhere else, that is reported as an
    error instead of silently deleting the module (which used to hide a multi-source mix).
    """
    root = os.path.abspath(root)
    if not _is_ultralytics_source(root):
        raise PathResolutionError(f"not an ultralytics source root (no ultralytics/__init__.py): {root}")

    existing = sys.modules.get("ultralytics")
    if existing is not None:
        resolved = os.path.abspath(getattr(existing, "__file__", "") or "")
        if not resolved.startswith(root + os.sep):
            raise PathResolutionError(
                f"'ultralytics' was already imported from {resolved}, which is outside the "
                f"requested source root {root}. Restart the process with the correct "
                "--ultralytics-root instead of mixing two sources."
            )
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)

    import ultralytics  # noqa: F401  (local import on purpose)

    resolved = os.path.abspath(ultralytics.__file__)
    if not resolved.startswith(root + os.sep):
        raise PathResolutionError(f"imported ultralytics from {resolved}, outside the pinned source root {root}")
    return resolved


def ultralytics_import_origin() -> dict:
    """Where ``ultralytics`` actually resolves from (empty dict when not imported)."""
    mod = sys.modules.get("ultralytics")
    if mod is None:
        return {"imported": False}
    return {
        "imported": True,
        "file": os.path.abspath(getattr(mod, "__file__", "") or ""),
        "version": getattr(mod, "__version__", None),
        "sys_path_heads": [p for p in sys.path[:4]],
    }


# --------------------------------------------------------------------------------------
# data / checkpoints
# --------------------------------------------------------------------------------------
def resolve_sibling_overlock_root(explicit: Optional[str] = None, *, project_root_: Optional[str] = None) -> Optional[str]:
    """Official ``OverLoCK-main`` checkout (holds the downloaded T/B checkpoints), or ``None``."""
    if explicit:
        path = os.path.abspath(expand_path_vars(explicit, project_root_ or _PROJECT_ROOT))
        return path if os.path.isdir(path) else None
    env = os.environ.get(ENV_OVERLOCK_ROOT)
    if env:
        path = os.path.abspath(os.path.expanduser(env))
        return path if os.path.isdir(path) else None
    root = project_root_ or _PROJECT_ROOT
    sibling = os.path.normpath(os.path.join(root, os.pardir, "OverLoCK-main"))
    return sibling if os.path.isdir(sibling) else None


def resolve_checkpoint_path(
    variant: str,
    explicit: Optional[str] = None,
    *,
    project_root_: Optional[str] = None,
    overlock_root: Optional[str] = None,
) -> Optional[str]:
    """Absolute checkpoint path for ``variant``: explicit, then ``checkpoints/<dir>``, then ``None``.

    ``None`` means "the file is not present" -- callers must fail loudly or run an explicitly
    random structure test; nothing is ever downloaded.
    """
    name = f"overlock_{variant}_in1k_224.pth"
    root = project_root_ or _PROJECT_ROOT
    if explicit:
        # An explicitly requested path is returned as-is (even when missing) so callers can
        # report "you asked for this file and it is not there" instead of silently falling back
        # to a different checkpoint.
        return os.path.abspath(expand_path_vars(explicit, root))
    src = resolve_sibling_overlock_root(overlock_root, project_root_=root)
    candidates = []
    if src:
        candidates.append(os.path.join(src, "checkpoints", name))
    candidates.append(os.path.join(root, "checkpoints", name))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def resolve_data_yaml(explicit: Optional[str] = None, *, project_root_: Optional[str] = None) -> Optional[str]:
    """SODA10M ``soda10m.yaml`` (COCO-format custom config)."""
    both = resolve_data_yaml_with_candidates(explicit, project_root_=project_root_)
    return both["resolved"]


def resolve_data_yaml_with_candidates(explicit: Optional[str] = None, *, project_root_: Optional[str] = None) -> dict:
    root = project_root_ or _PROJECT_ROOT
    candidates: List[tuple] = []
    if explicit:
        candidates.append(("cli", os.path.normpath(os.path.abspath(expand_path_vars(explicit, root)))))
    env = os.environ.get(ENV_DATA_ROOT)
    if env:
        candidates.append(("env:" + ENV_DATA_ROOT, os.path.join(os.path.abspath(expand_path_vars(env, root)), "soda10m.yaml")))
    candidates.append(("sibling-data", os.path.normpath(os.path.join(root, os.pardir, "data", "SODA10M", "soda10m.yaml"))))
    candidates.append(("local-data", os.path.join(root, "data", "soda10m.yaml")))
    for source, path in candidates:
        if os.path.isfile(path):
            return {"resolved": path, "source": source, "probed": [list(c) for c in candidates]}
    return {"resolved": None, "source": None, "probed": [list(c) for c in candidates]}


def default_out_dir(project_root_: Optional[str] = None) -> str:
    """Default output directory for views/artifacts (always inside the project)."""
    return os.path.join(project_root_ or _PROJECT_ROOT, "artifacts", "v2")


def describe_resolution() -> dict:
    """Compact, machine-readable summary of the resolved roots (for report meta blocks)."""
    root = _PROJECT_ROOT
    ultra = None
    try:
        ultra = resolve_ultralytics_root(project_root_=root)
    except PathResolutionError as exc:  # keep the reason, do not crash diagnostics
        ultra = {"error": str(exc)}
    return {
        "project_root": root,
        "cwd": os.getcwd(),
        "ultralytics_root": ultra,
        "sibling_overlock_root": resolve_sibling_overlock_root(project_root_=root),
        "data_yaml": resolve_data_yaml_with_candidates(project_root_=root),
        "env": {k: os.environ.get(k) for k in (ENV_PROJECT_ROOT, ENV_ULTRALYTICS_ROOT, ENV_OVERLOCK_ROOT, ENV_DATA_ROOT)},
    }
