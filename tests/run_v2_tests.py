#!/usr/bin/env python
"""Standalone runner for ``tests/test_v2_contract.py`` (pytest is not installed locally).

Runs every ``test_*`` function and every ``@pytest.mark.parametrize`` case, reporting the same
verdicts the pytest suite would::

    OMP_NUM_THREADS=2 python tests/run_v2_tests.py
    OMP_NUM_THREADS=2 python tests/run_v2_tests.py --out reports/v2/regression_tests.json

``pytest`` is imported by the test module, so a tiny shim is installed first when it is missing;
this keeps the test file itself pytest-compatible for the server environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def install_pytest_shim() -> bool:
    """Provide the minimal ``pytest`` surface the test module uses. Returns True if shimmed."""
    try:
        import pytest  # noqa: F401

        return False
    except ImportError:
        pass

    import types

    shim = types.ModuleType("pytest")

    class _Skip(Exception):
        pass

    class _Mark:
        @staticmethod
        def parametrize(argnames, argvalues, **kwargs):
            names = [a.strip() for a in argnames.split(",")] if isinstance(argnames, str) else list(argnames)

            def deco(fn):
                fn._parametrize = (names, list(argvalues))
                return fn

            return deco

    def _raises(exc, **kwargs):
        class _Ctx:
            value = None

            def __enter__(self):
                return self

            def __exit__(self, et, ev, tb):
                if et is None:
                    raise AssertionError(f"DID NOT RAISE {exc}")
                if not issubclass(et, exc):
                    return False
                self.value = ev
                return True

        return _Ctx()

    def _skip(reason=""):
        raise _Skip(reason)

    shim.mark = _Mark()
    shim.raises = _raises
    shim.skip = _skip
    shim.Skip = _Skip
    sys.modules["pytest"] = shim
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the V2 regression tests without pytest")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "v2", "regression_tests.json"))
    ap.add_argument("--only", default=None, help="substring filter on the test name")
    args = ap.parse_args(argv)

    shimmed = install_pytest_shim()
    import pytest

    path = os.path.join(ROOT, "tests", "test_v2_contract.py")
    spec = importlib.util.spec_from_file_location("test_v2_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    results = []
    t0 = time.time()
    for name in sorted(n for n in dir(module) if n.startswith("test_")):
        fn = getattr(module, name)
        if not callable(fn):
            continue
        if args.only and args.only not in name:
            continue
        params = getattr(fn, "_parametrize", None)
        cases = [{}] if params is None else [dict(zip(params[0], v if isinstance(v, (tuple, list)) else (v,))) for v in params[1]]
        for case in cases:
            label = name + ("" if not case else "[" + ",".join(f"{k}={v}" for k, v in case.items()) + "]")
            start = time.time()
            try:
                fn(**case)
                status, error = "pass", None
            except getattr(pytest, "Skip", Exception) as exc:  # type: ignore[misc]
                status, error = "skipped", str(exc)
            except Exception as exc:  # noqa: BLE001
                status, error = "fail", f"{type(exc).__name__}: {exc}"
            entry = {"test": label, "status": status, "seconds": round(time.time() - start, 3), "error": error}
            if status == "fail":
                entry["traceback"] = traceback.format_exc()
            results.append(entry)
            flag = {"pass": "PASS", "fail": "FAIL", "skipped": "SKIP"}[status]
            print(f"[{flag}] {label}" + (f" ({entry['seconds']}s)" if status != "fail" else ""), flush=True)

    counts = {k: sum(1 for r in results if r["status"] == k) for k in ("pass", "fail", "skipped")}
    payload = {
        "meta": {
            "generated_by": "tests/run_v2_tests.py",
            "module": "tests/test_v2_contract.py",
            "pytest_available": not shimmed,
            "pytest_shim_used": shimmed,
            "device": "cpu",
            "dtype": "float32",
            "total_seconds": round(time.time() - t0, 2),
        },
        "summary": {**counts, "total": len(results), "verdict": "PASS" if counts["fail"] == 0 else "FAIL"},
        "tests": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\ntests -> {args.out}\n{json.dumps(payload['summary'])}")
    return 0 if counts["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
