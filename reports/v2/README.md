# `reports/v2` — V2 evidence

JSON is the source of truth; `HANDOFF.md` is the readable summary.  Nothing in this directory
claims a target-GPU or accuracy result.

| File | Produced by | Meaning |
|---|---|---|
| `environment.json` | `scripts/report_environment.py` | local interpreter/torch/threads + differences from the server target |
| `compatibility.json` | `scripts/report_environment.py` | 40 combinations, variant tables, native tail measurements, vendored-source digest, target stack + wheel evidence |
| `checkpoint_t.json`, `checkpoint_b.json` | `scripts/smoke.py` M05 | full per-key checkpoint audit (T, B) |
| `checkpoint_manifest.json` | hand-written | which checkpoints are expected where, with hashes |
| `variant_matrix.json` | `scripts/profile_model.py --matrix --verify-matrix` | 40-cell parameter matrix + assembly cross-checks |
| `profile_*.json` / `.md` | `scripts/profile_model.py` | parameters, partial MACs, analytic `na2d_av` MACs |
| `validation.json` | `scripts/smoke.py` | M01–M11 evidence |
| `regression_tests.json` | `tests/run_v2_tests.py` | targeted regression tests for the V1 defects |
| `data_view.json` | `scripts/prepare_data.py` | generated smoke view contents |

Regenerate everything with::

    python scripts/smoke.py --out reports/v2/validation.json
    python tests/run_v2_tests.py
    python scripts/profile_model.py --matrix --verify-matrix
    python scripts/profile_model.py --config configs/overlock_t_yolo11s_soda.yaml
    python scripts/report_environment.py
