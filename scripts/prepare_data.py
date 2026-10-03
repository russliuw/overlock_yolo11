#!/usr/bin/env python
"""Build an isolated native-YOLO view of SODA10M (COCO -> YOLO txt).

Examples
--------
Smoke view (used by the local CPU checks; 2 images per split, per-file symlinks, no source
writes)::

    python scripts/prepare_data.py --out data/soda_smoke --limit-per-split 2 --force

Full conversion (NOT run locally; implemented and available for the server)::

    python scripts/prepare_data.py --out data/soda_full

``images/<split>`` is a real directory holding one symlink per selected image, so
``--limit-per-split N`` restricts the visible images *and* the labels together and no
``labels.cache`` can be written next to the source annotations.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from overlock_yolo.data import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
