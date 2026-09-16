"""Local, provenance-checked SimToolReal assets vendored from upstream.

The robot (KUKA iiwa14 + SHARPA hand) and table URDFs ship with the
repository under ``assets/simtoolreal/`` so a fresh clone needs no second
checkout; the upstream repository stays the source of truth (see
``assets/simtoolreal/LICENSE-upstream-simtoolreal`` and the manifest's
per-file ``source``/``license`` provenance).  The 1200-tool pool is not an
asset: it is generated procedurally at first use by
``tasks/simtoolreal/tool_urdf_writer.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ASSET_ROOT = Path(__file__).parent / "assets" / "simtoolreal"

_UPSTREAM = "https://github.com/tylerlum/simtoolreal"
_UPSTREAM_COMMIT = "313d5aea"  # frozen snapshot the migration verified against


def verify_assets() -> dict[str, str]:
    """Fail closed when any vendored asset is missing or corrupted."""
    manifest = json.loads((ASSET_ROOT / "manifest.json").read_text())
    for relative, expected in manifest["sha256"].items():
        path = ASSET_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required SimToolReal asset is missing: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"SimToolReal asset checksum mismatch: {path}")
    return manifest["sha256"]


def main() -> None:
    print(f"Verified {len(verify_assets())} local SimToolReal assets in {ASSET_ROOT}")
    print(f"source: {_UPSTREAM} @ {_UPSTREAM_COMMIT} (MIT)")


if __name__ == "__main__":
    main()
