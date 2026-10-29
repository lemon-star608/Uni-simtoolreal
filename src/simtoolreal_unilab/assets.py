"""Local, provenance-checked assets for the two shipped products.

SimToolReal ships the KUKA iiwa14 + SHARPA robot/table URDF sources, while
DexToolBench ships its complete static-scene, object, trajectory, and viewer
asset set. Fresh clones therefore need no second checkout. The 1200-tool
SimToolReal pool is generated procedurally as entity-split MJCF and is not a
manifest asset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ASSET_ROOT = Path(__file__).parent / "assets" / "simtoolreal"
DEXTOOLBENCH_ASSET_ROOT = Path(__file__).parent / "assets" / "dextoolbench"


def _verify_manifest(root: Path, label: str) -> dict[str, str]:
    """Fail closed when a committed asset is missing or corrupted."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256"].items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required {label} asset is missing: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{label} asset checksum mismatch: {path}")
    return manifest["sha256"]


def verify_assets() -> dict[str, str]:
    """Return verified SimToolReal source-asset checksums."""
    return _verify_manifest(ASSET_ROOT, "SimToolReal")


def verify_dextoolbench_assets() -> dict[str, str]:
    """Return verified DexToolBench static-asset checksums."""
    return _verify_manifest(DEXTOOLBENCH_ASSET_ROOT, "DexToolBench")


def main() -> None:
    simtoolreal_count = len(verify_assets())
    dextoolbench_count = len(verify_dextoolbench_assets())
    print(f"Verified {simtoolreal_count} local SimToolReal assets in {ASSET_ROOT}")
    print(f"Verified {dextoolbench_count} local DexToolBench assets in {DEXTOOLBENCH_ASSET_ROOT}")
    for root in (ASSET_ROOT, DEXTOOLBENCH_ASSET_ROOT):
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        print(f"source: {manifest['source']} @ {manifest['source_commit']} (MIT)")


if __name__ == "__main__":
    main()
