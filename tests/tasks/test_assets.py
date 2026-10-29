"""Integrity and provenance checks for committed product assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simtoolreal_unilab import assets


@pytest.mark.parametrize(
    ("root", "label", "expected_count"),
    [
        (assets.ASSET_ROOT, "SimToolReal", 46),
        (assets.DEXTOOLBENCH_ASSET_ROOT, "DexToolBench", 312),
    ],
)
def test_committed_asset_manifests_match_every_file(
    root: Path, label: str, expected_count: int
) -> None:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    checksums = manifest["sha256"]
    assert len(checksums) == expected_count

    for relative, expected in checksums.items():
        path = root / relative
        assert path.is_file(), f"{label} asset is missing: {relative}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, f"{label} asset checksum mismatch: {relative}"


def test_asset_command_reports_provenance_for_both_products(capsys) -> None:
    assets.main()
    output = capsys.readouterr().out

    assert "Verified 46 local SimToolReal assets" in output
    assert "Verified 312 local DexToolBench assets" in output
    assert output.count("source:") == 2
    assert "source: https://github.com/tylerlum/simtoolreal @ 313d5aea (MIT)" in output
    assert (
        "source: https://github.com/tylerlum/simtoolreal "
        "@ 313d5aea1f507c6cfe097b672b62945d7b0bbff5 (MIT)" in output
    )


def test_asset_documentation_covers_both_products_without_migration_history() -> None:
    text = Path(assets.__file__).read_text(encoding="utf-8").lower()
    assert "simtoolreal" in text
    assert "dextoolbench" in text
    assert "migration" not in text
