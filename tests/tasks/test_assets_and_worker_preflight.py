"""Vendored assets and the IsaacSim worker preflight.

The assets distributed with the package (``simtoolreal_unilab/assets``)
carry a SHA-256 manifest so a corrupted or partial clone fails at
``simtoolreal-assets`` instead of deep inside the MJCF transcription
chain.  The worker preflight auto-detects the conventional
``setup_isaacsim_env.sh`` installation so training needs no shell
configuration.
"""

from __future__ import annotations

import json
import os

import pytest

import simtoolreal_unilab._isaacsim_worker as preflight
from simtoolreal_unilab.assets import ASSET_ROOT, verify_assets


def test_vendored_assets_verify_against_the_manifest():
    # The manifest covers every required asset (robot URDF + meshes, table)
    # with exact checksums; a partial or corrupted tree fails closed.
    table = verify_assets()
    assert "urdf/table_narrow.urdf" in table
    assert "urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf" in table
    manifest = json.loads((ASSET_ROOT / "manifest.json").read_text())
    assert manifest["source"].startswith("https://github.com/tylerlum/simtoolreal")
    assert (ASSET_ROOT / "LICENSE-upstream-simtoolreal").is_file()


def test_worker_preflight_autodetects_the_conventional_install(monkeypatch, tmp_path):
    for name in (
        "UNISIM_ISAACSIM_PYTHON",
        "UNILAB_ISAACSIM_PYTHON",
        "UNISIM_ISAACSIM_HOME",
        "UNILAB_ISAACSIM_HOME",
    ):
        monkeypatch.delenv(name, raising=False)
    fake = tmp_path / "venv" / "bin" / "python"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setattr(preflight, "CONVENTIONAL_WORKER_PYTHON", fake)
    resolved = preflight.ensure_worker_runtime()
    assert resolved == str(fake)
    assert os.environ["UNISIM_ISAACSIM_PYTHON"] == str(fake)


def test_worker_preflight_explicit_interpreter_wins(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit-python"
    explicit.write_text("#!/bin/sh\n")
    monkeypatch.setenv("UNISIM_ISAACSIM_PYTHON", str(explicit))
    assert preflight.ensure_worker_runtime() == str(explicit)


def test_worker_preflight_missing_explicit_interpreter_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("UNISIM_ISAACSIM_PYTHON", str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match="UNISIM_ISAACSIM_PYTHON"):
        preflight.ensure_worker_runtime()


def test_worker_preflight_without_any_runtime_names_the_setup_command(monkeypatch, tmp_path):
    for name in (
        "UNISIM_ISAACSIM_PYTHON",
        "UNILAB_ISAACSIM_PYTHON",
        "UNISIM_ISAACSIM_HOME",
        "UNILAB_ISAACSIM_HOME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(preflight, "CONVENTIONAL_WORKER_PYTHON", tmp_path / "absent")
    with pytest.raises(RuntimeError, match="setup_isaacsim_env.sh"):
        preflight.ensure_worker_runtime()


def test_worker_preflight_home_layout_passes_through(monkeypatch, tmp_path):
    # An exported home root is unisim's own discovery path; the preflight
    # must not second-guess it.
    monkeypatch.delenv("UNISIM_ISAACSIM_PYTHON", raising=False)
    monkeypatch.delenv("UNILAB_ISAACSIM_PYTHON", raising=False)
    monkeypatch.setenv("UNISIM_ISAACSIM_HOME", str(tmp_path))
    assert preflight.ensure_worker_runtime() == str(tmp_path)


def test_setup_command_points_at_the_official_unilab_installer():
    # The repository ships no installer of its own: the worker runtime comes
    # from UniLab's official script (scripts/tools/setup_isaacsim_env.sh in a
    # points at the vendored installer (scripts/setup_isaacsim_env.sh)
    assert "scripts/setup_isaacsim_env.sh" in preflight._SETUP_COMMAND
