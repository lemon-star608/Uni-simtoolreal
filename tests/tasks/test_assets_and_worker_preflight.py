"""Vendored assets and the IsaacSim worker preflight.

The assets distributed with the package (``simtoolreal_unilab/assets``)
carry a SHA-256 manifest so a corrupted or partial clone fails at
``simtoolreal-assets`` instead of deep inside the URDF converter, and the
scene resolver prefers them over any sibling checkout.  The worker
preflight auto-detects the conventional ``setup_isaacsim_env.sh``
installation so training needs no shell configuration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

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


def test_scene_resolver_prefers_vendored_assets(monkeypatch, tmp_path):
    # Resolution order: SIMTOOLREAL_ASSETS_ROOT override first, then the
    # vendored package assets, then a sibling checkout.  A sibling directory
    # next to a fake workspace must never shadow the vendored tree.
    from simtoolreal_unilab.tasks.simtoolreal.scene import resolve_assets_root

    monkeypatch.delenv("SIMTOOLREAL_ASSETS_ROOT", raising=False)
    assert resolve_assets_root() == ASSET_ROOT

    override = tmp_path / "override-assets"
    (override / "urdf").mkdir(parents=True)
    (override / "urdf" / "table_narrow.urdf").write_text("<robot/>")
    monkeypatch.setenv("SIMTOOLREAL_ASSETS_ROOT", str(override))
    # The override wins only when it carries the robot asset; a partial
    # override falls through to the vendored tree instead of dangling.
    assert resolve_assets_root() == ASSET_ROOT
    robot_rel = "urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    (override / robot_rel).parent.mkdir(parents=True, exist_ok=True)
    (override / robot_rel).write_text("<robot/>")
    assert resolve_assets_root() == override


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


def test_setup_script_exists_and_pins_the_validated_stack():
    script = Path(__file__).resolve().parents[2] / "scripts" / "setup_isaacsim_env.sh"
    text = script.read_text()
    assert script.stat().st_mode & 0o111, "setup script must be executable"
    # The validated worker stack (see README evidence): IsaacSim 5.1.0,
    # IsaacLab 2.3.2.post1 wheels, torch 2.7.0 on the cu126 index.
    for pin in (
        "ISAACSIM_VERSION:-5.1.0",
        "ISAACLAB_VERSION:-2.3.2.post1",
        "TORCH_VERSION:-2.7.0",
        "TORCH_INDEX:-cu126",
    ):
        assert pin in text, pin
