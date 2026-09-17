"""Full-pool (12 x 100 = 1200) compile gate and URDF-pool parity (M-J1).

Every composed MJCF in the materialized pool must compile
(``MjModel.from_xml_path`` — the compiler is its own format authority, D5)
with the composed-model contract (36/35/29/62) and an ``object`` body whose
compiled mass equals the manifest entry.  The manifest's masses/scales must
equal the URDF pool's value-for-value (both derive from the same catalog;
``test_simtoolreal_tool_pool_full_parity.py`` pins the URDF side against the
original generator byte-for-byte).
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from simtoolreal_unilab.tasks.simtoolreal.env_cfg import _materialize_pool
from simtoolreal_unilab.tasks.simtoolreal.tool_pool_mjcf import (
    compose_scene_mjcf,
    materialize_tool_pool_mjcf,
)

NUM_PER_TYPE = 100
SEED = 42
EXPECTED_POOL_SIZE = 1200
EXPECTED_CONTRACT = (36, 35, 29, 62)  # nq, nv, nu, nmesh


@pytest.fixture(scope="module")
def mjcf_pool(tmp_path_factory):
    root = tmp_path_factory.mktemp("mjcf_pool_root")
    paths, masses, scales = materialize_tool_pool_mjcf(
        seed=SEED, num_per_type=NUM_PER_TYPE, configured_root=str(root)
    )
    return root, paths, masses, scales


@pytest.fixture(scope="module")
def urdf_pool(tmp_path_factory):
    root = tmp_path_factory.mktemp("urdf_pool_root")
    return _materialize_pool(seed=SEED, num_per_type=NUM_PER_TYPE, configured_root=str(root))


def test_pool_size_and_layout(mjcf_pool):
    root, paths, masses, scales = mjcf_pool
    pool_dir = (
        root / f"seed_{SEED}_n{NUM_PER_TYPE}"
    )  # configured root replaces the base, mirroring env_cfg._pool_root
    assert pool_dir.is_dir()
    assert (pool_dir / "base_scene.xml").is_file()
    assert len(paths) == EXPECTED_POOL_SIZE == len(masses) == len(scales)
    for path in paths:
        assert path.parent == pool_dir and path.suffix == ".xml"
    manifest = json.loads((pool_dir / "manifest.json").read_text())
    assert manifest["spec"] == {
        "seed": SEED,
        "num_per_type": NUM_PER_TYPE,
        "count": EXPECTED_POOL_SIZE,
    }
    assert manifest["base_scene"] == "base_scene.xml"
    assert manifest["files"] == [path.name for path in paths]


def test_pool_reuse_is_byte_stable(mjcf_pool):
    """A complete manifest is reused: file bytes and returned tables identical."""
    root, paths, masses, scales = mjcf_pool
    before = {path: path.read_bytes() for path in paths[:8]}
    paths2, masses2, scales2 = materialize_tool_pool_mjcf(
        seed=SEED, num_per_type=NUM_PER_TYPE, configured_root=str(root)
    )
    assert paths2 == paths and masses2 == masses and scales2 == scales
    for path, content in before.items():
        assert path.read_bytes() == content


def test_pool_rebuilds_when_partial(mjcf_pool):
    root, paths, _, _ = mjcf_pool
    paths[0].unlink()
    paths3, _, _ = materialize_tool_pool_mjcf(
        seed=SEED, num_per_type=NUM_PER_TYPE, configured_root=str(root)
    )
    assert paths3[0].is_file()


def test_manifest_matches_urdf_pool_value_for_value(mjcf_pool, urdf_pool):
    _, _, masses, scales = mjcf_pool
    urdf_paths, urdf_masses, urdf_scales = urdf_pool
    assert len(urdf_paths) == EXPECTED_POOL_SIZE
    assert masses == urdf_masses  # exact: same catalog, same float pipeline
    assert scales == urdf_scales


def test_all_composed_mjcf_compile_with_object_mass(mjcf_pool):
    _, paths, masses, _ = mjcf_pool
    for index, path in enumerate(paths):
        model = mujoco.MjModel.from_xml_path(str(path))
        assert (model.nq, model.nv, model.nu, model.nmesh) == EXPECTED_CONTRACT
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_root")
        assert body_id >= 1
        assert model.body_mass[body_id] == pytest.approx(masses[index], rel=1e-12)
        assert model.nkey == 1
        assert model.key_qpos.shape[1] == model.nq  # home keyframe extended


def test_composed_scene_has_no_include(mjcf_pool):
    """D3: one complete single-model file; meshes referenced absolutely."""
    _, paths, _, _ = mjcf_pool
    for path in paths[:16]:
        text = path.read_text()
        assert "<include" not in text
        assert "<flag" in text  # scene-owned option carried through
    sample = paths[0].read_text()
    for line in sample.splitlines():
        if 'file="' in line:
            assert line.split('file="')[1].split('"')[0].startswith("/")


def test_step_smoke_on_representative_variants(mjcf_pool):
    """NaN smoke over the three tool topologies plus a mid-pool variant."""
    _, paths, _, _ = mjcf_pool
    picks = [paths[0], paths[1], paths[2], paths[600], paths[-1]]
    for path in picks:
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        for _ in range(4):
            mujoco.mj_step(model, data)
            assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
