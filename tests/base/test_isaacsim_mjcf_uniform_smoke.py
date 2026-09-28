"""L2 smoke: SimToolReal IsaacSim backend with the D8 uniform pool.

Regression gate for the 2026-09-28 IsaacSim INIT failure: IsaacLab's MJCF
importer cannot consume inline ``vertex``/``face`` meshes (it treats them as
file paths), so the uniform pool ships file-backed OBJ companions.  This
suite builds the real IsaacSim worker path at small scale through the
production wiring (Hydra compose of the isaacsim owner leaf with
``env.tool_pool_layout=uniform_public_layout`` -> BackendAdapter override ->
registry.make): worker INIT must pass, the scene composes with the uniform
bootstrap, and the variant binding is UNIFORM_PUBLIC_LAYOUT.

Requires the external worker runtime (UniLab's official installer,
``scripts/tools/setup_isaacsim_env.sh``) and a GPU; skipped on hosts
without them (the ``isaacsim`` marker).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

NUM_ENVS = 8
NUM_PER_TYPE = 1  # 12 variants

pytestmark = pytest.mark.isaacsim


def _worker_available() -> bool:
    if os.environ.get("UNISIM_ISAACSIM_PYTHON") or os.environ.get("UNILAB_ISAACSIM_PYTHON"):
        return True
    from simtoolreal_unilab._isaacsim_worker import CONVENTIONAL_WORKER_PYTHON

    return CONVENTIONAL_WORKER_PYTHON.is_file()


def _build_env(tmp_path):
    from hydra import compose, initialize_config_dir
    from unilab.base import registry
    from unilab.base.config_adapter import BackendAdapter

    import simtoolreal_unilab.tasks  # noqa: F401  (registers env + resolvers)
    from simtoolreal_unilab._isaacsim_worker import ensure_worker_runtime
    from simtoolreal_unilab.config import CONF_ROOT

    ensure_worker_runtime()
    with initialize_config_dir(str(CONF_ROOT), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=simtoolreal/isaacsim",
                # The real CLI knob: resolves both the layout-aware bootstrap
                # resolver interpolation and the env cfg field.
                "env.tool_pool_layout=uniform_public_layout",
            ],
        )
    adapter = BackendAdapter(cfg, root_dir=str(tmp_path))
    override = adapter.build_task_env_cfg_override()
    override["tool_pool_num_per_type"] = NUM_PER_TYPE
    override["tool_pool_root"] = str(tmp_path / "pool")
    return registry.make(
        "SimToolReal", sim_backend="isaacsim", env_cfg_override=override, num_envs=NUM_ENVS
    )


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    if not _worker_available():
        pytest.skip(
            "IsaacSim worker runtime not installed (UniLab scripts/tools/setup_isaacsim_env.sh)"
        )
    built = _build_env(tmp_path_factory.mktemp("isaacsim_uniform_smoke"))
    yield built
    built.close()


def test_worker_init_binds_uniform_layout(env):
    """The worker INIT passed (fixture built the env); the plan is uniform."""
    from unisim.dr.types import FixedVariantLayout

    binding = env.cfg.scene.entity_variant
    assert binding is not None
    assert binding.plan.layout is FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT
    assert env.cfg.tool_pool_layout == "uniform_public_layout"
    # The object bootstrap source resolved through the layout-aware resolver.
    object_spec = next(s for s in env.cfg.scene.entity_assets if s.name == "object")
    assert "entity_bootstrap_uniform_public_layout" in str(object_spec.source.model_file)


def test_uniform_pool_ships_file_backed_meshes(env):
    from pathlib import Path

    root = Path(env.cfg.tool_pool_files[0]).parent.parent
    assert root.name.endswith("_uniform")
    meshes = sorted((root / "tools" / "meshes").glob("*.obj"))
    assert meshes  # handle OBJ per tool + head OBJ per headed tool
    assert "vertex=" not in Path(env.cfg.tool_pool_files[0]).read_text()


def test_reset_obs_contract(env):
    obs, info = env.reset()
    assert isinstance(info, dict)
    assert obs["obs"].shape == (NUM_ENVS, 140)  # OBS_FIELD_SIZES pins
    assert obs["critic"].shape == (NUM_ENVS, 162)
    assert all(np.isfinite(value).all() for value in obs.values())


def test_step_50_no_nan(env):
    for _ in range(50):
        state = env.step(np.zeros((NUM_ENVS, 29), dtype=np.float32))
    assert all(np.isfinite(value).all() for value in state.obs.values())
    assert np.isfinite(state.reward).all()


def test_variant_assignment_binds_catalog_masses(env):
    """Round-robin over the uniform variants; the wrench mass table follows."""
    import simtoolreal_unilab.tasks.simtoolreal.env_cfg as env_cfg_module

    assignment = env_cfg_module.fixed_variant_plan_assignment(env.cfg, NUM_ENVS)
    assert assignment is not None and assignment.shape == (NUM_ENVS,)
    catalog_masses = np.asarray(env.cfg.tool_pool_masses)
    assert catalog_masses.shape == (12,)

    term = env.event_manager._mode_class_term_cfgs["interval"][0].func
    assert term._object_masses is not None
    np.testing.assert_allclose(term._object_masses, catalog_masses[assignment], rtol=1e-6)
