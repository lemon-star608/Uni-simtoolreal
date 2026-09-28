"""L2 smoke: SimToolReal on the real MuJoCo backend with the D8 uniform pool.

Mirror of ``test_mujoco_backend_smoke.py`` with the layout knob flipped:
``tool_pool_layout=uniform_public_layout`` flows through the production
wiring (Hydra compose -> BackendAdapter override -> registry.make with
sim_backend="mujoco") into the layout-aware bootstrap resolver, the uniform
pool cache (sibling ``*_uniform`` directory), and
``bind_entity_variant``'s ``FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT``
mapping.  No mocks: the UniSim MuJoCo backend composes the full-mesh entity
scene, validates the uniform contract, and steps mjbatch on CPU.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

NUM_ENVS = 8
NUM_PER_TYPE = 1  # 12 variants
NUM_VARIANTS = 12


def _build_env(tmp_path):
    from hydra import compose, initialize_config_dir
    from unilab.base import registry
    from unilab.base.config_adapter import BackendAdapter

    import simtoolreal_unilab.tasks  # noqa: F401  (registers env + resolvers)
    from simtoolreal_unilab.config import CONF_ROOT

    with initialize_config_dir(str(CONF_ROOT), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=simtoolreal/mujoco",
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
        "SimToolReal", sim_backend="mujoco", env_cfg_override=override, num_envs=NUM_ENVS
    )


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    built = _build_env(tmp_path_factory.mktemp("mujoco_uniform_smoke"))
    yield built
    built.close()


def test_uniform_layout_bound_and_accepted(env):
    """The knob maps to UNIFORM_PUBLIC_LAYOUT and the backend declares no
    rejection for the bound plan."""
    from unisim.dr.types import FixedVariantLayout

    binding = env.cfg.scene.entity_variant
    assert binding is not None
    assert binding.plan.layout is FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT
    assert env.cfg.tool_pool_layout == "uniform_public_layout"
    rejections = env._backend.get_dr_capabilities().fixed_variant_rejections(binding.plan)
    assert rejections == ()
    # The object bootstrap source resolved through the layout-aware resolver.
    object_spec = next(s for s in env.cfg.scene.entity_assets if s.name == "object")
    assert "entity_bootstrap_uniform_public_layout" in str(object_spec.source.model_file)


def test_uniform_pool_materialized_in_sibling_directory(env):
    from pathlib import Path

    root = Path(env.cfg.tool_pool_files[0]).parent.parent
    assert root.name.endswith("_uniform")
    assert (root / "manifest.json").is_file()
    import json

    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["layout"] == "uniform_public_layout"


def test_reset_obs_contract(env):
    obs, info = env.reset()
    assert isinstance(info, dict)
    assert obs["obs"].shape == (NUM_ENVS, 140)  # OBS_FIELD_SIZES pins
    assert obs["critic"].shape == (NUM_ENVS, 162)
    assert all(np.isfinite(value).all() for value in obs.values())


def test_step_100_no_nan(env):
    for _ in range(100):
        state = env.step(np.zeros((NUM_ENVS, 29), dtype=np.float32))
    assert all(np.isfinite(value).all() for value in state.obs.values())
    assert np.isfinite(state.reward).all()


def test_variant_assignment_binds_catalog_masses(env):
    """Round-robin assignment over 12 uniform variants; per-env masses match
    the same ToolSpec table the same-layout pool uses (layout-independent)."""
    import simtoolreal_unilab.tasks.simtoolreal.env_cfg as env_cfg_module

    assignment = env_cfg_module.fixed_variant_plan_assignment(env.cfg, NUM_ENVS)
    assert assignment is not None
    assert assignment.shape == (NUM_ENVS,)
    catalog_masses = np.asarray(env.cfg.tool_pool_masses)
    assert catalog_masses.shape == (NUM_VARIANTS,)

    backend = env._backend
    object_id = mujoco.mj_name2id(backend._model, mujoco.mjtObj.mjOBJ_BODY, "object/object_root")
    assert object_id > 0
    per_env = np.asarray(backend._base_body_mass)[:, object_id]
    # mjbatch stores per-variant tables in float32; the catalog is float64.
    np.testing.assert_allclose(per_env, catalog_masses[assignment], rtol=1e-5)
    assert len(np.unique(catalog_masses[assignment])) > 1


def test_uniform_variants_realize_mesh_geoms(env):
    """The bound pool's variant files are full-mesh: every slot mesh-typed,
    erasers headless.  (The pool files are the sources the VariantPack maps
    per env; the compose-level contract gate pins eraser omission.)"""
    from pathlib import Path

    from simtoolreal_unilab.tasks.simtoolreal.tool_catalog import build_tool_catalog

    catalog = build_tool_catalog(num_per_type=NUM_PER_TYPE, seed=42, shuffle=True)
    for spec, path in zip(catalog, env.cfg.tool_pool_files, strict=True):
        model = mujoco.MjModel.from_xml_path(str(Path(path)))
        assert all(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH for g in range(model.ngeom))
        assert model.ngeom == (2 if spec.head_scale is None else 4)
