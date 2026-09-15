"""Step-0 skeleton tests for the SimToolReal task registration."""

from __future__ import annotations

import numpy as np
from unilab.base import registry
from unilab.base.scene import SceneCfg

from simtoolreal_unilab.tasks.simtoolreal import config as simtoolreal_config
from simtoolreal_unilab.tasks.simtoolreal.scene import resolve_asset


def test_simtoolreal_registered_isaacsim_only():
    registry.ensure_registries()
    meta = registry._envs["SimToolReal"]
    assert set(meta.env_factory_dict) == {"isaacsim"}, (
        "step-0 registers only isaacsim; mujoco lands in step 2 after verification"
    )


def test_simtoolreal_asset_resolver_finds_robot_urdf():
    path = resolve_asset(simtoolreal_config.AssetsCfg().robot_urdf)
    assert path.endswith(".urdf")


def test_canonical_layout_constants():
    assert simtoolreal_config.NUM_JOINTS == 29
    assert len(simtoolreal_config.JOINT_NAMES_CANONICAL) == 29
    assert len(simtoolreal_config.ARM_JOINT_NAMES) == 7
    # PD gain tables cover every canonical joint exactly once.
    stiffness = {
        **simtoolreal_config.ARM_JOINT_STIFFNESS,
        **simtoolreal_config.HAND_JOINT_STIFFNESS,
    }
    assert set(stiffness) == set(simtoolreal_config.JOINT_NAMES_CANONICAL)


def test_tool_catalog_default_pool_size():
    catalog = __import__(
        "simtoolreal_unilab.tasks.simtoolreal.tool_catalog", fromlist=["build_tool_catalog"]
    )
    specs = catalog.build_tool_catalog()
    assert len(specs) == 1200  # 12 distributions x num_per_type=100
    assert np.isfinite([s.mass for s in specs]).all()


def test_obs_width_pins_match_obs_cfg_field_lists():
    """140/162 are code-asserted at assembly; here the hand-computed pins are
    tied to the configured field lists (DESIGN.md section 1, obs_utils.py:
    35-54 field sizes over simtoolreal_env_cfg.py:112-143 lists)."""
    from simtoolreal_unilab.tasks.simtoolreal import observations as obs_mod

    obs = simtoolreal_config.ObsCfg()
    assert sum(obs_mod.OBS_FIELD_SIZES[f] for f in obs.obs_list) == obs_mod.POLICY_OBS_DIM == 140
    assert sum(obs_mod.OBS_FIELD_SIZES[f] for f in obs.state_list) == obs_mod.CRITIC_OBS_DIM == 162


def test_multi_asset_scene_materializes_typed_entity_specs():
    """Hydra-shaped mappings become backend-owned cold-path declarations."""
    robot = resolve_asset(simtoolreal_config.AssetsCfg().robot_urdf)
    object_path = resolve_asset("generated/object.urdf")
    goalviz_path = resolve_asset("generated/goalviz.urdf")
    scene = SceneCfg(
        model_file=robot,
        entities={
            "robot": {"root_body_name": simtoolreal_config.ROOT_BODY_NAME},
            "table": {"root_body_name": "box"},
            "object": {"root_body_name": "object_root", "body_names": ["object_root"]},
            "goalviz": {"root_body_name": "goalviz_root", "body_names": ["goalviz_root"]},
        },
        entity_assets=[
            {
                "name": "robot",
                "model_file": robot,
                "asset_format": "urdf",
                "materialization": "articulation",
                "root_mode": "fixed",
                "contact_friction": [0.5, 0.5, 0.0],
                "contact_friction_by_body": [
                    {"body_name": "left_index_DP", "friction": [1.5, 1.5, 0.0]}
                ],
            },
            {
                "name": "table",
                "model_file": resolve_asset(simtoolreal_config.AssetsCfg().table_urdf),
                "asset_format": "urdf",
                "materialization": "rigid",
                "root_mode": "kinematic",
            },
            {
                "name": "object",
                "model_file": object_path,
                "asset_format": "urdf",
                "materialization": "rigid",
                "root_mode": "floating",
            },
            {
                "name": "goalviz",
                "model_file": goalviz_path,
                "asset_format": "urdf",
                "materialization": "rigid",
                "root_mode": "kinematic",
            },
        ],
    )
    assert tuple(spec.name for spec in scene.entity_assets) == (
        "robot",
        "table",
        "object",
        "goalviz",
    )
    assert all(type(spec).__name__ == "SceneEntitySpec" for spec in scene.entity_assets)
    assert scene.entity_assets[0].contact_friction_by_body[0].body_name == "left_index_DP"
    assert scene.entity_assets[2].root_mode == "floating"
    assert scene.entity_assets[2].model_file != scene.entity_assets[3].model_file
