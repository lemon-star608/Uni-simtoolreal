"""Step-0 skeleton tests for the SimToolReal task registration."""

from __future__ import annotations

import numpy as np
from unilab.base import registry

from simtoolreal_unilab.tasks.simtoolreal import config as simtoolreal_config


def test_simtoolreal_registered_backends():
    registry.ensure_registries()
    meta = registry._envs["SimToolReal"]
    assert set(meta.env_factory_dict) == {"mujoco"}


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
    """Tie the hand-computed width pins to the configured field lists."""
    from simtoolreal_unilab.tasks.simtoolreal import observations as obs_mod

    obs = simtoolreal_config.ObsCfg()
    assert sum(obs_mod.OBS_FIELD_SIZES[f] for f in obs.obs_list) == obs_mod.POLICY_OBS_DIM == 140
    assert sum(obs_mod.OBS_FIELD_SIZES[f] for f in obs.state_list) == obs_mod.CRITIC_OBS_DIM == 162
