"""Config-adapter forwarding gate for the fast-iteration no-DR profile."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir
from unilab.base.config_adapter import BackendAdapter

import simtoolreal_unilab.tasks  # noqa: F401  -- register task/resolvers
from simtoolreal_unilab.config import CONF_ROOT


@pytest.mark.parametrize("backend", ["mujoco", "isaacsim"])
def test_no_dr_profile_reaches_task_env_override(tmp_path, backend):
    """The owner YAML values must survive Hydra and BackendAdapter assembly."""

    with initialize_config_dir(str(CONF_ROOT), version_base=None):
        cfg = compose(config_name="config", overrides=[f"task=simtoolreal/{backend}"])

    override = BackendAdapter(cfg, root_dir=str(tmp_path)).build_task_env_cfg_override()
    policy = override["observations"]["policy"]["terms"]["simtoolreal"]["params"]
    action = override["actions"]["joint_pos"]
    wrench = override["events"]["wrench_dr"]["params"]

    assert policy["use_object_state_delay_noise"] is False
    assert policy["object_state_delay_max"] == 0
    assert policy["object_state_xyz_noise_std"] == 0.0
    assert policy["object_state_rotation_noise_degrees"] == 0.0
    assert policy["joint_velocity_obs_noise_std"] == 0.0
    assert policy["use_obs_delay"] is False
    assert policy["obs_delay_max"] == 0

    assert action["arm_moving_average"] == 1.0
    assert action["hand_moving_average"] == 1.0
    assert action["use_action_delay"] is False
    assert action["action_delay_max"] == 0

    assert wrench["force_scale"] == 0.0
    assert wrench["torque_scale"] == 0.0

    goal = override["commands"]["goal"]
    assert goal["object_scale_noise_multiplier_range"] == [1.0, 1.0]

    if backend == "isaacsim":
        assert cfg.env.isaacsim_max_depenetration_velocity == 1000.0
