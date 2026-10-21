from __future__ import annotations

import numpy as np

from simtoolreal_unilab.tasks.dextoolbench.catalog import find_task, load_trajectory
from simtoolreal_unilab.tasks.dextoolbench.env_cfg import (
    build_dextoolbench_env_cfg_override,
)


def _build_override() -> dict:
    return build_dextoolbench_env_cfg_override("hammer", "claw_hammer", "swing_down")


def test_override_pins_fixed_scene_and_no_dr(tmp_path, monkeypatch) -> None:
    generated_root = tmp_path / "must_not_exist"
    monkeypatch.setenv("SIMTOOLREAL_GENERATED_ASSETS_ROOT", str(generated_root))
    override = _build_override()

    assert override["tool_pool_enabled"] is False
    assert "entity_assets" not in override["scene"]
    assert "entity_variant" not in override["scene"]
    assert override["scene"]["model_file"].endswith(
        "dextoolbench/scenes/hammer/claw_hammer/swing_down.xml"
    )
    assert override["events"]["reset_robot"]["params"]["dof_pos_noise_arm"] == 0.0
    assert override["events"]["reset_robot"]["params"]["dof_pos_noise_hand"] == 0.0
    assert override["events"]["reset_robot"]["params"]["dof_vel_noise"] == 0.0
    assert override["events"]["wrench_dr"]["params"]["force_scale"] == 0.0
    assert override["events"]["wrench_dr"]["params"]["torque_scale"] == 0.0
    assert override["events"]["wrench_dr"]["params"]["force_prob_range"] == (0.0001, 0.0001)
    assert override["events"]["wrench_dr"]["params"]["torque_prob_range"] == (0.0001, 0.0001)
    assert override["terminations"]["success_tracker"]["params"]["success_steps"] == 1
    assert override["terminations"]["max_consecutive_successes"]["params"][
        "max_consecutive_successes"
    ] == len(override["fixed_goal_states"])
    assert not generated_root.exists()


def test_registry_builds_single_env_and_keeps_obs_action_contract(tmp_path, monkeypatch) -> None:
    from unilab.base import registry

    import simtoolreal_unilab.tasks  # noqa: F401

    monkeypatch.setenv("SIMTOOLREAL_GENERATED_ASSETS_ROOT", str(tmp_path / "must_not_exist"))
    env = registry.make(
        "DexToolBench",
        sim_backend="mujoco",
        env_cfg_override=_build_override(),
        num_envs=1,
    )
    try:
        obs, _info = env.reset()
        assert obs["obs"].shape == (1, 140)
        assert obs["critic"].shape == (1, 162)
        state = env.step(np.zeros((1, 29), dtype=np.float32))
        assert state.obs["obs"].shape == (1, 140)
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.reward).all()

        robot = env.scene["robot"]
        expected_arm = np.array([-1.571, 1.3964670740, 0.0, 1.5505329252, 0.0, 1.485, 1.308])
        np.testing.assert_allclose(robot.data.default_joint_pos[0, :7], expected_arm, atol=1e-5)
        goal_hub = env.command_manager.get_term("goal")
        np.testing.assert_allclose(goal_hub.object_scale_per_env[0], (2.5, 0.5625, 0.375))
        trajectory = load_trajectory(find_task("hammer", "claw_hammer", "swing_down"))
        np.testing.assert_allclose(goal_hub.goal[0], trajectory["goals_wxyz"][0], atol=1e-6)
        goal_hub._advance_goal(np.array([0], dtype=np.intp))  # noqa: SLF001
        np.testing.assert_allclose(goal_hub.goal[0], trajectory["goals_wxyz"][1], atol=1e-6)
        assert goal_hub.goal_count[0] == 1

        object_pose = env.scene["object"].data.root_link_pose_w[0].copy()
        object_pose[:3] -= env.scene.env_origins[0, :3]
        np.testing.assert_allclose(object_pose[:2], trajectory["start_pose_wxyz"][:2], atol=1e-5)
        assert object_pose[2] < trajectory["start_pose_wxyz"][2]
        assert object_pose[2] > trajectory["start_pose_wxyz"][2] - 0.03
    finally:
        env.close()
    assert not (tmp_path / "must_not_exist").exists()
