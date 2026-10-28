"""Real MuJoCo runtime contract for SimToolReal on UniLab 1.5."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

NUM_ENVS = 8
NUM_PER_TYPE = 1
NUM_VARIANTS = 12


def _build_env(tmp_path):
    from hydra import compose, initialize_config_dir
    from unilab.base import registry
    from unilab.base.config_adapter import EnvConfigAdapter

    import simtoolreal_unilab.tasks  # noqa: F401
    from simtoolreal_unilab.config import CONF_ROOT

    with initialize_config_dir(str(CONF_ROOT), version_base=None):
        cfg = compose(config_name="config", overrides=["task=simtoolreal/mujoco"])
    adapter = EnvConfigAdapter(cfg, root_dir=str(tmp_path))
    override = adapter.build_task_env_cfg_override()
    override["tool_pool_num_per_type"] = NUM_PER_TYPE
    override["tool_pool_root"] = str(tmp_path / "pool")
    return registry.make(
        "SimToolReal", sim_backend="mujoco", env_cfg_override=override, num_envs=NUM_ENVS
    )


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    built = _build_env(tmp_path_factory.mktemp("mujoco_runtime"))
    yield built
    built.close()


def test_reset_and_step_obs_contract(env):
    obs, info = env.reset()
    assert isinstance(info, dict)
    assert obs["obs"].shape == (NUM_ENVS, 140)
    assert obs["critic"].shape == (NUM_ENVS, 162)
    assert all(bool(torch.isfinite(value).all()) for value in obs.values())
    state = env.step(torch.zeros((NUM_ENVS, 29), dtype=torch.float32, device=env.device))
    assert state.obs["obs"].shape == (NUM_ENVS, 140)
    assert bool(torch.isfinite(state.reward).all())


def test_policy_joint_positions_remain_normalized(env):
    obs, _info = env.reset()
    robot = env.scene["robot"]
    raw = torch.from_numpy(env._backend._qpos_view[:, :29]).to(
        device=env.device, dtype=torch.float32
    )
    limits = robot.data.soft_joint_pos_limits(env.device)
    expected = 2.0 * (raw - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 1.0
    torch.testing.assert_close(obs["obs"][:, :29], expected, rtol=0.0, atol=1e-6)


def test_bootstrap_reset_starts_every_episode_counter_at_zero(env):
    env.reset()
    assert torch.equal(env._state.info["steps"], torch.zeros_like(env.episode_length_buf))
    assert torch.equal(env.episode_length_buf, torch.zeros_like(env.episode_length_buf))


def test_packed_body_reads_match_final_mujoco_kinematics(env):
    env.reset()
    state = env.step(torch.zeros((NUM_ENVS, 29), dtype=torch.float32, device=env.device))
    assert bool(torch.isfinite(state.obs["obs"]).all())

    backend = env._backend
    scratch = mujoco.MjData(backend._model)
    scratch.qpos[:] = backend._qpos_view[0]
    mujoco.mj_forward(backend._model, scratch)
    origin = env.scene.env_origins(env.device)[0]
    robot_name = "robot/iiwa14_link_7"
    object_name = "object/object_root"
    robot_id = mujoco.mj_name2id(backend._model, mujoco.mjtObj.mjOBJ_BODY, robot_name)
    object_id = mujoco.mj_name2id(backend._model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    robot_pos = (
        env.scene.read_plan.body_tensor_view(env.scene["robot"], (robot_name,)).pos_w[0, 0] - origin
    )
    object_pos = (
        env.scene.read_plan.body_tensor_view(env.scene["object"], (object_name,)).pos_w[0, 0]
        - origin
    )
    torch.testing.assert_close(
        robot_pos,
        torch.tensor(scratch.xpos[robot_id], dtype=torch.float32, device=env.device),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        object_pos,
        torch.tensor(scratch.xpos[object_id], dtype=torch.float32, device=env.device),
        rtol=0.0,
        atol=1e-6,
    )


def test_reset_joints_land_in_sampling_envelope(env):
    env.reset()
    robot = env.scene["robot"]
    joints = env.scene.read_plan.joint_tensor_view(robot)
    q = joints.joint_pos.detach().cpu().numpy()
    default = robot.data.default_joint_pos(env.device)[0].detach().cpu().numpy()
    limits = robot.data.soft_joint_pos_limits(env.device).detach().cpu().numpy()
    lo = default - 0.1 * (default - limits[:, 0])
    hi = default + 0.1 * (limits[:, 1] - default)
    assert (q >= lo - 1e-4).all() and (q <= hi + 1e-4).all()
    assert limits[0, 1] > 2.9
    assert np.abs(q[:, 0] - (-np.pi / 2)).max() < 0.46


def test_variant_assignment_binds_distinct_masses(env):
    from simtoolreal_unilab.tasks.simtoolreal.env_cfg import fixed_variant_plan_assignment

    assignment = fixed_variant_plan_assignment(env.cfg, NUM_ENVS)
    assert assignment is not None and assignment.shape == (NUM_ENVS,)
    catalog_masses = np.asarray(env.cfg.tool_pool_masses)
    assert catalog_masses.shape == (NUM_VARIANTS,)
    backend = env._backend
    object_id = mujoco.mj_name2id(backend._model, mujoco.mjtObj.mjOBJ_BODY, "object/object_root")
    assert object_id > 0
    per_env = np.asarray(backend._base_body_mass)[:, object_id]
    np.testing.assert_allclose(per_env, catalog_masses[assignment], rtol=1e-5)
    assert len(np.unique(catalog_masses[assignment])) > 1


def test_table_z_randomization(env):
    env.reset()
    table_z = np.asarray(env.scene["table"].read_mocap_pose())[:, 2]
    assert np.all(table_z >= 0.37 - 1e-6)
    assert np.all(table_z <= 0.39 + 1e-6)
    assert np.unique(np.round(table_z, 6)).size > 1


def test_wrench_event_measurably_perturbs_object(env):
    env.reset()
    backend = env._backend
    for _ in range(90):
        env.step(torch.zeros((NUM_ENVS, 29), dtype=torch.float32, device=env.device))

    term = env.event_manager._mode_class_term_cfgs["interval"][0].func
    assert type(term).__name__ == "SimToolRealWrenchEvent"
    original_force_scale = term._force_scale
    original_torque_scale = term._torque_scale
    term._force_scale = 20.0
    term._torque_scale = 2.0
    assert term._object_masses is not None
    assert term._object_masses.shape == (NUM_ENVS,)

    env.command_manager.get_term("goal").lifted[:] = True
    term._force_prob.zero_()
    term._torque_prob.zero_()
    for _ in range(10):
        env.step(torch.zeros((NUM_ENVS, 29), dtype=torch.float32, device=env.device))
    control = np.linalg.norm(backend._qvel_view[:, 29:32], axis=1).mean()

    term._force_prob.fill_(1.0)
    term._torque_prob.fill_(1.0)
    for _ in range(10):
        env.step(torch.zeros((NUM_ENVS, 29), dtype=torch.float32, device=env.device))
    active = np.linalg.norm(backend._qvel_view[:, 29:32], axis=1).mean()

    assert active > 100 * max(control, 1e-9)
    term._force_scale = original_force_scale
    term._torque_scale = original_torque_scale
    env.reset()
