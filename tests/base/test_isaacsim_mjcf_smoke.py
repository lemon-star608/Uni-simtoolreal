"""L2 smoke: SimToolReal on the real IsaacSim backend (Isaac/MJCF line).

Small pool (tool_pool_num_per_type=1 -> 12 tool variants) and num_envs=8,
built through the production wiring (Hydra compose of the isaacsim owner
leaf -> BackendAdapter override -> registry.make with
sim_backend="isaacsim").  No mocks: the UniSim IsaacSim worker composes the
mapped entity scene (B form: robot/table/floor/object entity assets
+ the object entity variant binding) and steps PhysX on the GPU.

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
NUM_VARIANTS = 12

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
        cfg = compose(config_name="config", overrides=["task=simtoolreal/isaacsim"])
    adapter = BackendAdapter(cfg, root_dir=str(tmp_path))
    override = adapter.build_task_env_cfg_override()
    override["tool_pool_num_per_type"] = NUM_PER_TYPE
    override["tool_pool_root"] = str(tmp_path / "pool")
    built = registry.make(
        "SimToolReal", sim_backend="isaacsim", env_cfg_override=override, num_envs=NUM_ENVS
    )
    assert {spec.name for spec in built.cfg.scene.entity_assets} == {
        "robot",
        "table",
        "floor",
        "object",
    }
    return built


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    if not _worker_available():
        pytest.skip(
            "IsaacSim worker runtime not installed (UniLab scripts/tools/setup_isaacsim_env.sh)"
        )
    built = _build_env(tmp_path_factory.mktemp("isaacsim_smoke"))
    yield built
    built.close()


def test_reset_obs_contract(env):
    obs, info = env.reset()
    assert isinstance(info, dict)
    assert obs["obs"].shape == (NUM_ENVS, 140)  # OBS_FIELD_SIZES pins
    assert obs["critic"].shape == (NUM_ENVS, 162)
    assert all(np.isfinite(value).all() for value in obs.values())


def test_reset_joints_land_near_home(env):
    """Post-reset joints = home + sampling noise, not the zero pose.

    Same sampling envelope as the mujoco smoke (events.py _reset_robot,
    reset_dof_pos_random_interval=0.1), on the Isaac/MJCF entity scene.
    """
    obs, info = env.reset()
    robot = env.scene["robot"]
    q = np.asarray(robot.data.joint_pos)  # (num_envs, 29), canonical order
    default = np.asarray(robot.data.default_joint_pos)  # (num_envs, 29), home
    limits = np.asarray(robot.data.soft_joint_pos_limits)  # (29, 2)
    lo = default - 0.1 * (default - limits[:, 0])
    hi = default + 0.1 * (limits[:, 1] - default)
    assert (q >= lo - 1e-4).all() and (q <= hi + 1e-4).all()
    assert limits[0, 1] > 2.9  # radian joint ranges, not a degree cage
    assert np.abs(q[:, 0] - (-1.571)).max() < 0.46


def test_step_100_no_nan(env):
    for _ in range(100):
        state = env.step(np.zeros((NUM_ENVS, 29), dtype=np.float32))
    assert all(np.isfinite(value).all() for value in state.obs.values())
    assert np.isfinite(state.reward).all()


def test_variant_assignment_binds_distinct_masses(env):
    """Round-robin assignment over 12 variants; wrench mass table follows it."""
    import simtoolreal_unilab.tasks.simtoolreal.env_cfg as env_cfg_module

    assignment = env_cfg_module.fixed_variant_plan_assignment(env.cfg, NUM_ENVS)
    assert assignment is not None
    assert assignment.shape == (NUM_ENVS,)
    catalog_masses = np.asarray(env.cfg.tool_pool_masses)
    assert catalog_masses.shape == (NUM_VARIANTS,)
    assert len(np.unique(catalog_masses[assignment])) > 1

    term = env.event_manager._mode_class_term_cfgs["interval"][0].func
    assert type(term).__name__ == "SimToolRealWrenchEvent"
    # The wrench masses come from the task-side catalog table (D5), indexed
    # by the same assignment.
    assert term._object_masses is not None
    np.testing.assert_allclose(term._object_masses, catalog_masses[assignment], rtol=1e-6)


def test_wrench_event_measurably_perturbs_object(env):
    """Force the interval wrench to fire and compare against a control phase."""
    env.reset()
    for _ in range(90):  # let the object settle onto the table
        env.step(np.zeros((NUM_ENVS, 29), dtype=np.float32))

    term = env.event_manager._mode_class_term_cfgs["interval"][0].func
    # The production owner YAML disables wrench DR for the 2026-09-30 no-DR
    # baseline.  This test validates the event mechanism itself, so temporarily
    # restore the nonzero upstream scales and restore them before leaving.
    original_force_scale = term._force_scale
    original_torque_scale = term._torque_scale
    term._force_scale = 20.0
    term._torque_scale = 2.0
    hub = env.command_manager.get_term("goal")
    hub.lifted[:] = True
    term._force_prob[:] = 0.0
    term._torque_prob[:] = 0.0
    for _ in range(10):
        env.step(np.zeros((NUM_ENVS, 29), dtype=np.float32))
    control = np.asarray(env.scene["object"].data.root_link_vel_w)[:, :3]
    control = np.linalg.norm(control, axis=1).mean()

    term._force_prob[:] = 1.0
    term._torque_prob[:] = 1.0
    for _ in range(10):
        env.step(np.zeros((NUM_ENVS, 29), dtype=np.float32))
    active = np.asarray(env.scene["object"].data.root_link_vel_w)[:, :3]
    active = np.linalg.norm(active, axis=1).mean()

    assert active > 10 * max(control, 1e-3), (
        f"wrench event had no measurable effect: control={control}, active={active}"
    )
    term._force_scale = original_force_scale
    term._torque_scale = original_torque_scale
    env.reset()
