"""L2 smoke: SimToolReal on the real MuJoCo backend (roadmap M-J2/M-J5).

Small pool (tool_pool_num_per_type=1 -> 12 tool variants) and num_envs=8,
built through the production wiring (Hydra compose of the mujoco owner leaf
-> BackendAdapter override -> registry.make with sim_backend="mujoco").  No
mocks: the UniSim MuJoCo backend composes the entity scene (B form, M-J5:
robot/table/floor/object entity assets + the object entity variant
binding), compiles the variant pack, and steps mjbatch on CPU.

History: this suite was xfail-blocked by unilabsim/mjbatch_uni#33 (VariantPack
treated the compiler-derived ``body_simple``/``dof_simplenum`` flags as shared
parameters, rejecting mass-varying same-layout pools).  The fix shipped in
mjbatch-uni 0.2.3 and the pin moved on 2026-09-21; the xfail is retired and
these tests must pass outright.
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
        cfg = compose(config_name="config", overrides=["task=simtoolreal/mujoco"])
    adapter = BackendAdapter(cfg, root_dir=str(tmp_path))
    override = adapter.build_task_env_cfg_override()
    override["tool_pool_num_per_type"] = NUM_PER_TYPE
    override["tool_pool_root"] = str(tmp_path / "pool")
    return registry.make(
        "SimToolReal", sim_backend="mujoco", env_cfg_override=override, num_envs=NUM_ENVS
    )


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    built = _build_env(tmp_path_factory.mktemp("mujoco_smoke"))
    yield built
    built.close()


def test_reset_obs_contract(env):
    obs, info = env.reset()
    assert isinstance(info, dict)
    assert obs["obs"].shape == (NUM_ENVS, 140)  # OBS_FIELD_SIZES pins
    assert obs["critic"].shape == (NUM_ENVS, 162)
    assert all(np.isfinite(value).all() for value in obs.values())


def test_headless_scene_has_four_entities(env):
    """GoalViz is intentionally absent from the production B-form scene."""
    assert {spec.name for spec in env.cfg.scene.entity_assets} == {
        "robot",
        "table",
        "floor",
        "object",
    }
    assert "goalviz" not in env.cfg.scene.entities


def test_reset_joints_land_near_home(env):
    """Post-reset joints = home + sampling noise, not the zero pose.

    Regression gate for the 2026-09-17 angle="radian" finding: with the
    range cage the reset term clipped home into ±0.05 rad and the arm started
    vertical every episode.  The asserted band is the term's exact sampling
    envelope: default + 0.1 * (limit - default) per side (events.py
    _reset_robot, reset_dof_pos_random_interval=0.1).
    """
    obs, info = env.reset()
    robot = env.scene["robot"]
    q = np.asarray(robot.data.joint_pos)  # (num_envs, 29), canonical order
    default = np.asarray(robot.data.default_joint_pos)  # (num_envs, 29), home
    limits = np.asarray(robot.data.soft_joint_pos_limits)  # (29, 2)
    lo = default - 0.1 * (default - limits[:, 0])
    hi = default + 0.1 * (limits[:, 1] - default)
    assert (q >= lo - 1e-4).all() and (q <= hi + 1e-4).all()
    # Cage signature guard: iiwa14_joint_1's home is -1.571 rad and its real
    # range is ±2.967 rad; the caged model's limit was ±0.0518 rad.  0.46 is
    # the joint's sampling envelope half-width (0.1 * (2.967 + 1.571)) + slack.
    assert limits[0, 1] > 2.9
    assert np.abs(q[:, 0] - (-1.571)).max() < 0.46


def test_step_100_no_nan(env):
    for _ in range(100):
        state = env.step(np.zeros((NUM_ENVS, 29), dtype=np.float32))
    assert all(np.isfinite(value).all() for value in state.obs.values())
    assert np.isfinite(state.reward).all()


def test_variant_assignment_binds_distinct_masses(env):
    """Round-robin assignment over 12 variants; per-env masses match catalog."""
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


def test_table_z_randomization_restored(env):
    """M-J5: the table is a kinematic entity again — z randomization is back.

    A-form composed scenes baked the table as a static world body (the M-J2
    known divergence); the entity reset channel samples table z in
    table_reset_z ± table_reset_z_range (0.38 ± 0.01) per env at reset.  The
    kinematic root is static between resets, so the pose readable after the
    previous tests' steps is still the sampled value — no extra reset here
    (keeping the shared env's rng stream stable for the wrench test).
    """
    table_z = np.asarray(env.scene["table"].data.root_link_pos_w)[:, 2]
    assert np.all(table_z >= 0.37 - 1e-6) and np.all(table_z <= 0.39 + 1e-6)
    assert np.unique(np.round(table_z, 6)).size > 1, "table z must vary across envs"


def test_wrench_event_measurably_perturbs_object(env):
    """Force the interval wrench to fire and compare against a control phase."""
    env.reset()
    backend = env._backend
    for _ in range(90):  # let the object settle onto the table
        env.step(np.zeros((NUM_ENVS, 29), dtype=np.float32))

    term = env.event_manager._mode_class_term_cfgs["interval"][0].func
    assert type(term).__name__ == "SimToolRealWrenchEvent"
    # The wrench masses come from the task-side catalog table (D5).
    assert term._object_masses is not None
    assert term._object_masses.shape == (NUM_ENVS,)

    hub = env.command_manager.get_term("goal")
    hub.lifted[:] = True
    term._force_prob[:] = 0.0
    term._torque_prob[:] = 0.0
    for _ in range(10):
        env.step(np.zeros((NUM_ENVS, 29), dtype=np.float32))
    control = np.linalg.norm(backend._qvel_view[:, 29:32], axis=1).mean()

    term._force_prob[:] = 1.0
    term._torque_prob[:] = 1.0
    for _ in range(10):
        env.step(np.zeros((NUM_ENVS, 29), dtype=np.float32))
    active = np.linalg.norm(backend._qvel_view[:, 29:32], axis=1).mean()

    assert active > 100 * max(control, 1e-9), (
        f"wrench event had no measurable effect: control={control}, active={active}"
    )
    env.reset()
