"""L2 smoke: SimToolReal on the real IsaacSim backend (Isaac/MJCF line).

Small pool (tool_pool_num_per_type=1 -> 12 tool variants) and num_envs=8,
built through the production wiring (Hydra compose of the isaacsim owner
leaf -> EnvConfigAdapter override -> registry.make with
sim_backend="isaacsim").  No mocks: the UniSim IsaacSim worker composes the
mapped entity scene (B form: robot/table/floor/object entity assets
+ the object entity variant binding) and steps PhysX on the GPU.

Requires the external worker runtime (UniLab's official installer,
``scripts/tools/setup_isaacsim_env.sh``) and a GPU; skipped on hosts
without them (the ``isaacsim`` marker).
"""

from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest
import torch

NUM_ENVS = 8
NUM_PER_TYPE = 1  # 12 variants
NUM_VARIANTS = 12

pytestmark = pytest.mark.isaacsim


def _worker_available() -> bool:
    if os.environ.get("UNISIM_ISAACSIM_PYTHON") or os.environ.get("UNILAB_ISAACSIM_PYTHON"):
        return True
    from simtoolreal_unilab._isaacsim_worker import CONVENTIONAL_WORKER_PYTHON

    if not CONVENTIONAL_WORKER_PYTHON.is_file():
        return False
    probe = subprocess.run(
        [
            str(CONVENTIONAL_WORKER_PYTHON),
            "-c",
            "import importlib.metadata as m; print(m.version('unisim-core'))",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return probe.returncode == 0 and probe.stdout.strip().startswith("2.1.")


def _build_env(tmp_path, *, tool_pool_enabled: bool):
    from hydra import compose, initialize_config_dir
    from unilab.base import registry
    from unilab.base.config_adapter import EnvConfigAdapter

    import simtoolreal_unilab.tasks  # noqa: F401  (registers env + resolvers)
    from simtoolreal_unilab._isaacsim_worker import ensure_worker_runtime
    from simtoolreal_unilab.config import CONF_ROOT

    ensure_worker_runtime()
    with initialize_config_dir(str(CONF_ROOT), version_base=None):
        cfg = compose(config_name="config", overrides=["task=simtoolreal/isaacsim"])
    adapter = EnvConfigAdapter(cfg, root_dir=str(tmp_path))
    override = adapter.build_task_env_cfg_override()
    override["tool_pool_enabled"] = tool_pool_enabled
    if tool_pool_enabled:
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
def basic_env(tmp_path_factory):
    if not _worker_available():
        pytest.skip(
            "IsaacSim worker runtime not installed (UniLab scripts/tools/setup_isaacsim_env.sh)"
        )
    built = _build_env(
        tmp_path_factory.mktemp("isaacsim_smoke_no_variants"), tool_pool_enabled=False
    )
    yield built
    built.close()


@pytest.fixture(scope="module")
def variant_env(tmp_path_factory):
    if not _worker_available():
        pytest.skip(
            "IsaacSim worker runtime not installed (UniLab scripts/tools/setup_isaacsim_env.sh)"
        )
    built = _build_env(tmp_path_factory.mktemp("isaacsim_smoke_variants"), tool_pool_enabled=True)
    yield built
    built.close()


def test_reset_obs_contract(basic_env):
    obs, info = basic_env.reset()
    assert isinstance(info, dict)
    assert obs["obs"].shape == (NUM_ENVS, 140)  # OBS_FIELD_SIZES pins
    assert obs["critic"].shape == (NUM_ENVS, 162)
    assert all(bool(torch.isfinite(value).all()) for value in obs.values())


def test_reset_joints_land_near_home(basic_env):
    """Post-reset joints = home + sampling noise, not the zero pose.

    Same sampling envelope as the mujoco smoke (events.py _reset_robot,
    reset_dof_pos_random_interval=0.1), on the Isaac/MJCF entity scene.
    """
    obs, info = basic_env.reset()
    robot = basic_env.scene["robot"]
    q = (
        basic_env.scene.read_plan.joint_tensor_view(robot).joint_pos.detach().cpu().numpy()
    )  # (num_envs, 29), canonical order
    default = (
        robot.data.default_joint_pos(basic_env.device)[0].detach().cpu().numpy()
    )  # (29,), home
    limits = robot.data.soft_joint_pos_limits(basic_env.device).detach().cpu().numpy()
    lo = default - 0.1 * (default - limits[:, 0])
    hi = default + 0.1 * (limits[:, 1] - default)
    assert (q >= lo - 1e-4).all() and (q <= hi + 1e-4).all()
    assert limits[0, 1] > 2.9  # radian joint ranges, not a degree cage
    assert np.abs(q[:, 0] - (-1.571)).max() < 0.46


def test_step_100_no_nan(basic_env):
    for _ in range(100):
        state = basic_env.step(
            torch.zeros((NUM_ENVS, 29), dtype=torch.float32, device=basic_env.device)
        )
    assert all(bool(torch.isfinite(value).all()) for value in state.obs.values())
    assert bool(torch.isfinite(state.reward).all())


def test_variant_assignment_binds_distinct_masses(variant_env):
    """Round-robin assignment over 12 variants; wrench mass table follows it."""
    import simtoolreal_unilab.tasks.simtoolreal.env_cfg as env_cfg_module

    assignment = env_cfg_module.fixed_variant_plan_assignment(variant_env.cfg, NUM_ENVS)
    assert assignment is not None
    assert assignment.shape == (NUM_ENVS,)
    catalog_masses = np.asarray(variant_env.cfg.tool_pool_masses)
    assert catalog_masses.shape == (NUM_VARIANTS,)
    assert len(np.unique(catalog_masses[assignment])) > 1

    term = variant_env.event_manager._mode_class_term_cfgs["interval"][0].func
    assert type(term).__name__ == "SimToolRealWrenchEvent"
    # The wrench masses come from the task-side catalog table (D5), indexed
    # by the same assignment.
    assert term._object_masses is not None
    np.testing.assert_allclose(
        term._object_masses.detach().cpu().numpy(),
        catalog_masses[assignment],
        rtol=1e-6,
    )


@pytest.mark.xfail(
    reason="UniSim 2.1 IsaacSim CUDA IPC does not implement stepping with fixed variants",
    strict=False,
)
def test_fixed_variant_step_is_supported(variant_env):
    variant_env.step(torch.zeros((NUM_ENVS, 29), dtype=torch.float32, device=variant_env.device))


def test_wrench_event_measurably_perturbs_object(basic_env):
    """Force the interval wrench to fire and compare against a control phase."""
    previous_autoreset = basic_env._autoreset
    basic_env.set_autoreset(False)
    basic_env.reset()
    # Keep the object awake: IsaacLab does not wake a sleeping rigid body merely
    # because an external wrench tensor is nonzero. A fresh reset provides the
    # correct active-state test for wrench dispatch.
    for _ in range(1):
        basic_env.step(torch.zeros((NUM_ENVS, 29), dtype=torch.float32, device=basic_env.device))

    term = basic_env.event_manager._mode_class_term_cfgs["interval"][0].func
    # The production owner YAML disables wrench DR for the 2026-09-30 no-DR
    # baseline.  This test validates the event mechanism itself, so temporarily
    # restore the nonzero upstream scales and restore them before leaving.
    original_force_scale = term._force_scale
    original_torque_scale = term._torque_scale
    # Amplify only the diagnostic wrench so its impulse is distinguishable
    # from IsaacSim's post-reset contact transient.
    term._force_scale = 1_000_000.0
    term._torque_scale = 100_000.0
    hub = basic_env.command_manager.get_term("goal")
    hub.lifted[:] = True
    term._force_prob.zero_()
    term._torque_prob.zero_()
    for _ in range(1):
        basic_env.step(torch.zeros((NUM_ENVS, 29), dtype=torch.float32, device=basic_env.device))
    object_name = hub.object_body_name
    object_entity = basic_env.scene["object"]
    control = (
        basic_env.scene.read_plan.body_tensor_view(object_entity, (object_name,))
        .lin_vel_w[:, 0, :]
        .detach()
        .cpu()
        .numpy()
    )
    control = np.linalg.norm(control, axis=1).mean()

    term._force_prob.fill_(1.0)
    term._torque_prob.fill_(1.0)
    # Exercise the task term's formal dispatch boundary directly. Isaac's
    # interval timer is sampled in float64 and can land a fraction of a control
    # step after the boundary; that scheduler precision is independent of
    # whether the term maps its sampled wrench into the backend.
    term(basic_env, torch.arange(NUM_ENVS, dtype=torch.int64, device=basic_env.device))
    basic_env.step(torch.zeros((NUM_ENVS, 29), dtype=torch.float32, device=basic_env.device))
    active = (
        basic_env.scene.read_plan.body_tensor_view(object_entity, (object_name,))
        .lin_vel_w[:, 0, :]
        .detach()
        .cpu()
        .numpy()
    )
    active = np.linalg.norm(active, axis=1).mean()

    assert active > control + 0.02, (
        f"wrench event had no measurable effect: control={control}, active={active}"
    )
    term._force_scale = original_force_scale
    term._torque_scale = original_torque_scale
    basic_env.set_autoreset(previous_autoreset)
    basic_env.reset()
