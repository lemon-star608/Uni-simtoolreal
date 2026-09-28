"""Step-1.4b-1 tests for the SimToolReal command hub (goal advancement).

Fake-backend pattern follows tests/tasks/test_simtoolreal_actions_observations.py:
a real ``EntityScene`` over a fake ``SimBackend`` — here with a floating
``goalviz_root`` (root-state layout + ``set_state`` capture) so the GoalViz
commit path is exercised through the production ``Entity`` facade and the
base-owned ``ResetStateTransaction``, mirroring how ``ManagerBasedRlEnv``
wraps ``command_manager.compute``/``reset`` in a transaction scope
(manager_based_rl_env.py:533-537, 584-591).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from unilab.base.entity import EntityCfg, EntityScene
from unilab.base.reset_state import ResetStateTransaction
from unilab.managers._types import ManagerBasedRlEnv
from unilab.utils.rotation import np_quat_error_magnitude_batched
from unisim.backend.base import BackendRootStateLayout, SimBackend

from simtoolreal_unilab.tasks.simtoolreal.commands import (
    SimToolRealCommandCfg,
    sample_absolute_goal_pose,
    sample_delta_goal_pose,
)
from simtoolreal_unilab.tasks.simtoolreal.config import (
    FINGERTIP_LINK_NAMES,
    PALM_BODY_NAME,
)

NUM_ENVS = 8
STEP_DT = 1.0 / 60.0

ROBOT_BODY_NAMES = (PALM_BODY_NAME, *FINGERTIP_LINK_NAMES)
GOALVIZ_BODY_NAME = "goalviz_root"
ALL_BODY_NAMES = (*ROBOT_BODY_NAMES, GOALVIZ_BODY_NAME)
GOALVIZ_BODY_ID = ALL_BODY_NAMES.index(GOALVIZ_BODY_NAME)

# Backend generalized state: a single floating root (the goalviz body).
_QPOS_WIDTH = 7
_QVEL_WIDTH = 6


class _FakeBackend:
    backend_type = "fake"
    num_envs = NUM_ENVS
    num_actuators = 0

    def __init__(self) -> None:
        self.body_pos = np.zeros((NUM_ENVS, len(ALL_BODY_NAMES), 3), dtype=np.float32)
        self.body_quat = np.zeros((NUM_ENVS, len(ALL_BODY_NAMES), 4), dtype=np.float32)
        self.body_quat[..., 0] = 1.0
        self.set_state_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # --- bodies ---
    def get_body_ids(self, names) -> np.ndarray:
        return np.asarray([ALL_BODY_NAMES.index(n) for n in names], dtype=np.int32)

    # --- joints (none; Entity init still validates the empty layouts) ---
    def get_dof_pos(self) -> np.ndarray:
        return np.zeros((NUM_ENVS, 0), dtype=np.float32)

    def get_dof_vel(self) -> np.ndarray:
        return np.zeros((NUM_ENVS, 0), dtype=np.float32)

    def get_body_pos_w(self, ids) -> np.ndarray:
        return self.body_pos[:, np.asarray(ids)]

    def get_body_quat_w(self, ids) -> np.ndarray:
        return self.body_quat[:, np.asarray(ids)]

    def get_body_lin_vel_w(self, ids) -> np.ndarray:
        return np.zeros((NUM_ENVS, len(ids), 3), dtype=np.float32)

    def get_body_ang_vel_w(self, ids) -> np.ndarray:
        return np.zeros((NUM_ENVS, len(ids), 3), dtype=np.float32)

    def get_body_lin_vel_b(self, ids) -> np.ndarray:
        return np.zeros((NUM_ENVS, len(ids), 3), dtype=np.float32)

    def get_body_ang_vel_b(self, ids) -> np.ndarray:
        return np.zeros((NUM_ENVS, len(ids), 3), dtype=np.float32)

    # --- root-state layout / reset transaction ---
    def get_root_state_layout(self, root_body_name: str) -> BackendRootStateLayout:
        if root_body_name != GOALVIZ_BODY_NAME:
            raise ValueError(f"unknown root body {root_body_name}")
        return BackendRootStateLayout(
            qpos_indices=tuple(range(_QPOS_WIDTH)),
            qvel_indices=tuple(range(_QVEL_WIDTH)),
        )

    def get_default_qpos(self) -> np.ndarray:
        return np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_init_qvel(self) -> np.ndarray:
        return np.zeros(_QVEL_WIDTH, dtype=np.float32)

    def set_state(self, env_ids, qpos, qvel, randomization=None) -> dict:
        self.set_state_calls.append((np.asarray(env_ids).copy(), qpos.copy(), qvel.copy()))
        # The committed pose is visible to later reads, like a real sim.
        self.body_pos[env_ids, GOALVIZ_BODY_ID] = qpos[:, 0:3]
        self.body_quat[env_ids, GOALVIZ_BODY_ID] = qpos[:, 3:7]
        return {"timing": {"set_state_ms": 0.0}}


def _make_env(seed: int = 7, with_goalviz: bool = True, **cfg_overrides):
    backend = _FakeBackend()
    transaction = ResetStateTransaction(cast(SimBackend, backend))
    entities = {"robot": EntityCfg(body_names=ROBOT_BODY_NAMES)}
    if with_goalviz:
        entities["goalviz"] = EntityCfg(root_body_name=GOALVIZ_BODY_NAME)
    scene = EntityScene(entities, cast(SimBackend, backend), reset_state=transaction)
    env = SimpleNamespace(
        num_envs=NUM_ENVS,
        rng=np.random.default_rng(seed),
        step_dt=STEP_DT,
        scene=scene,
        episode_length_buf=np.zeros(NUM_ENVS, dtype=np.int64),
        common_step_counter=0,
    )
    # Mirror of ManagerBasedRlEnv.set_episode_length_buf: the env-owned
    # info["steps"] mirror is zeroed together with episode_length_buf.
    steps_mirror = np.zeros(NUM_ENVS, dtype=np.int64)

    def set_episode_length_buf(values: np.ndarray) -> None:
        np.copyto(env.episode_length_buf, values)
        np.copyto(steps_mirror, values)

    env.set_episode_length_buf = set_episode_length_buf
    env.steps_mirror = steps_mirror
    cfg = SimToolRealCommandCfg(resampling_time_range=(1e6, 1e6), **cfg_overrides)
    hub = cfg.build(cast(ManagerBasedRlEnv, env))
    env.command_manager = SimpleNamespace(get_term=lambda name: hub)
    return env, backend, transaction, hub


def _reset_all(env, transaction, hub):
    ids = np.arange(NUM_ENVS, dtype=np.int64)
    with transaction.scoped(ids):
        hub.reset(ids)
    return ids


def _compute_step(env, transaction, hub):
    """Per-step command compute inside the env's transaction scope."""
    env.common_step_counter += 1
    with transaction.scoped(np.arange(NUM_ENVS, dtype=np.int64)):
        hub.compute(dt=STEP_DT, env_ids=None)


# ----------------------------------------------------------------------------
# Pure samplers (goal_sampling.py:21-62)
# ----------------------------------------------------------------------------


def test_absolute_goal_sampling_bounds_and_unit_quat():
    rng = np.random.default_rng(0)
    mins, maxs = (-0.35, -0.2, 0.6), (0.35, 0.2, 0.95)
    pos, quat = sample_absolute_goal_pose(rng, mins, maxs, scale=1.0, n_envs=256)
    assert pos.shape == (256, 3)
    assert np.all(pos >= np.asarray(mins) - 1e-6)
    assert np.all(pos <= np.asarray(maxs) + 1e-6)
    np.testing.assert_allclose(np.linalg.norm(quat, axis=-1), 1.0, atol=1e-6)


def test_absolute_goal_sampling_region_scale_shrinks_about_center():
    rng = np.random.default_rng(1)
    mins, maxs = (-0.35, -0.2, 0.6), (0.35, 0.2, 0.95)
    center = 0.5 * (np.asarray(mins) + np.asarray(maxs))
    half = 0.5 * (np.asarray(maxs) - np.asarray(mins)) * 0.5
    pos, _ = sample_absolute_goal_pose(rng, mins, maxs, scale=0.5, n_envs=256)
    assert np.all(pos >= center - half - 1e-6)
    assert np.all(pos <= center + half + 1e-6)


def test_delta_goal_sampling_walk_bounds_and_angle_cap():
    rng = np.random.default_rng(2)
    mins, maxs = (-0.35, -0.2, 0.6), (0.35, 0.2, 0.95)
    prev_pos = np.zeros((256, 3), dtype=np.float32)  # outside the volume: clamp pulls in
    prev_quat = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (256, 1))
    new_pos, new_quat = sample_delta_goal_pose(
        rng,
        prev_pos=prev_pos,
        prev_quat_wxyz=prev_quat,
        delta_distance=0.1,
        delta_rotation_degrees=90.0,
        mins=mins,
        maxs=maxs,
        scale=1.0,
    )
    # Position clamped into the volume (goal_sampling.py:53).
    assert np.all(new_pos >= np.asarray(mins) - 1e-6)
    assert np.all(new_pos <= np.asarray(maxs) + 1e-6)
    # In-volume starts move at most delta_distance per axis.
    center = (0.5 * (np.asarray(mins) + np.asarray(maxs))).astype(np.float32)
    prev_in = np.tile(center, (256, 1))
    stepped, _ = sample_delta_goal_pose(
        rng,
        prev_pos=prev_in,
        prev_quat_wxyz=prev_quat,
        delta_distance=0.1,
        delta_rotation_degrees=90.0,
        mins=mins,
        maxs=maxs,
        scale=1.0,
    )
    assert np.all(np.abs(stepped - prev_in) <= 0.1 + 1e-6)
    # Rotation perturbation is capped at delta_rotation_degrees.
    angle = np_quat_error_magnitude_batched(new_quat, prev_quat)
    assert np.all(angle <= np.radians(90.0) + 1e-4)
    np.testing.assert_allclose(np.linalg.norm(new_quat, axis=-1), 1.0, atol=1e-6)


# ----------------------------------------------------------------------------
# Reset-path sampling (reset_utils.py:388)
# ----------------------------------------------------------------------------


def test_reset_samples_absolute_goal_and_clears_episode_state():
    env, backend, transaction, hub = _make_env(goal_sampling_type="delta")
    ids = _reset_all(env, transaction, hub)
    mins = np.asarray(hub.cfg.target_volume_mins)
    maxs = np.asarray(hub.cfg.target_volume_maxs)
    assert np.all(hub.goal[:, :3] >= mins - 1e-6)
    assert np.all(hub.goal[:, :3] <= maxs + 1e-6)
    np.testing.assert_allclose(np.linalg.norm(hub.goal[:, 3:], axis=-1), 1.0, atol=1e-6)
    # The committed GoalViz pose matches the hub record (env origins are zero).
    assert len(backend.set_state_calls) == 1
    env_ids, qpos, _ = backend.set_state_calls[0]
    np.testing.assert_array_equal(np.sort(env_ids), ids)
    np.testing.assert_allclose(qpos[:, :7], hub.goal[env_ids], atol=1e-6)

    # Dirty every episode-scope field, then reset a subset.
    hub.goal_count[:] = 5
    hub.pending_success[:] = True
    hub.lifted[:] = True
    hub.near_goal[:] = True
    hub.near_goal_steps[:] = 7
    hub.closest_keypoint_max_dist[:] = 0.5
    hub.closest_fingertip_dist[:] = 0.5
    sub = np.asarray([1, 4], dtype=np.int64)
    with transaction.scoped(sub):
        hub.reset(sub)
    np.testing.assert_array_equal(hub.prev_episode_successes[sub], 5)
    np.testing.assert_array_equal(hub.goal_count[sub], 0)
    assert not hub.pending_success[sub].any()
    assert not hub.lifted[sub].any()
    assert not hub.near_goal[sub].any()
    np.testing.assert_array_equal(hub.near_goal_steps[sub], 0)
    np.testing.assert_array_equal(hub.closest_keypoint_max_dist[sub], -1.0)
    np.testing.assert_array_equal(hub.closest_fingertip_dist[sub], -1.0)
    # Untouched envs keep their state.
    np.testing.assert_array_equal(hub.goal_count[[0, 2]], 5)


def test_log_extras_uses_computed_max_success_termination_result():
    env, _backend, _transaction, hub = _make_env()
    hub.goal_count[:] = np.arange(NUM_ENVS)
    hub.prev_episode_successes[:] = np.arange(NUM_ENVS) + 10
    max_successes = np.zeros(NUM_ENVS, dtype=bool)
    max_successes[[2, 6]] = True

    class RewardManager:
        def step_reward_vectors(self):
            return {}

    class TerminationManager:
        def get_term(self, name):
            if name == "max_consecutive_successes":
                return max_successes
            return np.zeros(NUM_ENVS, dtype=bool)

    extras = hub.log_extras(
        reward_manager=RewardManager(),
        termination_manager=TerminationManager(),
    )
    np.testing.assert_array_equal(extras["episode_final"]["all_goals_hit"], max_successes)
    np.testing.assert_array_equal(extras["episode_final"]["successes"], np.arange(NUM_ENVS))
    np.testing.assert_array_equal(extras["successes"], np.arange(NUM_ENVS) + 10)


# ----------------------------------------------------------------------------
# Success-driven advancement (termination_utils.py:46-51, DESIGN.md §6.5)
# ----------------------------------------------------------------------------


def _dirty_trackers(hub, ids):
    hub.closest_keypoint_max_dist[ids] = 0.5
    hub.closest_fingertip_dist[ids] = 0.5
    hub.near_goal_steps[ids] = 7


def test_success_advance_consumes_pending_success():
    env, backend, transaction, hub = _make_env(seed=11)
    _reset_all(env, transaction, hub)
    env.episode_length_buf[:] = 40
    env.steps_mirror[:] = 39
    prev_goal = hub.goal.copy()

    succeeded = np.asarray([1, 4], dtype=np.int64)
    hub.pending_success[succeeded] = True
    _dirty_trackers(hub, succeeded)
    _compute_step(env, transaction, hub)

    # termination_utils.py:46: the counter advances for succeeded envs only.
    np.testing.assert_array_equal(hub.goal_count[succeeded], 1)
    np.testing.assert_array_equal(hub.goal_count[[0, 2, 3]], 0)
    # reset_utils.py:376-380: per-goal trackers cleared for succeeded envs.
    np.testing.assert_array_equal(hub.closest_keypoint_max_dist[succeeded], -1.0)
    np.testing.assert_array_equal(hub.closest_fingertip_dist[succeeded], -1.0)
    np.testing.assert_array_equal(hub.near_goal_steps[succeeded], 0)
    assert hub.closest_keypoint_max_dist[0] == -1.0  # untouched default
    # The goal advanced (delta walk from the previous pose).
    assert not np.allclose(hub.goal[succeeded], prev_goal[succeeded])
    np.testing.assert_allclose(hub.goal[[0, 2, 3]], prev_goal[[0, 2, 3]], rtol=1e-6)
    delta = np.abs(hub.goal[succeeded, :3] - prev_goal[succeeded, :3])
    assert np.all(delta <= hub.cfg.delta_goal_distance + 1e-6)
    # termination_utils.py:51: both episode counters zeroed via the dual write.
    np.testing.assert_array_equal(env.episode_length_buf[succeeded], 0)
    np.testing.assert_array_equal(env.steps_mirror[succeeded], 0)
    np.testing.assert_array_equal(env.episode_length_buf[[0, 2, 3]], 40)
    # Mid-episode advance is hub-only: the reset transaction's full-row
    # staging would reset robot/object state (module docstring), so no sim
    # commit happens outside the reset path.
    assert len(backend.set_state_calls) == 1  # the reset commit only


def test_pending_success_scoped_on_reset_path_compute():
    """compute(dt=0, env_ids) only consumes pending_success inside env_ids."""
    env, backend, transaction, hub = _make_env()
    _reset_all(env, transaction, hub)
    hub.pending_success[[2, 5]] = True
    ids = np.asarray([2], dtype=np.int64)
    with transaction.scoped(ids):
        hub.compute(dt=0.0, env_ids=ids)
    np.testing.assert_array_equal(hub.goal_count[[2, 5]], [1, 0])
    assert hub.pending_success[5]  # left for the next per-step compute


def test_goalviz_degrades_to_hub_only_when_missing():
    """Pre-1.4s: without the goalviz entity the hub pose is the only record."""
    env, backend, transaction, hub = _make_env(with_goalviz=False)
    ids = np.arange(NUM_ENVS, dtype=np.int64)
    hub.reset(ids)  # no transaction scope needed: nothing is committed
    prev_goal = hub.goal.copy()
    hub.pending_success[[0, 3]] = True
    env.common_step_counter += 1
    hub.compute(dt=STEP_DT, env_ids=None)
    np.testing.assert_array_equal(hub.goal_count[[0, 3]], 1)
    assert not np.allclose(hub.goal[[0, 3], :3], prev_goal[[0, 3], :3])
    assert backend.set_state_calls == []


def test_delta_mode_walks_from_hub_goal_not_stale_goalviz():
    """The delta walk chains from the hub's current goal (reaudit fix P3a).

    The original chains from the GoalViz sim pose, which its per-advance
    ``write_root_pose_to_sim`` keeps fresh (reset_utils.py:351-353); UniLab
    defers that sim write to the reset path, so the hub buffer is the
    authoritative current goal and the delta walk must read it.
    """
    env, backend, transaction, hub = _make_env(seed=13)
    _reset_all(env, transaction, hub)
    # Simulate a stale GoalViz sim pose (mid-episode commits are deferred):
    # the walk must ignore it and chain from the hub's current goal.
    hub.goal[:, :3] = [0.0, 0.0, 0.7]
    hub.goal[:, 3] = 1.0
    hub.pending_success[:] = True
    _compute_step(env, transaction, hub)
    delta = np.abs(hub.goal[:, :3] - np.asarray([0.0, 0.0, 0.7]))
    assert np.all(delta <= hub.cfg.delta_goal_distance + 1e-6)


def test_delta_mode_chains_across_successive_advances():
    """Two successive advances chain: the second walks from the first's goal.

    Anchoring every advance to the episode-start goal (the pre-P3a behavior)
    would freeze the intra-episode goal walk.
    """
    env, backend, transaction, hub = _make_env(seed=17)
    _reset_all(env, transaction, hub)
    first = np.asarray([0])
    hub.pending_success[first] = True
    _compute_step(env, transaction, hub)
    chained_goal = hub.goal[first].copy()
    # Second advance on the same rows (fresh success frame).
    hub.near_goal_steps[first] = hub.cfg.success_steps
    hub.pending_success[first] = True
    _compute_step(env, transaction, hub)
    delta = np.abs(hub.goal[first, :3] - chained_goal[0, :3])
    assert np.all(delta <= hub.cfg.delta_goal_distance + 1e-6)


# ----------------------------------------------------------------------------
# Tolerance curriculum (termination_utils.py:10-36)
#
# The hub owns the curriculum state; the per-step trigger moved to the
# termination layer's SuccessTracker (audit fix A1.4), so these cadence tests
# drive hub.tick_tolerance_curriculum() directly.  The same-frame semantics
# are pinned in tests/tasks/test_simtoolreal_terminations.py.
# ----------------------------------------------------------------------------


def _curriculum_env(**overrides):
    cfg = {
        "tolerance_curriculum_interval": 10,
        "tolerance_curriculum_success_threshold": 3.0,
        "tolerance_curriculum_increment": 0.9,
        **overrides,
    }
    env, backend, transaction, hub = _make_env(**cfg)
    _reset_all(env, transaction, hub)  # park the resample timer at 1e6 s
    return env, backend, transaction, hub


def test_tolerance_curriculum_interval_and_shrink():
    env, _, transaction, hub = _curriculum_env()
    del transaction
    start = hub.current_success_tolerance
    assert start == hub.cfg.success_tolerance

    # Before the interval elapses: no shrink even with enough successes.
    hub.prev_episode_successes[:] = 5
    env.common_step_counter = 9
    hub.tick_tolerance_curriculum()
    assert hub.current_success_tolerance == start

    # Interval reached and threshold met: shrink by the increment.
    env.common_step_counter = 10
    hub.tick_tolerance_curriculum()
    assert hub.current_success_tolerance == pytest.approx(start * 0.9)
    assert hub._last_curriculum_step == 10

    # The next window is measured from the last shrink.
    env.common_step_counter = 19
    hub.tick_tolerance_curriculum()
    assert hub.current_success_tolerance == pytest.approx(start * 0.9)
    env.common_step_counter = 20
    hub.tick_tolerance_curriculum()
    assert hub.current_success_tolerance == pytest.approx(start * 0.9**2)


def test_tolerance_curriculum_below_threshold_keeps_checking():
    """termination_utils.py:14-30: a missed window does not advance
    ``_last_curriculum_update``, so the check re-runs every step."""
    env, _, transaction, hub = _curriculum_env()
    del transaction
    hub.prev_episode_successes[:] = 0  # below threshold
    env.common_step_counter = 30
    hub.tick_tolerance_curriculum()
    assert hub.current_success_tolerance == hub.cfg.success_tolerance
    assert hub._last_curriculum_step == 0
    # The very next step can shrink once the threshold is met.
    hub.prev_episode_successes[:] = 3
    env.common_step_counter = 31
    hub.tick_tolerance_curriculum()
    assert hub.current_success_tolerance == pytest.approx(hub.cfg.success_tolerance * 0.9)
    assert hub._last_curriculum_step == 31


def test_tolerance_curriculum_floor():
    env, _, transaction, hub = _curriculum_env()
    del transaction
    hub.current_success_tolerance = 0.011
    hub.prev_episode_successes[:] = 10
    env.common_step_counter = 10
    hub.tick_tolerance_curriculum()
    # 0.011 * 0.9 = 0.0099 -> clamped to the 0.01 floor.
    assert hub.current_success_tolerance == hub.cfg.target_success_tolerance


def test_eval_tolerance_pins_curriculum():
    env, _, transaction, hub = _curriculum_env(eval_success_tolerance=0.02)
    del transaction
    assert hub.current_success_tolerance == 0.02  # pinned at construction
    hub.prev_episode_successes[:] = 10
    env.common_step_counter = 10
    hub.tick_tolerance_curriculum()
    assert hub.current_success_tolerance == 0.02


def test_curriculum_not_ticked_by_command_phase():
    """Audit fix A1.4: the command phase never ticks the curriculum — the
    per-step trigger lives in the termination layer's SuccessTracker, so
    neither the reset-path recompute nor a per-step compute can double-tick
    within one frame."""
    env, _, transaction, hub = _curriculum_env()
    hub.prev_episode_successes[:] = 10
    env.common_step_counter = 50
    ids = np.asarray([0], dtype=np.int64)
    with transaction.scoped(ids):
        hub.compute(dt=0.0, env_ids=ids)
    assert hub.current_success_tolerance == hub.cfg.success_tolerance
    assert hub._last_curriculum_step == 0
    with transaction.scoped(np.arange(NUM_ENVS, dtype=np.int64)):
        hub.compute(dt=STEP_DT, env_ids=None)
    assert hub.current_success_tolerance == hub.cfg.success_tolerance
    assert hub._last_curriculum_step == 0


def test_object_scale_per_env_expands_from_tool_pool_tables():
    """scene_utils.py:1638-1655: each env's scale comes from its variant id."""
    env, _backend, _transaction, hub = _make_env()
    del hub
    env.cfg = SimpleNamespace(
        tool_pool_scales=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
        scene=SimpleNamespace(
            entity_variant=SimpleNamespace(
                plan=SimpleNamespace(
                    assignment=np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int32),
                    variants=("a.xml", "b.xml"),
                )
            )
        ),
    )
    rebuilt = SimToolRealCommandCfg(resampling_time_range=(1e6, 1e6)).build(env)
    expected = np.asarray([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]] * 4, dtype=np.float32)
    np.testing.assert_array_equal(rebuilt.object_scale_per_env, expected)


def test_object_scale_per_env_defaults_to_ones_without_pool_tables():
    """Lightweight fixtures keep the unit-scale default."""
    env, _backend, _transaction, hub = _make_env()
    assert getattr(env, "cfg", None) is None
    np.testing.assert_array_equal(
        hub.object_scale_per_env, np.ones((NUM_ENVS, 3), dtype=np.float32)
    )


def test_object_scale_per_env_rejects_scale_table_below_pool():
    env, _backend, _transaction, _hub = _make_env()
    env.cfg = SimpleNamespace(
        tool_pool_scales=((1.0, 2.0, 3.0),),
        scene=SimpleNamespace(
            entity_variant=SimpleNamespace(
                plan=SimpleNamespace(
                    assignment=np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32),
                    variants=("a.xml", "b.xml"),
                )
            )
        ),
    )
    with pytest.raises(ValueError, match="does not cover the fixed variant plan"):
        SimToolRealCommandCfg(resampling_time_range=(1e6, 1e6)).build(env)


def test_object_scale_per_env_rejects_plan_assignment_length_mismatch():
    env, _backend, _transaction, _hub = _make_env()
    env.cfg = SimpleNamespace(
        tool_pool_scales=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
        scene=SimpleNamespace(
            entity_variant=SimpleNamespace(
                plan=SimpleNamespace(
                    assignment=np.asarray([0, 1], dtype=np.int32),
                    variants=("a.xml", "b.xml"),
                )
            )
        ),
    )
    with pytest.raises(ValueError, match="one entry per environment"):
        SimToolRealCommandCfg(resampling_time_range=(1e6, 1e6)).build(env)
