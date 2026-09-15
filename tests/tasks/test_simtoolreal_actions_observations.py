"""Step-1 tests for the SimToolReal action and observation terms.

Fake-backend pattern follows tests/envs/mdp/test_joint_position_action.py: a
real ``EntityScene`` over a fake ``SimBackend`` (29 DoF in a rotated canonical
order, palm + fingertip + object_root + goalviz_root bodies) plus a real
``SimToolRealCommand`` hub, so permutation, delay, smoothing, and per-field
observation assembly are exercised through the production facades.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from unisim.backend.base import BackendRootStateLayout, SimBackend

from unilab.base.entity import EntityCfg, EntityScene
from unilab.base.reset_state import ResetStateTransaction
from unilab.managers._types import ManagerBasedRlEnv
from unilab.managers.manager_base import ManagerTermBaseCfg
from simtoolreal_unilab.tasks.simtoolreal.actions import (
    SimToolRealAction,
    SimToolRealActionCfg,
)
from simtoolreal_unilab.tasks.simtoolreal.commands import (
    SimToolRealCommandCfg,
)
from simtoolreal_unilab.tasks.simtoolreal.config import (
    FINGERTIP_LINK_NAMES,
    FINGERTIP_OFFSET,
    JOINT_NAMES_CANONICAL,
    KEYPOINT_CORNERS,
    NUM_ARM_JOINTS,
    NUM_FINGERTIPS,
    NUM_JOINTS,
    PALM_BODY_NAME,
    PALM_CENTER_OFFSET,
    ObsCfg,
)
from simtoolreal_unilab.tasks.simtoolreal.observations import (
    CRITIC_OBS_DIM,
    OBS_FIELD_SIZES,
    POLICY_OBS_DIM,
    SimToolRealCriticObservation,
    SimToolRealPolicyObservation,
)
from unilab.utils.rotation import np_quat_apply

NUM_ENVS = 8
STEP_DT = 1.0 / 60.0

# Backend joint order is the canonical order rotated by 3: a non-involutive
# permutation, so gather/scatter direction bugs in the terms are detectable.
_SHIFT = 3
BACKEND_JOINT_NAMES = tuple(
    JOINT_NAMES_CANONICAL[(k + _SHIFT) % NUM_JOINTS] for k in range(NUM_JOINTS)
)
# CANON_TO_BACKEND[k] = backend column holding canonical joint k.
CANON_TO_BACKEND = np.asarray(
    [BACKEND_JOINT_NAMES.index(name) for name in JOINT_NAMES_CANONICAL], dtype=np.intp
)

ROBOT_BODY_NAMES = (PALM_BODY_NAME, *FINGERTIP_LINK_NAMES)
ALL_BODY_NAMES = (*ROBOT_BODY_NAMES, "object_root", "goalviz_root")
PALM_BACKEND_ID = ALL_BODY_NAMES.index(PALM_BODY_NAME)
FINGERTIP_BACKEND_IDS = [ALL_BODY_NAMES.index(n) for n in FINGERTIP_LINK_NAMES]
OBJECT_BODY_ID = ALL_BODY_NAMES.index("object_root")
GOALVIZ_BODY_ID = ALL_BODY_NAMES.index("goalviz_root")

# Distinct per-joint ranges make order/clamp mix-ups detectable; all ranges
# include zero so the zero-initialized targets stay valid.
_JOINT_LOWER = -2.0 - np.arange(NUM_JOINTS, dtype=np.float32) * 0.1
_JOINT_UPPER = 2.0 + np.arange(NUM_JOINTS, dtype=np.float32) * 0.1
# Canonical-order limits (what the terms resolve through the facade).
_CANON_LOWER = _JOINT_LOWER[CANON_TO_BACKEND]
_CANON_UPPER = _JOINT_UPPER[CANON_TO_BACKEND]


class _FakeBackend:
    backend_type = "fake"
    num_envs = NUM_ENVS
    num_actuators = NUM_JOINTS

    def __init__(self) -> None:
        self.dof_pos = np.zeros((NUM_ENVS, NUM_JOINTS), dtype=np.float32)
        self.dof_vel = np.zeros((NUM_ENVS, NUM_JOINTS), dtype=np.float32)
        self.body_pos = np.zeros((NUM_ENVS, len(ALL_BODY_NAMES), 3), dtype=np.float32)
        self.body_quat = np.zeros((NUM_ENVS, len(ALL_BODY_NAMES), 4), dtype=np.float32)
        self.body_quat[..., 0] = 1.0
        self.body_lin_vel = np.zeros((NUM_ENVS, len(ALL_BODY_NAMES), 3), dtype=np.float32)
        self.body_ang_vel = np.zeros((NUM_ENVS, len(ALL_BODY_NAMES), 3), dtype=np.float32)
        self.set_state_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # --- joints / actuators ---
    def get_joint_dof_pos_indices(self, names) -> np.ndarray:
        return np.asarray([BACKEND_JOINT_NAMES.index(n) for n in names], dtype=np.int32)

    def get_joint_dof_vel_indices(self, names) -> np.ndarray:
        return self.get_joint_dof_pos_indices(names)

    def get_dof_pos(self) -> np.ndarray:
        return self.dof_pos

    def get_dof_vel(self) -> np.ndarray:
        return self.dof_vel

    def get_default_dof_pos(self) -> np.ndarray:
        return np.zeros(NUM_JOINTS, dtype=np.float32)

    def get_joint_range(self) -> np.ndarray:
        return np.stack([_JOINT_LOWER, _JOINT_UPPER], axis=1)

    def get_actuator_names(self) -> tuple[str, ...]:
        return BACKEND_JOINT_NAMES

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        return BACKEND_JOINT_NAMES

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return np.tile(np.asarray([[-100.0, 100.0]], dtype=np.float32), (NUM_JOINTS, 1))

    # --- bodies ---
    def get_body_ids(self, names) -> np.ndarray:
        return np.asarray([ALL_BODY_NAMES.index(n) for n in names], dtype=np.int32)

    def get_body_pos_w(self, ids) -> np.ndarray:
        return self.body_pos[:, np.asarray(ids)]

    def get_body_quat_w(self, ids) -> np.ndarray:
        return self.body_quat[:, np.asarray(ids)]

    def get_body_lin_vel_w(self, ids) -> np.ndarray:
        return self.body_lin_vel[:, np.asarray(ids)]

    def get_body_ang_vel_w(self, ids) -> np.ndarray:
        return self.body_ang_vel[:, np.asarray(ids)]

    def get_body_lin_vel_b(self, ids) -> np.ndarray:
        return np.zeros((NUM_ENVS, len(ids), 3), dtype=np.float32)

    def get_body_ang_vel_b(self, ids) -> np.ndarray:
        return np.zeros((NUM_ENVS, len(ids), 3), dtype=np.float32)

    def get_root_state_layout(self, root_body_name: str):
        # Generalized-state layout: 29 robot dofs, then the object root
        # (qpos 29..35, qvel 29..34), then the goalviz root (qpos 36..42,
        # qvel 35..40).
        if root_body_name == "object_root":
            return BackendRootStateLayout(
                qpos_indices=tuple(range(29, 36)), qvel_indices=tuple(range(29, 35))
            )
        if root_body_name == "goalviz_root":
            return BackendRootStateLayout(
                qpos_indices=tuple(range(36, 43)), qvel_indices=tuple(range(35, 41))
            )
        raise ValueError(f"unknown root body {root_body_name}")

    def get_default_qpos(self) -> np.ndarray:
        qpos = np.zeros(43, dtype=np.float32)
        qpos[32] = 1.0  # object root qw
        qpos[39] = 1.0  # goalviz root qw
        return qpos

    def get_init_qvel(self) -> np.ndarray:
        return np.zeros(41, dtype=np.float32)

    def set_state(self, env_ids, qpos, qvel, randomization=None) -> dict:
        self.set_state_calls.append((np.asarray(env_ids).copy(), qpos.copy(), qvel.copy()))
        # Committed root poses are visible to later reads, like a real sim.
        self.body_pos[env_ids, OBJECT_BODY_ID] = qpos[:, 29:32]
        self.body_quat[env_ids, OBJECT_BODY_ID] = qpos[:, 32:36]
        self.body_pos[env_ids, GOALVIZ_BODY_ID] = qpos[:, 36:39]
        self.body_quat[env_ids, GOALVIZ_BODY_ID] = qpos[:, 39:43]
        return {"timing": {"set_state_ms": 0.0}}


class _ScriptedRng:
    """Generator wrapper with a scripted ``integers`` stream for delay lags."""

    def __init__(self, lag_script: list[np.ndarray], seed: int = 0) -> None:
        self._lags = list(lag_script)
        self._rng = np.random.default_rng(seed)

    def integers(self, low, high=None, size=None, dtype=np.int64):
        assert self._lags, "lag script exhausted"
        return self._lags.pop(0)

    def standard_normal(self, shape, dtype=np.float32):
        return self._rng.standard_normal(shape, dtype=dtype)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self._rng.uniform(low, high, size)


def _make_env(seed: int = 7, **action_cfg_overrides):
    backend = _FakeBackend()
    control = np.zeros((NUM_ENVS, NUM_JOINTS), dtype=np.float32)
    transaction = ResetStateTransaction(cast(SimBackend, backend))
    scene = EntityScene(
        {
            "robot": EntityCfg(
                joint_names=BACKEND_JOINT_NAMES,
                body_names=ROBOT_BODY_NAMES,
                actuator_names=BACKEND_JOINT_NAMES,
            ),
            "object": EntityCfg(root_body_name="object_root"),
            "goalviz": EntityCfg(root_body_name="goalviz_root"),
        },
        cast(SimBackend, backend),
        control,
        reset_state=transaction,
    )
    env = SimpleNamespace(
        num_envs=NUM_ENVS,
        rng=np.random.default_rng(seed),
        step_dt=STEP_DT,
        scene=scene,
        episode_length_buf=np.ones(NUM_ENVS, dtype=np.int64),
        reward_buf=np.zeros(NUM_ENVS, dtype=np.float32),
        common_step_counter=0,
        # The env-owned reset transaction (manager_based_rl_env.py:584-591
        # wraps command_manager.reset in it); command hub reset commits the
        # GoalViz pose through it.
        reset_transaction=transaction,
    )
    hub = SimToolRealCommandCfg(resampling_time_range=(1e6, 1e6)).build(
        cast(ManagerBasedRlEnv, env)
    )
    env.command_manager = SimpleNamespace(get_term=lambda name: hub)
    action = SimToolRealActionCfg(entity_name="robot", **action_cfg_overrides).build(
        cast(ManagerBasedRlEnv, env)
    )
    env.action_manager = SimpleNamespace(get_term=lambda name: action)
    return env, backend, control, hub, action


def _policy_term(env, **params) -> SimToolRealPolicyObservation:
    cfg = ManagerTermBaseCfg(func=SimToolRealPolicyObservation, params=params)
    return SimToolRealPolicyObservation(cfg, cast(ManagerBasedRlEnv, env))


def _critic_term(env, **params) -> SimToolRealCriticObservation:
    cfg = ManagerTermBaseCfg(func=SimToolRealCriticObservation, params=params)
    return SimToolRealCriticObservation(cfg, cast(ManagerBasedRlEnv, env))


_NO_DR = {
    "use_object_state_delay_noise": False,
    "joint_velocity_obs_noise_std": 0.0,
    "use_obs_delay": False,
}


def _field_slice(fields: tuple[str, ...], name: str) -> slice:
    offset = 0
    for field in fields:
        width = OBS_FIELD_SIZES[field]
        if field == name:
            return slice(offset, offset + width)
        offset += width
    raise KeyError(name)


def _canon(values_backend_order: np.ndarray) -> np.ndarray:
    """Gather backend-order (N, 29) rows into canonical policy order."""
    return values_backend_order[:, CANON_TO_BACKEND]


def _rows(values) -> np.ndarray:
    """Tile a per-env-constant expected vector to (NUM_ENVS, width)."""
    return np.repeat(np.atleast_2d(np.asarray(values, dtype=np.float32)), NUM_ENVS, axis=0)


# ----------------------------------------------------------------------------
# Action term
# ----------------------------------------------------------------------------


def test_action_no_delay_arm_delta_hand_absolute_and_permutation():
    """action_utils.py:50-74 with delay off."""
    env, backend, control, hub, action = _make_env(use_action_delay=False)
    assert isinstance(action, SimToolRealAction)
    env.episode_length_buf[:] = 3  # mid-episode: no flush even if delay on

    raw = np.linspace(-1.0, 1.0, NUM_JOINTS, dtype=np.float32)[None, :]
    raw = np.repeat(raw, NUM_ENVS, axis=0)
    action.process_actions(raw)
    action.apply_actions()

    # Arm: prev=0 -> arm_raw = 1.5 * dt * a; smoothed = 0.1 * arm_raw.
    arm_expected = 0.1 * (1.5 * STEP_DT * raw[:, :NUM_ARM_JOINTS])
    # Hand: raw = lower + 0.5*(a+1)*(upper-lower); smoothed = 0.1*raw.
    lower = _CANON_LOWER[NUM_ARM_JOINTS:]
    upper = _CANON_UPPER[NUM_ARM_JOINTS:]
    hand_raw = lower + 0.5 * (raw[:, NUM_ARM_JOINTS:] + 1.0) * (upper - lower)
    hand_expected = 0.1 * hand_raw
    expected_canonical = np.concatenate([arm_expected, hand_expected], axis=1)
    np.testing.assert_allclose(action.prev_targets, expected_canonical, rtol=1e-6)
    np.testing.assert_allclose(action.raw_action, raw)

    # Facade write scatters canonical targets onto the rotated backend order:
    # control[:, CANON_TO_BACKEND[k]] = target_k.
    expected_control = np.zeros_like(control)
    expected_control[:, CANON_TO_BACKEND] = expected_canonical
    np.testing.assert_allclose(control, expected_control, rtol=1e-6)


def test_action_no_clip_minus_one_one():
    """Step-1 decision 1.1-e: the task term never clips actions to [-1, 1]."""
    env, backend, control, hub, action = _make_env(use_action_delay=False)
    env.episode_length_buf[:] = 3
    raw = np.full((NUM_ENVS, NUM_JOINTS), 5.0, dtype=np.float32)
    action.process_actions(raw)
    np.testing.assert_array_equal(action.raw_action, raw)
    # action_utils.py:60-69 clamps the hand only once, after smoothing: a=5
    # drives hand_raw past the limits (5 * upper for these symmetric ranges),
    # and the smoothed 0.5 * upper still sits inside, so the clamp is a no-op.
    lower_hand = _CANON_LOWER[NUM_ARM_JOINTS:]
    upper_hand = _CANON_UPPER[NUM_ARM_JOINTS:]
    hand_raw = lower_hand + 0.5 * (5.0 + 1.0) * (upper_hand - lower_hand)
    hand_expected = np.clip(0.1 * hand_raw, lower_hand, upper_hand)
    np.testing.assert_allclose(
        action.prev_targets[:, NUM_ARM_JOINTS:],
        np.repeat(hand_expected[None, :], NUM_ENVS, axis=0),
        rtol=1e-6,
    )


def test_action_delay_queue_flush_roll_and_lag():
    """action_utils.py:36-48: episode-start flush, roll, per-env lag sample.

    The lag script makes the delay draws deterministic: step 2 lags 0 (newest
    frame), step 3 lags 2 (oldest queued frame).
    """
    env, backend, control, hub, action = _make_env(seed=11)
    lag0 = np.zeros(NUM_ENVS, dtype=np.int64)
    lag2 = np.full(NUM_ENVS, 2, dtype=np.int64)
    env.rng = _ScriptedRng([lag0.copy(), lag0.copy(), lag2.copy()])
    env.episode_length_buf[:] = 0  # fresh episode
    hub.goal_count[:] = 0

    a1 = np.full((NUM_ENVS, NUM_JOINTS), 0.5, dtype=np.float32)
    action.process_actions(a1)
    # Flush filled every slot with a1: any lag returns a1.
    for slot in range(3):
        np.testing.assert_allclose(action._action_queue[:, slot], a1)
    prev1 = action.prev_targets.copy()

    env.episode_length_buf[:] = 1
    a2 = np.full((NUM_ENVS, NUM_JOINTS), -0.25, dtype=np.float32)
    action.process_actions(a2)
    # Roll then push: slot 0 newest, slots 1-2 hold the flushed a1.
    np.testing.assert_allclose(action._action_queue[:, 0], a2)
    np.testing.assert_allclose(action._action_queue[:, 1], a1)
    np.testing.assert_allclose(action._action_queue[:, 2], a1)
    # Lag 0 -> the delayed arm input is a2; the delta accumulates on prev.
    prev2 = action.prev_targets.copy()
    np.testing.assert_allclose(
        prev2[:, :NUM_ARM_JOINTS],
        0.1 * (prev1[:, :NUM_ARM_JOINTS] + 1.5 * STEP_DT * a2[:, :NUM_ARM_JOINTS])
        + 0.9 * prev1[:, :NUM_ARM_JOINTS],
        rtol=1e-6,
    )

    env.episode_length_buf[:] = 2
    a3 = np.full((NUM_ENVS, NUM_JOINTS), 0.75, dtype=np.float32)
    action.process_actions(a3)
    np.testing.assert_allclose(action._action_queue[:, 0], a3)
    np.testing.assert_allclose(action._action_queue[:, 1], a2)
    np.testing.assert_allclose(action._action_queue[:, 2], a1)
    # Lag 2 -> the delayed arm input is a1 (oldest slot).
    np.testing.assert_allclose(
        action.prev_targets[:, :NUM_ARM_JOINTS],
        0.1 * (prev2[:, :NUM_ARM_JOINTS] + 1.5 * STEP_DT * a1[:, :NUM_ARM_JOINTS])
        + 0.9 * prev2[:, :NUM_ARM_JOINTS],
        rtol=1e-6,
    )


def test_action_delay_not_flushed_after_goal_advance():
    """_episode_start mask: a success advance zeroes the episode length but
    keeps the queue (action_utils.py:38, the ``_successes`` half)."""
    env, backend, control, hub, action = _make_env(seed=3)
    env.episode_length_buf[:] = 0
    hub.goal_count[:] = 1  # intra-episode goal advance, not a fresh episode
    action.process_actions(np.full((NUM_ENVS, NUM_JOINTS), 0.9, dtype=np.float32))
    # No flush: slots 1 and 2 still hold the zero-initialized history.
    np.testing.assert_array_equal(action._action_queue[:, 1:], 0.0)


def test_action_moving_average_two_steps():
    """EMA recursion over two steps matches action_utils.py:54-57 exactly."""
    env, _, _, _, action = _make_env(use_action_delay=False)
    env.episode_length_buf[:] = 5
    a = np.full((NUM_ENVS, NUM_JOINTS), 1.0, dtype=np.float32)
    action.process_actions(a)
    step1 = action.prev_targets.copy()
    action.process_actions(a)
    step2 = action.prev_targets
    arm_raw = step1[:, :NUM_ARM_JOINTS] + 1.5 * STEP_DT * a[:, :NUM_ARM_JOINTS]
    expected = 0.1 * arm_raw + 0.9 * step1[:, :NUM_ARM_JOINTS]
    np.testing.assert_allclose(step2[:, :NUM_ARM_JOINTS], expected, rtol=1e-6)


def test_action_joint_limits_clamp():
    """action_utils.py:53-58, 69: arm delta clamps twice, hand once."""
    env, _, _, _, action = _make_env(use_action_delay=False)
    env.episode_length_buf[:] = 5
    huge = np.full((NUM_ENVS, NUM_JOINTS), 1000.0, dtype=np.float32)
    for _ in range(300):  # accumulate past the arm upper limits
        action.process_actions(huge)
    # float32 geometric EMA convergence: a few ULPs below the exact limit.
    np.testing.assert_allclose(
        action.prev_targets,
        _rows(_CANON_UPPER),
        rtol=1e-5,
        atol=1e-5,
    )
    neg = np.full((NUM_ENVS, NUM_JOINTS), -1000.0, dtype=np.float32)
    for _ in range(300):
        action.process_actions(neg)
    np.testing.assert_allclose(
        action.prev_targets,
        _rows(_CANON_LOWER),
        rtol=1e-5,
        atol=1e-5,
    )


def test_action_reset_seeds_targets_from_post_reset_pose():
    """reset_utils.py:218-220, 399: queue zeroed; prev/cur targets read back
    from the post-reset joint state in canonical order."""
    env, backend, control, hub, action = _make_env()
    reset_pos_backend = np.linspace(-0.5, 0.5, NUM_JOINTS, dtype=np.float32)
    backend.dof_pos[:] = reset_pos_backend[None, :]
    ids = np.asarray([1, 3], dtype=np.int64)
    action.reset(ids)
    expected = np.tile(reset_pos_backend[CANON_TO_BACKEND], (len(ids), 1))
    np.testing.assert_allclose(action.prev_targets[ids], expected, rtol=1e-6)
    np.testing.assert_allclose(action._cur_targets[ids], expected, rtol=1e-6)
    np.testing.assert_array_equal(action._action_queue[ids], 0.0)
    np.testing.assert_array_equal(action.raw_action[ids], 0.0)
    # Untouched envs keep their previous (zero) buffers.
    np.testing.assert_array_equal(action.prev_targets[0], 0.0)


# ----------------------------------------------------------------------------
# Observation width pins
# ----------------------------------------------------------------------------


def test_obs_width_pins_match_field_lists():
    obs = ObsCfg()
    assert sum(OBS_FIELD_SIZES[f] for f in obs.obs_list) == POLICY_OBS_DIM == 140
    assert sum(OBS_FIELD_SIZES[f] for f in obs.state_list) == CRITIC_OBS_DIM == 162


def test_obs_terms_reach_pinned_widths():
    env, *_ = _make_env()
    policy = _policy_term(env, **_NO_DR)
    critic = _critic_term(env)
    assert policy(env).shape == (NUM_ENVS, POLICY_OBS_DIM)
    assert critic(env).shape == (NUM_ENVS, CRITIC_OBS_DIM)


def test_obs_terms_fail_fast_without_scene_entities():
    env, *_ = _make_env()
    robot_only = EntityScene(
        {
            "robot": EntityCfg(
                joint_names=BACKEND_JOINT_NAMES,
                body_names=ROBOT_BODY_NAMES,
                actuator_names=BACKEND_JOINT_NAMES,
            )
        },
        cast(SimBackend, _FakeBackend()),
        np.zeros((NUM_ENVS, NUM_JOINTS), dtype=np.float32),
    )
    env.scene = robot_only
    with pytest.raises(KeyError, match="object"):
        _policy_term(env, **_NO_DR)
    with pytest.raises(KeyError, match="object"):
        _critic_term(env)


# ----------------------------------------------------------------------------
# Observation field-level assembly (all DR off)
# ----------------------------------------------------------------------------


def _populate_fake_state(env, backend, hub, action):
    """Deterministic, recognizable state across every obs source field."""
    backend.dof_pos[:] = np.linspace(-1.0, 1.0, NUM_JOINTS, dtype=np.float32)[None, :]
    backend.dof_vel[:] = np.linspace(0.0, 0.58, NUM_JOINTS, dtype=np.float32)[None, :]
    backend.body_pos[:, PALM_BACKEND_ID] = [0.1, 0.2, 0.3]
    for k, body_id in enumerate(FINGERTIP_BACKEND_IDS):
        backend.body_pos[:, body_id] = [0.01 * (k + 1), -0.02 * (k + 1), 0.5]
    backend.body_lin_vel[:, PALM_BACKEND_ID] = [1.0, 2.0, 3.0]
    backend.body_ang_vel[:, PALM_BACKEND_ID] = [4.0, 5.0, 6.0]
    backend.body_pos[:, OBJECT_BODY_ID] = [0.4, -0.1, 0.7]
    backend.body_lin_vel[:, OBJECT_BODY_ID] = [7.0, 8.0, 9.0]
    # Kept below clamp_abs_observations (10.0) so the field-level tests see the
    # unclamped values; the clamp path is covered by test_obs_clamp_abs.
    backend.body_ang_vel[:, OBJECT_BODY_ID] = [9.1, 9.4, 9.7]
    backend.body_pos[:, GOALVIZ_BODY_ID] = [-0.3, 0.6, 0.9]
    # Goal readers consume the hub buffer (reaudit fix P3a); a real commit
    # keeps the hub and the GoalViz sim pose in sync.
    hub.goal[:, :3] = [-0.3, 0.6, 0.9]
    hub.closest_keypoint_max_dist[:] = 0.123
    hub.closest_fingertip_dist[:] = np.arange(5, dtype=np.float32) * 0.01
    hub.lifted[:] = True
    hub.goal_count[:] = 2
    env.episode_length_buf[:] = 40
    env.reward_buf[:] = 250.0
    # prev_action_targets: mark each canonical joint with its index / 100.
    action.prev_targets[:] = np.arange(NUM_JOINTS, dtype=np.float32)[None, :] / 100.0


def _make_env_and_state():
    env, backend, control, hub, action = _make_env()
    _populate_fake_state(env, backend, hub, action)
    return env, backend, hub, action


def test_policy_obs_field_by_field():
    """obs_utils.py:308-337 with every DR switch off: exact field values."""
    env, backend, hub, action = _make_env_and_state()
    policy = _policy_term(env, **_NO_DR)
    frame = policy(env)
    fields = ObsCfg().obs_list

    joint_pos_expected = (
        2.0 * (_canon(backend.dof_pos) - _CANON_LOWER) / (_CANON_UPPER - _CANON_LOWER) - 1.0
    )
    np.testing.assert_allclose(
        frame[:, _field_slice(fields, "joint_pos")], joint_pos_expected, rtol=1e-5
    )
    np.testing.assert_allclose(
        frame[:, _field_slice(fields, "joint_vel")], _canon(backend.dof_vel), rtol=1e-6
    )
    np.testing.assert_allclose(
        frame[:, _field_slice(fields, "prev_action_targets")],
        action.prev_targets,
        rtol=1e-6,
    )
    # Palm center offset with identity rotation lands directly on the offset.
    palm_center = np.asarray([0.1, 0.2, 0.3]) + np.asarray(PALM_CENTER_OFFSET)
    np.testing.assert_allclose(
        frame[:, _field_slice(fields, "palm_pos")], _rows(palm_center), rtol=1e-6
    )
    # wxyz identity -> xyzw identity (obs_utils.py:303-306).
    np.testing.assert_allclose(
        frame[:, _field_slice(fields, "palm_rot")], _rows([0.0, 0.0, 0.0, 1.0]), atol=1e-7
    )
    np.testing.assert_allclose(
        frame[:, _field_slice(fields, "object_rot")], _rows([0.0, 0.0, 0.0, 1.0]), atol=1e-7
    )
    # Fingertip pad centers relative to the palm center (identity rotations).
    ft_expected = np.concatenate(
        [
            np.asarray([0.01 * (k + 1), -0.02 * (k + 1), 0.5])
            + np.asarray(FINGERTIP_OFFSET)
            - palm_center
            for k in range(NUM_FINGERTIPS)
        ]
    )
    np.testing.assert_allclose(
        frame[:, _field_slice(fields, "fingertip_pos_rel_palm")],
        _rows(ft_expected),
        rtol=1e-6,
    )
    # Keypoints: unit scales -> corners * 0.04 * 1.5 * 0.5 = corners * 0.03.
    corners = np.asarray(KEYPOINT_CORNERS, dtype=np.float32) * 0.03
    obj_kp = np.asarray([0.4, -0.1, 0.7]) + corners
    goal_kp = np.asarray([-0.3, 0.6, 0.9]) + corners
    np.testing.assert_allclose(
        frame[:, _field_slice(fields, "keypoints_rel_palm")],
        _rows((obj_kp - palm_center).reshape(-1)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        frame[:, _field_slice(fields, "keypoints_rel_goal")],
        _rows((obj_kp - goal_kp).reshape(-1)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(frame[:, _field_slice(fields, "object_scales")], 1.0, rtol=1e-6)


def test_critic_obs_field_by_field():
    """obs_utils.py:308-327: clean fields plus hub success pre-caches."""
    env, backend, hub, action = _make_env_and_state()
    critic = _critic_term(env)
    frame = critic(env)
    fields = ObsCfg().state_list

    np.testing.assert_allclose(
        frame[:, _field_slice(fields, "palm_vel")],
        _rows([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        frame[:, _field_slice(fields, "object_vel")],
        _rows([7.0, 8.0, 9.0, 9.1, 9.4, 9.7]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        frame[:, _field_slice(fields, "closest_keypoint_max_dist")], 0.123, rtol=1e-6
    )
    np.testing.assert_allclose(
        frame[:, _field_slice(fields, "closest_fingertip_dist")],
        _rows(np.arange(5, dtype=np.float32) * 0.01),
        rtol=1e-6,
    )
    np.testing.assert_allclose(frame[:, _field_slice(fields, "lifted_object")], 1.0, rtol=1e-6)
    np.testing.assert_allclose(
        frame[:, _field_slice(fields, "progress")], np.log(40.0 / 10.0 + 1.0), rtol=1e-6
    )
    np.testing.assert_allclose(
        frame[:, _field_slice(fields, "successes")], np.log(2.0 + 1.0), rtol=1e-6
    )
    np.testing.assert_allclose(frame[:, _field_slice(fields, "reward")], 250.0 * 0.01, rtol=1e-6)
    joint_pos_expected = (
        2.0 * (_canon(backend.dof_pos) - _CANON_LOWER) / (_CANON_UPPER - _CANON_LOWER) - 1.0
    )
    np.testing.assert_allclose(
        frame[:, _field_slice(fields, "joint_pos")], joint_pos_expected, rtol=1e-5
    )


def test_obs_clamp_abs():
    """obs_utils.py:345-347: both frames clamp to +/-clamp_abs_observations."""
    env, backend, hub, action = _make_env_and_state()
    backend.body_lin_vel[:, OBJECT_BODY_ID] = 500.0
    policy = _policy_term(env, **_NO_DR)
    critic = _critic_term(env)
    assert np.abs(policy(env)).max() <= 10.0
    critic_frame = critic(env)
    assert np.abs(critic_frame).max() <= 10.0
    np.testing.assert_allclose(
        critic_frame[:, _field_slice(ObsCfg().state_list, "object_vel")],
        _rows([10.0, 10.0, 10.0, 9.1, 9.4, 9.7]),
    )


def test_palm_and_fingertip_offsets_rotate_with_body():
    """obs_utils.py:88-100: local offsets are rotated by the body orientation."""
    env, backend, hub, action = _make_env_and_state()
    # 90-degree rotation about world +Z (wxyz): x -> y, y -> -x.
    backend.body_quat[:, PALM_BACKEND_ID] = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    policy = _policy_term(env, **_NO_DR)
    frame = policy(env)
    rotated_offset = np.asarray(
        [-PALM_CENTER_OFFSET[1], PALM_CENTER_OFFSET[0], PALM_CENTER_OFFSET[2]]
    )
    np.testing.assert_allclose(
        frame[:, _field_slice(ObsCfg().obs_list, "palm_pos")],
        _rows(np.asarray([0.1, 0.2, 0.3]) + rotated_offset),
        atol=1e-6,
    )
    # palm_rot carries the xyzw conversion of the rotated quaternion.
    np.testing.assert_allclose(
        frame[:, _field_slice(ObsCfg().obs_list, "palm_rot")],
        _rows([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]),
        atol=1e-6,
    )


# ----------------------------------------------------------------------------
# Observation DR paths
# ----------------------------------------------------------------------------


def test_object_state_delay_flush_and_roll():
    """obs_utils.py:119-133, 210-220: episode-start flush then rolling queue."""
    env, backend, hub, action = _make_env_and_state()
    policy = _policy_term(
        env,
        use_object_state_delay_noise=True,
        object_state_delay_max=10,
        object_state_xyz_noise_std=0.0,
        object_state_rotation_noise_degrees=0.0,
        joint_velocity_obs_noise_std=0.0,
        use_obs_delay=False,
    )
    env.episode_length_buf[:] = 0
    hub.goal_count[:] = 0
    env.common_step_counter = 1
    policy(env)
    # Flush: all 10 slots hold the current object state (pos 0.4, -0.1, 0.7).
    np.testing.assert_allclose(
        policy._object_state_queue[:, :, 0:3],
        np.broadcast_to(np.asarray([0.4, -0.1, 0.7], dtype=np.float32), (NUM_ENVS, 10, 3)),
        rtol=1e-6,
    )
    # Next step with a moved object: roll puts the new state in slot 0.
    backend.body_pos[:, OBJECT_BODY_ID] = [9.0, 9.0, 9.0]
    env.episode_length_buf[:] = 1
    env.common_step_counter = 2
    policy(env)
    np.testing.assert_allclose(
        policy._object_state_queue[:, 0, 0:3],
        np.broadcast_to(np.asarray([9.0, 9.0, 9.0], dtype=np.float32), (NUM_ENVS, 3)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        policy._object_state_queue[:, 1, 0:3],
        np.broadcast_to(np.asarray([0.4, -0.1, 0.7], dtype=np.float32), (NUM_ENVS, 3)),
        rtol=1e-6,
    )


def test_object_state_noise_actor_fields_only():
    """xyz std 0.01 / rot 5 deg land on the actor object fields; the noisy
    quaternion stays unit-length (obs_utils.py:217-218, 330-333)."""
    env, backend, hub, action = _make_env_and_state()
    env.episode_length_buf[:] = 0
    hub.goal_count[:] = 0
    env.common_step_counter = 1
    policy = _policy_term(
        env,
        use_object_state_delay_noise=True,
        object_state_delay_max=10,
        use_obs_delay=False,
        joint_velocity_obs_noise_std=0.0,
    )
    fields = ObsCfg().obs_list
    noisy = policy(env)
    # Fresh-episode flush makes the delayed state the current state; noise is
    # applied on top, so object_rot stays a unit quaternion.
    norms = np.linalg.norm(noisy[:, _field_slice(fields, "object_rot")], axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)
    # With zero-mean noise the noisy keypoints stay near the clean ones.
    clean_frame = _policy_term(env, **_NO_DR)(env)
    sl = _field_slice(fields, "keypoints_rel_goal")
    diff = noisy[:, sl] - clean_frame[:, sl]
    assert not np.allclose(diff, 0.0)
    assert np.abs(diff).mean() < 0.02  # ~0.01 xyz std on 0.03-scale keypoints


def test_joint_velocity_noise_actor_only():
    """obs_utils.py:334-337: actor joint_vel gets std-0.1 noise, critic none."""
    env, backend, hub, action = _make_env_and_state()
    env.episode_length_buf[:] = 0
    hub.goal_count[:] = 0
    env.common_step_counter = 1
    policy = _policy_term(env, **{**_NO_DR, "joint_velocity_obs_noise_std": 0.1})
    critic = _critic_term(env)
    policy_frame = policy(env)
    critic_frame = critic(env)
    psl = _field_slice(ObsCfg().obs_list, "joint_vel")
    csl = _field_slice(ObsCfg().state_list, "joint_vel")
    diff = policy_frame[:, psl] - critic_frame[:, csl]
    assert not np.allclose(diff, 0.0)
    assert abs(float(diff.mean())) < 0.02
    assert 0.05 < float(diff.std()) < 0.15


def test_policy_obs_delay_flush_and_roll():
    """obs_utils.py:223-228: whole-frame delay queue keeps the 140 width."""
    env, backend, hub, action = _make_env_and_state()
    env.episode_length_buf[:] = 0
    hub.goal_count[:] = 0
    env.common_step_counter = 1
    policy = _policy_term(env, **{**_NO_DR, "use_obs_delay": True, "obs_delay_max": 3})
    frame1 = policy(env)
    assert frame1.shape == (NUM_ENVS, POLICY_OBS_DIM)
    # Fresh-episode flush: every slot holds the current frame.
    np.testing.assert_allclose(policy._obs_queue[:, 0], frame1, rtol=1e-6)
    np.testing.assert_allclose(policy._obs_queue[:, 2], frame1, rtol=1e-6)
    # Change state and step: the queue rolls; slot 1 keeps the old frame.
    backend.body_pos[:, OBJECT_BODY_ID] = [1.0, 1.0, 1.0]
    env.episode_length_buf[:] = 1
    env.common_step_counter = 2
    frame2 = policy(env)
    assert frame2.shape == (NUM_ENVS, POLICY_OBS_DIM)
    np.testing.assert_allclose(policy._obs_queue[:, 1], frame1, rtol=1e-6)
    # Slot 0 holds the freshly pushed frame (no noise -> the clean frame).
    clean_now = _policy_term(env, **_NO_DR)(env)
    np.testing.assert_allclose(policy._obs_queue[:, 0], clean_now, rtol=1e-6)
    # The served frame is one of the queued frames per env.
    for row in range(NUM_ENVS):
        served = any(
            np.allclose(frame2[row], policy._obs_queue[row, slot], rtol=1e-5) for slot in range(3)
        )
        assert served, f"row {row} not served from the queue"


def test_same_step_reentry_refreshes_only_reset_rows():
    """Autoreset-tail recompute: non-reset envs keep the step's delayed frame,
    reset envs are flushed to the post-reset state (module docstring memo)."""
    env, backend, hub, action = _make_env_and_state()
    policy = _policy_term(
        env,
        use_object_state_delay_noise=True,
        object_state_delay_max=10,
        object_state_xyz_noise_std=0.0,
        object_state_rotation_noise_degrees=0.0,
        joint_velocity_obs_noise_std=0.0,
        use_obs_delay=True,
        obs_delay_max=3,
    )
    env.episode_length_buf[:] = 5
    hub.goal_count[:] = 0
    env.common_step_counter = 4
    first = policy(env)
    queue_after_first = policy._obs_queue.copy()
    # Autoreset tail: same step, env 2 resets with a new object pose.  The
    # pose is staged through the facade inside the reset transaction, mirroring
    # the reset event that owns the object pose in production (the transaction
    # commits full rows, so poking the backend directly would be overwritten).
    reset_ids = np.asarray([2])
    env.episode_length_buf[reset_ids] = 0
    with env.reset_transaction.scoped(reset_ids):
        env.scene["object"].write_root_link_pose_to_sim(
            np.asarray([[-5.0, -5.0, -5.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            env_ids=reset_ids,
        )
        hub.reset(reset_ids)
    second = policy(env)
    # Non-reset rows: identical output, and their queue rows were not pushed.
    np.testing.assert_allclose(second[[0, 1, 3]], first[[0, 1, 3]], rtol=1e-6)
    np.testing.assert_allclose(
        policy._obs_queue[[0, 1, 3]], queue_after_first[[0, 1, 3]], rtol=1e-6
    )
    # The reset row was flushed: its delayed object state is the new pose, and
    # the goal is the freshly resampled absolute pose committed to GoalViz
    # (reset_utils.py:388: full reset → always absolute).
    sl = _field_slice(ObsCfg().obs_list, "keypoints_rel_goal")
    assert not np.allclose(second[reset_ids][:, sl], first[reset_ids][:, sl])
    corners = np.asarray(KEYPOINT_CORNERS, dtype=np.float32) * 0.03
    obj_kp = np.asarray([-5.0, -5.0, -5.0]) + corners
    goal_pos = backend.body_pos[2, GOALVIZ_BODY_ID]
    goal_quat = backend.body_quat[2, GOALVIZ_BODY_ID]
    goal_kp = goal_pos + np_quat_apply(
        goal_quat[None, :].astype(np.float64), corners.astype(np.float64)
    )
    np.testing.assert_allclose(
        second[reset_ids][0, sl], (obj_kp - goal_kp).reshape(-1), rtol=1e-5, atol=1e-6
    )


def test_row_scoped_observation_reentry_avoids_full_term_recompute():
    """The reset-row fast path preserves delay queues for untouched rows."""
    env, backend, hub, action = _make_env_and_state()
    policy = _policy_term(
        env,
        use_object_state_delay_noise=True,
        object_state_delay_max=10,
        object_state_xyz_noise_std=0.0,
        object_state_rotation_noise_degrees=0.0,
        joint_velocity_obs_noise_std=0.0,
        use_obs_delay=True,
        obs_delay_max=3,
    )
    env.episode_length_buf[:] = 5
    hub.goal_count[:] = 0
    env.common_step_counter = 4
    policy(env)
    queue_after_first = policy._obs_queue.copy()
    reset_ids = np.asarray([2])
    env.episode_length_buf[reset_ids] = 0
    with env.reset_transaction.scoped(reset_ids):
        env.scene["object"].write_root_link_pose_to_sim(
            np.asarray([[-5.0, -5.0, -5.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            env_ids=reset_ids,
        )
        hub.reset(reset_ids)
    rows = policy.compute_rows(env, reset_ids)
    assert rows.shape == (1, POLICY_OBS_DIM)
    np.testing.assert_allclose(
        policy._obs_queue[[0, 1, 3]], queue_after_first[[0, 1, 3]], rtol=1e-6
    )
    sl = _field_slice(ObsCfg().obs_list, "keypoints_rel_goal")
    assert not np.allclose(rows[:, sl], queue_after_first[reset_ids, 0, sl])


def test_row_scoped_critic_observation_matches_full_slice():
    env, backend, hub, action = _make_env_and_state()
    critic = _critic_term(env, **_NO_DR)
    full = critic(env)
    ids = np.asarray([1, 6], dtype=np.int32)
    rows = critic.compute_rows(env, ids)
    np.testing.assert_array_equal(rows, full[ids])


def test_row_scoped_policy_matches_full_post_reset_rows():
    """Selected policy rows match the original full post-reset calculation."""
    params = {
        "use_object_state_delay_noise": True,
        "object_state_delay_max": 10,
        "object_state_xyz_noise_std": 0.01,
        "object_state_rotation_noise_degrees": 5.0,
        "joint_velocity_obs_noise_std": 0.1,
        "use_obs_delay": True,
        "obs_delay_max": 3,
    }
    env_rows, backend_rows, hub_rows, action_rows = _make_env_and_state()
    env_full, backend_full, hub_full, action_full = _make_env_and_state()
    policy_rows = _policy_term(env_rows, **params)
    policy_full = _policy_term(env_full, **params)
    env_rows.episode_length_buf[:] = 5
    env_full.episode_length_buf[:] = 5
    hub_rows.goal_count[:] = 0
    hub_full.goal_count[:] = 0
    env_rows.common_step_counter = 4
    env_full.common_step_counter = 4
    np.testing.assert_array_equal(policy_rows(env_rows), policy_full(env_full))

    ids = np.asarray([2, 5], dtype=np.int32)
    for env, hub in ((env_rows, hub_rows), (env_full, hub_full)):
        env.episode_length_buf[ids] = 0
        with env.reset_transaction.scoped(ids):
            hub.reset(ids)

    rng_state = env_full.rng.bit_generator.state
    row_output = policy_rows.compute_rows(env_rows, ids)
    env_full.rng.bit_generator.state = rng_state
    full_output = policy_full(env_full)[ids]
    np.testing.assert_array_equal(row_output, full_output)


def test_policy_reset_zeroes_queues():
    """reset_utils.py:399-401: observation delay queues zeroed per env."""
    env, backend, hub, action = _make_env_and_state()
    policy = _policy_term(env, **{**_NO_DR, "use_obs_delay": True})
    env.episode_length_buf[:] = 0
    hub.goal_count[:] = 0
    env.common_step_counter = 1
    policy(env)
    ids = np.asarray([0, 4])
    policy.reset(ids)
    np.testing.assert_array_equal(policy._object_state_queue[ids], 0.0)
    np.testing.assert_array_equal(policy._obs_queue[ids], 0.0)


def test_object_scale_multiplier_feeds_scales_and_keypoints():
    """obs_utils.py:275, 301: scales obs = scale_per_env * multiplier and the
    keypoint offsets scale per axis by the same product."""
    env, backend, hub, action = _make_env_and_state()
    hub.object_scale_per_env[:] = [2.0, 3.0, 4.0]
    hub.object_scale_multiplier[:] = [1.0, 0.5, 2.0]
    policy = _policy_term(env, **_NO_DR)
    frame = policy(env)
    np.testing.assert_allclose(
        frame[:, _field_slice(ObsCfg().obs_list, "object_scales")],
        _rows([2.0, 1.5, 8.0]),
        rtol=1e-6,
    )
    corners = np.asarray(KEYPOINT_CORNERS, dtype=np.float32)
    kp_offsets = corners * 0.03 * np.asarray([2.0, 1.5, 8.0])
    obj_kp = np.asarray([0.4, -0.1, 0.7]) + kp_offsets
    goal_kp = np.asarray([-0.3, 0.6, 0.9]) + kp_offsets
    np.testing.assert_allclose(
        frame[:, _field_slice(ObsCfg().obs_list, "keypoints_rel_goal")],
        _rows((obj_kp - goal_kp).reshape(-1)),
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# Reset-row stale-palm replication (reaudit fix P4 / R2)
# ---------------------------------------------------------------------------


def _stale_env_with_terminal_rows():
    """Env whose step frame is memoized and rows 1,2 marked terminal."""
    from unilab.base.np_env import NpEnvState

    env, backend, hub, action = _make_env_and_state()
    env.common_step_counter = 3
    policy = _policy_term(env, **_NO_DR)
    critic = _critic_term(env, **_NO_DR)
    policy(env)
    critic(env)
    mask = np.zeros(NUM_ENVS, dtype=bool)
    mask[[1, 2]] = True
    env.state = NpEnvState(
        obs={"obs": np.zeros((NUM_ENVS, POLICY_OBS_DIM), dtype=np.float32),
             "critic": np.zeros((NUM_ENVS, CRITIC_OBS_DIM), dtype=np.float32)},
        reward=np.zeros(NUM_ENVS, dtype=np.float32),
        terminated=np.zeros(NUM_ENVS, dtype=bool),
        truncated=np.zeros(NUM_ENVS, dtype=bool),
        info={"_final_observation": mask},
    )
    # Post-reset world: the reset transaction re-posed the hand and the
    # object, so post-reset palm/object values differ from the terminal frame.
    backend.body_pos[:, PALM_BACKEND_ID] = [0.3, 0.55, 0.62]
    backend.body_pos[:, OBJECT_BODY_ID] = [0.4, -0.1, 0.65]
    backend.body_quat[:, OBJECT_BODY_ID] = [0.70710678, 0.70710678, 0.0, 0.0]
    env.episode_length_buf[:] = 0
    hub.goal_count[:] = 0
    return env, backend, hub, policy, critic


def test_reset_row_policy_obs_replicates_stale_palm():
    """Terminal reset rows keep the pre-reset palm fields; fresh rows do not.

    The original's reset-row observation reads palm/fingertip body state from
    buffers last collected before the reset writes (direct_rl_env step order),
    while joints/object/goal come from the reset (reaudit R2/D1).
    """
    env, backend, hub, policy, _ = _stale_env_with_terminal_rows()
    raw = policy._raw_frame
    assert raw is not None
    ids = np.asarray([0, 1, 2])
    rows = policy.compute_rows(env, ids)

    palm_sl = _field_slice(ObsCfg().obs_list, "palm_pos")
    ft_sl = _field_slice(ObsCfg().obs_list, "fingertip_pos_rel_palm")
    # Terminal rows: stale palm values from the pre-reset frame.
    np.testing.assert_allclose(rows[1, palm_sl], raw[1, palm_sl], rtol=1e-6)
    np.testing.assert_allclose(rows[2, palm_sl], raw[2, palm_sl], rtol=1e-6)
    np.testing.assert_allclose(rows[1, ft_sl], raw[1, ft_sl], rtol=1e-6)
    # Fresh row: palm re-derived from the (unchanged) scene state, not raw.
    assert not np.allclose(rows[0, palm_sl], raw[0, palm_sl])

    # keypoints_rel_palm recomputes fresh object keypoints against the STALE
    # palm position (the original's mixed frame).  The fresh object carries a
    # 90-degree x rotation, so the corners rotate with it: (x, y, z) -> (x, -z, y).
    corners = (np.asarray(KEYPOINT_CORNERS, dtype=np.float32) * 0.03).astype(np.float64)
    rotated = np.stack([corners[:, 0], -corners[:, 2], corners[:, 1]], axis=-1)
    obj_kp = np.asarray([0.4, -0.1, 0.65]) + rotated
    kp_sl = _field_slice(ObsCfg().obs_list, "keypoints_rel_palm")
    expected = (obj_kp - raw[1, palm_sl]).reshape(-1)
    np.testing.assert_allclose(rows[1, kp_sl], expected, rtol=1e-4, atol=1e-6)


def test_reset_row_critic_obs_replicates_stale_palm_and_vel():
    env, backend, hub, _, critic = _stale_env_with_terminal_rows()
    raw = critic._raw_frame
    ids = np.asarray([1, 2])
    rows = critic.compute_rows(env, ids)

    for name in ("palm_pos", "palm_rot", "palm_vel", "fingertip_pos_rel_palm"):
        sl = _field_slice(ObsCfg().state_list, name)
        np.testing.assert_allclose(rows[0, sl], raw[1, sl], rtol=1e-6)
    # Fresh fields: object rot comes from the post-reset state (90° about x),
    # not the pre-reset identity frame.
    obj_sl = _field_slice(ObsCfg().state_list, "object_rot")
    np.testing.assert_allclose(
        rows[0, obj_sl], np.asarray([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float32),
        rtol=1e-5,
    )


def test_reset_row_fresh_mode_keeps_post_reset_palm():
    """reset_row_body_state='fresh' restores the pre-P4 coherent frame."""
    env, backend, hub, _, critic = _stale_env_with_terminal_rows()
    fresh = _critic_term(env, reset_row_body_state="fresh", clamp_abs_observations=10.0)
    rows = fresh.compute_rows(env, np.asarray([1, 2]))
    raw = critic._raw_frame
    palm_sl = _field_slice(ObsCfg().state_list, "palm_pos")
    assert not np.allclose(rows[0, palm_sl], raw[1, palm_sl])


def test_manual_reset_rows_stay_fresh_without_terminal_mask():
    """No terminal mask (manual/cold reset): rows keep post-reset values."""
    env, backend, hub, action = _make_env_and_state()
    env.common_step_counter = 5
    critic = _critic_term(env, **_NO_DR)
    critic(env)
    # Simulate a manual reset that re-posed the hand after the last step.
    backend.body_pos[:, PALM_BACKEND_ID] = [0.3, 0.55, 0.62]
    rows = critic.compute_rows(env, np.asarray([1]))
    raw = critic._raw_frame
    palm_sl = _field_slice(ObsCfg().state_list, "palm_pos")
    assert not np.allclose(rows[0, palm_sl], raw[1, palm_sl])
