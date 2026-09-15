"""Step-1.4b-3 tests for the SimToolReal termination terms.

Same fake-backend pattern as tests/tasks/test_simtoolreal_rewards.py: a real
``EntityScene`` over a fake ``SimBackend`` (palm + 5 fingertip + object_root +
goalviz_root bodies; joint state unused here) plus the real
``SimToolRealCommand`` hub, so the success chain, the pending_success writes
and the time-out mask are exercised through the production facades.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
from unisim.backend.base import SimBackend

from unilab.base.entity import EntityCfg, EntityScene
from unilab.managers import TerminationManager, TerminationTermCfg
from unilab.managers._types import ManagerBasedRlEnv
from unilab.managers.manager_base import ManagerTermBaseCfg
from simtoolreal_unilab.tasks.simtoolreal.commands import SimToolRealCommandCfg
from simtoolreal_unilab.tasks.simtoolreal.config import (
    FINGERTIP_LINK_NAMES,
    NUM_JOINTS,
    PALM_BODY_NAME,
)
from simtoolreal_unilab.tasks.simtoolreal.terminations import (
    Fall,
    HandFar,
    MaxConsecutiveSuccesses,
    SuccessTracker,
    time_out,
)

NUM_ENVS = 8
STEP_DT = 1.0 / 60.0
MAX_EPISODE_LENGTH = 600

ROBOT_BODY_NAMES = (PALM_BODY_NAME, *FINGERTIP_LINK_NAMES)
ALL_BODY_NAMES = (*ROBOT_BODY_NAMES, "object_root", "goalviz_root")
FINGERTIP_BACKEND_IDS = [ALL_BODY_NAMES.index(n) for n in FINGERTIP_LINK_NAMES]
OBJECT_BODY_ID = ALL_BODY_NAMES.index("object_root")
GOALVIZ_BODY_ID = ALL_BODY_NAMES.index("goalviz_root")

OBJECT_POS = np.asarray([0.4, -0.1, 0.7], dtype=np.float32)
GOAL_POS = np.asarray([-0.3, 0.6, 0.9], dtype=np.float32)
FIXED_SIZE = [0.141, 0.03025, 0.0271]
SUCCESS_PARAMS = {
    "success_steps": 10,
    "force_consecutive": False,
    "keypoint_scale": 1.5,
    "fixed_size": FIXED_SIZE,
}


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

    def get_joint_dof_pos_indices(self, names) -> np.ndarray:
        return np.arange(len(names), dtype=np.int32)

    def get_joint_dof_vel_indices(self, names) -> np.ndarray:
        return self.get_joint_dof_pos_indices(names)

    def get_dof_pos(self) -> np.ndarray:
        return self.dof_pos

    def get_dof_vel(self) -> np.ndarray:
        return self.dof_vel

    def get_default_dof_pos(self) -> np.ndarray:
        return np.zeros(NUM_JOINTS, dtype=np.float32)

    def get_joint_range(self) -> np.ndarray:
        return np.tile(np.asarray([[-2.0, 2.0]], dtype=np.float32), (NUM_JOINTS, 1))

    def get_actuator_names(self) -> tuple[str, ...]:
        return tuple(f"joint_{k}" for k in range(NUM_JOINTS))

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        return self.get_actuator_names()

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
        return np.zeros((NUM_ENVS, len(ids), 3), dtype=np.float32)

    def get_body_ang_vel_w(self, ids) -> np.ndarray:
        return np.zeros((NUM_ENVS, len(ids), 3), dtype=np.float32)

    def get_body_lin_vel_b(self, ids) -> np.ndarray:
        return np.zeros((NUM_ENVS, len(ids), 3), dtype=np.float32)

    def get_body_ang_vel_b(self, ids) -> np.ndarray:
        return np.zeros((NUM_ENVS, len(ids), 3), dtype=np.float32)


def _make_env(seed: int = 7):
    backend = _FakeBackend()
    control = np.zeros((NUM_ENVS, NUM_JOINTS), dtype=np.float32)
    scene = EntityScene(
        {
            "robot": EntityCfg(
                joint_names=backend.get_actuator_names(),
                body_names=ROBOT_BODY_NAMES,
                actuator_names=backend.get_actuator_names(),
            ),
            "object": EntityCfg(root_body_name="object_root"),
            "goalviz": EntityCfg(root_body_name="goalviz_root"),
        },
        cast(SimBackend, backend),
        control,
    )
    env = SimpleNamespace(
        num_envs=NUM_ENVS,
        rng=np.random.default_rng(seed),
        step_dt=STEP_DT,
        scene=scene,
        episode_length_buf=np.ones(NUM_ENVS, dtype=np.int64),
        max_episode_length=MAX_EPISODE_LENGTH,
        common_step_counter=0,
    )
    hub = SimToolRealCommandCfg(resampling_time_range=(1e6, 1e6)).build(
        cast(ManagerBasedRlEnv, env)
    )
    env.command_manager = SimpleNamespace(get_term=lambda name: hub)
    backend.body_pos[:, OBJECT_BODY_ID] = OBJECT_POS
    backend.body_pos[:, GOALVIZ_BODY_ID] = GOAL_POS
    # Goal readers consume the hub buffer (reaudit fix P3a); a real commit
    # keeps the hub and the GoalViz sim pose in sync.
    hub.goal[:, :3] = GOAL_POS
    return env, backend, hub


def _term(cls, env, **params):
    cfg = ManagerTermBaseCfg(func=cls, params=params)
    return cls(cfg=cfg, env=cast(ManagerBasedRlEnv, env))


def _place_object_near_goal(backend, distance: float) -> None:
    """Identity rotations on both bodies: keypoint offsets cancel, so the
    keypoint max distance equals the center distance."""
    backend.body_pos[:, GOALVIZ_BODY_ID] = GOAL_POS
    backend.body_pos[:, OBJECT_BODY_ID] = GOAL_POS + np.asarray(
        [distance, 0.0, 0.0], dtype=np.float32
    )


# ----------------------------------------------------------------------------
# Success chain (obs_utils.py:190-205, reward_utils.py:79-84)
# ----------------------------------------------------------------------------


def test_success_chain_amortized_boundaries():
    env, backend, hub = _make_env()
    term = _term(SuccessTracker, env, **SUCCESS_PARAMS)
    hub.current_success_tolerance = 0.1  # tol = 0.1 * keypoint_scale = 0.15
    _place_object_near_goal(backend, 0.16)  # beyond tolerance
    out = term(env)
    assert out.dtype == np.bool_ and not out.any()  # success never terminates
    assert not hub.near_goal.any()
    np.testing.assert_array_equal(hub.near_goal_steps, 0)
    assert not hub.pending_success.any()
    # Inside tolerance: the counter accumulates; success at step 10.
    _place_object_near_goal(backend, 0.14)
    for step in range(1, 11):
        term(env)
        assert hub.near_goal.all()
        np.testing.assert_array_equal(hub.near_goal_steps, step)
        assert hub.pending_success.all() == (step >= 10)
    # Non-consecutive mode: leaving the goal region does not reset the counter
    # (reward_utils.py:82-83); pending_success persists until the command stage
    # clears near_goal_steps on goal advance.
    _place_object_near_goal(backend, 0.16)
    term(env)
    assert not hub.near_goal.any()
    np.testing.assert_array_equal(hub.near_goal_steps, 10)
    assert hub.pending_success.all()


def test_success_chain_force_consecutive_resets_on_gap():
    env, backend, hub = _make_env()
    params = {**SUCCESS_PARAMS, "force_consecutive": True}
    term = _term(SuccessTracker, env, **params)
    hub.current_success_tolerance = 0.1
    _place_object_near_goal(backend, 0.14)
    for _ in range(5):
        term(env)
    np.testing.assert_array_equal(hub.near_goal_steps, 5)
    # A single far step zeroes the counter (reward_utils.py:81-82).
    _place_object_near_goal(backend, 0.16)
    term(env)
    np.testing.assert_array_equal(hub.near_goal_steps, 0)
    _place_object_near_goal(backend, 0.14)
    for _ in range(9):
        term(env)
    assert not hub.pending_success.any()
    term(env)  # 10th consecutive near-goal step
    assert hub.pending_success.all()


def test_success_chain_tolerance_uses_keypoint_scale_and_curriculum():
    """tol = hub.current_success_tolerance * keypoint_scale (obs_utils.py:193-194)."""
    env, backend, hub = _make_env()
    term = _term(SuccessTracker, env, **SUCCESS_PARAMS)
    hub.current_success_tolerance = 0.1
    _place_object_near_goal(backend, 0.149)
    term(env)
    assert hub.near_goal.all()  # 0.149 <= 0.15
    _place_object_near_goal(backend, 0.151)
    term(env)
    assert not hub.near_goal.any()
    # Curriculum shrink (written by the command hub) tightens the criterion.
    hub.current_success_tolerance = 0.05  # tol = 0.075
    _place_object_near_goal(backend, 0.08)
    term(env)
    assert not hub.near_goal.any()
    _place_object_near_goal(backend, 0.07)
    term(env)
    assert hub.near_goal.all()


def test_success_chain_per_env_independence():
    env, backend, hub = _make_env()
    term = _term(SuccessTracker, env, **SUCCESS_PARAMS)
    hub.current_success_tolerance = 0.1
    _place_object_near_goal(backend, 0.14)
    backend.body_pos[3, OBJECT_BODY_ID] = GOAL_POS + np.asarray([0.5, 0.0, 0.0])
    term(env)
    expected_near = np.ones(NUM_ENVS, dtype=bool)
    expected_near[3] = False
    np.testing.assert_array_equal(hub.near_goal, expected_near)
    np.testing.assert_array_equal(hub.near_goal_steps, expected_near.astype(np.int64))


# ----------------------------------------------------------------------------
# fall / hand_far (termination_utils.py:54-55, :62)
# ----------------------------------------------------------------------------


def test_fall_threshold():
    env, backend, hub = _make_env()
    term = _term(Fall, env, fall_height=0.1)
    assert not term(env).any()  # object at z=0.7
    backend.body_pos[:, OBJECT_BODY_ID, 2] = 0.1
    assert not term(env).any()  # strict <
    backend.body_pos[:, OBJECT_BODY_ID, 2] = 0.099
    assert term(env).all()
    backend.body_pos[2, OBJECT_BODY_ID, 2] = 0.5
    np.testing.assert_array_equal(term(env), [True, True, False] + [True] * 5)


def test_hand_far_threshold():
    env, backend, hub = _make_env()
    term = _term(HandFar, env, hand_far_distance=1.5)
    for body_id in FINGERTIP_BACKEND_IDS:
        backend.body_pos[:, body_id] = OBJECT_POS
    assert not term(env).any()  # all fingertips on the object
    # Exactly at the threshold does not trigger (strict >).
    backend.body_pos[:, FINGERTIP_BACKEND_IDS[0]] = OBJECT_POS + [1.5, 0.0, 0.0]
    assert not term(env).any()
    backend.body_pos[4, FINGERTIP_BACKEND_IDS[2]] = OBJECT_POS + [0.0, 1.6, 0.0]
    expected = np.zeros(NUM_ENVS, dtype=bool)
    expected[4] = True
    np.testing.assert_array_equal(term(env), expected)


# ----------------------------------------------------------------------------
# max_consecutive_successes (termination_utils.py:57-60 with :46 fold-in)
# ----------------------------------------------------------------------------


def test_max_consecutive_successes_counts_pending_same_step():
    env, backend, hub = _make_env()
    term = _term(MaxConsecutiveSuccesses, env, max_consecutive_successes=50)
    hub.goal_count[:] = 49
    assert not term(env).any()
    # The 50th success terminates on the same step (termination_utils.py:46
    # increments before the :58 comparison; goal_count only increments in the
    # later command stage, so the term folds in pending_success).
    hub.pending_success[1] = True
    expected = np.zeros(NUM_ENVS, dtype=bool)
    expected[1] = True
    np.testing.assert_array_equal(term(env), expected)
    hub.goal_count[:] = 50
    hub.pending_success[:] = False
    assert term(env).all()


def test_max_consecutive_successes_disabled_nonpositive():
    env, backend, hub = _make_env()
    for cap in (0, -1):
        term = _term(MaxConsecutiveSuccesses, env, max_consecutive_successes=cap)
        hub.goal_count[:] = 100
        hub.pending_success[:] = True
        assert not term(env).any()


# ----------------------------------------------------------------------------
# time_out mask (termination_utils.py:50-51, 65)
# ----------------------------------------------------------------------------


def test_time_out_masks_success_frame():
    env, backend, hub = _make_env()
    env.episode_length_buf[:] = MAX_EPISODE_LENGTH
    hub.pending_success[:] = False
    assert time_out(env).all()  # ordinary truncation
    hub.pending_success[0] = True
    hub.pending_success[5] = True
    expected = np.ones(NUM_ENVS, dtype=bool)
    expected[[0, 5]] = False  # success frames never truncate
    np.testing.assert_array_equal(time_out(env), expected)
    env.episode_length_buf[:] = MAX_EPISODE_LENGTH - 1
    assert not time_out(env).any()


# ----------------------------------------------------------------------------
# TerminationManager integration: terminated/truncated split
# ----------------------------------------------------------------------------


def test_termination_manager_split_and_hub_writes():
    env, backend, hub = _make_env()
    cfg = {
        "success_tracker": TerminationTermCfg(func=SuccessTracker, params=SUCCESS_PARAMS),
        "fall": TerminationTermCfg(func=Fall, params={"fall_height": 0.1}),
        "hand_far": TerminationTermCfg(func=HandFar, params={"hand_far_distance": 1.5}),
        "max_consecutive_successes": TerminationTermCfg(
            func=MaxConsecutiveSuccesses, params={"max_consecutive_successes": 50}
        ),
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
    }
    manager = TerminationManager(cfg, cast(ManagerBasedRlEnv, env))

    # All envs exactly at the goal (kmd 0).  env0: fallen object; env1: hand
    # far; env2: at max length on its 10th near-goal step (success frame,
    # masked from truncation); env3: at max length (real truncation).
    hub.current_success_tolerance = 0.1
    _place_object_near_goal(backend, 0.0)
    backend.body_pos[0, OBJECT_BODY_ID, 2] = GOAL_POS[2] - 0.85  # z 0.05: fallen
    backend.body_pos[1, FINGERTIP_BACKEND_IDS[0]] = GOAL_POS + [2.0, 0.0, 0.0]
    hub.near_goal_steps[2] = 9
    env.episode_length_buf[[2, 3]] = MAX_EPISODE_LENGTH

    dones = manager.compute()
    np.testing.assert_array_equal(manager.terminated, [True, True] + [False] * 6)
    np.testing.assert_array_equal(manager.time_outs, [False] * 3 + [True] + [False] * 4)
    np.testing.assert_array_equal(dones, manager.terminated | manager.time_outs)
    expected_fall = np.zeros(NUM_ENVS, dtype=bool)
    expected_fall[0] = True
    np.testing.assert_array_equal(manager.get_term("fall"), expected_fall)
    expected_hand_far = np.zeros(NUM_ENVS, dtype=bool)
    expected_hand_far[1] = True
    np.testing.assert_array_equal(manager.get_term("hand_far"), expected_hand_far)
    # The tracker ran first and refreshed the hub snapshot for the step.
    expected_near = np.ones(NUM_ENVS, dtype=bool)
    expected_near[0] = False  # fallen env is 0.85 from the goal
    np.testing.assert_array_equal(hub.near_goal, expected_near)
    expected_steps = np.ones(NUM_ENVS, dtype=np.int64)
    expected_steps[0] = 0
    expected_steps[2] = 10
    np.testing.assert_array_equal(hub.near_goal_steps, expected_steps)
    expected_success = np.zeros(NUM_ENVS, dtype=bool)
    expected_success[2] = True
    np.testing.assert_array_equal(hub.pending_success, expected_success)
