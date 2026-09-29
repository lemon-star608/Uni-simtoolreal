"""Step-1.4b-2 tests for the SimToolReal reward terms.

Fake-backend pattern follows tests/tasks/test_simtoolreal_actions_observations.py:
a real ``EntityScene`` over a fake ``SimBackend`` (29 DoF in a rotated canonical
order, palm + 5 fingertip + object_root + goalviz_root bodies) plus the real
``SimToolRealCommand`` hub, so gates, latches, tracker updates and the
fixed-size keypoint branch are exercised through the production facades.
Reward terms write no sim state, so no reset transaction is needed here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
from unilab.base.entity import EntityCfg, EntityScene
from unilab.managers import RewardManager, RewardTermCfg
from unilab.managers._types import ManagerBasedRlEnv
from unilab.managers.manager_base import ManagerTermBaseCfg
from unisim.backend.base import SimBackend

from simtoolreal_unilab.tasks.simtoolreal.commands import SimToolRealCommandCfg
from simtoolreal_unilab.tasks.simtoolreal.config import (
    FINGERTIP_LINK_NAMES,
    JOINT_NAMES_CANONICAL,
    KEYPOINT_CORNERS,
    NUM_ARM_JOINTS,
    NUM_JOINTS,
    PALM_BODY_NAME,
)
from simtoolreal_unilab.tasks.simtoolreal.rewards import (
    ArmJointVelocityPenalty,
    FingertipDistanceDelta,
    HandJointVelocityPenalty,
    KeypointDistanceDelta,
    LiftingBonus,
    LiftProgress,
    ReachBonus,
)

NUM_ENVS = 8
STEP_DT = 1.0 / 60.0

# Backend joint order is the canonical order rotated by 3 (non-involutive
# permutation) so gather-direction bugs in the penalty terms are detectable.
_SHIFT = 3
BACKEND_JOINT_NAMES = tuple(
    JOINT_NAMES_CANONICAL[(k + _SHIFT) % NUM_JOINTS] for k in range(NUM_JOINTS)
)
CANON_TO_BACKEND = np.asarray(
    [BACKEND_JOINT_NAMES.index(name) for name in JOINT_NAMES_CANONICAL], dtype=np.intp
)

ROBOT_BODY_NAMES = (PALM_BODY_NAME, *FINGERTIP_LINK_NAMES)
ALL_BODY_NAMES = (*ROBOT_BODY_NAMES, "object_root", "goalviz_root")
FINGERTIP_BACKEND_IDS = [ALL_BODY_NAMES.index(n) for n in FINGERTIP_LINK_NAMES]
OBJECT_BODY_ID = ALL_BODY_NAMES.index("object_root")
GOALVIZ_BODY_ID = ALL_BODY_NAMES.index("goalviz_root")

OBJECT_POS = np.asarray([0.4, -0.1, 0.7], dtype=np.float32)
GOAL_POS = np.asarray([-0.3, 0.6, 0.9], dtype=np.float32)


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
        return np.tile(np.asarray([[-2.0, 2.0]], dtype=np.float32), (NUM_JOINTS, 1))

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
                joint_names=BACKEND_JOINT_NAMES,
                body_names=ROBOT_BODY_NAMES,
                actuator_names=BACKEND_JOINT_NAMES,
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
        common_step_counter=0,
    )
    hub = SimToolRealCommandCfg(resampling_time_range=(1e6, 1e6)).build(
        cast(ManagerBasedRlEnv, env)
    )
    env.command_manager = SimpleNamespace(get_term=lambda name: hub)
    # Recognizable default geometry: object and goalviz at fixed poses.  The
    # goal also mirrors into the hub buffer: a real commit keeps the two in
    # sync, and the goal readers consume the hub (reaudit fix P3a).
    backend.body_pos[:, OBJECT_BODY_ID] = OBJECT_POS
    backend.body_pos[:, GOALVIZ_BODY_ID] = GOAL_POS
    hub.goal[:, :3] = GOAL_POS
    hub.object_init_z[:] = OBJECT_POS[2]
    return env, backend, hub


def _term(cls, env, **params):
    cfg = ManagerTermBaseCfg(func=cls, params=params)
    return cls(cfg=cfg, env=cast(ManagerBasedRlEnv, env))


def _set_object_z(backend, hub, z_lift: float) -> None:
    """Place the object so that z_lift == the requested value (init z fixed)."""
    backend.body_pos[:, OBJECT_BODY_ID, 2] = hub.object_init_z + (z_lift - 0.05)


# ----------------------------------------------------------------------------
# Lift progress / lifting bonus / latch (reward_utils.py:8-23)
# ----------------------------------------------------------------------------


def test_lift_progress_before_crossing_and_clip():
    env, backend, hub = _make_env()
    progress = _term(LiftProgress, env, bonus_threshold=0.15)
    _set_object_z(backend, hub, 0.10)
    np.testing.assert_allclose(progress(env), 0.10, rtol=1e-6)
    assert not hub.lifted.any()  # the progress term never writes the latch
    # The 0.5 clip is reachable when a custom threshold exceeds 0.5.
    loose = _term(LiftProgress, env, bonus_threshold=0.6)
    _set_object_z(backend, hub, 0.55)
    np.testing.assert_allclose(loose(env), 0.5, rtol=1e-6)


def test_lifting_bonus_one_shot_and_latch_write():
    env, backend, hub = _make_env()
    bonus = _term(LiftingBonus, env, bonus_threshold=0.15)
    _set_object_z(backend, hub, 0.10)  # below threshold
    np.testing.assert_array_equal(bonus(env), 0.0)
    assert not hub.lifted.any()
    _set_object_z(backend, hub, 0.16)  # crossing step
    np.testing.assert_array_equal(bonus(env), 1.0)
    assert hub.lifted.all()
    np.testing.assert_array_equal(bonus(env), 0.0)  # latched: no second bonus


def test_lift_terms_are_manager_order_independent():
    """The crossing step zeroes progress and pays the bonus in both orders."""
    for first, second in ((LiftProgress, LiftingBonus), (LiftingBonus, LiftProgress)):
        env, backend, hub = _make_env()
        _set_object_z(backend, hub, 0.16)
        term_a = _term(first, env, bonus_threshold=0.15)
        term_b = _term(second, env, bonus_threshold=0.15)
        values = np.stack([term_a(env), term_b(env)])
        # One term returns 0 (progress, gated by the fresh latch), the other 1.
        np.testing.assert_allclose(sorted(values[:, 0]), [0.0, 1.0], rtol=1e-6)
        assert hub.lifted.all()


# ----------------------------------------------------------------------------
# Fingertip distance delta (reward_utils.py:26-40, obs_utils.py:184-188)
# ----------------------------------------------------------------------------


def _place_fingertips(backend, distances: np.ndarray) -> None:
    """Fingertip k sits at object + (distances[k], 0, 0)."""
    for k, body_id in enumerate(FINGERTIP_BACKEND_IDS):
        backend.body_pos[:, body_id] = OBJECT_POS + np.asarray(
            [distances[k], 0.0, 0.0], dtype=np.float32
        )


def test_fingertip_delta_sentinel_progress_gate_and_tracker():
    env, backend, hub = _make_env()
    term = _term(FingertipDistanceDelta, env, bonus_threshold=0.15)
    _place_fingertips(backend, np.asarray([1.0, 0.9, 0.8, 0.7, 0.6]))
    # Sentinel -1: the first observed distances become closest-so-far, no reward.
    np.testing.assert_array_equal(term(env), 0.0)
    np.testing.assert_allclose(hub.closest_fingertip_dist[0], [1.0, 0.9, 0.8, 0.7, 0.6], rtol=1e-6)
    # Every fingertip closes 0.05: reward is the summed delta (not lifted).
    _place_fingertips(backend, np.asarray([0.95, 0.85, 0.75, 0.65, 0.55]))
    np.testing.assert_allclose(term(env), 0.25, rtol=1e-6)
    np.testing.assert_allclose(
        hub.closest_fingertip_dist[0], [0.95, 0.85, 0.75, 0.65, 0.55], rtol=1e-6
    )
    # Moving away yields no negative reward and does not raise the tracker.
    _place_fingertips(backend, np.asarray([1.0, 0.9, 0.8, 0.7, 0.6]))
    np.testing.assert_array_equal(term(env), 0.0)
    np.testing.assert_allclose(
        hub.closest_fingertip_dist[0], [0.95, 0.85, 0.75, 0.65, 0.55], rtol=1e-6
    )
    # Lifted gate: reward zeroed, tracker still updates (compute_rewards:104).
    hub.lifted[:] = True
    _place_fingertips(backend, np.asarray([0.90, 0.80, 0.70, 0.60, 0.50]))
    np.testing.assert_array_equal(term(env), 0.0)
    np.testing.assert_allclose(
        hub.closest_fingertip_dist[0], [0.90, 0.80, 0.70, 0.60, 0.50], rtol=1e-6
    )


def test_fingertip_delta_clip_upper_bound():
    env, backend, hub = _make_env()
    term = _term(FingertipDistanceDelta, env, bonus_threshold=0.15)
    _place_fingertips(backend, np.full(5, 20.0))
    term(env)  # seed the tracker
    _place_fingertips(backend, np.zeros(5))  # delta 20 -> clipped to 10
    np.testing.assert_allclose(term(env), 5 * 10.0, rtol=1e-6)


def test_fingertip_delta_zeroes_success_frame():
    """The original clears this tracker before rewards on a goal success."""
    env, backend, hub = _make_env()
    term = _term(FingertipDistanceDelta, env, bonus_threshold=0.15)
    _place_fingertips(backend, np.full(5, 1.0))
    term(env)  # seed the tracker

    hub.pending_success[:] = False
    hub.pending_success[1] = True
    _place_fingertips(backend, np.full(5, 0.8))
    values = term(env)
    np.testing.assert_allclose(values[0], 5 * 0.2, rtol=1e-6)
    np.testing.assert_array_equal(values[1], 0.0)
    np.testing.assert_allclose(hub.closest_fingertip_dist, 0.8, rtol=1e-6)


# ----------------------------------------------------------------------------
# Keypoint distance delta (reward_utils.py:42-51; fixed-size branch, §3.5)
# ----------------------------------------------------------------------------


def _expected_fixed_kp_max_dist(obj_pos, obj_quat, goal_pos, goal_quat) -> float:
    from unilab.utils.rotation import np_quat_apply

    offsets = np.asarray(KEYPOINT_CORNERS, dtype=np.float64) * (
        0.5 * 1.5 * np.asarray([0.141, 0.03025, 0.0271])
    )
    obj_kp = obj_pos + np_quat_apply(np.broadcast_to(obj_quat, (4, 4)), offsets)
    goal_kp = goal_pos + np_quat_apply(np.broadcast_to(goal_quat, (4, 4)), offsets)
    return float(np.linalg.norm(obj_kp - goal_kp, axis=-1).max())


def test_keypoint_delta_fixed_size_branch_not_observation_scale():
    """A 180-degree goal flip exposes the fixed_size offsets (0.75 * fixed_size),
    not the observation branch's 0.03-scale offsets (DESIGN.md section 3.5)."""
    env, backend, hub = _make_env()
    # Object and goalviz centers coincide; the goal is flipped about z.
    backend.body_pos[:, GOALVIZ_BODY_ID] = OBJECT_POS
    backend.body_quat[:, GOALVIZ_BODY_ID] = [0.0, 0.0, 1.0, 0.0]  # wxyz 180° z
    hub.goal[:, :3] = OBJECT_POS
    hub.goal[:, 3:] = [0.0, 0.0, 1.0, 0.0]
    term = _term(
        KeypointDistanceDelta,
        env,
        bonus_threshold=0.15,
        keypoint_scale=1.5,
        fixed_size=[0.141, 0.03025, 0.0271],
    )
    expected = _expected_fixed_kp_max_dist(
        OBJECT_POS.astype(np.float64),
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        OBJECT_POS.astype(np.float64),
        np.asarray([0.0, 0.0, 1.0, 0.0]),
    )
    assert expected > 0.2  # fixed branch; the obs branch would give ~0.085
    hub.lifted[:] = True
    term(env)  # sentinel fill: tracker = expected, no reward
    np.testing.assert_allclose(hub.closest_keypoint_max_dist, expected, rtol=1e-5)


def test_keypoint_delta_gate_tracker_and_clip():
    env, backend, hub = _make_env()
    term = _term(
        KeypointDistanceDelta,
        env,
        bonus_threshold=0.15,
        keypoint_scale=1.5,
        fixed_size=[0.141, 0.03025, 0.0271],
    )
    # Identity rotations: every corner offset is equal, so the max distance is
    # the center distance. Start 0.2 away on x (move the goal next to the
    # object for a clean center-distance readout; the goal readers consume the
    # hub buffer, reaudit fix P3a).
    hub.goal[:, :3] = OBJECT_POS
    backend.body_pos[:, GOALVIZ_BODY_ID] = OBJECT_POS
    backend.body_pos[:, OBJECT_BODY_ID] = OBJECT_POS + np.asarray([0.2, 0.0, 0.0])
    hub.lifted[:] = False
    np.testing.assert_array_equal(term(env), 0.0)  # sentinel: no reward
    np.testing.assert_allclose(hub.closest_keypoint_max_dist, 0.2, rtol=1e-5)
    # Pre-lift gate: closing to 0.08 pays nothing but lowers the tracker.
    backend.body_pos[:, OBJECT_BODY_ID] = OBJECT_POS + np.asarray([0.08, 0.0, 0.0])
    np.testing.assert_array_equal(term(env), 0.0)
    np.testing.assert_allclose(hub.closest_keypoint_max_dist, 0.08, rtol=1e-5)
    # Post-lift: closing to 0.03 pays 0.05.
    hub.lifted[:] = True
    backend.body_pos[:, OBJECT_BODY_ID] = OBJECT_POS + np.asarray([0.03, 0.0, 0.0])
    np.testing.assert_allclose(term(env), 0.05, rtol=1e-5)
    np.testing.assert_allclose(hub.closest_keypoint_max_dist, 0.03, rtol=1e-5)
    # Moving away pays nothing and keeps the tracker.
    backend.body_pos[:, OBJECT_BODY_ID] = OBJECT_POS + np.asarray([0.30, 0.0, 0.0])
    np.testing.assert_array_equal(term(env), 0.0)
    np.testing.assert_allclose(hub.closest_keypoint_max_dist, 0.03, rtol=1e-5)


def test_keypoint_delta_zeroes_success_frame():
    """Audit fix A1.2 (termination_utils.py:46-51 -> reward_utils.py:117-123):
    the original clears the per-goal trackers inside _get_dones before rewards
    run, so the success frame's keypoint delta clamps to 0; the pending_success
    snapshot written by the termination stage reproduces that here."""
    env, backend, hub = _make_env()
    term = _term(
        KeypointDistanceDelta,
        env,
        bonus_threshold=0.15,
        keypoint_scale=1.5,
        fixed_size=[0.141, 0.03025, 0.0271],
    )
    hub.goal[:, :3] = OBJECT_POS
    backend.body_pos[:, GOALVIZ_BODY_ID] = OBJECT_POS
    hub.lifted[:] = True
    # Seed the tracker at 0.2 (sentinel step pays nothing).
    backend.body_pos[:, OBJECT_BODY_ID] = OBJECT_POS + np.asarray([0.2, 0.0, 0.0])
    np.testing.assert_array_equal(term(env), 0.0)
    # Success frame for envs 1 and 4: closing to 0.08 pays the delta on the
    # other rows only; the success rows clamp to 0.
    backend.body_pos[:, OBJECT_BODY_ID] = OBJECT_POS + np.asarray([0.08, 0.0, 0.0])
    hub.pending_success[[1, 4]] = True
    rew = term(env)
    expected = np.full(NUM_ENVS, 0.12, dtype=np.float32)
    expected[[1, 4]] = 0.0
    np.testing.assert_allclose(rew, expected, rtol=1e-5)
    # The tracker still min-updates on the success frame (the command stage
    # clears it on the goal advance).
    np.testing.assert_allclose(hub.closest_keypoint_max_dist, 0.08, rtol=1e-5)
    # Following frame: the snapshot was consumed, every row pays normally.
    hub.pending_success[:] = False
    backend.body_pos[:, OBJECT_BODY_ID] = OBJECT_POS + np.asarray([0.03, 0.0, 0.0])
    np.testing.assert_allclose(term(env), 0.05, rtol=1e-5)


# ----------------------------------------------------------------------------
# Joint-velocity penalties (reward_utils.py:54-64)
# ----------------------------------------------------------------------------


def test_joint_velocity_penalties_use_canonical_split():
    env, backend, hub = _make_env()
    backend.dof_vel[:] = np.linspace(-1.4, 1.4, NUM_JOINTS, dtype=np.float32)[None, :]
    canon_vel = backend.dof_vel[:, CANON_TO_BACKEND]
    arm = _term(ArmJointVelocityPenalty, env)
    hand = _term(HandJointVelocityPenalty, env)
    # Positive L1 sums; the YAML weights (-0.03 / -0.003) carry the sign.
    np.testing.assert_allclose(
        arm(env), np.abs(canon_vel[:, :NUM_ARM_JOINTS]).sum(axis=-1), rtol=1e-6
    )
    np.testing.assert_allclose(
        hand(env), np.abs(canon_vel[:, NUM_ARM_JOINTS:]).sum(axis=-1), rtol=1e-6
    )


# ----------------------------------------------------------------------------
# Reach bonus (reward_utils.py:86-98)
# ----------------------------------------------------------------------------


def test_reach_bonus_amortized_near_goal():
    env, backend, hub = _make_env()
    term = _term(ReachBonus, env, success_steps=10, force_consecutive=False)
    hub.near_goal[:] = [True, False, True, False, False, False, True, False]
    # Raw value is near_goal / success_steps; weight 1000 completes
    # near_goal * (reach_goal_bonus / success_steps).
    np.testing.assert_allclose(term(env), [0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0])
    # Success does not pay extra in amortized mode.
    hub.pending_success[:] = True
    np.testing.assert_allclose(term(env), [0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0])


def test_reach_bonus_one_shot_on_success():
    env, backend, hub = _make_env()
    term = _term(ReachBonus, env, success_steps=10, force_consecutive=True)
    hub.pending_success[:] = [True, False, False, True, False, False, False, False]
    hub.near_goal[:] = True  # ignored in this mode
    np.testing.assert_array_equal(term(env), [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])


# ----------------------------------------------------------------------------
# Reward-manager integration: weights + scale_rewards_by_dt 口径
# ----------------------------------------------------------------------------


def test_reward_manager_applies_yaml_weights_and_dt():
    """reward_buf = raw * weight * step_dt — 1/60 of the original magnitudes."""
    env, backend, hub = _make_env()
    cfg = {
        "lift_progress": RewardTermCfg(func=LiftProgress, weight=20.0),
        "lifting_bonus": RewardTermCfg(
            func=LiftingBonus, weight=300.0, params={"bonus_threshold": 0.15}
        ),
        "fingertip_distance_delta": RewardTermCfg(
            func=FingertipDistanceDelta, weight=50.0, params={"bonus_threshold": 0.15}
        ),
        "keypoint_distance_delta": RewardTermCfg(
            func=KeypointDistanceDelta, weight=200.0, params={"bonus_threshold": 0.15}
        ),
        "arm_joint_velocity_penalty": RewardTermCfg(func=ArmJointVelocityPenalty, weight=-0.03),
        "hand_joint_velocity_penalty": RewardTermCfg(func=HandJointVelocityPenalty, weight=-0.003),
        "reach_bonus": RewardTermCfg(func=ReachBonus, weight=1000.0),
    }
    manager = RewardManager(cfg, cast(ManagerBasedRlEnv, env), scale_by_dt=True)

    # State: no lift (z_lift 0.10 -> progress 0.10, no bonus, no latch),
    # fingertips 0.05 past the seeded tracker (delta sum 0.25), zero keypoint
    # delta (sentinel), known joint velocities, one near-goal env.
    _set_object_z(backend, hub, 0.10)
    _place_fingertips(backend, np.full(5, 1.0))
    backend.dof_vel[:] = 0.5
    hub.near_goal[0] = True
    manager.compute(dt=STEP_DT)  # seeds the fingertip tracker (sentinel)
    _place_fingertips(backend, np.full(5, 0.95))
    reward = manager.compute(dt=STEP_DT)

    # _set_object_z lifted the object 0.05 above OBJECT_POS[2], while
    # _place_fingertips anchors on OBJECT_POS: each fingertip distance is
    # hypot(d, 0.05), so the summed delta is not exactly 5 * 0.05.
    ft_delta = 5.0 * (np.hypot(1.0, 0.05) - np.hypot(0.95, 0.05))
    expected_raw = (
        20.0 * 0.10  # lift progress
        + 300.0 * 0.0  # no crossing
        + 50.0 * ft_delta  # fingertip delta
        + 200.0 * 0.0  # keypoint delta sentinel step
        + (-0.03) * (NUM_ARM_JOINTS * 0.5)
        + (-0.003) * ((NUM_JOINTS - NUM_ARM_JOINTS) * 0.5)
        + 1000.0 * (1.0 / 10.0) * np.asarray([1.0] + [0.0] * 7)
    )
    np.testing.assert_allclose(reward, STEP_DT * expected_raw, rtol=1e-5)
    # Same call without dt scaling is exactly 60x larger. The fingertip tracker
    # is hub state, so close the fingers another 0.05 for a non-zero delta.
    _place_fingertips(backend, np.full(5, 0.90))
    ft_delta_2 = 5.0 * (np.hypot(0.95, 0.05) - np.hypot(0.90, 0.05))
    expected_raw_2 = expected_raw - 50.0 * ft_delta + 50.0 * ft_delta_2
    manager_no_dt = RewardManager(cfg, cast(ManagerBasedRlEnv, env), scale_by_dt=False)
    np.testing.assert_allclose(manager_no_dt.compute(dt=STEP_DT), expected_raw_2, rtol=1e-5)
