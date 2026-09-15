"""SimToolReal reward terms.

Literal NumPy port of ``reward_utils.py:8-153`` (original snapshot 313d5aea)
into UniLab class-based reward terms: lift progress + one-shot lifting bonus
with the latch (:8-23), fingertip/keypoint distance deltas with the pre/post
lift gates (:26-51), arm/hand joint-velocity L1 penalties (:54-64 — the
upstream field name says "actions" but reads joint velocities), and the reach
bonus (:86-98).  ``update_near_goal_steps`` (:79-84) is part of the success
criterion and belongs to the termination layer (1.4b-3), not to these terms.

Weights live in the owner YAML ``reward:`` section (``weight`` field), which
the RewardManager multiplies as ``raw * weight * dt``; the original applies
the scales inside the functions.  Term order in the manager config is free:
every gate reads the latch through the idempotent view ``(z_lift > threshold)
| hub.lifted`` instead of assuming a side effect ran first.

Registered deviations (DESIGN.md section 3.5 convention):

- Reward dt scaling was reverted (training-pipeline-reaudit fix P2):
  ``scale_rewards_by_dt: false`` (base.yaml) keeps ``reward_buf`` at the
  original DirectRLEnv magnitudes (no per-term ``step_dt`` factor), so the
  critic-observation ``reward`` field — which reads ``env.reward_buf`` —
  matches the original scale as well.  The manager-level contract is
  unchanged: ``RewardManager`` still supports both modes; only the owner
  YAML choice moved from the framework default (``true``) to parity
  (``false``).
- The ``lifted`` latch write moves from ``compute_rewards``
  (reward_utils.py:97) into :class:`LiftingBonus`, the term that needs the
  ``just_crossed`` edge; all other consumers re-derive ``lifted_now``
  idempotently, so manager term order cannot change numerics.
- The -1 sentinel fill (``obs_utils.py:180-188``) moves from
  ``compute_intermediate_values`` into the two tracker-owning terms; it runs
  same-step before the min-update, identical numerics.
- Success/reward keypoints always use the fixed-size branch
  (``reset_utils.py:79-83``: corners * 0.5 * keypoint_scale * fixed_size);
  the per-env-scale branch (``fixed_size_keypoint_reward=False``) is an
  observation-path feature (DESIGN.md section 3.5) and is not ported here.
- ``near_goal``/``is_success`` are read from the hub (``hub.near_goal`` /
  ``hub.pending_success``), written by the termination layer earlier in the
  same step (DESIGN.md section 6.5); the original reads
  ``env._near_goal``/``env._is_success`` from ``compute_intermediate_values``
  in ``_get_dones`` — same relative order.  The
  ``_keypoint_success_tolerance_m`` subclass hook (obs_utils.py:190-193) is
  not ported.
- The gym-era hand-delta and object velocity penalties stay retired
  (zero-coefficient terms, DESIGN.md section 3.5).
"""

from __future__ import annotations

from typing import cast

import numpy as np

from unilab.base.entity import Entity
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg

from .commands import goal_command
from .config import (
    FINGERTIP_LINK_NAMES,
    JOINT_NAMES_CANONICAL,
    KEYPOINT_CORNERS,
    NUM_ARM_JOINTS,
)
from .observations import (
    _cached_fingertip_distances,
    _cached_keypoint_max_dist,
)  # shared literal port (obs_utils.py:103-112)

# Defaults mirror simtoolreal_env_cfg.py:331-350 / 437-438.
_LIFTING_BONUS_THRESHOLD = 0.15
_KEYPOINT_SCALE = 1.5
_FIXED_SIZE = (0.141, 0.03025, 0.0271)
_SUCCESS_STEPS = 10


def _check_params(cfg: ManagerTermBaseCfg, allowed: frozenset[str]) -> None:
    unexpected = set(cfg.params) - allowed
    if unexpected:
        raise TypeError(
            f"{cfg.func.__name__} received unsupported parameters: {sorted(unexpected)}"
        )


def _object_pos_local(env, object_entity: Entity) -> np.ndarray:
    return object_entity.data.root_link_pos_w - env.scene.env_origins


def _lifted_now(env, hub, object_entity: Entity, threshold: float) -> np.ndarray:
    """Idempotent post-update latch view (reward_utils.py:17, 97 ordering)."""
    z_lift = 0.05 + _object_pos_local(env, object_entity)[:, 2] - hub.object_init_z
    return (z_lift > threshold) | hub.lifted


class LiftProgress(ManagerTermBase):
    """Lift progress reward (reward_utils.py:8-23, progress half).

    ``clip(0.05 + z - z_init, 0, 0.5)``, zeroed once the latch holds
    (including the crossing step); the YAML weight is ``lifting_rew_scale``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(cfg, frozenset({"bonus_threshold"}))
        self._threshold = float(cfg.params.get("bonus_threshold", _LIFTING_BONUS_THRESHOLD))
        self._object = cast(Entity, env.scene["object"])

    def __call__(self, env, **params) -> np.ndarray:
        del params
        hub = goal_command(env)
        z_lift = 0.05 + _object_pos_local(env, self._object)[:, 2] - hub.object_init_z
        lift_rew = np.clip(z_lift, 0.0, 0.5)  # reward_utils.py:16
        lifted_now = (z_lift > self._threshold) | hub.lifted  # :17
        return (lift_rew * (~lifted_now).astype(np.float32)).astype(np.float32)  # :21-23


class LiftingBonus(ManagerTermBase):
    """One-shot bonus when the lift latch crosses the threshold
    (reward_utils.py:17-20); owns the latch write (compute_rewards:97)."""

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(cfg, frozenset({"bonus_threshold"}))
        self._threshold = float(cfg.params.get("bonus_threshold", _LIFTING_BONUS_THRESHOLD))
        self._object = cast(Entity, env.scene["object"])

    def __call__(self, env, **params) -> np.ndarray:
        del params
        hub = goal_command(env)
        prev = hub.lifted
        z_lift = 0.05 + _object_pos_local(env, self._object)[:, 2] - hub.object_init_z
        lifted_now = (z_lift > self._threshold) | prev  # reward_utils.py:17
        just_crossed = lifted_now & ~prev  # :18
        hub.lifted[:] = lifted_now  # compute_rewards:97
        return just_crossed.astype(np.float32)  # :19-20 (bonus via YAML weight)


class FingertipDistanceDelta(ManagerTermBase):
    """Fingertip approach delta, gated off once lifted (reward_utils.py:26-40,
    compute_rewards:99-104); owns the ``closest_fingertip_dist`` tracker."""

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(cfg, frozenset({"bonus_threshold"}))
        self._threshold = float(cfg.params.get("bonus_threshold", _LIFTING_BONUS_THRESHOLD))
        self._object = cast(Entity, env.scene["object"])
        self._robot = cast(Entity, env.scene["robot"])
        ids, _ = self._robot.find_bodies(list(FINGERTIP_LINK_NAMES), preserve_order=True)
        self._fingertip_ids = np.asarray(ids, dtype=np.intp)

    def __call__(self, env, **params) -> np.ndarray:
        del params
        hub = goal_command(env)
        # obs_utils.py:166-170: fingertip-to-object-center distances.  The
        # original computes this intermediate once for both reward and done.
        curr = _cached_fingertip_distances(
            env, hub, self._object, self._robot, self._fingertip_ids
        )  # (N, 5)
        lifted_now = _lifted_now(env, hub, self._object, self._threshold)
        # obs_utils.py:184-188: first observed value becomes closest-so-far.
        closest = np.where(hub.closest_fingertip_dist < 0.0, curr, hub.closest_fingertip_dist)
        deltas = np.clip(closest - curr, 0.0, 10.0)  # reward_utils.py:33-35
        rew = deltas.sum(axis=-1) * (~lifted_now).astype(np.float32)  # :36
        # :34, compute_rewards:104 — the tracker updates regardless of the gate.
        hub.closest_fingertip_dist[:] = np.minimum(closest, curr)
        return rew.astype(np.float32)


class KeypointDistanceDelta(ManagerTermBase):
    """Keypoint max-distance delta, gated on once lifted (reward_utils.py:42-51,
    compute_rewards:106-110); owns the ``closest_keypoint_max_dist`` tracker.

    Keypoints use the fixed-size branch (reset_utils.py:79-83), never the
    per-env-scale observation branch (DESIGN.md section 3.5).
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(cfg, frozenset({"bonus_threshold", "keypoint_scale", "fixed_size"}))
        self._threshold = float(cfg.params.get("bonus_threshold", _LIFTING_BONUS_THRESHOLD))
        keypoint_scale = float(cfg.params.get("keypoint_scale", _KEYPOINT_SCALE))
        fixed_size = np.asarray(cfg.params.get("fixed_size", _FIXED_SIZE), dtype=np.float32)
        if fixed_size.shape != (3,) or not np.isfinite(fixed_size).all():
            raise ValueError("KeypointDistanceDelta fixed_size must be a finite 3-D vector")
        # reset_utils.py:79-83: corners * 0.5 * keypoint_scale * fixed_size.
        self._kp_offsets = np.asarray(KEYPOINT_CORNERS, dtype=np.float32) * (
            0.5 * keypoint_scale * fixed_size
        )
        self._object = cast(Entity, env.scene["object"])

    def __call__(self, env, **params) -> np.ndarray:
        del params
        hub = goal_command(env)
        kp_offsets = np.broadcast_to(
            self._kp_offsets[None, :, :], (env.num_envs, *self._kp_offsets.shape)
        )
        # obs_utils.py:172-178 (fixed branch).  The goal pose comes from the
        # hub buffer (reaudit fix P3a): the original reads back the GoalViz
        # pose, which its per-advance writes keep equal to the current goal.
        kmd = _cached_keypoint_max_dist(env, hub, self._object, kp_offsets)
        lifted_now = _lifted_now(env, hub, self._object, self._threshold)
        # obs_utils.py:180-183: first observed value becomes closest-so-far.
        closest = np.where(hub.closest_keypoint_max_dist < 0.0, kmd, hub.closest_keypoint_max_dist)
        delta = np.clip(closest - kmd, 0.0, 100.0)  # reward_utils.py:45-47
        rew = delta * lifted_now.astype(np.float32)  # :48
        hub.closest_keypoint_max_dist[:] = np.minimum(closest, kmd)  # :46, compute_rewards:110
        return rew.astype(np.float32)


class _JointVelocityPenalty(ManagerTermBase):
    """L1 joint-velocity penalty (reward_utils.py:54-64); the negative sign
    comes from the YAML weight (``kuka``/``hand_actions_penalty_scale``)."""

    _joint_names: tuple[str, ...] = ()

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(cfg, frozenset())
        self._robot = cast(Entity, env.scene["robot"])
        ids, _ = self._robot.find_joints(list(self._joint_names), preserve_order=True)
        self._joint_ids = np.asarray(ids, dtype=np.intp)

    def __call__(self, env, **params) -> np.ndarray:
        del params
        return (
            np.abs(self._robot.data.joint_vel[:, self._joint_ids]).sum(axis=-1).astype(np.float32)
        )


class ArmJointVelocityPenalty(_JointVelocityPenalty):
    """Arm 7-DoF L1 velocity penalty (reward_utils.py:62, canonical order)."""

    _joint_names = JOINT_NAMES_CANONICAL[:NUM_ARM_JOINTS]


class HandJointVelocityPenalty(_JointVelocityPenalty):
    """Hand 22-DoF L1 velocity penalty (reward_utils.py:63, canonical order)."""

    _joint_names = JOINT_NAMES_CANONICAL[NUM_ARM_JOINTS:]


class ReachBonus(ManagerTermBase):
    """Reach-goal bonus (reward_utils.py:86-98): amortized near-goal bonus
    (default) or one-shot success bonus (``force_consecutive`` mode).

    Reads the hub snapshot written by the termination layer earlier in the
    step (DESIGN.md section 6.5); the YAML weight is ``reach_goal_bonus``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(cfg, frozenset({"success_steps", "force_consecutive"}))
        self._success_steps = int(cfg.params.get("success_steps", _SUCCESS_STEPS))
        self._force_consecutive = bool(cfg.params.get("force_consecutive", False))

    def __call__(self, env, **params) -> np.ndarray:
        del params
        hub = goal_command(env)
        # reward_utils.py:95-96: lump-sum on success.
        if self._force_consecutive:
            return hub.pending_success.astype(np.float32)
        # :97-98: near_goal * (bonus / success_steps); bonus via YAML weight.
        return hub.near_goal.astype(np.float32) * np.float32(1.0 / self._success_steps)


__all__ = [
    "ArmJointVelocityPenalty",
    "FingertipDistanceDelta",
    "HandJointVelocityPenalty",
    "KeypointDistanceDelta",
    "LiftProgress",
    "LiftingBonus",
    "ReachBonus",
]
