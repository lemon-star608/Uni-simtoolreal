"""SimToolReal's seven Torch reward terms."""

from __future__ import annotations

import torch
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg

from .commands import SimToolRealCommand, goal_command
from .config import (
    JOINT_NAMES_CANONICAL,
    KEYPOINT_CORNERS,
    NUM_ARM_JOINTS,
    declared_names,
    match_entity_names,
)
from .observations import _cached_fingertip_distances, _cached_keypoint_max_dist
from .tensor_utils import index_tensor, require_body_view, require_joint_view

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


def _object_pos_local(env, hub: SimToolRealCommand) -> torch.Tensor:
    view = require_body_view(env, hub.object, (hub.object_body_name,))
    return view.pos_w[:, 0, :] - env.scene.env_origins(env.device)


def _lifted_now(env, hub, threshold: float) -> torch.Tensor:
    z_lift = 0.05 + _object_pos_local(env, hub)[:, 2] - hub.object_init_z
    return (z_lift > threshold) | hub.lifted


class LiftProgress(ManagerTermBase):
    """Lift progress before the one-way lifted latch holds."""

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(cfg, frozenset({"bonus_threshold"}))
        self._threshold = float(cfg.params.get("bonus_threshold", _LIFTING_BONUS_THRESHOLD))

    @property
    def entity_name(self) -> str:
        return "object"

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return (goal_command(self._env).object_body_name,)

    def __call__(self, env, **params) -> torch.Tensor:
        del params
        hub = goal_command(env)
        z_lift = 0.05 + _object_pos_local(env, hub)[:, 2] - hub.object_init_z
        lift_rew = torch.clamp(z_lift, min=0.0, max=0.5)
        return lift_rew * ~_lifted_now(env, hub, self._threshold)


class LiftingBonus(ManagerTermBase):
    """One-shot bonus when the lift latch crosses its threshold."""

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(cfg, frozenset({"bonus_threshold"}))
        self._threshold = float(cfg.params.get("bonus_threshold", _LIFTING_BONUS_THRESHOLD))

    @property
    def entity_name(self) -> str:
        return "object"

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return (goal_command(self._env).object_body_name,)

    def __call__(self, env, **params) -> torch.Tensor:
        del params
        hub = goal_command(env)
        prev = hub.lifted
        lifted_now = _lifted_now(env, hub, self._threshold)
        just_crossed = lifted_now & ~prev
        hub.lifted.copy_(lifted_now)
        return just_crossed.to(torch.float32)


class FingertipDistanceDelta(ManagerTermBase):
    """Fingertip approach-delta reward and closest-distance tracker."""

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(cfg, frozenset({"bonus_threshold"}))
        self._threshold = float(cfg.params.get("bonus_threshold", _LIFTING_BONUS_THRESHOLD))

    @property
    def entity_name(self) -> str:
        return "object"

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return (goal_command(self._env).object_body_name,)

    def __call__(self, env, **params) -> torch.Tensor:
        del params
        hub = goal_command(env)
        curr = _cached_fingertip_distances(env, hub, hub.object, hub.robot, ())
        lifted_now = _lifted_now(env, hub, self._threshold)
        closest = torch.where(hub.closest_fingertip_dist < 0.0, curr, hub.closest_fingertip_dist)
        deltas = torch.clamp(closest - curr, min=0.0, max=10.0)
        rew = deltas.sum(dim=-1) * (~lifted_now).to(torch.float32)
        hub.closest_fingertip_dist.copy_(torch.minimum(closest, curr))
        return rew


class KeypointDistanceDelta(ManagerTermBase):
    """Keypoint max-distance delta after lift and closest-distance tracker."""

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(cfg, frozenset({"bonus_threshold", "keypoint_scale", "fixed_size"}))
        self._threshold = float(cfg.params.get("bonus_threshold", _LIFTING_BONUS_THRESHOLD))
        keypoint_scale = float(cfg.params.get("keypoint_scale", _KEYPOINT_SCALE))
        fixed_size = torch.tensor(cfg.params.get("fixed_size", _FIXED_SIZE), dtype=torch.float32)
        if fixed_size.shape != (3,) or not bool(torch.isfinite(fixed_size).all()):
            raise ValueError("KeypointDistanceDelta fixed_size must be a finite 3-D vector")
        corners = torch.tensor(KEYPOINT_CORNERS, dtype=torch.float32)
        self._kp_offsets = corners * (0.5 * keypoint_scale * fixed_size)

    @property
    def entity_name(self) -> str:
        return "object"

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return (goal_command(self._env).object_body_name,)

    def __call__(self, env, **params) -> torch.Tensor:
        del params
        hub = goal_command(env)
        offsets = self._kp_offsets.to(env.device)[None, :, :].expand(env.num_envs, -1, -1)
        kmd = _cached_keypoint_max_dist(env, hub, hub.object, offsets)
        lifted_now = _lifted_now(env, hub, self._threshold)
        closest = torch.where(
            hub.closest_keypoint_max_dist < 0.0, kmd, hub.closest_keypoint_max_dist
        )
        delta = torch.clamp(closest - kmd, min=0.0, max=100.0)
        rew = delta * lifted_now.to(torch.float32)
        hub.closest_keypoint_max_dist.copy_(torch.minimum(closest, kmd))
        return rew


class _JointVelocityPenalty(ManagerTermBase):
    """L1 joint-velocity penalty."""

    _joint_names: tuple[str, ...] = ()

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(cfg, frozenset())
        self._robot = env.scene["robot"]
        ids, _ = self._robot.find_joints(
            match_entity_names(
                declared_names(self._robot, "joint"), self._joint_names, label="joint"
            ),
            preserve_order=True,
        )
        self._joint_ids = index_tensor(ids, env.device)

    def __call__(self, env, **params) -> torch.Tensor:
        del params
        velocity = require_joint_view(env, self._robot).joint_vel
        return velocity.index_select(1, self._joint_ids).abs().sum(dim=-1)


class ArmJointVelocityPenalty(_JointVelocityPenalty):
    _joint_names = JOINT_NAMES_CANONICAL[:NUM_ARM_JOINTS]


class HandJointVelocityPenalty(_JointVelocityPenalty):
    _joint_names = JOINT_NAMES_CANONICAL[NUM_ARM_JOINTS:]


class ReachBonus(ManagerTermBase):
    """Amortized near-goal bonus or one-shot success bonus."""

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(cfg, frozenset({"success_steps", "force_consecutive"}))
        self._success_steps = int(cfg.params.get("success_steps", _SUCCESS_STEPS))
        self._force_consecutive = bool(cfg.params.get("force_consecutive", False))

    def __call__(self, env, **params) -> torch.Tensor:
        del params
        hub = goal_command(env)
        if self._force_consecutive:
            return hub.pending_success.to(torch.float32)
        return hub.near_goal.to(torch.float32) * (1.0 / self._success_steps)


__all__ = [
    "ArmJointVelocityPenalty",
    "FingertipDistanceDelta",
    "HandJointVelocityPenalty",
    "KeypointDistanceDelta",
    "LiftProgress",
    "LiftingBonus",
    "ReachBonus",
]
