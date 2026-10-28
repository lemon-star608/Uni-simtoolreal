"""SimToolReal termination terms."""

from __future__ import annotations

import torch
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg

from .commands import goal_command
from .config import KEYPOINT_CORNERS
from .observations import _cached_fingertip_distances, _cached_keypoint_max_dist
from .tensor_utils import require_body_view

_KEYPOINT_SCALE = 1.5
_FIXED_SIZE = (0.141, 0.03025, 0.0271)
_SUCCESS_STEPS = 10
_FALL_HEIGHT = 0.05
_HAND_FAR_DISTANCE = 0.55
_MAX_CONSECUTIVE_SUCCESSES = 5


def _check_params(cfg: ManagerTermBaseCfg, allowed: frozenset[str]) -> None:
    unexpected = set(cfg.params) - allowed
    if unexpected:
        raise TypeError(
            f"{cfg.func.__name__} received unsupported parameters: {sorted(unexpected)}"
        )


def _object_body_name(term: ManagerTermBase) -> str:
    return goal_command(term._env).object_body_name


class SuccessTracker(ManagerTermBase):
    """Near-goal tracker whose output is always false; success advances goals."""

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(
            cfg,
            frozenset(
                {
                    "success_tolerance",
                    "keypoint_scale",
                    "fixed_size",
                    "success_steps",
                    "force_consecutive",
                }
            ),
        )
        self._success_tolerance = float(
            cfg.params.get("success_tolerance", goal_command(env).cfg.success_tolerance)
        )
        self._keypoint_scale = float(cfg.params.get("keypoint_scale", _KEYPOINT_SCALE))
        self._success_steps = int(cfg.params.get("success_steps", _SUCCESS_STEPS))
        self._force_consecutive = bool(
            cfg.params.get(
                "force_consecutive", goal_command(env).cfg.force_consecutive_near_goal_steps
            )
        )
        fixed_size = torch.tensor(cfg.params.get("fixed_size", _FIXED_SIZE), dtype=torch.float32)
        if fixed_size.shape != (3,) or not bool(torch.isfinite(fixed_size).all()):
            raise ValueError("SuccessTracker fixed_size must be a finite 3-D vector")
        corners = torch.tensor(KEYPOINT_CORNERS, dtype=torch.float32)
        self._kp_offsets = corners * (0.5 * self._keypoint_scale * fixed_size)

    @property
    def entity_name(self) -> str:
        return "object"

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return (_object_body_name(self),)

    def __call__(self, env, **params) -> torch.Tensor:
        del params
        hub = goal_command(env)
        offsets = self._kp_offsets.to(env.device)[None].expand(env.num_envs, -1, -1)
        kmd = _cached_keypoint_max_dist(env, hub, hub.object, offsets)
        near_goal = kmd <= hub.current_success_tolerance * self._keypoint_scale
        ng = near_goal.to(torch.int64)
        steps = (
            (hub.near_goal_steps + ng) * ng
            if self._force_consecutive
            else (hub.near_goal_steps + ng)
        )
        hub.near_goal.copy_(near_goal)
        hub.near_goal_steps.copy_(steps)
        hub.pending_success.copy_(steps >= self._success_steps)
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


class Fall(ManagerTermBase):
    """Object dropped below the local fall-height threshold."""

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(cfg, frozenset({"fall_height"}))
        self._fall_height = float(cfg.params.get("fall_height", _FALL_HEIGHT))

    @property
    def entity_name(self) -> str:
        return "object"

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return (_object_body_name(self),)

    def __call__(self, env, **params) -> torch.Tensor:
        del params
        hub = goal_command(env)
        object_z = require_body_view(env, hub.object, (hub.object_body_name,)).pos_w[:, 0, 2]
        return (object_z - env.scene.env_origins(env.device)[:, 2]) < self._fall_height


class HandFar(ManagerTermBase):
    """Any fingertip becomes farther than the threshold from the object."""

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(cfg, frozenset({"hand_far_distance"}))
        self._distance = float(cfg.params.get("hand_far_distance", _HAND_FAR_DISTANCE))

    @property
    def entity_name(self) -> str:
        return "object"

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return (_object_body_name(self),)

    def __call__(self, env, **params) -> torch.Tensor:
        del params
        hub = goal_command(env)
        distances = _cached_fingertip_distances(env, hub, hub.object, hub.robot, ())
        return distances.amax(dim=-1) > self._distance


class MaxConsecutiveSuccesses(ManagerTermBase):
    """Episode cap on chained goals."""

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(cfg, frozenset({"max_consecutive_successes"}))
        self._max = int(cfg.params.get("max_consecutive_successes", _MAX_CONSECUTIVE_SUCCESSES))

    def __call__(self, env, **params) -> torch.Tensor:
        del params
        if self._max <= 0:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        hub = goal_command(env)
        return (hub.goal_count + hub.pending_success) >= self._max


def time_out(env) -> torch.Tensor:
    """Truncation masked by this step's success snapshot."""
    hub = goal_command(env)
    return (env.episode_length_buf >= env.max_episode_length) & ~hub.pending_success


__all__ = [
    "Fall",
    "HandFar",
    "MaxConsecutiveSuccesses",
    "SuccessTracker",
    "time_out",
]
