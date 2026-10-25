"""SimToolReal's 29-DoF joint-position-target action pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from unilab.managers import ActionTerm, ActionTermCfg

from .commands import goal_command
from .config import (
    JOINT_NAMES_CANONICAL,
    NUM_ARM_JOINTS,
    NUM_JOINTS,
    declared_names,
    match_entity_names,
)
from .tensor_utils import index_tensor

if TYPE_CHECKING:
    from unilab.base.entity import Entity


@dataclass(kw_only=True)
class SimToolRealActionCfg(ActionTermCfg):
    """Action pipeline knobs (SimToolReal action_utils.py)."""

    arm_moving_average: float = 0.1
    hand_moving_average: float = 0.1
    dof_speed_scale: float = 1.5
    use_action_delay: bool = True
    action_delay_max: int = 3

    def build(self, env):
        return SimToolRealAction(self, env)


class SimToolRealAction(ActionTerm):
    """Canonical-order Torch action delay, accumulation, scaling, and smoothing."""

    cfg: SimToolRealActionCfg
    _entity: Entity

    def __init__(self, cfg: SimToolRealActionCfg, env):
        super().__init__(cfg, env)
        joint_ids, self._resolved_joint_names = self._entity.find_joints(
            match_entity_names(
                declared_names(self._entity, "joint"), JOINT_NAMES_CANONICAL, label="joint"
            ),
            preserve_order=True,
        )
        if len(joint_ids) != NUM_JOINTS:
            raise ValueError(
                f"SimToolRealAction expected {NUM_JOINTS} canonical joints, "
                f"resolved {len(joint_ids)}: {self._resolved_joint_names}"
            )
        self._joint_ids = tuple(int(value) for value in joint_ids)
        self._joint_ids_t = index_tensor(self._joint_ids, self._env.device)

        limits = self._entity.data.soft_joint_pos_limits(self._env.device)[self._joint_ids_t]
        self._arm_lower = limits[:NUM_ARM_JOINTS, 0].clone()
        self._arm_upper = limits[:NUM_ARM_JOINTS, 1].clone()
        self._hand_lower = limits[NUM_ARM_JOINTS:, 0].clone()
        self._hand_upper = limits[NUM_ARM_JOINTS:, 1].clone()

        self._delay_steps = max(1, int(cfg.action_delay_max))
        shape = (self.num_envs, self._delay_steps, NUM_JOINTS)
        self._action_queue = torch.zeros(shape, dtype=torch.float32, device=self._env.device)
        self._prev_targets = torch.zeros(
            (self.num_envs, NUM_JOINTS), dtype=torch.float32, device=self._env.device
        )
        self._cur_targets = torch.zeros_like(self._prev_targets)
        self._raw_action = torch.zeros_like(self._prev_targets)

    @property
    def action_dim(self) -> int:
        return NUM_JOINTS

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_action

    @property
    def prev_targets(self) -> torch.Tensor:
        """Previous smoothed targets in canonical policy order."""
        return self._prev_targets

    def process_actions(self, actions: torch.Tensor) -> None:
        cfg = self.cfg
        if actions.shape != (self.num_envs, NUM_JOINTS) or actions.dtype != torch.float32:
            raise TypeError(
                f"SimToolReal actions must be float32 ({self.num_envs}, {NUM_JOINTS}); "
                f"got {tuple(actions.shape)} {actions.dtype}"
            )
        self._raw_action.copy_(actions)
        act = actions.clone()

        if cfg.use_action_delay and cfg.action_delay_max > 0:
            hub = goal_command(self._env)
            episode_start = (self._env.episode_length_buf == 0) & (hub.goal_count == 0)
            if bool(episode_start.any()):
                selected = torch.nonzero(episode_start, as_tuple=False).flatten()
                self._action_queue[selected] = act[selected, None, :]
            self._action_queue = torch.roll(self._action_queue, shifts=1, dims=1)
            self._action_queue[:, 0, :].copy_(act)
            delay_idx = self._env.rng.integers(0, self._delay_steps, size=self.num_envs)
            rows = torch.arange(self.num_envs, device=act.device)[:, None]
            act = self._action_queue[rows, delay_idx[:, None]]

        dt = self._env.step_dt
        prev = self._prev_targets
        arm_action = act[:, :NUM_ARM_JOINTS]
        prev_arm = prev[:, :NUM_ARM_JOINTS]
        arm_raw = prev_arm + cfg.dof_speed_scale * dt * arm_action
        arm_raw = torch.clamp(arm_raw, min=self._arm_lower, max=self._arm_upper)
        arm_ma = cfg.arm_moving_average
        arm_smoothed = arm_ma * arm_raw + (1.0 - arm_ma) * prev_arm
        arm_smoothed = torch.clamp(arm_smoothed, min=self._arm_lower, max=self._arm_upper)

        hand_action = act[:, NUM_ARM_JOINTS:]
        prev_hand = prev[:, NUM_ARM_JOINTS:]
        hand_raw = self._hand_lower + 0.5 * (hand_action + 1.0) * (
            self._hand_upper - self._hand_lower
        )
        hand_ma = cfg.hand_moving_average
        hand_smoothed = hand_ma * hand_raw + (1.0 - hand_ma) * prev_hand
        hand_smoothed = torch.clamp(hand_smoothed, min=self._hand_lower, max=self._hand_upper)

        self._cur_targets[:, :NUM_ARM_JOINTS].copy_(arm_smoothed)
        self._cur_targets[:, NUM_ARM_JOINTS:].copy_(hand_smoothed)
        self._prev_targets.copy_(self._cur_targets)

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(self._cur_targets, joint_ids=self._joint_ids)

    def seed_reset_targets(self, env_ids: torch.Tensor, positions: torch.Tensor) -> None:
        """Receive the joint positions staged by SimToolReal's reset event.

        Manager reset dispatches reset events before action reset, but the
        packed reset transaction commits only after all manager resets. The
        selected-row read plan is therefore not publishable yet; forwarding
        the already-sampled canonical positions keeps the old post-reset
        target seeding semantics without crossing that transaction boundary.
        """
        self._prev_targets[env_ids] = positions
        self._cur_targets[env_ids] = positions

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            raise ValueError("SimToolRealAction reset requires concrete env_ids")
        ids = (
            torch.arange(self.num_envs, device=self._env.device)[env_ids]
            if isinstance(env_ids, slice)
            else env_ids
        )
        self._action_queue[ids].zero_()
        self._raw_action[ids].zero_()


__all__ = ["SimToolRealAction", "SimToolRealActionCfg"]
