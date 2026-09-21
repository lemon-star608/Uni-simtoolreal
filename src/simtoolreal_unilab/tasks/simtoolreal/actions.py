"""SimToolReal action term.

Literal NumPy port of ``apply_action_pipeline``
(``simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/action_utils.py:18-75``,
snapshot 313d5aea) into the UniLab ``ActionTerm`` contract: canonical policy
order -> backend joint order mapping, per-env random action delay queue, arm
velocity-delta accumulation, hand absolute ``[-1, 1]`` scaling, moving
averages, and joint-limit clamps.

Registered deviations (DESIGN.md section 3.5 convention):

- No ``[-1, 1]`` action clip here: clipping is owned by the rlgames wrapper
  (``clip_actions: 1.0``, step-1 decision 1.1-e).  The original IsaacSim
  utility does not clip either (action_utils.py:31-34 only permutes).
- All internal buffers (delay-queue frames, ``prev``/``cur`` targets, joint
  limits) are kept in *canonical* (policy) order; the canonical->backend
  mapping is applied once at the ``Entity`` facade write via
  ``find_joints(..., preserve_order=True)`` + ``set_joint_position_target``,
  instead of permuting the incoming action to Lab order up front
  (action_utils.py:34).  The pipeline is elementwise per joint, so this is a
  relabeling with identical numerics; observations read ``prev_targets``
  without a further permutation (obs_utils.py:145).
- Joint limits come from the model-level ``soft_joint_pos_limits`` (29, 2);
  the original reads a per-env tensor (reset_utils.py:52-62) whose rows are
  identical across envs.
- Delay lag indices are drawn from ``env.rng`` (NumPy Generator) instead of
  ``torch.randint`` (action_utils.py:43-45): same uniform distribution over
  ``{0, ..., action_delay_max - 1}``, different stream.
- The debug-replay escape hatch (action_utils.py:20-25,
  ``_replay_target_lab_order``) is a tooling-only subclass hook and is not
  ported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from unilab.managers import ActionTerm, ActionTermCfg

from .commands import goal_command
from .config import (
    JOINT_NAMES_CANONICAL,
    NUM_ARM_JOINTS,
    NUM_JOINTS,
    declared_names,
    match_entity_names,
)

if TYPE_CHECKING:
    from unilab.base.entity import Entity


@dataclass(kw_only=True)
class SimToolRealActionCfg(ActionTermCfg):
    """Action pipeline knobs (simtoolreal_env_cfg.py:317-322 + DR delays)."""

    arm_moving_average: float = 0.1
    hand_moving_average: float = 0.1
    dof_speed_scale: float = 1.5
    # domain_randomization action latency (simtoolreal_env_cfg.py:462-464).
    use_action_delay: bool = True
    action_delay_max: int = 3

    def build(self, env):
        return SimToolRealAction(self, env)


class SimToolRealAction(ActionTerm):
    """29-DoF joint-position-target pipeline with delay and smoothing."""

    cfg: SimToolRealActionCfg
    _entity: Entity

    def __init__(self, cfg: SimToolRealActionCfg, env):
        super().__init__(cfg, env)
        # Cold path: resolve the canonical 29-DoF policy order against the
        # backend's joint order once (action_utils.py:31-34 permutation,
        # reset_utils.py:40-50).  ``_joint_ids[k]`` is the entity-local joint
        # column of canonical joint k; the facade write maps it to the
        # backend actuator order.
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
        self._joint_ids = np.asarray(joint_ids, dtype=np.intp)

        # Canonical-order joint limits (reset_utils.py:52-62).  The original
        # slices per-env rows of ``joint_pos_limits``; the facade exposes one
        # model-level table whose values are identical across envs.
        limits = self._entity.data.soft_joint_pos_limits[self._joint_ids]  # (29, 2)
        self._arm_lower = limits[:NUM_ARM_JOINTS, 0]
        self._arm_upper = limits[:NUM_ARM_JOINTS, 1]
        self._hand_lower = limits[NUM_ARM_JOINTS:, 0]
        self._hand_upper = limits[NUM_ARM_JOINTS:, 1]

        # Delay queue (reset_utils.py:145-150): max(1, action_delay_max)
        # slots, allocated even when the delay is disabled.
        self._delay_steps = max(1, cfg.action_delay_max)
        self._action_queue = np.zeros(
            (self.num_envs, self._delay_steps, NUM_JOINTS), dtype=np.float32
        )
        # Previous/current targets in canonical order (reset_utils.py:64-67).
        self._prev_targets = np.zeros((self.num_envs, NUM_JOINTS), dtype=np.float32)
        self._cur_targets = np.zeros((self.num_envs, NUM_JOINTS), dtype=np.float32)
        self._raw_action = np.zeros((self.num_envs, NUM_JOINTS), dtype=np.float32)

    @property
    def action_dim(self) -> int:
        return NUM_JOINTS

    @property
    def raw_action(self) -> np.ndarray:
        return self._raw_action

    @property
    def prev_targets(self) -> np.ndarray:
        """Previous smoothed targets in canonical order (obs_utils.py:145)."""
        return self._prev_targets

    def process_actions(self, actions: np.ndarray) -> None:
        cfg = self.cfg
        # Raw-action bookkeeping in canonical order, before delay (the rlgames
        # wrapper has already applied clip_actions=1.0; no clip here).
        self._raw_action[:] = actions
        # action_utils.py:31 ``actions.clone().to(env.device)``.
        act = np.array(actions, dtype=np.float32, copy=True)

        # Action delay (action_utils.py:36-48): flush freshly reset episodes
        # with the current action, roll the queue, push, sample a per-env lag.
        if cfg.use_action_delay and cfg.action_delay_max > 0:
            hub = goal_command(self._env)
            episode_start = (self._env.episode_length_buf == 0) & (hub.goal_count == 0)
            if episode_start.any():
                self._action_queue[episode_start] = act[episode_start, None, :]
            self._action_queue = np.roll(self._action_queue, shift=1, axis=1)
            self._action_queue[:, 0, :] = act
            delay_idx = self._env.rng.integers(0, self._action_queue.shape[1], size=self.num_envs)
            act = self._action_queue[np.arange(self.num_envs), delay_idx]

        dt = np.float32(self._env.step_dt)  # policy step (1/60 s), action_utils.py:29
        prev = self._prev_targets

        # Arm: velocity-delta accumulator, clamped twice (action_utils.py:50-58).
        arm_action = act[:, :NUM_ARM_JOINTS]
        prev_arm = prev[:, :NUM_ARM_JOINTS]
        arm_raw = prev_arm + np.float32(cfg.dof_speed_scale) * dt * arm_action
        arm_raw = np.clip(arm_raw, self._arm_lower, self._arm_upper)
        arm_ma = np.float32(cfg.arm_moving_average)
        arm_smoothed = arm_ma * arm_raw + (np.float32(1.0) - arm_ma) * prev_arm
        arm_smoothed = np.clip(arm_smoothed, self._arm_lower, self._arm_upper)

        # Hand: absolute [-1, 1] -> [lower, upper], clamped once
        # (action_utils.py:60-69).
        hand_action = act[:, NUM_ARM_JOINTS:]
        prev_hand = prev[:, NUM_ARM_JOINTS:]
        hand_raw = self._hand_lower + np.float32(0.5) * (hand_action + np.float32(1.0)) * (
            self._hand_upper - self._hand_lower
        )
        hand_ma = np.float32(cfg.hand_moving_average)
        hand_smoothed = hand_ma * hand_raw + (np.float32(1.0) - hand_ma) * prev_hand
        hand_smoothed = np.clip(hand_smoothed, self._hand_lower, self._hand_upper)

        # Publish canonical-order targets, then snapshot for the next step
        # (action_utils.py:71-74; the copy stops prev/cur from aliasing).
        self._cur_targets[:, :NUM_ARM_JOINTS] = arm_smoothed
        self._cur_targets[:, NUM_ARM_JOINTS:] = hand_smoothed
        self._prev_targets[:] = self._cur_targets

    def apply_actions(self) -> None:
        # Canonical-order targets mapped to backend actuator order by the
        # facade (action_utils.py:72-73 write, decimation-invariant: the
        # position target is applied once per control step).
        self._entity.set_joint_position_target(self._cur_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids) -> None:
        ids = (
            np.arange(self.num_envs)
            if isinstance(env_ids, slice)
            else np.asarray(env_ids, dtype=np.intp).reshape(-1)
        )
        # reset_utils.py:399: clear the delay queue rows.
        self._action_queue[ids] = 0.0
        self._raw_action[ids] = 0.0
        # reset_utils.py:218-220: seed previous/current targets from the
        # post-reset joint state.  The original reads back the pose sampled by
        # the reset event; UniLab's reset transaction commits before the
        # action manager reset runs (ManagerBasedRlEnv.reset order), so the
        # facade read returns the post-reset pose.
        joint_pos = self._entity.data.joint_pos  # (num_envs, num_joints), entity order
        pose_canonical = joint_pos[np.ix_(ids, self._joint_ids)]
        self._prev_targets[ids] = pose_canonical
        self._cur_targets[ids] = pose_canonical


__all__ = ["SimToolRealAction", "SimToolRealActionCfg"]
