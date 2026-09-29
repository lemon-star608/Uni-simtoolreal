"""SimToolReal termination terms.

Literal NumPy port of ``termination_utils.py:39-72`` (original snapshot
313d5aea) plus the success criterion chain (``obs_utils.py:190-205`` and
``reward_utils.py:79-84``) into UniLab manager terms:

- :class:`SuccessTracker` runs the full success chain each step — keypoint
  max distance against the (curriculum-adjusted) tolerance, near-goal step
  counting, ``is_success`` — and writes the ``near_goal``/``near_goal_steps``/
  ``pending_success`` snapshot onto the command hub for the reward and
  command stages (DESIGN.md section 6.5).  Success never terminates the
  episode, so the term always returns all-False.
- :class:`Fall`, :class:`HandFar` and :class:`MaxConsecutiveSuccesses` are
  the real ``terminated`` causes (termination_utils.py:54-64).
- :func:`time_out` is the task-specific truncation variant: the original
  zeroes ``episode_length_buf`` for successful envs *before* the truncation
  check (termination_utils.py:50-51, 65); UniLab zeroes it in the later
  command stage, so the term masks the success frame with
  ``pending_success`` instead (step-1.4b-1 registered constraint).

Registered deviations (DESIGN.md section 3.5 convention):

- The success chain runs in the termination stage; the original computes it
  in ``compute_intermediate_values`` inside ``_get_dones`` — the same
  position relative to rewards.  State lands on the command hub instead of
  ``env._near_goal``/``env._near_goal_steps``/``env._is_success``.
- Success/reward keypoints always use the fixed-size branch
  (``SimToolReal.yaml:89`` ``fixed_size_keypoint_reward: true``;
  ``reset_utils.py:79-83``); the ``_keypoint_success_tolerance_m`` subclass
  hook (``obs_utils.py:190-193``) is not ported, matching rewards.py.
- :class:`MaxConsecutiveSuccesses` compares ``goal_count + pending_success``:
  the original increments ``_successes`` (termination_utils.py:46) before
  the ``>=`` check (:57-60) in the same call, while the UniLab
  ``hub.goal_count`` increments in the later command stage.  Folding the
  pending snapshot in reproduces the original same-step termination.
- :func:`time_out` masks ``pending_success`` instead of reading a pre-zeroed
  ``episode_length_buf`` (see above); subsequent steps see the zeroed length
  written by the command stage, exactly like the original.
- The ``_termination_reasons`` dict (termination_utils.py:66-71) is not
  ported: ``TerminationManager`` keeps per-term done buffers and logs
  ``Episode_Termination/<name>``.
- The tolerance curriculum state (termination_utils.py:10-36) lives in the
  command hub (step 1.4b-1); :class:`SuccessTracker` triggers the per-step
  tick ahead of its success comparison, so a tightening is effective on the
  same frame — the original runs ``update_tolerance_curriculum`` before the
  success judgment inside ``_get_dones`` (audit fix A1.4).
- ``fall_height``/``hand_far_distance`` are hardcoded literals in the
  original (:55, :62); here they are cfg-backed params whose defaults are
  those literals (config.py:347-349).
"""

from __future__ import annotations

from typing import cast

import numpy as np
from unilab.base.entity import Entity
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg

from .commands import goal_command
from .config import FINGERTIP_LINK_NAMES, KEYPOINT_CORNERS, declared_names, match_entity_names
from .observations import (
    _cached_fingertip_distances,
    _cached_keypoint_max_dist,
)  # shared literal port (obs_utils.py:103-112)

# Shared literal-port helpers/defaults (simtoolreal_env_cfg.py:337, 437-438;
# termination_utils.py:55, 62).
from .rewards import _FIXED_SIZE, _KEYPOINT_SCALE, _SUCCESS_STEPS, _check_params

_FALL_HEIGHT = 0.1  # termination_utils.py:55
_HAND_FAR_DISTANCE = 1.5  # termination_utils.py:62
_MAX_CONSECUTIVE_SUCCESSES = 50  # simtoolreal_env_cfg.py:438


class SuccessTracker(ManagerTermBase):
    """Per-step success criterion snapshot (obs_utils.py:190-205); never done.

    Triggers the hub's tolerance-curriculum tick ahead of the success
    comparison (termination_utils.py:10-36 ordering; audit fix A1.4), then
    writes ``hub.near_goal``/``hub.near_goal_steps``/``hub.pending_success``
    for the reward and command stages later in the same control step
    (DESIGN.md section 6.5).
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(
            cfg, frozenset({"success_steps", "force_consecutive", "keypoint_scale", "fixed_size"})
        )
        self._success_steps = int(cfg.params.get("success_steps", _SUCCESS_STEPS))
        self._force_consecutive = bool(cfg.params.get("force_consecutive", False))
        keypoint_scale = float(cfg.params.get("keypoint_scale", _KEYPOINT_SCALE))
        self._keypoint_scale = keypoint_scale
        fixed_size = np.asarray(cfg.params.get("fixed_size", _FIXED_SIZE), dtype=np.float32)
        if fixed_size.shape != (3,) or not np.isfinite(fixed_size).all():
            raise ValueError("SuccessTracker fixed_size must be a finite 3-D vector")
        # reset_utils.py:79-83: corners * 0.5 * keypoint_scale * fixed_size.
        self._kp_offsets = np.asarray(KEYPOINT_CORNERS, dtype=np.float32) * (
            0.5 * keypoint_scale * fixed_size
        )
        self._object = cast(Entity, env.scene["object"])

    def __call__(self, env, **params) -> np.ndarray:
        del params
        hub = goal_command(env)
        # termination_utils.py:10-36 runs the tolerance curriculum before the
        # success judgment (same-frame effective); the hub owns the state,
        # this term owns the per-step trigger (audit fix A1.4).
        hub.tick_tolerance_curriculum()
        kp_offsets = np.broadcast_to(
            self._kp_offsets[None, :, :], (env.num_envs, *self._kp_offsets.shape)
        )
        # obs_utils.py:176-178 (fixed branch).
        kmd = _cached_keypoint_max_dist(env, hub, self._object, kp_offsets)
        # obs_utils.py:193-196: tolerance scaled by keypoint_scale.
        near_goal = kmd <= hub.current_success_tolerance * self._keypoint_scale
        # reward_utils.py:79-84 (update_near_goal_steps).
        ng = near_goal.astype(np.int64)
        if self._force_consecutive:
            steps = (hub.near_goal_steps + ng) * ng
        else:
            steps = hub.near_goal_steps + ng
        # obs_utils.py:204-205.
        hub.near_goal[:] = near_goal
        hub.near_goal_steps[:] = steps
        hub.pending_success[:] = steps >= self._success_steps
        # termination_utils.py:46-51 semantics (goal advance, tracker reset)
        # run in the command stage; success itself never terminates.
        return np.zeros(env.num_envs, dtype=bool)


class Fall(ManagerTermBase):
    """Object dropped below ``fall_height`` (termination_utils.py:54-55)."""

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(cfg, frozenset({"fall_height"}))
        self._fall_height = float(cfg.params.get("fall_height", _FALL_HEIGHT))
        self._object = cast(Entity, env.scene["object"])

    def __call__(self, env, **params) -> np.ndarray:
        del params
        # termination_utils.py:54-55: env-local object root z.
        object_z_local = self._object.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
        return object_z_local < self._fall_height


class HandFar(ManagerTermBase):
    """Any fingertip beyond ``hand_far_distance`` from the object center
    (termination_utils.py:62; distances per obs_utils.py:166-170)."""

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(cfg, frozenset({"hand_far_distance"}))
        self._distance = float(cfg.params.get("hand_far_distance", _HAND_FAR_DISTANCE))
        self._object = cast(Entity, env.scene["object"])
        self._robot = cast(Entity, env.scene["robot"])
        ids, _ = self._robot.find_bodies(
            match_entity_names(
                declared_names(self._robot, "body"), FINGERTIP_LINK_NAMES, label="body"
            ),
            preserve_order=True,
        )
        self._fingertip_ids = np.asarray(ids, dtype=np.intp)

    def __call__(self, env, **params) -> np.ndarray:
        del params
        hub = goal_command(env)
        distances = _cached_fingertip_distances(
            env, hub, self._object, self._robot, self._fingertip_ids
        )  # (N, 5)
        return distances.max(axis=-1) > self._distance  # termination_utils.py:62


class MaxConsecutiveSuccesses(ManagerTermBase):
    """Episode cap on chained goals (termination_utils.py:57-60)."""

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        _check_params(cfg, frozenset({"max_consecutive_successes"}))
        self._max = int(cfg.params.get("max_consecutive_successes", _MAX_CONSECUTIVE_SUCCESSES))

    def __call__(self, env, **params) -> np.ndarray:
        del params
        if self._max <= 0:
            return np.zeros(env.num_envs, dtype=bool)  # termination_utils.py:59-60
        hub = goal_command(env)
        # :46 increments _successes before the :58 comparison in the same
        # call; hub.goal_count increments in the later command stage, so fold
        # this step's pending successes in (module docstring, deviation list).
        return (hub.goal_count + hub.pending_success) >= self._max


def time_out(env) -> np.ndarray:
    """Truncation masked by the success snapshot (termination_utils.py:50-51, 65).

    The original zeroes ``episode_length_buf`` for successful envs before the
    truncation check; UniLab zeroes it in the command stage (after this
    manager runs), so the success frame is masked with ``pending_success``
    instead.  Registered with ``time_out: true`` so SAPG's
    ``value_bootstrap`` treats it as truncation, not termination.
    """
    hub = goal_command(env)
    return (env.episode_length_buf >= env.max_episode_length) & ~hub.pending_success


__all__ = [
    "Fall",
    "HandFar",
    "MaxConsecutiveSuccesses",
    "SuccessTracker",
    "time_out",
]
