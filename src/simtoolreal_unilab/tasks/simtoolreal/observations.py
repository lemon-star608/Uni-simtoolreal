"""SimToolReal observation terms.

Literal NumPy port of ``build_observations`` and its helpers
(``simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/obs_utils.py:75-348``,
snapshot 313d5aea) into two UniLab observation terms:

- ``SimToolRealPolicyObservation`` — actor frame (``obs_list``,
  simtoolreal_env_cfg.py:132-143): noisy/delayed object state path, joint
  velocity noise, whole-frame policy obs delay, width asserted against
  ``POLICY_OBS_DIM`` (140).
- ``SimToolRealCriticObservation`` — critic frame (``state_list``,
  simtoolreal_env_cfg.py:112-131): clean state path plus the success
  pre-caches read from the command hub, width asserted against
  ``CRITIC_OBS_DIM`` (162).

The clean/noisy field split follows obs_utils.py:308-337 exactly: the actor
overrides ``object_rot``/``object_vel``/``keypoints_rel_palm``/
``keypoints_rel_goal`` with the delayed+noisy object state and adds joint
velocity noise; every other field is the same clean value in both frames.
The critic consumes no RNG, matching the original single-pass assembly.

Registered deviations:

- Quaternions: internal math is wxyz (UniLab/Isaac Lab convention); the
  policy-facing ``palm_rot``/``object_rot`` fields are converted to the legacy
  xyzw layout (obs_utils.py:303-306, ``convert_quat(to="xyzw")``).
- RNG draws use ``env.rng`` (NumPy Generator) instead of torch
  (obs_utils.py:78-84, 131, 217, 336): same distributions, different stream.
- The per-env goal-yaw obs-noise hook (obs_utils.py:281-290,
  ``goal_yaw_obs_noise``) is a subclass ablation the base task never sets;
  it is not ported and ``noisy_goal_kp == goal_kp`` always.
- Delay queues live in the policy term (the original keeps them on the env,
  reset_utils.py:132-150).  UniLab recomputes observations for reset envs on
  the autoreset tail within the same control step. The row-scoped fast path
  now selects those rows before geometry assembly, while the term still
  memoizes the per-step push on ``env.common_step_counter``: a same-step
  re-entry only flushes the rows with ``episode_length_buf == 0 and
  goal_count == 0`` (the original's ``_episode_start`` mask,
  obs_utils.py:115-116) and leaves other rows' queues untouched, preserving
  the one-push-per-step cadence.
- Critic ``reward`` field on the autoreset-tail recompute reads the pre-reset
  step's ``env.reward_buf`` (UniLab does not zero it before the reset-path
  observation rebuild); the original's ``_reset_idx`` zeroes it first
  (reset_utils.py:196-197 allocation, DirectRLEnv reset).  Affects only the
  reset rows of the reset-path critic frame.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

from unilab.base.entity import Entity
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.utils.rotation import (
    np_quat_apply_batched,
    np_quat_from_angle_axis,
    np_quat_mul,
)

from .commands import SimToolRealCommand, goal_command
from .config import (
    FINGERTIP_LINK_NAMES,
    FINGERTIP_OFFSET,
    JOINT_NAMES_CANONICAL,
    KEYPOINT_CORNERS,
    NUM_FINGERTIPS,
    NUM_JOINTS,
    NUM_KEYPOINTS,
    PALM_BODY_NAME,
    PALM_CENTER_OFFSET,
    DomainRandomizationCfg,
    ObsCfg,
    RewardCfg,
)

# DESIGN.md section 1 (obs_utils.py:231-348): actor 140 / critic 162 fields.
# Hand-computed from OBS_FIELD_SIZES over the configured field lists; both are
# pinned by tests and code-asserted at assembly time.
POLICY_OBS_DIM = 140
CRITIC_OBS_DIM = 162

# obs_utils.py:35-54.
OBS_FIELD_SIZES: dict[str, int] = {
    "joint_pos": NUM_JOINTS,
    "joint_vel": NUM_JOINTS,
    "prev_action_targets": NUM_JOINTS,
    "palm_pos": 3,
    "palm_rot": 4,
    "palm_vel": 6,
    "object_rot": 4,
    "object_vel": 6,
    "fingertip_pos_rel_palm": 3 * NUM_FINGERTIPS,  # 15
    "keypoints_rel_palm": 3 * NUM_KEYPOINTS,  # 12
    "keypoints_rel_goal": 3 * NUM_KEYPOINTS,  # 12
    "object_scales": 3,
    "closest_keypoint_max_dist": 1,
    "closest_fingertip_dist": NUM_FINGERTIPS,  # 5
    "lifted_object": 1,
    "progress": 1,
    "successes": 1,
    "reward": 1,
}

_POLICY_FIELDS = ObsCfg().obs_list
_CRITIC_FIELDS = ObsCfg().state_list

# Object-state delay frame layout (reset_utils.py:132-138): pos(3) +
# quat(4, wxyz) + lin_vel(3) + ang_vel(3).
_OBJECT_STATE_DIM = 13


# ----------------------------------------------------------------------------
# Quaternion / keypoint / delay helpers (obs_utils.py:75-133)
# ----------------------------------------------------------------------------


def _perturb_quat(rng: np.random.Generator, q_wxyz: np.ndarray, max_deg: float) -> np.ndarray:
    """Apply random-axis rotation noise to wxyz quaternions (obs_utils.py:75-85)."""
    n = q_wxyz.shape[0]
    axis = rng.standard_normal((n, 3), dtype=np.float32)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    angle = rng.uniform(-max_deg, max_deg, n).astype(np.float32) * np.float32(math.pi / 180.0)
    dq = np_quat_from_angle_axis(angle, axis)
    return np_quat_mul(dq, q_wxyz).astype(np.float32)


def _perturb_quat_rows(
    rng: np.random.Generator,
    q_wxyz: np.ndarray,
    max_deg: float,
    num_envs: int,
    env_ids: np.ndarray,
) -> np.ndarray:
    """Perturb selected rows while preserving the full-batch RNG draw shape."""
    axis = rng.standard_normal((num_envs, 3), dtype=np.float32)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    angle = rng.uniform(-max_deg, max_deg, num_envs).astype(np.float32) * np.float32(
        math.pi / 180.0
    )
    dq = np_quat_from_angle_axis(angle[env_ids], axis[env_ids])
    return np_quat_mul(dq, q_wxyz).astype(np.float32)


def _apply_local_offset(
    pos_w: np.ndarray,
    rot_wxyz: np.ndarray,
    offset: tuple[float, float, float],
    batch_shape: tuple[int, ...],
) -> np.ndarray:
    """Apply one local-frame offset to batched world poses (obs_utils.py:88-100)."""
    offset_t = np.broadcast_to(np.asarray(offset, dtype=np.float32), (*batch_shape, 3))
    return pos_w + np_quat_apply_batched(rot_wxyz, offset_t)


def _keypoints_world(
    center_pos: np.ndarray,  # (N, 3)
    center_rot: np.ndarray,  # (N, 4) wxyz
    kp_offsets: np.ndarray,  # (N, K, 3)
) -> np.ndarray:
    """Rotate + translate object-frame keypoints (obs_utils.py:103-112)."""
    rotated = np_quat_apply_batched(center_rot[:, None, :], kp_offsets)
    return center_pos[:, None, :] + rotated


def _cached_keypoint_max_dist(
    env,
    hub,
    object_entity: Entity,
    kp_offsets: np.ndarray,
) -> np.ndarray:
    """Return the per-step keypoint distance cache used by reward and done terms.

    The original task computes this geometry once in ``compute_intermediate_values``
    and both ``_get_dones`` and ``_get_rewards`` consume that result.  Manager terms
    are independent by design, so keep the same sharing boundary in the task hub.
    The cache is keyed by the control-step counter and by the literal offsets, which
    keeps custom term configurations correct if they do not use the owner defaults.
    The goal pose comes from the hub buffer (env-local): the original reads back
    the GoalViz pose, which its per-advance writes keep equal to the current
    goal; UniLab defers that sim write, so the hub is the authoritative record
    (reaudit fix P3a).
    """
    # Small term-level tests use a lightweight SimpleNamespace and intentionally
    # mutate the fake backend between direct calls at the same counter.  The
    # production manager lifecycle always has a live NpEnv state object, which
    # is the scope where the original one-intermediate-per-step contract applies.
    if not hasattr(env, "_state"):
        origins = env.scene.env_origins
        obj_pos = object_entity.data.root_link_pos_w - origins
        obj_kp = _keypoints_world(obj_pos, object_entity.data.root_link_quat_w, kp_offsets)
        goal_kp = _keypoints_world(hub.goal[:, :3], hub.goal[:, 3:], kp_offsets)
        return np.linalg.norm(obj_kp - goal_kp, axis=-1).max(axis=-1)

    step = int(env.common_step_counter)
    if getattr(hub, "_geometry_cache_step", None) != step:
        hub._geometry_cache_step = step
        hub._geometry_keypoint_offsets = None
        hub._geometry_keypoint_max_dist = None
        hub._geometry_fingertip_ids = None
        hub._geometry_fingertip_distances = None

    offsets = np.asarray(kp_offsets[0], dtype=np.float32)
    cached_offsets = getattr(hub, "_geometry_keypoint_offsets", None)
    cached = getattr(hub, "_geometry_keypoint_max_dist", None)
    if cached is None or cached_offsets is None or not np.array_equal(cached_offsets, offsets):
        origins = env.scene.env_origins
        obj_pos = object_entity.data.root_link_pos_w - origins
        obj_kp = _keypoints_world(obj_pos, object_entity.data.root_link_quat_w, kp_offsets)
        goal_kp = _keypoints_world(hub.goal[:, :3], hub.goal[:, 3:], kp_offsets)
        cached = np.linalg.norm(obj_kp - goal_kp, axis=-1).max(axis=-1)
        hub._geometry_keypoint_offsets = offsets.copy()
        hub._geometry_keypoint_max_dist = cached
    return cached


def _cached_fingertip_distances(
    env,
    hub,
    object_entity: Entity,
    robot: Entity,
    fingertip_ids: np.ndarray,
) -> np.ndarray:
    """Return the per-step fingertip-to-object distance cache.

    This is the NumPy equivalent of the original ``_curr_fingertip_distances``
    intermediate buffer.  Both the approach reward and ``hand_far`` termination
    read the same frame, so no second norm is needed in the manager path.
    """
    if not hasattr(env, "_state"):
        origins = env.scene.env_origins
        obj_pos = object_entity.data.root_link_pos_w - origins
        ft_pos = robot.data.body_link_pos_w[:, fingertip_ids] - origins[:, None, :]
        return np.linalg.norm(ft_pos - obj_pos[:, None, :], axis=-1)

    step = int(env.common_step_counter)
    if getattr(hub, "_geometry_cache_step", None) != step:
        hub._geometry_cache_step = step
        hub._geometry_keypoint_offsets = None
        hub._geometry_keypoint_max_dist = None
        hub._geometry_fingertip_ids = None
        hub._geometry_fingertip_distances = None

    cached_ids = getattr(hub, "_geometry_fingertip_ids", None)
    cached = getattr(hub, "_geometry_fingertip_distances", None)
    ids = np.asarray(fingertip_ids, dtype=np.intp)
    if cached is None or cached_ids is None or not np.array_equal(cached_ids, ids):
        origins = env.scene.env_origins
        obj_pos = object_entity.data.root_link_pos_w - origins
        ft_pos = robot.data.body_link_pos_w[:, ids] - origins[:, None, :]
        cached = np.linalg.norm(ft_pos - obj_pos[:, None, :], axis=-1)
        hub._geometry_fingertip_ids = ids.copy()
        hub._geometry_fingertip_distances = cached
    return cached


def _sample_delay(
    queue: np.ndarray,
    values: np.ndarray,
    rng: np.random.Generator,
    flush: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Push current values into a rolling queue and sample per-env delay.

    obs_utils.py:119-133: freshly reset rows are flushed with the current
    value, then the queue rolls, slot 0 takes the current value, and each env
    draws a lag uniformly from ``range(queue_len)``.
    """
    if flush.any():
        queue[flush] = values[flush][:, None, :]
    queue = np.roll(queue, shift=1, axis=1)
    queue[:, 0, :] = values
    idx = rng.integers(0, queue.shape[1], size=values.shape[0])
    delayed = queue[np.arange(values.shape[0]), idx]
    return queue, delayed


def _episode_start(env, hub: SimToolRealCommand) -> np.ndarray:
    """obs_utils.py:115-116: fresh-episode mask that survives goal advances."""
    return (env.episode_length_buf == 0) & (hub.goal_count == 0)


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Legacy policy layout conversion (obs_utils.py:303-306, convert_quat)."""
    return np.ascontiguousarray(q[..., [1, 2, 3, 0]])


def _field_slice(field_list: tuple[str, ...], name: str) -> slice:
    """Stacked-frame column slice of one field (OBS_FIELD_SIZES order)."""
    offset = 0
    for field in field_list:
        width = OBS_FIELD_SIZES[field]
        if field == name:
            return slice(offset, offset + width)
        offset += width
    raise KeyError(f"unknown observation field {name!r}")


def _stale_terminal_positions(env, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows of ``ids`` whose terminal observation the current autoreset saved.

    ``NpEnv._reset_done_envs`` records the pre-reset frame and marks exactly
    the done rows in ``state.info["_final_observation"]``; manual resets and
    the initial reset leave the mask cleared for their rows, matching the
    original repository's fresh explicit-reset observations.
    """
    state = getattr(env, "state", None)
    info = getattr(state, "info", None)
    mask = info.get("_final_observation") if isinstance(info, dict) else None
    if not isinstance(mask, np.ndarray) or mask.shape != (env.num_envs,):
        empty = np.empty(0, dtype=np.intp)
        return empty, empty
    local = mask[ids]
    return np.flatnonzero(local), np.asarray(ids, dtype=np.intp)[local]


def _stack(fields: dict[str, np.ndarray], field_list: tuple[str, ...]) -> np.ndarray:
    """Concatenate named fields in config order (obs_utils.py:62-67)."""
    parts = [fields[name].reshape(fields[name].shape[0], -1) for name in field_list]
    return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)


# ----------------------------------------------------------------------------
# Cold-path geometry cache + clean-field assembly (obs_utils.py:136-327)
# ----------------------------------------------------------------------------


class _Geometry:
    """Resolved entity/body/joint views shared by the actor and critic terms."""

    def __init__(self, env) -> None:
        # Fail-fast entity resolution by name (cold path): a scene without
        # robot/object raises KeyError here instead of silently producing
        # zero frames.  The goal pose is read from the command hub (fix P3a),
        # so no goalviz entity is resolved here.
        self.robot = cast(Entity, env.scene["robot"])
        self.object = cast(Entity, env.scene["object"])
        self.hub = goal_command(env)

        joint_ids, _ = self.robot.find_joints(list(JOINT_NAMES_CANONICAL), preserve_order=True)
        if len(joint_ids) != NUM_JOINTS:
            raise ValueError(
                f"SimToolReal observations expected {NUM_JOINTS} canonical joints, "
                f"resolved {len(joint_ids)}"
            )
        self.joint_ids = np.asarray(joint_ids, dtype=np.intp)

        # Canonical-order limits for the joint_pos normalization
        # (reset_utils.py:52-56).  Model-level table; identical across envs.
        limits = self.robot.data.soft_joint_pos_limits[self.joint_ids]  # (29, 2)
        self.joint_lower_canon = limits[:, 0]
        self.joint_upper_canon = limits[:, 1]

        self.palm_id = int(self.robot.find_bodies([PALM_BODY_NAME])[0][0])
        fingertip_ids, _ = self.robot.find_bodies(list(FINGERTIP_LINK_NAMES), preserve_order=True)
        assert len(fingertip_ids) == NUM_FINGERTIPS  # reset_utils.py:38
        self.fingertip_ids = np.asarray(fingertip_ids, dtype=np.intp)

        self._corners = np.asarray(KEYPOINT_CORNERS, dtype=np.float32)  # (4, 3)


def _assemble_clean(
    geo: _Geometry, env, env_ids: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Clean shared geometry and per-field values (obs_utils.py:236-327).

    Field values are env-local (world minus ``env_origins``) float32 arrays.
    """
    hub = geo.hub
    if env_ids is None:
        ids = None
        num_rows = env.num_envs
    else:
        ids = np.asarray(env_ids, dtype=np.intp)
        if ids.ndim != 1:
            raise ValueError(f"observation env_ids must be 1-D, got {ids.shape}")
        num_rows = int(ids.size)

    def rows(value: np.ndarray) -> np.ndarray:
        return value if ids is None else value[ids]

    origins = env.scene.env_origins if ids is None else env.scene.env_origins[ids]

    # Canonical joint obs (obs_utils.py:136-150).  ``prev_targets`` is stored
    # canonical-order by the action term, so no permutation is needed here
    # (the original applies ``_perm_lab_to_canon`` at :145).
    joint_pos_raw = rows(geo.robot.data.joint_pos)[:, geo.joint_ids]
    joint_pos = (
        2.0
        * (joint_pos_raw - geo.joint_lower_canon)
        / (geo.joint_upper_canon - geo.joint_lower_canon)
        - 1.0
    )
    joint_vel = rows(geo.robot.data.joint_vel)[:, geo.joint_ids]
    prev_targets = rows(env.action_manager.get_term("joint_pos").prev_targets)

    # Palm state with the trained palm-center offset (obs_utils.py:238-246).
    palm_pos_w = rows(geo.robot.data.body_link_pos_w)[:, geo.palm_id]
    palm_rot = rows(geo.robot.data.body_link_quat_w)[:, geo.palm_id]  # wxyz
    palm_vel = np.concatenate(
        (
            rows(geo.robot.data.body_link_lin_vel_w)[:, geo.palm_id],
            rows(geo.robot.data.body_link_ang_vel_w)[:, geo.palm_id],
        ),
        axis=-1,
    )
    palm_center_pos_w = _apply_local_offset(palm_pos_w, palm_rot, PALM_CENTER_OFFSET, (num_rows,))
    palm_pos = palm_center_pos_w - origins

    # Fingertip pad centers (obs_utils.py:248-257).
    ft_body_pos_w = rows(geo.robot.data.body_link_pos_w)[:, geo.fingertip_ids]
    ft_body_rot_w = rows(geo.robot.data.body_link_quat_w)[:, geo.fingertip_ids]
    ft_pos_w = _apply_local_offset(
        ft_body_pos_w, ft_body_rot_w, FINGERTIP_OFFSET, (num_rows, NUM_FINGERTIPS)
    )

    # Object / goal root state (obs_utils.py:259-266).
    obj_pos = rows(geo.object.data.root_link_pos_w) - origins
    obj_rot = rows(geo.object.data.root_link_quat_w)  # wxyz
    obj_vel = np.concatenate(
        (rows(geo.object.data.root_link_lin_vel_w), rows(geo.object.data.root_link_ang_vel_w)),
        axis=-1,
    )
    # The goal comes from the hub buffer (env-local): the original reads back
    # the GoalViz pose, which its per-advance writes keep equal to the current
    # goal; UniLab defers that sim write to the reset path, so the hub is the
    # authoritative record (reaudit fix P3a).
    goal = rows(hub.goal)
    goal_pos = goal[:, :3]
    goal_rot = goal[:, 3:]  # wxyz

    # Keypoint offsets: per-env tool scale (reset_utils.py:74-77) scaled by
    # the DR multiplier (obs_utils.py:275).  Reward constants come from the
    # typed mirror (SimToolReal.yaml reward: section).
    rew = RewardCfg()
    kp_offsets = (
        geo._corners[None, :, :]
        * (
            rows(hub.object_scale_per_env)
            * np.float32(rew.object_base_size * rew.keypoint_scale * 0.5)
        )[:, None, :]
    )
    kp_offsets = kp_offsets * rows(hub.object_scale_multiplier)[:, None, :]

    obj_kp = _keypoints_world(obj_pos, obj_rot, kp_offsets)  # (N, 4, 3)
    goal_kp = _keypoints_world(goal_pos, goal_rot, kp_offsets)

    # obs_utils.py:297-301.
    fingertip_pos_rel_palm = (ft_pos_w - origins[:, None, :]) - palm_pos[:, None, :]
    object_scales = rows(hub.object_scale_per_env) * rows(hub.object_scale_multiplier)

    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "prev_action_targets": prev_targets,
        "palm_pos": palm_pos,
        "palm_rot": palm_rot,
        "palm_vel": palm_vel,
        "obj_pos": obj_pos,
        "obj_rot": obj_rot,
        "obj_vel": obj_vel,
        "obj_kp": obj_kp,
        "goal_kp": goal_kp,
        "kp_offsets": kp_offsets,
        "fingertip_pos_rel_palm": fingertip_pos_rel_palm,
        "object_scales": object_scales,
    }


def _assemble_clean_cached(
    geo: _Geometry, env, env_ids: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Share one clean frame between the policy and critic terms.

    ``build_observations`` in the original repository creates one clean dict and
    derives both actor and critic fields from it.  The two manager terms are kept
    separate for the public manager contract, but their pure clean assembly can be
    shared safely within one step.  A reset-row call uses a separate cache key, so
    it cannot reuse a full-step frame or another reset batch.
    """
    hub = geo.hub
    ids = None if env_ids is None else np.asarray(env_ids, dtype=np.intp)
    if ids is not None and ids.ndim != 1:
        raise ValueError(f"observation env_ids must be 1-D, got {ids.shape}")
    # Lightweight term fixtures do not have the NpEnv lifecycle state and may
    # intentionally mutate their fake backend between direct calls.  Keep the
    # pre-cache direct-call semantics there; the production manager always owns
    # a live state object when actor/critic are evaluated together.
    if not hasattr(env, "_state"):
        return _assemble_clean(geo, env, ids)
    step = int(env.common_step_counter)
    cached_step = getattr(hub, "_obs_clean_cache_step", None)
    cached_ids = getattr(hub, "_obs_clean_cache_ids", None)
    cached = getattr(hub, "_obs_clean_cache", None)
    same_ids = (cached_ids is None and ids is None) or (
        cached_ids is not None and ids is not None and np.array_equal(cached_ids, ids)
    )
    if cached is not None and cached_step == step and same_ids:
        return cached

    clean = _assemble_clean(geo, env, ids)
    hub._obs_clean_cache_step = step
    hub._obs_clean_cache_ids = None if ids is None else ids.copy()
    hub._obs_clean_cache = clean
    return clean


# ----------------------------------------------------------------------------
# Observation terms (obs_utils.py:231-349)
# ----------------------------------------------------------------------------


class SimToolRealPolicyObservation(ManagerTermBase):
    """Actor frame: noisy/delayed object state path (obs_utils.py:329-346).

    Owns the object-state delay queue (13-wide, obs_utils.py:210-220) and the
    whole-frame policy obs delay queue (obs_utils.py:223-228), both allocated
    like reset_utils.py:132-150 and zeroed per env on reset
    (reset_utils.py:399-401).
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        self._geo = _Geometry(env)
        dr = DomainRandomizationCfg()
        obs = ObsCfg()
        params: dict[str, Any] = cfg.params
        self._use_object_state_delay_noise = bool(
            params.get("use_object_state_delay_noise", dr.use_object_state_delay_noise)
        )
        self._object_state_delay_max = int(
            params.get("object_state_delay_max", dr.object_state_delay_max)
        )
        self._object_state_xyz_noise_std = float(
            params.get("object_state_xyz_noise_std", dr.object_state_xyz_noise_std)
        )
        self._object_state_rotation_noise_degrees = float(
            params.get(
                "object_state_rotation_noise_degrees",
                dr.object_state_rotation_noise_degrees,
            )
        )
        self._joint_velocity_obs_noise_std = float(
            params.get("joint_velocity_obs_noise_std", dr.joint_velocity_obs_noise_std)
        )
        self._use_obs_delay = bool(params.get("use_obs_delay", dr.use_obs_delay))
        self._obs_delay_max = int(params.get("obs_delay_max", dr.obs_delay_max))
        self._clamp_abs = float(params.get("clamp_abs_observations", obs.clamp_abs_observations))
        # Reset-row palm replication (reaudit fix P4 / R2): the original's
        # reset-row frame reads palm/fingertip body state collected before the
        # reset writes.  "stale" (default) replicates that from this term's
        # pre-delay frame; "fresh" keeps fully post-reset values.
        self._reset_row_body_state = (
            str(params.get("reset_row_body_state", "stale")).strip().lower()
        )
        if self._reset_row_body_state not in {"stale", "fresh"}:
            raise ValueError(
                f"reset_row_body_state must be 'stale' or 'fresh', got {self._reset_row_body_state!r}"
            )
        self._raw_frame: np.ndarray | None = None
        self._raw_frame_step: int = -1

        # reset_utils.py:132-144: max(1, delay_max) slots, always allocated.
        self._object_state_queue = np.zeros(
            (self.num_envs, max(1, self._object_state_delay_max), _OBJECT_STATE_DIM),
            dtype=np.float32,
        )
        self._obs_queue = np.zeros(
            (self.num_envs, max(1, self._obs_delay_max), POLICY_OBS_DIM),
            dtype=np.float32,
        )
        # Same-step memo (see module docstring): the autoreset tail recomputes
        # observations for reset envs within the same control step.  Each queue
        # tracks its own step key so a disabled object-state path does not
        # stall the whole-frame obs delay.
        self._object_state_step_key: int | None = None
        self._obs_step_key: int | None = None
        self._delayed_object_state = np.zeros((self.num_envs, _OBJECT_STATE_DIM), dtype=np.float32)
        self._delayed_policy_obs = np.zeros((self.num_envs, POLICY_OBS_DIM), dtype=np.float32)

    def __call__(self, env, **params) -> np.ndarray:
        del params
        return self._compute(env, None)

    def compute_rows(self, env, env_ids: np.ndarray, **params) -> np.ndarray:
        """Compute only reset rows after the full step has populated delay state."""
        del params
        ids = np.asarray(env_ids, dtype=np.intp)
        if ids.ndim != 1:
            raise ValueError(f"observation env_ids must be 1-D, got {ids.shape}")
        if ids.size == self.num_envs:
            return self._compute(env, None)
        step_key = env.common_step_counter
        if (self._use_object_state_delay_noise and self._object_state_step_key != step_key) or (
            self._use_obs_delay and self._obs_step_key != step_key
        ):
            # Cold/manual partial resets have no full-step delay frame to
            # reuse. Keep the established full-batch behavior in that case.
            return self._compute(env, None)[ids]
        return self._compute(env, ids)

    def _compute(self, env, env_ids: np.ndarray | None) -> np.ndarray:
        geo = self._geo
        hub = geo.hub
        rng = env.rng
        ids = None if env_ids is None else np.asarray(env_ids, dtype=np.intp)
        clean = _assemble_clean_cached(geo, env, ids)
        row_count = env.num_envs if ids is None else int(ids.size)

        # Object-state delay + noise (obs_utils.py:210-220, 268-273).
        obj_state = np.concatenate(
            (
                clean["obj_pos"],
                clean["obj_rot"],
                clean["obj_vel"][:, 0:3],
                clean["obj_vel"][:, 3:6],
            ),
            axis=-1,
        )
        flush = _episode_start(env, hub)
        if ids is not None:
            flush = flush[ids]
        if self._use_object_state_delay_noise:
            if ids is None:
                delayed = self._advance_object_state_delay(obj_state, flush)
                xyz_noise = rng.standard_normal((self.num_envs, 3), dtype=np.float32)
            else:
                delayed = self._advance_object_state_delay_rows(ids, obj_state, flush)
                xyz_noise = rng.standard_normal((self.num_envs, 3), dtype=np.float32)[ids]
            noisy_obj_pos = delayed[:, 0:3] + xyz_noise * np.float32(
                self._object_state_xyz_noise_std
            )
            if ids is None:
                noisy_obj_rot = _perturb_quat(
                    rng, delayed[:, 3:7], self._object_state_rotation_noise_degrees
                )
            else:
                noisy_obj_rot = _perturb_quat_rows(
                    rng,
                    delayed[:, 3:7],
                    self._object_state_rotation_noise_degrees,
                    self.num_envs,
                    ids,
                )
            # delayed[:, 7:13] (object velocity) has no actor-side consumer:
            # the policy field list carries no object_vel (obs_utils.py:308-337).
        else:
            noisy_obj_pos = clean["obj_pos"]
            noisy_obj_rot = clean["obj_rot"]

        # obs_utils.py:275-299: noisy keypoints share the clean kp offsets;
        # noisy_goal_kp == goal_kp (the goal-yaw hook is not ported).
        noisy_obj_kp = _keypoints_world(noisy_obj_pos, noisy_obj_rot, clean["kp_offsets"])
        palm_pos = clean["palm_pos"]
        keypoints_rel_palm_noisy = noisy_obj_kp - palm_pos[:, None, :]
        keypoints_rel_goal_noisy = noisy_obj_kp - clean["goal_kp"]

        # obs_utils.py:308-337: clean dict, then the actor-side overrides.
        joint_vel = clean["joint_vel"]
        if self._joint_velocity_obs_noise_std > 0:
            if ids is not None:
                joint_noise = rng.standard_normal(
                    (self.num_envs, joint_vel.shape[1]), dtype=np.float32
                )[ids]
            else:
                joint_noise = rng.standard_normal(joint_vel.shape, dtype=np.float32)
            joint_vel = joint_vel + joint_noise * np.float32(self._joint_velocity_obs_noise_std)

        fields = {
            "joint_pos": clean["joint_pos"],
            "joint_vel": joint_vel,
            "prev_action_targets": clean["prev_action_targets"],
            "palm_pos": palm_pos,
            "palm_rot": _wxyz_to_xyzw(clean["palm_rot"]),
            "object_rot": _wxyz_to_xyzw(noisy_obj_rot),
            "fingertip_pos_rel_palm": clean["fingertip_pos_rel_palm"],
            "keypoints_rel_palm": keypoints_rel_palm_noisy,
            "keypoints_rel_goal": keypoints_rel_goal_noisy,
            "object_scales": clean["object_scales"],
        }
        step_key = int(env.common_step_counter)
        if ids is not None and self._reset_row_body_state == "stale":
            # Reset-row palm replication (reaudit fix P4 / R2): terminal rows
            # keep the pre-reset palm/fingertip fields, and the palm-relative
            # keypoints re-derive against the stale palm position — the
            # original's reset-row frame mixes fresh object state with palm
            # body state collected before the reset writes.
            positions, gids = _stale_terminal_positions(env, ids)
            if positions.size and self._raw_frame is not None and self._raw_frame_step == step_key:
                for name in ("palm_pos", "palm_rot", "fingertip_pos_rel_palm"):
                    target = fields[name][positions]
                    stale = self._raw_frame[gids][:, _field_slice(_POLICY_FIELDS, name)]
                    fields[name][positions] = stale.reshape(target.shape)
                fields["keypoints_rel_palm"] = fields["keypoints_rel_palm"].copy()
                fields["keypoints_rel_palm"][positions] = (
                    noisy_obj_kp[positions] - fields["palm_pos"][positions][:, None, :]
                )
        policy_tensor = _stack(fields, _POLICY_FIELDS)
        if ids is None:
            # Pre-delay raw frame: the stale source for this step's reset-row
            # recomputes (the saved final_observation is post-delay).
            self._raw_frame = policy_tensor.copy()
            self._raw_frame_step = step_key

        # Whole-frame policy obs delay (obs_utils.py:342-343).
        if self._use_obs_delay:
            if ids is None:
                policy_tensor = self._advance_obs_delay(policy_tensor, flush)
            else:
                policy_tensor = self._advance_obs_delay_rows(ids, policy_tensor, flush)

        # obs_utils.py:345-347: symmetric clamp applied to both frames.
        policy_tensor = np.clip(policy_tensor, -self._clamp_abs, self._clamp_abs)
        assert policy_tensor.shape == (row_count, POLICY_OBS_DIM), (
            f"SimToolReal actor frame width {policy_tensor.shape} != "
            f"({row_count}, {POLICY_OBS_DIM})"
        )
        return policy_tensor.astype(np.float32, copy=False)

    def _advance_object_state_delay(self, obj_state: np.ndarray, flush: np.ndarray) -> np.ndarray:
        step_key = self._env.common_step_counter
        if step_key != self._object_state_step_key:
            self._object_state_queue, delayed = _sample_delay(
                self._object_state_queue, obj_state, self._env.rng, flush
            )
            self._object_state_step_key = step_key
            self._delayed_object_state = delayed
            return delayed
        # Same-step re-entry (autoreset-tail recompute): refresh only the
        # freshly reset rows; other rows keep this step's delayed frame.
        delayed = self._delayed_object_state.copy()
        if flush.any():
            ids = np.flatnonzero(flush)
            self._object_state_queue[flush] = obj_state[flush][:, None, :]
            idx = self._env.rng.integers(0, self._object_state_queue.shape[1], size=ids.size)
            delayed[ids] = self._object_state_queue[ids, idx]
        self._delayed_object_state = delayed
        return delayed

    def _advance_object_state_delay_rows(
        self, env_ids: np.ndarray, obj_state: np.ndarray, flush: np.ndarray
    ) -> np.ndarray:
        """Refresh reset rows without rebuilding the full object-state frame."""
        if self._object_state_step_key != self._env.common_step_counter:
            raise RuntimeError("row-scoped object delay requires a populated full-step frame")
        delayed = self._delayed_object_state[env_ids].copy()
        if flush.any():
            local_ids = np.flatnonzero(flush)
            global_ids = env_ids[local_ids]
            self._object_state_queue[global_ids] = obj_state[local_ids, None, :]
            idx = self._env.rng.integers(0, self._object_state_queue.shape[1], size=local_ids.size)
            delayed[local_ids] = self._object_state_queue[global_ids, idx]
            self._delayed_object_state[global_ids] = delayed[local_ids]
        return delayed

    def _advance_obs_delay(self, policy_tensor: np.ndarray, flush: np.ndarray) -> np.ndarray:
        step_key = self._env.common_step_counter
        if step_key != self._obs_step_key:
            self._obs_queue, delayed = _sample_delay(
                self._obs_queue, policy_tensor, self._env.rng, flush
            )
            self._obs_step_key = step_key
            self._delayed_policy_obs = delayed
            return delayed
        delayed = self._delayed_policy_obs.copy()
        if flush.any():
            ids = np.flatnonzero(flush)
            self._obs_queue[flush] = policy_tensor[flush][:, None, :]
            idx = self._env.rng.integers(0, self._obs_queue.shape[1], size=ids.size)
            delayed[ids] = self._obs_queue[ids, idx]
        self._delayed_policy_obs = delayed
        return delayed

    def _advance_obs_delay_rows(
        self, env_ids: np.ndarray, policy_tensor: np.ndarray, flush: np.ndarray
    ) -> np.ndarray:
        """Refresh reset rows without rebuilding the full policy delay frame."""
        if self._obs_step_key != self._env.common_step_counter:
            raise RuntimeError("row-scoped observation delay requires a populated full-step frame")
        delayed = self._delayed_policy_obs[env_ids].copy()
        if flush.any():
            local_ids = np.flatnonzero(flush)
            global_ids = env_ids[local_ids]
            self._obs_queue[global_ids] = policy_tensor[local_ids, None, :]
            idx = self._env.rng.integers(0, self._obs_queue.shape[1], size=local_ids.size)
            delayed[local_ids] = self._obs_queue[global_ids, idx]
            self._delayed_policy_obs[global_ids] = delayed[local_ids]
        return delayed

    def reset(self, env_ids) -> None:
        # reset_utils.py:399-401: clear the delay queue rows.
        self._object_state_queue[env_ids] = 0.0
        self._obs_queue[env_ids] = 0.0


class SimToolRealCriticObservation(ManagerTermBase):
    """Critic frame: clean state path without actor-side noise/delay.

    obs_utils.py:308-327, 339, 347 (``state_list`` + clamp).  Success
    pre-caches (``closest_*``, ``lifted_object``, ``successes``) are read from
    the command hub (``goal_command(env)``); the termination/reward layer owns
    their per-step updates (step-1 stage 2).
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        self._geo = _Geometry(env)
        obs = ObsCfg()
        self._clamp_abs = float(
            cfg.params.get("clamp_abs_observations", obs.clamp_abs_observations)
        )
        # Reset-row palm replication (reaudit fix P4 / R2); see the policy
        # term for the mode contract.
        self._reset_row_body_state = (
            str(cfg.params.get("reset_row_body_state", "stale")).strip().lower()
        )
        if self._reset_row_body_state not in {"stale", "fresh"}:
            raise ValueError(
                f"reset_row_body_state must be 'stale' or 'fresh', got {self._reset_row_body_state!r}"
            )
        self._raw_frame: np.ndarray | None = None
        self._raw_frame_step: int = -1

    def __call__(self, env, **params) -> np.ndarray:
        del params
        return self._compute(env, None)

    def compute_rows(self, env, env_ids: np.ndarray, **params) -> np.ndarray:
        del params
        ids = np.asarray(env_ids, dtype=np.intp)
        if ids.ndim != 1:
            raise ValueError(f"observation env_ids must be 1-D, got {ids.shape}")
        return self._compute(env, None if ids.size == self.num_envs else ids)

    def _compute(self, env, env_ids: np.ndarray | None) -> np.ndarray:
        geo = self._geo
        hub = geo.hub
        ids = None if env_ids is None else np.asarray(env_ids, dtype=np.intp)
        clean = _assemble_clean_cached(geo, env, ids)
        row_count = env.num_envs if ids is None else int(ids.size)

        def rows(value: np.ndarray) -> np.ndarray:
            return value if ids is None else value[ids]

        palm_pos = clean["palm_pos"]
        fields = {
            "joint_pos": clean["joint_pos"],
            "joint_vel": clean["joint_vel"],
            "prev_action_targets": clean["prev_action_targets"],
            "palm_pos": palm_pos,
            "palm_rot": _wxyz_to_xyzw(clean["palm_rot"]),
            "palm_vel": clean["palm_vel"],
            "object_rot": _wxyz_to_xyzw(clean["obj_rot"]),
            "object_vel": clean["obj_vel"],
            "fingertip_pos_rel_palm": clean["fingertip_pos_rel_palm"],
            "keypoints_rel_palm": clean["obj_kp"] - palm_pos[:, None, :],
            "keypoints_rel_goal": clean["obj_kp"] - clean["goal_kp"],
            "object_scales": clean["object_scales"],
            # Success pre-caches via the command hub (obs_utils.py:321-326).
            "closest_keypoint_max_dist": rows(hub.closest_keypoint_max_dist)[:, None],
            "closest_fingertip_dist": rows(hub.closest_fingertip_dist),
            "lifted_object": rows(hub.lifted).astype(np.float32)[:, None],
            "progress": np.log(
                rows(env.episode_length_buf).astype(np.float32) / np.float32(10.0) + np.float32(1.0)
            )[:, None],
            "successes": np.log(rows(hub.goal_count).astype(np.float32) + np.float32(1.0))[:, None],
            "reward": (rows(env.reward_buf) * np.float32(0.01))[:, None],
        }
        step_key = int(env.common_step_counter)
        if ids is not None and self._reset_row_body_state == "stale":
            # Reset-row palm replication (reaudit fix P4 / R2): terminal rows
            # keep the pre-reset palm fields, and keypoints_rel_palm re-derives
            # fresh object keypoints against the stale palm position.
            positions, gids = _stale_terminal_positions(env, ids)
            if positions.size and self._raw_frame is not None and self._raw_frame_step == step_key:
                for name in ("palm_pos", "palm_rot", "palm_vel", "fingertip_pos_rel_palm"):
                    target = fields[name][positions]
                    stale = self._raw_frame[gids][:, _field_slice(_CRITIC_FIELDS, name)]
                    fields[name][positions] = stale.reshape(target.shape)
                fields["keypoints_rel_palm"] = fields["keypoints_rel_palm"].copy()
                fields["keypoints_rel_palm"][positions] = (
                    clean["obj_kp"][positions] - fields["palm_pos"][positions][:, None, :]
                )
        state_tensor = _stack(fields, _CRITIC_FIELDS)
        if ids is None:
            self._raw_frame = state_tensor.copy()
            self._raw_frame_step = step_key
        state_tensor = np.clip(state_tensor, -self._clamp_abs, self._clamp_abs)
        assert state_tensor.shape == (row_count, CRITIC_OBS_DIM), (
            f"SimToolReal critic frame width {state_tensor.shape} != "
            f"({row_count}, {CRITIC_OBS_DIM})"
        )
        return state_tensor.astype(np.float32, copy=False)


__all__ = [
    "CRITIC_OBS_DIM",
    "OBS_FIELD_SIZES",
    "POLICY_OBS_DIM",
    "SimToolRealCriticObservation",
    "SimToolRealPolicyObservation",
]
