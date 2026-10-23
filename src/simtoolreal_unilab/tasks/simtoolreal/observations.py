"""Torch-native SimToolReal actor and critic observations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import torch
from unilab.base.entity import Entity
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.rng import ManagerRng

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
    declared_names,
    match_entity_names,
)
from .tensor_utils import (
    apply_local_offset,
    index_tensor,
    keypoints_world,
    normalize_quat,
    quat_from_angle_axis,
    quat_mul,
    require_body_view,
    require_joint_view,
)

POLICY_OBS_DIM = 140
CRITIC_OBS_DIM = 162
OBS_FIELD_SIZES: dict[str, int] = {
    "joint_pos": NUM_JOINTS,
    "joint_vel": NUM_JOINTS,
    "prev_action_targets": NUM_JOINTS,
    "palm_pos": 3,
    "palm_rot": 4,
    "palm_vel": 6,
    "object_rot": 4,
    "object_vel": 6,
    "fingertip_pos_rel_palm": 3 * NUM_FINGERTIPS,
    "keypoints_rel_palm": 3 * NUM_KEYPOINTS,
    "keypoints_rel_goal": 3 * NUM_KEYPOINTS,
    "object_scales": 3,
    "closest_keypoint_max_dist": 1,
    "closest_fingertip_dist": NUM_FINGERTIPS,
    "lifted_object": 1,
    "progress": 1,
    "successes": 1,
    "reward": 1,
}
_POLICY_FIELDS = ObsCfg().obs_list
_CRITIC_FIELDS = ObsCfg().state_list
_OBJECT_STATE_DIM = 13


def _wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[:, 1:], quat[:, :1]), dim=-1)


def _stack(fields: dict[str, torch.Tensor], names: Sequence[str]) -> torch.Tensor:
    return torch.cat([fields[name].reshape(fields[name].shape[0], -1) for name in names], dim=-1)


def _field_slice(names: Sequence[str], field: str) -> tuple[int, int]:
    start = sum(OBS_FIELD_SIZES[name] for name in names[: names.index(field)])
    return start, start + OBS_FIELD_SIZES[field]


def _episode_start(env, hub: SimToolRealCommand) -> torch.Tensor:
    return (env.episode_length_buf == 0) & (hub.goal_count == 0)


def _cold_observation_probe(env, width: int) -> torch.Tensor | None:
    """Return the shape-only frame ObservationManager requests before plan compilation."""
    plan = getattr(getattr(env, "scene", None), "read_plan", None)
    if plan is not None and plan.ready:
        return None
    return torch.zeros((env.num_envs, width), dtype=torch.float32, device=env.device)


def _sample_delay(
    queue: torch.Tensor,
    values: torch.Tensor,
    rng: ManagerRng,
    flush: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bool(flush.any()):
        rows = torch.nonzero(flush, as_tuple=False).flatten()
        queue[rows] = values[rows, None, :]
    queue = torch.roll(queue, shifts=1, dims=1)
    queue[:, 0, :].copy_(values)
    lag = rng.integers(0, queue.shape[1], size=values.shape[0])
    rows = torch.arange(values.shape[0], device=values.device)[:, None]
    return queue, queue[rows, lag[:, None]]


def _refresh_delay_on_reentry(
    queue: torch.Tensor,
    cached: torch.Tensor,
    values: torch.Tensor,
    rng: ManagerRng,
    flush: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    delayed = cached.clone()
    if bool(flush.any()):
        rows = torch.nonzero(flush, as_tuple=False).flatten()
        queue[rows] = values[rows, None, :]
        lag = rng.integers(0, queue.shape[1], size=rows.numel())
        delayed[rows] = queue[rows, lag[:, None]]
    return queue, delayed


def _perturb_quat(rng: ManagerRng, quat: torch.Tensor, max_deg: float) -> torch.Tensor:
    axis = rng.standard_normal((quat.shape[0], 3))
    axis = normalize_quat(axis)
    angle = rng.uniform(-max_deg, max_deg, quat.shape[0]) * (math.pi / 180.0)
    return quat_mul(quat_from_angle_axis(angle, axis), quat)


class _Geometry:
    """Cold entity/name resolution and one-per-step shared geometry caches."""

    def __init__(self, env):
        self.env = env
        self.hub = goal_command(env)
        self.robot = cast(Entity, env.scene["robot"])
        self.object = cast(Entity, env.scene["object"])
        joint_ids, _ = self.robot.find_joints(
            match_entity_names(
                declared_names(self.robot, "joint"), JOINT_NAMES_CANONICAL, label="joint"
            ),
            preserve_order=True,
        )
        if len(joint_ids) != NUM_JOINTS:
            raise ValueError(f"SimToolReal expected {NUM_JOINTS} joints, got {len(joint_ids)}")
        self.joint_ids = index_tensor(joint_ids, env.device)
        self.palm_body_name = self.hub.palm_body_name
        self.fingertip_body_names = self.hub.fingertip_body_names
        self.robot_body_names = (self.palm_body_name, *self.fingertip_body_names)
        self.object_body_name = self.hub.object_body_name
        self._corners = torch.tensor(KEYPOINT_CORNERS, dtype=torch.float32, device=env.device)

    def joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        state = require_joint_view(self.env, self.robot)
        return state.joint_pos.index_select(1, self.joint_ids), state.joint_vel.index_select(
            1, self.joint_ids
        )

    def robot_body_state(self):
        return require_body_view(self.env, self.robot, self.robot_body_names)

    def object_body_state(self):
        return require_body_view(self.env, self.object, (self.object_body_name,))

    def origins(self) -> torch.Tensor:
        return self.env.scene.env_origins(self.env.device)


def _object_pose(geo: _Geometry) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    view = geo.object_body_state()
    pos = view.pos_w[:, 0, :] - geo.origins()
    rot = view.quat_w[:, 0, :]
    vel = torch.cat((view.lin_vel_w[:, 0, :], view.ang_vel_w[:, 0, :]), dim=-1)
    return pos, rot, vel


def _assemble_clean(geo: _Geometry, env) -> dict[str, torch.Tensor]:
    hub = geo.hub
    origins = geo.origins()
    joint_pos, joint_vel = geo.joint_state()
    prev_targets = env.action_manager.get_term("joint_pos").prev_targets

    robot = geo.robot_body_state()
    palm_pos_w = robot.pos_w[:, 0, :]
    palm_rot = robot.quat_w[:, 0, :]
    palm_vel = torch.cat((robot.lin_vel_w[:, 0, :], robot.ang_vel_w[:, 0, :]), dim=-1)
    palm_center_w = apply_local_offset(palm_pos_w, palm_rot, PALM_CENTER_OFFSET)
    palm_pos = palm_center_w - origins

    ft_pos_w = robot.pos_w[:, 1:, :]
    ft_rot_w = robot.quat_w[:, 1:, :]
    ft_pos_w = apply_local_offset(ft_pos_w, ft_rot_w, FINGERTIP_OFFSET)
    fingertip_pos_rel_palm = (ft_pos_w - origins[:, None, :]) - palm_pos[:, None, :]

    obj_pos, obj_rot, obj_vel = _object_pose(geo)
    goal = hub.goal
    goal_pos, goal_rot = goal[:, :3], goal[:, 3:]

    rew = RewardCfg()
    scale_factor = rew.object_base_size * rew.keypoint_scale * 0.5
    kp_offsets = (
        geo._corners[None, :, :]
        * (hub.object_scale_per_env * hub.object_scale_multiplier * scale_factor)[:, None, :]
    )
    obj_kp = keypoints_world(obj_pos, obj_rot, kp_offsets)
    goal_kp = keypoints_world(goal_pos, goal_rot, kp_offsets)
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
        "object_scales": hub.object_scale_per_env * hub.object_scale_multiplier,
    }


def _cached_clean(geo: _Geometry, env) -> dict[str, torch.Tensor]:
    hub = geo.hub
    step = int(env.common_step_counter)
    if hub._obs_clean_cache_step != step or hub._obs_clean_cache is None:
        hub._obs_clean_cache = _assemble_clean(geo, env)
        hub._obs_clean_cache_step = step
        hub._obs_clean_cache_ids = None
    return hub._obs_clean_cache


def _cached_keypoint_max_dist(
    env,
    hub: SimToolRealCommand,
    object_entity: Entity,
    kp_offsets: torch.Tensor,
) -> torch.Tensor:
    del object_entity
    step = int(env.common_step_counter)
    if hub._geometry_cache_step != step:
        hub._geometry_cache_step = step
        hub._geometry_keypoint_offsets = None
        hub._geometry_keypoint_max_dist = None
        hub._geometry_fingertip_names = None
        hub._geometry_fingertip_distances = None
    offsets = kp_offsets[0]
    cached_offsets = hub._geometry_keypoint_offsets
    cached = hub._geometry_keypoint_max_dist
    if cached is None or cached_offsets is None or not torch.equal(cached_offsets, offsets):
        geo = _Geometry(env)
        obj_pos, obj_rot, _ = _object_pose(geo)
        obj_kp = keypoints_world(obj_pos, obj_rot, kp_offsets)
        goal_kp = keypoints_world(hub.goal[:, :3], hub.goal[:, 3:], kp_offsets)
        cached = torch.linalg.vector_norm(obj_kp - goal_kp, dim=-1).amax(dim=-1)
        hub._geometry_keypoint_offsets = offsets.detach().clone()
        hub._geometry_keypoint_max_dist = cached
    return cached


def _cached_fingertip_distances(
    env,
    hub: SimToolRealCommand,
    object_entity: Entity,
    robot: Entity,
    fingertip_ids,
) -> torch.Tensor:
    del object_entity, robot, fingertip_ids
    step = int(env.common_step_counter)
    if hub._geometry_cache_step != step:
        hub._geometry_cache_step = step
        hub._geometry_keypoint_offsets = None
        hub._geometry_keypoint_max_dist = None
        hub._geometry_fingertip_names = None
        hub._geometry_fingertip_distances = None
    cached_names = hub._geometry_fingertip_names
    cached = hub._geometry_fingertip_distances
    names = tuple(FINGERTIP_LINK_NAMES)
    if cached is None or cached_names != names:
        geo = _Geometry(env)
        origins = geo.origins()
        obj_pos, _, _ = _object_pose(geo)
        ft_pos = geo.robot_body_state().pos_w[:, 1:, :] - origins[:, None, :]
        cached = torch.linalg.vector_norm(ft_pos - obj_pos[:, None, :], dim=-1)
        hub._geometry_fingertip_names = names
        hub._geometry_fingertip_distances = cached
    return cached


def _stale_reset_fields(
    fields: dict[str, torch.Tensor],
    raw_frame: torch.Tensor,
    field_list: Sequence[str],
    names: Sequence[str],
    candidates: torch.Tensor,
) -> None:
    rows = torch.nonzero(candidates, as_tuple=False).flatten()
    if rows.numel() == 0:
        return
    for name in names:
        start, stop = _field_slice(field_list, name)
        fields[name][rows] = raw_frame[rows, start:stop].reshape(fields[name][rows].shape)


class SimToolRealPolicyObservation(ManagerTermBase):
    """Actor frame with optional object-state noise, delay, and frame delay."""

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
                "object_state_rotation_noise_degrees", dr.object_state_rotation_noise_degrees
            )
        )
        self._joint_velocity_obs_noise_std = float(
            params.get("joint_velocity_obs_noise_std", dr.joint_velocity_obs_noise_std)
        )
        self._use_obs_delay = bool(params.get("use_obs_delay", dr.use_obs_delay))
        self._obs_delay_max = int(params.get("obs_delay_max", dr.obs_delay_max))
        self._clamp_abs = float(params.get("clamp_abs_observations", obs.clamp_abs_observations))
        self._raw_frame: torch.Tensor | None = None
        self._raw_frame_step = -1
        shape = (self.num_envs, max(1, self._object_state_delay_max), _OBJECT_STATE_DIM)
        self._object_state_queue = torch.zeros(shape, dtype=torch.float32, device=self._env.device)
        shape = (self.num_envs, max(1, self._obs_delay_max), POLICY_OBS_DIM)
        self._obs_queue = torch.zeros(shape, dtype=torch.float32, device=self._env.device)
        self._object_state_step_key: int | None = None
        self._obs_step_key: int | None = None
        self._delayed_object_state = torch.zeros(
            (self.num_envs, _OBJECT_STATE_DIM), dtype=torch.float32, device=self._env.device
        )
        self._delayed_policy_obs = torch.zeros(
            (self.num_envs, POLICY_OBS_DIM), dtype=torch.float32, device=self._env.device
        )

    @property
    def entity_name(self) -> str:
        return "robot"

    @property
    def tensor_state_entity(self) -> str:
        return "robot"

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return self._geo.robot_body_names

    def __call__(self, env, **params) -> torch.Tensor:
        del params
        probe = _cold_observation_probe(env, POLICY_OBS_DIM)
        if probe is not None:
            return probe
        clean = _cached_clean(self._geo, env)
        hub = self._geo.hub
        rng = env.rng
        obj_state = torch.cat((clean["obj_pos"], clean["obj_rot"], clean["obj_vel"]), dim=-1)
        flush = _episode_start(env, hub)
        if self._use_object_state_delay_noise:
            step = int(env.common_step_counter)
            if step != self._object_state_step_key:
                self._object_state_queue, delayed = _sample_delay(
                    self._object_state_queue, obj_state, rng, flush
                )
                self._object_state_step_key = step
            else:
                self._object_state_queue, delayed = _refresh_delay_on_reentry(
                    self._object_state_queue,
                    self._delayed_object_state,
                    obj_state,
                    rng,
                    flush,
                )
            self._delayed_object_state = delayed
            xyz_noise = rng.standard_normal((self.num_envs, 3))
            noisy_obj_pos = delayed[:, :3] + xyz_noise * self._object_state_xyz_noise_std
            noisy_obj_rot = _perturb_quat(
                rng, delayed[:, 3:7], self._object_state_rotation_noise_degrees
            )
        else:
            noisy_obj_pos = clean["obj_pos"]
            noisy_obj_rot = clean["obj_rot"]

        noisy_obj_kp = keypoints_world(noisy_obj_pos, noisy_obj_rot, clean["kp_offsets"])
        palm_pos = clean["palm_pos"]
        joint_vel = clean["joint_vel"]
        if self._joint_velocity_obs_noise_std > 0:
            noise = rng.standard_normal(joint_vel.shape)
            joint_vel = joint_vel + noise * self._joint_velocity_obs_noise_std
        fields = {
            "joint_pos": clean["joint_pos"],
            "joint_vel": joint_vel,
            "prev_action_targets": clean["prev_action_targets"],
            "palm_pos": palm_pos,
            "palm_rot": _wxyz_to_xyzw(clean["palm_rot"]),
            "object_rot": _wxyz_to_xyzw(noisy_obj_rot),
            "fingertip_pos_rel_palm": clean["fingertip_pos_rel_palm"],
            "keypoints_rel_palm": noisy_obj_kp - palm_pos[:, None, :],
            "keypoints_rel_goal": noisy_obj_kp - clean["goal_kp"],
            "object_scales": clean["object_scales"],
        }
        step = int(env.common_step_counter)
        policy_tensor = _stack(fields, _POLICY_FIELDS)
        if step != self._raw_frame_step:
            self._raw_frame = policy_tensor.detach().clone()
            self._raw_frame_step = step
        elif self._raw_frame is not None:
            _stale_reset_fields(
                fields,
                self._raw_frame,
                _POLICY_FIELDS,
                ("palm_pos", "palm_rot", "fingertip_pos_rel_palm"),
                flush,
            )
            fields["keypoints_rel_palm"] = noisy_obj_kp - fields["palm_pos"][:, None, :]
            policy_tensor = _stack(fields, _POLICY_FIELDS)

        if self._use_obs_delay:
            if step != self._obs_step_key:
                self._obs_queue, policy_tensor = _sample_delay(
                    self._obs_queue, policy_tensor, rng, flush
                )
                self._obs_step_key = step
            else:
                self._obs_queue, policy_tensor = _refresh_delay_on_reentry(
                    self._obs_queue, self._delayed_policy_obs, policy_tensor, rng, flush
                )
            self._delayed_policy_obs = policy_tensor
        policy_tensor = torch.clamp(policy_tensor, -self._clamp_abs, self._clamp_abs)
        if policy_tensor.shape != (self.num_envs, POLICY_OBS_DIM):
            raise ValueError(f"invalid SimToolReal actor frame: {tuple(policy_tensor.shape)}")
        return policy_tensor

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is not None:
            self._object_state_queue[env_ids].zero_()
            self._obs_queue[env_ids].zero_()


class SimToolRealCriticObservation(ManagerTermBase):
    """Clean asymmetric critic frame."""

    def __init__(self, cfg: ManagerTermBaseCfg, env):
        super().__init__(env)
        self._geo = _Geometry(env)
        self._clamp_abs = float(
            cfg.params.get("clamp_abs_observations", ObsCfg().clamp_abs_observations)
        )
        self._raw_frame: torch.Tensor | None = None
        self._raw_frame_step = -1

    @property
    def entity_name(self) -> str:
        return "robot"

    @property
    def tensor_state_entity(self) -> str:
        return "robot"

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return self._geo.robot_body_names

    def __call__(self, env, **params) -> torch.Tensor:
        del params
        probe = _cold_observation_probe(env, CRITIC_OBS_DIM)
        if probe is not None:
            return probe
        clean = _cached_clean(self._geo, env)
        hub = self._geo.hub
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
            "closest_keypoint_max_dist": hub.closest_keypoint_max_dist[:, None],
            "closest_fingertip_dist": hub.closest_fingertip_dist,
            "lifted_object": hub.lifted.to(torch.float32)[:, None],
            "progress": torch.log(env.episode_length_buf.to(torch.float32) / 10.0 + 1.0)[:, None],
            "successes": torch.log(hub.goal_count.to(torch.float32) + 1.0)[:, None],
            "reward": (env.reward_buf * 0.01)[:, None],
        }
        state_tensor = _stack(fields, _CRITIC_FIELDS)
        step = int(env.common_step_counter)
        if step != self._raw_frame_step:
            self._raw_frame = state_tensor.detach().clone()
            self._raw_frame_step = step
        elif self._raw_frame is not None:
            _stale_reset_fields(
                fields,
                self._raw_frame,
                _CRITIC_FIELDS,
                ("palm_pos", "palm_rot", "palm_vel", "fingertip_pos_rel_palm"),
                _episode_start(env, hub),
            )
            fields["keypoints_rel_palm"] = clean["obj_kp"] - fields["palm_pos"][:, None, :]
            state_tensor = _stack(fields, _CRITIC_FIELDS)
        state_tensor = torch.clamp(state_tensor, -self._clamp_abs, self._clamp_abs)
        if state_tensor.shape != (self.num_envs, CRITIC_OBS_DIM):
            raise ValueError(f"invalid SimToolReal critic frame: {tuple(state_tensor.shape)}")
        return state_tensor


__all__ = [
    "CRITIC_OBS_DIM",
    "OBS_FIELD_SIZES",
    "POLICY_OBS_DIM",
    "SimToolRealCriticObservation",
    "SimToolRealPolicyObservation",
]
