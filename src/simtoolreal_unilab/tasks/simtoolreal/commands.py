"""SimToolReal goal-pose command term and task state hub."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from unilab.base.entity import Entity
from unilab.managers import CommandTerm, CommandTermCfg
from unilab.managers.rng import ManagerRng

from .config import FINGERTIP_LINK_NAMES, PALM_BODY_NAME, declared_names, match_entity_names
from .env_cfg import fixed_variant_plan_assignment
from .tensor_utils import quat_from_angle_axis, quat_mul

GOAL_DIM = 7


def _scale_workspace_bounds(
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
    scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    lo = torch.tensor(mins, dtype=torch.float32, device=device)
    hi = torch.tensor(maxs, dtype=torch.float32, device=device)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * scale
    return center - half, center + half


def sample_absolute_goal_pose(
    rng: ManagerRng,
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
    scale: float,
    n_envs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly sample a position and unit wxyz orientation."""
    lo, hi = _scale_workspace_bounds(mins, maxs, scale, rng.device)
    unit = rng.uniform(0.0, 1.0, (n_envs, 3))
    pos = lo + (hi - lo) * unit
    quat = rng.standard_normal((n_envs, 4))
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)
    return pos, quat


def sample_delta_goal_pose(
    rng: ManagerRng,
    prev_pos: torch.Tensor,
    prev_quat_wxyz: torch.Tensor,
    delta_distance: float,
    delta_rotation_degrees: float,
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perturb the previous goal by a bounded random walk."""
    n = prev_pos.shape[0]
    lo, hi = _scale_workspace_bounds(mins, maxs, scale, rng.device)
    pos_noise = (rng.uniform(0.0, 1.0, (n, 3)) * 2.0 - 1.0) * delta_distance
    new_pos = torch.clamp(prev_pos + pos_noise, min=lo, max=hi)
    axis = rng.standard_normal((n, 3))
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-12)
    angle = (rng.uniform(0.0, 1.0, n) * 2.0 - 1.0) * (delta_rotation_degrees * (math.pi / 180.0))
    dq = quat_from_angle_axis(angle, axis)
    return quat_mul(dq, prev_quat_wxyz), new_pos


@dataclass(kw_only=True)
class SimToolRealCommandCfg(CommandTermCfg):
    """Goal sampling, success, and tolerance-curriculum knobs."""

    goal_sampling_type: str = "delta"
    delta_goal_distance: float = 0.1
    delta_rotation_degrees: float = 90.0
    target_volume_mins: tuple[float, float, float] = (-0.35, -0.2, 0.6)
    target_volume_maxs: tuple[float, float, float] = (0.35, 0.2, 0.95)
    target_volume_region_scale: float = 1.0
    success_tolerance: float = 0.075
    target_success_tolerance: float = 0.01
    eval_success_tolerance: float | None = None
    success_steps: int = 10
    force_consecutive_near_goal_steps: bool = False
    tolerance_curriculum_increment: float = 0.9
    tolerance_curriculum_interval: int = 3000
    tolerance_curriculum_success_threshold: float = 3.0
    object_scale_noise_multiplier_range: tuple[float, float] = (1.0, 1.0)
    fixed_goal_states: tuple[tuple[float, ...], ...] | None = None
    object_scale: tuple[float, float, float] | None = None

    def build(self, env):
        return SimToolRealCommand(self, env)


class SimToolRealCommand(CommandTerm):
    """Goal sampling, success advancement, tolerance curriculum, and shared DR state."""

    cfg: SimToolRealCommandCfg

    def __init__(self, cfg: SimToolRealCommandCfg, env):
        super().__init__(cfg, env)
        self.robot = cast(Entity, env.scene["robot"])
        self.object = cast(Entity, env.scene["object"])
        self.palm_body_name = match_entity_names(
            declared_names(self.robot, "body"), [PALM_BODY_NAME], label="body"
        )[0]
        self.fingertip_body_names = tuple(
            match_entity_names(
                declared_names(self.robot, "body"), FINGERTIP_LINK_NAMES, label="body"
            )
        )
        self.object_body_name = match_entity_names(
            declared_names(self.object, "body"), ["object_root"], label="body"
        )[0]
        try:
            self._goalviz: Entity | None = cast(Entity, env.scene["goalviz"])
        except KeyError:
            self._goalviz = None

        device = self._device
        self.goal = torch.zeros((self.num_envs, GOAL_DIM), dtype=torch.float32, device=device)
        self.goal[:, 3] = 1.0
        self.goal_count = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.pending_success = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.object_init_z = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.near_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.near_goal_steps = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.closest_keypoint_max_dist = torch.full(
            (self.num_envs,), -1.0, dtype=torch.float32, device=device
        )
        self.closest_fingertip_dist = torch.full(
            (self.num_envs, len(FINGERTIP_LINK_NAMES)), -1.0, dtype=torch.float32, device=device
        )
        self._geometry_cache_step: int | None = None
        self._geometry_keypoint_offsets: torch.Tensor | None = None
        self._geometry_keypoint_max_dist: torch.Tensor | None = None
        self._geometry_fingertip_names: tuple[str, ...] | None = None
        self._geometry_fingertip_distances: torch.Tensor | None = None
        self._obs_clean_cache_step: int | None = None
        self._obs_clean_cache_ids: torch.Tensor | None = None
        self._obs_clean_cache: dict[str, torch.Tensor] | None = None
        self.current_success_tolerance = float(
            cfg.eval_success_tolerance
            if cfg.eval_success_tolerance is not None
            else cfg.success_tolerance
        )
        self.prev_episode_successes = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self._last_curriculum_step = 0
        self._fixed_goals: torch.Tensor | None = None
        if cfg.fixed_goal_states is not None:
            goals = torch.tensor(cfg.fixed_goal_states, dtype=torch.float32, device=device)
            if goals.ndim != 2 or goals.shape[1] != GOAL_DIM or goals.shape[0] == 0:
                raise ValueError("fixed_goal_states must be a non-empty (K, 7) table")
            if not bool(torch.isfinite(goals).all()):
                raise ValueError("fixed_goal_states must contain finite values")
            norms = torch.linalg.vector_norm(goals[:, 3:7], dim=-1)
            if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-5)):
                raise ValueError("fixed_goal_states quaternions must be unit wxyz values")
            self._fixed_goals = goals
        self._in_reset = False

        scales = getattr(getattr(env, "cfg", None), "tool_pool_scales", ())
        assignments = fixed_variant_plan_assignment(getattr(env, "cfg", None), self.num_envs)
        if scales and assignments is not None:
            scale_table = torch.tensor(scales, dtype=torch.float32, device=device)
            assign = torch.as_tensor(assignments, dtype=torch.int64, device=device)
            if scale_table.ndim != 2 or scale_table.shape[1] != 3:
                raise ValueError(f"tool_pool_scales must be (N, 3); got {tuple(scale_table.shape)}")
            if not bool(torch.isfinite(scale_table).all()) or bool((scale_table <= 0).any()):
                raise ValueError("tool_pool_scales entries must be finite positive values")
            if bool((assign < 0).any()) or bool((assign >= scale_table.shape[0]).any()):
                raise ValueError("tool_pool_scales does not cover the fixed variant assignment")
            self.object_scale_per_env = scale_table.index_select(0, assign)
        else:
            self.object_scale_per_env = torch.ones(
                (self.num_envs, 3), dtype=torch.float32, device=device
            )
        if cfg.object_scale is not None:
            static_scale = torch.tensor(cfg.object_scale, dtype=torch.float32, device=device)
            if static_scale.shape != (3,) or not bool(torch.isfinite(static_scale).all()):
                raise ValueError("object_scale must be a finite 3-D vector")
            if bool((static_scale <= 0).any()):
                raise ValueError("object_scale entries must be positive")
            self.object_scale_per_env[:] = static_scale[None, :]
        lo, hi = cfg.object_scale_noise_multiplier_range
        self.object_scale_multiplier = self._rng.uniform(lo, hi, (self.num_envs, 3))

        self.metrics = {
            "goal_count": torch.zeros(self.num_envs, dtype=torch.float32, device=device),
            "success_tolerance": torch.zeros(self.num_envs, dtype=torch.float32, device=device),
        }

    @property
    def command(self) -> torch.Tensor:
        return self.goal

    @property
    def entity_name(self) -> str:
        return "robot"

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return (self.palm_body_name, *self.fingertip_body_names)

    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        env_ids = torch.nonzero(reset_mask, as_tuple=False).flatten()
        if env_ids.numel():
            self._sample_goal(env_ids, mode="absolute")

    def _update_metrics(self) -> None:
        self.metrics["goal_count"].copy_(self.goal_count.to(torch.float32))
        self.metrics["success_tolerance"].fill_(self.current_success_tolerance)

    def _update_command(self) -> None:
        succeeded = torch.nonzero(self.pending_success, as_tuple=False).flatten()
        if succeeded.numel():
            self._advance_goal(succeeded)
        self._tick_tolerance_curriculum()

    def refresh_reset(self, reset_mask: torch.Tensor) -> None:
        """Consume the committed reset boundary; sampled goal state needs no rewrite."""
        del reset_mask

    def post_compute_reset(self) -> None:
        """Match the CommandTerm reset hook contract; no post-commit state exists."""

    def _advance_goal(self, env_ids: torch.Tensor) -> None:
        self.goal_count[env_ids] += 1
        self.closest_keypoint_max_dist[env_ids] = -1.0
        self.closest_fingertip_dist[env_ids] = -1.0
        self.near_goal_steps[env_ids] = 0
        self._sample_goal(env_ids, mode=self.cfg.goal_sampling_type)
        set_lengths = getattr(self._env, "set_episode_length_buf", None)
        if callable(set_lengths):
            lengths = self._env.episode_length_buf.clone()
            lengths[env_ids] = 0
            set_lengths(lengths)
        else:
            self._env.episode_length_buf[env_ids] = 0

    def _sample_goal(self, env_ids: torch.Tensor, mode: str) -> None:
        cfg = self.cfg
        rng = self._env.rng
        if self._fixed_goals is not None:
            indices = torch.clamp(self.goal_count[env_ids], max=len(self._fixed_goals) - 1)
            self.goal[env_ids, :3] = self._fixed_goals[indices, :3]
            self.goal[env_ids, 3:] = self._fixed_goals[indices, 3:]
            return
        if mode == "delta":
            new_pos, new_quat = sample_delta_goal_pose(
                rng,
                self.goal[env_ids, :3],
                self.goal[env_ids, 3:],
                cfg.delta_goal_distance,
                cfg.delta_rotation_degrees,
                cfg.target_volume_mins,
                cfg.target_volume_maxs,
                cfg.target_volume_region_scale,
            )
        elif mode == "absolute":
            new_pos, new_quat = sample_absolute_goal_pose(
                rng,
                cfg.target_volume_mins,
                cfg.target_volume_maxs,
                cfg.target_volume_region_scale,
                env_ids.numel(),
            )
        else:
            raise ValueError(f"unknown goal sampling mode: {mode}")
        self.goal[env_ids, :3] = new_pos
        self.goal[env_ids, 3:] = new_quat
        self._commit_goalviz(env_ids)

    def _commit_goalviz(self, env_ids: torch.Tensor) -> None:
        if self._goalviz is None or not self._in_reset:
            return
        pose = self.goal[env_ids].clone()
        pose[:, :3] += self._env.scene.env_origins(self._env.device)[env_ids]
        self._goalviz.write_root_link_pose_to_sim(pose, env_ids=env_ids)

    def _tick_tolerance_curriculum(self) -> None:
        cfg = self.cfg
        frame = int(self._env.common_step_counter)
        if frame - self._last_curriculum_step >= cfg.tolerance_curriculum_interval:
            successes = self.prev_episode_successes.to(torch.float32)
            mean_successes = float(successes.mean().detach().cpu())
            if mean_successes >= cfg.tolerance_curriculum_success_threshold:
                new_tol = self.current_success_tolerance * cfg.tolerance_curriculum_increment
                self.current_success_tolerance = max(
                    min(new_tol, cfg.success_tolerance), cfg.target_success_tolerance
                )
                self._last_curriculum_step = frame
        if cfg.eval_success_tolerance is not None:
            self.current_success_tolerance = float(cfg.eval_success_tolerance)

    def log_extras(self, *, reward_manager, termination_manager) -> dict[str, object]:
        reward_name_map = {
            "fingertip_distance_delta": "fingertip_delta_rew",
            "lift_progress": "lifting_rew",
            "lifting_bonus": "lift_bonus_rew",
            "keypoint_distance_delta": "keypoint_rew",
            "arm_joint_velocity_penalty": "kuka_actions_penalty",
            "hand_joint_velocity_penalty": "hand_actions_penalty",
            "reach_bonus": "bonus_rew",
        }
        names = reward_manager.step_reward_names
        rewards = reward_manager.step_rewards
        cumulative = {
            reward_name_map[name]: rewards[:, index].detach()
            for index, name in enumerate(names)
            if name in reward_name_map
        }
        if cumulative:
            cumulative["total_reward"] = torch.stack(tuple(cumulative.values()), dim=0).sum(dim=0)
        try:
            all_goals_hit = termination_manager.get_term("max_consecutive_successes")
        except (KeyError, ValueError):
            all_goals_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        final = {
            "successes": self.goal_count.to(torch.float32).clone(),
            "all_goals_hit": all_goals_hit.to(torch.float32),
        }
        for name, legacy_name in {
            "fall": "done_fall",
            "hand_far": "done_hand_far",
            "max_consecutive_successes": "done_max_successes",
            "time_out": "done_timeout",
        }.items():
            try:
                final[legacy_name] = termination_manager.get_term(name).to(torch.float32)
            except (KeyError, ValueError):
                continue
        return {
            "episode_cumulative": cumulative,
            "episode_final": final,
            "successes": self.prev_episode_successes.to(torch.float32).clone(),
            "current_success_tolerance": float(self.current_success_tolerance),
        }

    def reset(
        self, env_ids: torch.Tensor | slice | None = None, *, publish_metrics: bool = True
    ) -> dict[str, float]:
        self._in_reset = True
        try:
            extras = super().reset(env_ids, publish_metrics=publish_metrics)
        finally:
            self._in_reset = False
        self._geometry_cache_step = None
        self._geometry_keypoint_offsets = None
        self._geometry_keypoint_max_dist = None
        self._geometry_fingertip_names = None
        self._geometry_fingertip_distances = None
        self._obs_clean_cache_step = None
        self._obs_clean_cache_ids = None
        self._obs_clean_cache = None
        self.prev_episode_successes[env_ids] = self.goal_count[env_ids]
        self.goal_count[env_ids] = 0
        self.pending_success[env_ids] = False
        self.lifted[env_ids] = False
        self.near_goal[env_ids] = False
        self.near_goal_steps[env_ids] = 0
        self.closest_keypoint_max_dist[env_ids] = -1.0
        self.closest_fingertip_dist[env_ids] = -1.0
        return extras


def goal_command(env) -> SimToolRealCommand:
    return cast(SimToolRealCommand, env.command_manager.get_term("goal"))


__all__ = [
    "GOAL_DIM",
    "SimToolRealCommand",
    "SimToolRealCommandCfg",
    "goal_command",
    "sample_absolute_goal_pose",
    "sample_delta_goal_pose",
]
