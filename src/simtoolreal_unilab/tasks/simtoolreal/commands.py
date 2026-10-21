"""SimToolReal goal-pose command term — the task state hub.

Literal NumPy port of ``goal_sampling.py:21-62`` (absolute/delta goal pose
sampling, wxyz quaternions), the goal-tracker half of
``reset_utils.py:304-401``, and the tolerance curriculum of
``termination_utils.py:10-36`` (original snapshot 313d5aea) into the UniLab
``CommandTerm`` contract.  Per the lifecycle protocol fixed in DESIGN.md
section 6.5, this term is the hub for goal advancement, success counting, and
the success-tolerance curriculum:

- the termination layer records this step's ``pending_success`` snapshot;
- reward terms consume that snapshot (success-frame bonus);
- the command manager computes *after* the reward manager
  (``ManagerBasedRlEnv._update_state_in_read_phase``: termination -> reward ->
  events -> command -> observation), so ``_update_command`` advances the goal,
  clears per-goal trackers, and zeros the episode length for succeeded envs
  (``termination_utils.py:46-51``);
- true done envs still go through the generic reset transaction.

Cross-term access goes through ``env.<manager>.get_term(...)`` (wuji
``ReorientCommand`` precedent); no env-global state hub is introduced.

Registered deviations (DESIGN.md section 3.5 convention):

- ``env._successes += is_success`` moves from the dones phase
  (``termination_utils.py:46``) to this term's post-reward ``_update_command``.
  Rewards read the ``pending_success`` snapshot (not the counter) and
  observations compute after the command manager, so the relative order is
  preserved; ``goal_count`` stays "number of goals advanced this episode".
- ``env.episode_length_buf[ids] = 0`` (``termination_utils.py:51``) goes
  through the env's ``set_episode_length_buf`` dual-write setter when
  available, so the env-owned ``state.info["steps"]`` mirror is zeroed with
  it (``manager_based_rl_env.py:494`` recomputes the buffer from
  ``info["steps"]`` every step; a direct buffer write would not persist).  The truncation check
  has already run when this executes, so the task's time-out term (1.4b-3)
  must mask time-outs with ``pending_success`` to preserve the original
  "a success step never truncates" semantics.
- The ``fixed_goal_pose`` debug pin and the ``fixed_trajectory_file`` ablation
  (``reset_utils.py:311-350``) are tooling-only branches and are not ported;
  ``goal_sampling_type`` selects "delta" | "absolute" directly.
- The curriculum subclass hooks (``_curriculum_eligible_mask`` /
  ``_curriculum_success_threshold``, ``termination_utils.py:16-27``) are not
  ported; the curriculum ticks on the per-step path only (``env_ids is
  None``), not on the reset-path ``compute(dt=0, env_ids)`` recompute.
- ``eval_success_tolerance`` is pinned at construction as well as per step;
  the original pins only inside the per-step update
  (``reset_utils.py:114`` initialises from ``success_tolerance``).
- Random streams come from ``env.rng`` (NumPy Generator) instead of torch:
  same distributions (uniform positions, standard-normal quaternions and
  axes), different stream.
- GoalViz commit: the headless B-form scene deliberately has no ``goalviz``
  entity.  It is visualization-only and does not participate in training;
  materializing a full mirrored tool prototype pool would double the worker's
  mapped-scene cold path without affecting observations, rewards, resets, or
  goal advancement.  The hub ``goal`` buffer is therefore the authoritative
  current goal, and every goal reader — the delta walk here, the success
  criterion, and the observation keypoint assembly — reads it (reaudit fix
  P3a).  The historical optional entity hook remains solely for focused
  unit-test scenes that exercise the reset-path compatibility branch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from unilab.base.entity import Entity
from unilab.managers import CommandTerm, CommandTermCfg
from unilab.utils.rotation import np_quat_from_angle_axis, np_quat_mul

from .config import FINGERTIP_LINK_NAMES, PALM_BODY_NAME, declared_names, match_entity_names
from .env_cfg import fixed_variant_plan_assignment

# Goal pose layout: env-local position (3) + wxyz quaternion (4).
GOAL_DIM = 7


def _scale_workspace_bounds(
    mins: np.ndarray, maxs: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Scale workspace bounds about their center (goal_sampling.py:12-17)."""
    center = 0.5 * (mins + maxs)
    half = 0.5 * (maxs - mins) * scale
    return center - half, center + half


def sample_absolute_goal_pose(
    rng: np.random.Generator,
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
    scale: float,
    n_envs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample position and orientation inside the workspace.

    goal_sampling.py:21-33; the orientation is IsaacLab
    ``random_orientation``: a standard-normal 4-vector normalized to unit
    length (isaaclab utils/math.py:1335-1351), returned wxyz.
    """
    mins_t = np.asarray(mins, dtype=np.float32)
    maxs_t = np.asarray(maxs, dtype=np.float32)
    lo, hi = _scale_workspace_bounds(mins_t, maxs_t, scale)
    pos = lo + (hi - lo) * rng.uniform(0.0, 1.0, (n_envs, 3)).astype(np.float32)
    quat = rng.standard_normal((n_envs, 4)).astype(np.float32)
    quat /= np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)
    return pos, quat


def sample_delta_goal_pose(
    rng: np.random.Generator,
    prev_pos: np.ndarray,  # (N, 3)
    prev_quat_wxyz: np.ndarray,  # (N, 4)
    delta_distance: float,
    delta_rotation_degrees: float,
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Perturb the previous goal by a bounded random walk (goal_sampling.py:36-62)."""
    n = prev_pos.shape[0]
    mins_t = np.asarray(mins, dtype=np.float32)
    maxs_t = np.asarray(maxs, dtype=np.float32)
    lo, hi = _scale_workspace_bounds(mins_t, maxs_t, scale)

    pos_noise = (rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32) * 2.0 - 1.0) * np.float32(
        delta_distance
    )
    new_pos = np.clip(prev_pos + pos_noise, lo, hi)

    axis = rng.standard_normal((n, 3)).astype(np.float32)
    axis /= np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    angle = (rng.uniform(0.0, 1.0, n).astype(np.float32) * 2.0 - 1.0) * np.float32(
        delta_rotation_degrees * (math.pi / 180.0)
    )
    dq = np_quat_from_angle_axis(angle, axis)
    new_quat = np_quat_mul(dq, prev_quat_wxyz).astype(np.float32)
    return new_pos, new_quat


@dataclass(kw_only=True)
class SimToolRealCommandCfg(CommandTermCfg):
    """Goal sampling + success/curriculum knobs.

    ``goal_sampling_type``/``delta_*``/``target_volume_*`` mirror
    simtoolreal_env_cfg.py:388-394; the tolerance curriculum mirrors
    simtoolreal_env_cfg.py:433-444 (termination_utils.py:10-36).
    """

    goal_sampling_type: str = "delta"  # "delta" | "absolute"
    delta_goal_distance: float = 0.1
    delta_rotation_degrees: float = 90.0
    target_volume_mins: tuple[float, float, float] = (-0.35, -0.2, 0.6)
    target_volume_maxs: tuple[float, float, float] = (0.35, 0.2, 0.95)
    target_volume_region_scale: float = 1.0
    # Success criterion and tolerance curriculum.
    success_tolerance: float = 0.075  # curriculum start
    target_success_tolerance: float = 0.01  # curriculum floor
    eval_success_tolerance: float | None = None
    success_steps: int = 10
    force_consecutive_near_goal_steps: bool = False
    tolerance_curriculum_increment: float = 0.9
    tolerance_curriculum_interval: int = 3000
    tolerance_curriculum_success_threshold: float = 3.0
    # Object-scale obs noise range (simtoolreal_env_cfg.py:472-473).  The
    # multiplier is task DR state consumed by the observation terms; the hub
    # owns the buffer so actor and critic read the same per-env values.
    object_scale_noise_multiplier_range: tuple[float, float] = (1.0, 1.0)
    # DexToolBench evaluation advances through an authoring-time trajectory.
    # Entries use the hub layout: env-local xyz + unit wxyz quaternion.  The
    # value is ``None`` for the normal sampled-goal task.
    fixed_goal_states: tuple[tuple[float, ...], ...] | None = None
    # Static single-object tasks do not carry the training tool-pool catalog.
    # This normalized scale feeds only the actor's object_scales observation;
    # it is deliberately not a physical mesh scale (upstream objects.py).
    object_scale: tuple[float, float, float] | None = None

    def build(self, env):
        return SimToolRealCommand(self, env)


class SimToolRealCommand(CommandTerm):
    """Goal-pose hub: sampling, success counting, tolerance curriculum."""

    cfg: SimToolRealCommandCfg

    def __init__(self, cfg: SimToolRealCommandCfg, env):
        super().__init__(cfg, env)
        # Cold path: resolve the bodies the success criterion reads
        # (obs_utils.py:153-205 computes palm/fingertip/keypoint geometry).
        self.robot = cast(Entity, env.scene["robot"])
        self.palm_id = int(
            self.robot.find_bodies(
                match_entity_names(
                    declared_names(self.robot, "body"), [PALM_BODY_NAME], label="body"
                )
            )[0][0]
        )
        self.fingertip_ids = np.asarray(
            self.robot.find_bodies(
                match_entity_names(
                    declared_names(self.robot, "body"), FINGERTIP_LINK_NAMES, label="body"
                ),
                preserve_order=True,
            )[0],
            dtype=np.int32,
        )
        # GoalViz mirror entity (reset_utils.py:304-369).  The scene assembly
        # wires it in step 1.4s; until then the name lookup degrades to
        # hub-only pose bookkeeping (module docstring, deviation list).
        try:
            self._goalviz: Entity | None = cast(Entity, env.scene["goalviz"])
        except KeyError:
            self._goalviz = None

        # Goal pose in env-local frame, wxyz quaternion (goal_sampling.py).
        self.goal = np.zeros((self.num_envs, GOAL_DIM), dtype=np.float32)
        self.goal[:, 3] = 1.0
        # Per-episode success counter (termination_utils.py:46 `successes`).
        self.goal_count = np.zeros(self.num_envs, dtype=np.int64)
        # Success snapshot for this control step, written by the termination
        # layer and consumed here after rewards (DESIGN.md section 6.5).
        self.pending_success = np.zeros(self.num_envs, dtype=bool)
        # Lift latch (reward_utils.py:8-23; gates fingertip vs keypoint delta
        # rewards and the wrench DR lift gate).
        self.lifted = np.zeros(self.num_envs, dtype=bool)
        self.object_init_z = np.zeros(self.num_envs, dtype=np.float32)
        # Near-goal tracking (reward_utils.py:67-76, termination_utils.py).
        self.near_goal = np.zeros(self.num_envs, dtype=bool)
        self.near_goal_steps = np.zeros(self.num_envs, dtype=np.int64)
        # Per-goal closest-distance trackers (reset_utils.py:372-401).
        self.closest_keypoint_max_dist = np.full(self.num_envs, -1.0, dtype=np.float32)
        self.closest_fingertip_dist = np.full(
            (self.num_envs, len(FINGERTIP_LINK_NAMES)), -1.0, dtype=np.float32
        )
        # Step-local geometry shared by the termination and reward terms.  The
        # original DirectRLEnv fills these intermediates once in
        # ``compute_intermediate_values``; manager terms use the hub to retain
        # that boundary without coupling the generic managers.
        self._geometry_cache_step: int | None = None
        self._geometry_keypoint_offsets: np.ndarray | None = None
        self._geometry_keypoint_max_dist: np.ndarray | None = None
        self._geometry_fingertip_ids: np.ndarray | None = None
        self._geometry_fingertip_distances: np.ndarray | None = None
        # Clean observation assembly is pure and shared by actor/critic within
        # a step.  Reset-row calls use a separate env-id key.
        self._obs_clean_cache_step: int | None = None
        self._obs_clean_cache_ids: np.ndarray | None = None
        self._obs_clean_cache: dict[str, np.ndarray] | None = None
        # Tolerance curriculum state (termination_utils.py:10-36,
        # reset_utils.py:114-119).
        self.current_success_tolerance = float(
            cfg.eval_success_tolerance
            if cfg.eval_success_tolerance is not None
            else cfg.success_tolerance
        )
        self.prev_episode_successes = np.zeros(self.num_envs, dtype=np.int64)
        self._last_curriculum_step = 0
        self._fixed_goals: np.ndarray | None = None
        if cfg.fixed_goal_states is not None:
            goals = np.asarray(cfg.fixed_goal_states, dtype=np.float32)
            if goals.ndim != 2 or goals.shape[1] != GOAL_DIM or goals.shape[0] == 0:
                raise ValueError(
                    "fixed_goal_states must be a non-empty (K, 7) position/quaternion table"
                )
            if not np.isfinite(goals).all():
                raise ValueError("fixed_goal_states must contain finite values")
            if not np.allclose(np.linalg.norm(goals[:, 3:7], axis=-1), 1.0, atol=1e-5):
                raise ValueError("fixed_goal_states quaternions must be unit wxyz values")
            self._fixed_goals = goals
        # Reset-path marker for the GoalViz commit gate (module docstring):
        # True only while ``reset()`` runs inside the env's reset transaction.
        self._in_reset = False

        # Object-scale state for the observation keypoint/scale fields
        # (obs_utils.py:275,301; pure state fields, no hub logic change).
        # ``object_scale_per_env`` mirrors ``env._object_scale_per_env``
        # (scene_utils.py:1638-1655): the normalized scale of the tool variant
        # assigned to each env, expanded from the owner's immutable catalog
        # table by the scene plan's assignment — the same round-robin the
        # worker applies to the pool (scene_utils.py:1650-1655).  Scale stays
        # a task-side table; only the assignment moved to the scene plan.
        # Lightweight fixtures without the pool keep the ones default, exact
        # only for unit-scale tools.
        scales = getattr(getattr(env, "cfg", None), "tool_pool_scales", ())
        assignments = fixed_variant_plan_assignment(getattr(env, "cfg", None), self.num_envs)
        if scales and assignments is not None:
            scale_table = np.asarray(scales, dtype=np.float32)
            assign = np.asarray(assignments, dtype=np.int64)
            if scale_table.ndim != 2 or scale_table.shape[1] != 3:
                raise ValueError(
                    f"tool_pool_scales must be an (N, 3) table; got {scale_table.shape}"
                )
            if not np.isfinite(scale_table).all() or np.any(scale_table <= 0.0):
                raise ValueError("tool_pool_scales entries must be finite positive values")
            if np.any(assign < 0) or np.any(assign >= scale_table.shape[0]):
                raise ValueError(
                    "tool_pool_scales does not cover the fixed variant plan assignment; "
                    f"the plan assigns variant ids up to {int(assign.max())} but the table "
                    f"has {scale_table.shape[0]} entries"
                )
            self.object_scale_per_env = scale_table[assign]
        else:
            self.object_scale_per_env = np.ones((self.num_envs, 3), dtype=np.float32)
        if cfg.object_scale is not None:
            static_scale = np.asarray(cfg.object_scale, dtype=np.float32)
            if static_scale.shape != (3,) or not np.isfinite(static_scale).all():
                raise ValueError("object_scale must be a finite 3-D vector")
            if np.any(static_scale <= 0.0):
                raise ValueError("object_scale entries must be positive")
            self.object_scale_per_env[:] = static_scale[None, :]
        # ``object_scale_multiplier`` mirrors allocate_state_buffers:86-89
        # (seeded once at init); the stage-2 reset event re-samples it per
        # episode (reset_utils.py:415-418).
        lo, hi = cfg.object_scale_noise_multiplier_range
        self.object_scale_multiplier = self._env.rng.uniform(lo, hi, (self.num_envs, 3)).astype(
            np.float32
        )

        self.metrics = {
            "goal_count": np.zeros(self.num_envs, dtype=np.float32),
            "success_tolerance": np.zeros(self.num_envs, dtype=np.float32),
        }

    @property
    def command(self) -> np.ndarray:
        return self.goal

    def _resample_command(self, env_ids: np.ndarray) -> None:
        # reset_utils.py:388: a full episode reset always samples an absolute
        # goal pose, regardless of ``goal_sampling_type``.
        self._sample_goal(env_ids, mode="absolute")

    def _update_command(self, env_ids: np.ndarray | None) -> None:
        ids = (
            np.arange(self.num_envs, dtype=np.intp)
            if env_ids is None
            else np.asarray(env_ids, dtype=np.intp)
        )
        # DESIGN.md section 6.5: consume the termination layer's snapshot
        # after rewards (termination_utils.py:46-51 semantics).
        succeeded = ids[self.pending_success[ids]]
        if len(succeeded) > 0:
            self._advance_goal(succeeded)
        # termination_utils.py:10-36: the curriculum ticks once per policy
        # step, not on the reset-path recompute (env_ids given).
        if env_ids is None:
            self._tick_tolerance_curriculum()

    def _advance_goal(self, env_ids: np.ndarray) -> None:
        """Success-driven goal advance (termination_utils.py:46-51)."""
        # termination_utils.py:46: env._successes += is_success.
        self.goal_count[env_ids] += 1
        # reset_utils.py:376-380: clear per-goal trackers.
        self.closest_keypoint_max_dist[env_ids] = -1.0
        self.closest_fingertip_dist[env_ids] = -1.0
        self.near_goal_steps[env_ids] = 0
        # reset_utils.py:394: intra-episode advance uses cfg.goal_sampling_type.
        self._sample_goal(env_ids, mode=self.cfg.goal_sampling_type)
        # termination_utils.py:51: zero the episode length so truncation does
        # not fire for chained successes.  The env owns the buffer pair
        # (episode_length_buf is recomputed from state.info["steps"] each
        # step), so the write goes through the env's dual-write setter when
        # available; the direct-write fallback keeps SimpleNamespace test
        # doubles working (transient, buffer-only).
        env = self._env
        set_lengths = getattr(env, "set_episode_length_buf", None)
        if callable(set_lengths):
            lengths = np.array(env.episode_length_buf, copy=True)
            lengths[env_ids] = 0
            set_lengths(lengths)
        else:
            env.episode_length_buf[env_ids] = 0

    def _sample_goal(self, env_ids: np.ndarray, mode: str) -> None:
        """Resample the goal pose and commit it to GoalViz (reset_utils.py:304-369)."""
        cfg = self.cfg
        rng = self._env.rng
        if self._fixed_goals is not None:
            # DexToolBench's IsaacGym path indexes trajectory_states by
            # successes and terminates once all authored goals are reached.
            # goal_count is zeroed on reset and incremented immediately before
            # this call on an advance, so it is exactly the trajectory cursor.
            indices = np.minimum(self.goal_count[env_ids], len(self._fixed_goals) - 1)
            self.goal[env_ids, :3] = self._fixed_goals[indices, :3]
            self.goal[env_ids, 3:] = self._fixed_goals[indices, 3:]
            return
        if mode == "delta":
            # reset_utils.py:351-353 reads the previous pose back from the
            # GoalViz sim state, which the original keeps fresh by writing it
            # on every advance.  UniLab defers that sim write to the reset
            # path (see _commit_goalviz), so the hub buffer is the
            # authoritative current goal; reading it keeps the intra-episode
            # delta walk chained exactly like the original (reaudit fix P3a).
            prev_pos = self.goal[env_ids, :3]
            prev_quat = self.goal[env_ids, 3:]
            new_pos, new_quat = sample_delta_goal_pose(
                rng,
                prev_pos=prev_pos,
                prev_quat_wxyz=prev_quat,
                delta_distance=cfg.delta_goal_distance,
                delta_rotation_degrees=cfg.delta_rotation_degrees,
                mins=cfg.target_volume_mins,
                maxs=cfg.target_volume_maxs,
                scale=cfg.target_volume_region_scale,
            )
        elif mode == "absolute":
            new_pos, new_quat = sample_absolute_goal_pose(
                rng,
                mins=cfg.target_volume_mins,
                maxs=cfg.target_volume_maxs,
                scale=cfg.target_volume_region_scale,
                n_envs=len(env_ids),
            )
        else:
            raise ValueError(f"unknown goal sampling mode: {mode}")

        self.goal[env_ids, :3] = new_pos
        self.goal[env_ids, 3:] = new_quat
        self._commit_goalviz(env_ids)

    def _commit_goalviz(self, env_ids: np.ndarray) -> None:
        """Write the env-local goal pose when a compatibility-only GoalViz exists."""
        if self._goalviz is None or not self._in_reset:
            # Headless B-form bookkeeping is intentionally hub-only; the
            # optional entity exists only in focused unit-test scenes.
            # Mid-episode compatibility writes are also skipped because the
            # reset transaction's full-row default staging would reset the
            # robot/object state of succeeded envs (module docstring).
            return
        pose = np.array(self.goal[env_ids], copy=True)
        pose[:, :3] += self._env.scene.env_origins[env_ids]
        self._goalviz.write_root_link_pose_to_sim(pose, env_ids=env_ids)

    def _tick_tolerance_curriculum(self) -> None:
        """Shrink success tolerance on sustained performance (termination_utils.py:10-36)."""
        cfg = self.cfg
        frame = self._env.common_step_counter
        if frame - self._last_curriculum_step >= cfg.tolerance_curriculum_interval:
            successes = self.prev_episode_successes.astype(np.float32)
            if (
                successes.size > 0
                and float(successes.mean()) >= cfg.tolerance_curriculum_success_threshold
            ):
                new_tol = self.current_success_tolerance * cfg.tolerance_curriculum_increment
                new_tol = max(min(new_tol, cfg.success_tolerance), cfg.target_success_tolerance)
                self.current_success_tolerance = float(new_tol)
                self._last_curriculum_step = frame
        # Eval pins the success criterion (termination_utils.py:35-36).
        if cfg.eval_success_tolerance is not None:
            self.current_success_tolerance = float(cfg.eval_success_tolerance)

    def _update_metrics(self, env_ids: np.ndarray | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.metrics["goal_count"][ids] = self.goal_count[ids]
        self.metrics["success_tolerance"][ids] = self.current_success_tolerance

    def log_extras(self, *, reward_manager, termination_manager) -> dict[str, object]:
        """Expose the original SimToolReal TensorBoard logging contract.

        The values are read-only views assembled after the manager update.  The
        observer performs the episode accumulation and only consumes the rows
        reported as done, matching ``EnvStatsAlgoObserver`` in the original
        repository.  None of these values feed observations, rewards, or done
        decisions.
        """
        reward_name_map = {
            "fingertip_distance_delta": "fingertip_delta_rew",
            "lift_progress": "lifting_rew",
            "lifting_bonus": "lift_bonus_rew",
            "keypoint_distance_delta": "keypoint_rew",
            "arm_joint_velocity_penalty": "kuka_actions_penalty",
            "hand_joint_velocity_penalty": "hand_actions_penalty",
            "reach_bonus": "bonus_rew",
        }
        step_terms = reward_manager.step_reward_vectors()
        cumulative = {
            reward_name_map[name]: values
            for name, values in step_terms.items()
            if name in reward_name_map
        }
        if cumulative:
            total = np.zeros(self.num_envs, dtype=np.float32)
            for values in cumulative.values():
                total += values
            cumulative["total_reward"] = total

        try:
            all_goals_hit = termination_manager.get_term("max_consecutive_successes")
        except (KeyError, ValueError):
            all_goals_hit = np.zeros(self.num_envs, dtype=bool)
        final = {
            # Reset runs immediately after this log is assembled and clears
            # goal_count; retain the terminal-frame value for the observer.
            "successes": self.goal_count.astype(np.float32, copy=True),
            # This termination term is the already-computed equivalent of the
            # original ``successes >= max_consecutive_successes`` expression.
            "all_goals_hit": all_goals_hit.astype(np.float32, copy=True),
        }
        reason_names = {
            "fall": "done_fall",
            "hand_far": "done_hand_far",
            "max_consecutive_successes": "done_max_successes",
            "time_out": "done_timeout",
        }
        for name, legacy_name in reason_names.items():
            try:
                values = termination_manager.get_term(name)
            except (KeyError, ValueError):
                continue
            final[legacy_name] = values.astype(np.float32, copy=True)

        return {
            "episode_cumulative": cumulative,
            "episode_final": final,
            "successes": self.prev_episode_successes.astype(np.float32, copy=True),
            "current_success_tolerance": float(self.current_success_tolerance),
        }

    def reset(self, env_ids) -> dict[str, float]:
        self._in_reset = True
        try:
            extras = super().reset(env_ids)
        finally:
            self._in_reset = False
        # A manual/auto reset can happen without advancing the global step
        # counter; do not let a pre-reset geometry or clean-observation frame be
        # observed by a subsequent reset-path call at the same counter value.
        self._geometry_cache_step = None
        self._geometry_keypoint_offsets = None
        self._geometry_keypoint_max_dist = None
        self._geometry_fingertip_ids = None
        self._geometry_fingertip_distances = None
        self._obs_clean_cache_step = None
        self._obs_clean_cache_ids = None
        self._obs_clean_cache = None
        # reset_utils.py:380-401: episode-scope state cleared on full reset.
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
    """Return the task state hub (wuji ``task(env)`` precedent)."""
    return cast(SimToolRealCommand, env.command_manager.get_term("goal"))


__all__ = [
    "GOAL_DIM",
    "SimToolRealCommand",
    "SimToolRealCommandCfg",
    "goal_command",
    "sample_absolute_goal_pose",
    "sample_delta_goal_pose",
]
