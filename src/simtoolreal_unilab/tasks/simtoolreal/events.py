"""SimToolReal Torch reset and interval randomization events."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import torch
from unilab.base.entity import Entity
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.rng import ManagerRng

from .config import (
    ARM_DEFAULT_JOINT_POS,
    ARM_JOINT_NAMES,
    HAND_JOINT_NAMES,
    ResetCfg,
    declared_names,
    match_entity_names,
)
from .env_cfg import fixed_variant_plan_assignment
from .tensor_utils import normalize_quat


def _ids(env: Any, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, dtype=torch.int64, device=env.device)[env_ids]
    result = env_ids.reshape(-1)
    if bool((result < 0).any()) or bool((result >= env.num_envs).any()):
        raise ValueError("environment ids are out of range")
    if result.unique().numel() != result.numel():
        raise ValueError("environment ids contain duplicates")
    return result


def _entity(scene: Mapping[str, Any], name: str) -> Entity | None:
    try:
        return cast(Entity, scene[name])
    except (KeyError, TypeError):
        return None


def _declared_root_mode(env: Any, name: str) -> str | None:
    scene = getattr(getattr(env, "cfg", None), "scene", None)
    for spec in getattr(scene, "entity_assets", ()) or ():
        spec_name = (
            getattr(spec, "name", None) if not isinstance(spec, Mapping) else spec.get("name")
        )
        if spec_name == name:
            return spec.get("root_mode") if isinstance(spec, Mapping) else spec.root_mode
    return None


def _initial_entity_pose(env: Any, name: str) -> torch.Tensor:
    """Read one composed entity's fixed initial world pose."""
    scene = getattr(getattr(env, "cfg", None), "scene", None)
    for spec in getattr(scene, "entity_assets", ()) or ():
        spec_name = (
            getattr(spec, "name", None) if not isinstance(spec, Mapping) else spec.get("name")
        )
        if spec_name != name:
            continue
        state = getattr(spec, "initial_state", None)
        if state is None and isinstance(spec, Mapping):
            state = spec.get("initial_state")
        if state is None:
            break
        position = getattr(state, "position", None)
        quaternion = getattr(state, "quaternion", None)
        if position is None and isinstance(state, Mapping):
            position = state.get("position")
            quaternion = state.get("quaternion")
        if position is None or quaternion is None:
            break
        return torch.tensor(
            tuple(position) + tuple(quaternion), dtype=torch.float32, device=env.device
        )
    raise ValueError(f"SimToolRealReset cannot resolve initial pose for entity {name!r}")


def _unit_quaternions(rng: ManagerRng, count: int) -> torch.Tensor:
    return normalize_quat(rng.standard_normal((count, 4)))


def _log_uniform(
    rng: ManagerRng,
    bounds: tuple[float, float],
    count: int,
    *,
    label: str,
) -> torch.Tensor:
    lo, hi = float(bounds[0]), float(bounds[1])
    if not math.isfinite(lo) or not math.isfinite(hi) or lo <= 0.0 or lo > hi:
        raise ValueError(f"{label} must satisfy 0 < min <= max, got {bounds!r}")
    return torch.exp(rng.uniform(math.log(lo), math.log(hi), count))


class SimToolRealReset(ManagerTermBase):
    """Reset robot joints and optional table/object roots."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: Any):
        super().__init__(env)
        self._robot = cast(Entity, env.scene["robot"])
        names = list(ARM_JOINT_NAMES) + list(HAND_JOINT_NAMES)
        names = match_entity_names(declared_names(self._robot, "joint"), names, label="joint")
        self._joint_ids, _ = self._robot.find_joints(names, preserve_order=True)
        default = [ARM_DEFAULT_JOINT_POS[name] for name in ARM_JOINT_NAMES]
        default += [0.0] * len(HAND_JOINT_NAMES)
        params = cfg.params
        if bool(params.get("start_arm_higher", False)):
            default[1] -= math.radians(10.0)
            default[3] += math.radians(10.0)
        self._default_pos = torch.tensor(default, dtype=torch.float32, device=env.device)

        defaults = ResetCfg()
        self._pos_noise_arm = float(
            params.get("dof_pos_noise_arm", defaults.reset_dof_pos_random_interval_arm)
        )
        self._pos_noise_hand = float(
            params.get("dof_pos_noise_hand", defaults.reset_dof_pos_random_interval_fingers)
        )
        self._vel_noise = float(params.get("dof_vel_noise", defaults.reset_dof_vel_random_interval))
        self._reset_position_noise = torch.tensor(
            (
                float(params.get("reset_position_noise_x", defaults.reset_position_noise_x)),
                float(params.get("reset_position_noise_y", defaults.reset_position_noise_y)),
                float(params.get("reset_position_noise_z", defaults.reset_position_noise_z)),
            ),
            dtype=torch.float32,
            device=env.device,
        )
        self._table_reset_z = float(params.get("table_reset_z", defaults.table_reset_z))
        self._table_reset_z_range = float(
            params.get("table_reset_z_range", defaults.table_reset_z_range)
        )
        self._table_object_z_offset = float(
            params.get("table_object_z_offset", defaults.table_object_z_offset)
        )
        fixed = params.get("fixed_start_pose", defaults.fixed_start_pose)
        self._fixed_start_pose = (
            None if fixed is None else torch.tensor(fixed, dtype=torch.float32, device=env.device)
        )
        if self._fixed_start_pose is not None and self._fixed_start_pose.shape != (7,):
            raise ValueError("fixed_start_pose must contain (x, y, z, qw, qx, qy, qz)")
        multiplier_range = params.get("object_scale_noise_multiplier_range", (1.0, 1.0))
        if len(multiplier_range) != 2:
            raise ValueError("object_scale_noise_multiplier_range must be a finite pair")
        self._scale_range = (float(multiplier_range[0]), float(multiplier_range[1]))
        if self._scale_range[0] > self._scale_range[1]:
            raise ValueError("object_scale_noise_multiplier_range minimum exceeds maximum")

        self._table = _entity(env.scene, "table")
        self._object = _entity(env.scene, "object")
        self._table_mocap = (
            self._table is not None
            and getattr(self._table, "_physical_entity", None) is not None
            and _declared_root_mode(env, "table") == "kinematic"
        )
        self._table_mocap_bound = False

    def _reset_robot(self, env: Any, ids: torch.Tensor) -> None:
        count = ids.numel()
        scales = torch.full_like(self._default_pos, self._pos_noise_hand)
        scales[: len(ARM_JOINT_NAMES)] = self._pos_noise_arm
        try:
            raw_limits = self._robot.data.soft_joint_pos_limits(env.device).to(env.device)
        except (AttributeError, NotImplementedError):
            raw_limits = None
        if raw_limits is None:
            positions = self._default_pos[None, :].expand(count, -1).clone()
        else:
            expected_limit_shape = (*self._default_pos.shape, 2)
            if tuple(raw_limits.shape) != expected_limit_shape:
                raise ValueError(
                    "SimToolRealReset soft joint limits have shape "
                    f"{tuple(raw_limits.shape)}; expected {expected_limit_shape}"
                )
            limits = raw_limits[None, :, :].expand(count, -1, -1)
            sampled = env.rng.uniform(
                limits[..., 0], limits[..., 1], (count, self._default_pos.numel())
            )
            positions = self._default_pos[None, :] + scales[None, :] * (
                sampled - self._default_pos[None, :]
            )
            positions = torch.clamp(positions, min=limits[..., 0], max=limits[..., 1])
        velocities = env.rng.uniform(
            -self._vel_noise, self._vel_noise, (count, self._default_pos.numel())
        )
        full_positions = torch.zeros(
            (env.num_envs, self._default_pos.numel()), dtype=torch.float32, device=env.device
        )
        full_velocities = torch.zeros_like(full_positions)
        full_positions[ids] = positions
        full_velocities[ids] = velocities
        reset_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        reset_mask[ids] = True
        reset_state = getattr(env, "_reset_state", None)
        transaction = getattr(reset_state, "_packed_reset_transaction", None)
        if reset_state is not None and transaction is not None:
            # UniLab 1.5's public Entity mask wrapper does not yet cover mapped
            # physical-entity joints. Both direct and composed scenes expose
            # the same public state columns, so stage the canonical columns in
            # the backend-owned packed transaction and mark the owner boundary.
            self._robot._materialize_reset_joint_indices()
            qpos_columns = self._robot._reset_joint_qpos_ids[list(self._joint_ids)]
            qvel_columns = self._robot._reset_joint_qvel_ids[list(self._joint_ids)]
            transaction.write_state_mask(
                reset_mask,
                qpos_columns=qpos_columns,
                qpos_values=full_positions,
                qvel_columns=qvel_columns,
                qvel_values=full_velocities,
            )
            reset_state._tensor_has_writes = True
            reset_state._requesting_terms.add("SimToolRealReset.write_joint_state")
        else:
            self._robot.write_joint_state_to_sim_mask(
                full_positions, full_velocities, reset_mask, joint_ids=self._joint_ids
            )
        action_term = env.action_manager.get_term("joint_pos")
        seed_reset_targets = getattr(action_term, "seed_reset_targets", None)
        if callable(seed_reset_targets):
            seed_reset_targets(ids, positions)

    def _reset_table_and_object(self, env: Any, ids: torch.Tensor) -> None:
        if self._table is None and self._object is None:
            return
        count = ids.numel()
        if self._table is None:
            table_z = torch.full(
                (count,), self._table_reset_z, dtype=torch.float32, device=env.device
            )
        else:
            table_z = self._table_reset_z + env.rng.uniform(
                -self._table_reset_z_range, self._table_reset_z_range, count
            )
            table_pose = _initial_entity_pose(env, "table")[None, :].expand(count, -1).clone()
            table_pose[:, 2] = table_z
            if self._table_mocap:
                if not self._table_mocap_bound:
                    assert self._table is not None
                    self._table.bind_mocap_pose_write(
                        self._table.root_body_name, term_name="simtoolreal_table_reset"
                    )
                    self._table_mocap_bound = True
                self._table.write_mocap_pose_to_sim(
                    table_pose.detach().cpu().numpy(),
                    ids,
                    term_name="simtoolreal_table_reset",
                )
            else:
                reset_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                reset_mask[ids] = True
                table_state = torch.zeros(
                    (env.num_envs, 13), dtype=torch.float32, device=env.device
                )
                table_state[:, 6] = 1.0
                table_state[ids, :7] = table_pose
                self._table.write_root_state_tensor_to_sim(table_state, reset_mask)

        if self._object is None:
            return
        if self._fixed_start_pose is not None:
            object_pose = self._fixed_start_pose[None, :].expand(count, -1).clone()
        else:
            noise = env.rng.uniform(-1.0, 1.0, (count, 3))
            object_pose = torch.empty((count, 7), dtype=torch.float32, device=env.device)
            object_pose[:, :3] = noise * self._reset_position_noise
            object_pose[:, 2] += table_z + self._table_object_z_offset
            object_pose[:, 3:] = _unit_quaternions(env.rng, count)
        object_pose[:, :3] += env.scene.env_origins(env.device)[ids]
        object_state = torch.zeros((env.num_envs, 13), dtype=torch.float32, device=env.device)
        object_state[:, 6] = 1.0
        object_state[ids, :7] = object_pose
        reset_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        reset_mask[ids] = True
        self._object.write_root_state_tensor_to_sim(object_state, reset_mask)
        try:
            hub = env.command_manager.get_term("goal")
        except (AttributeError, KeyError, ValueError):
            hub = None
        if hub is not None:
            origins = env.scene.env_origins(env.device)
            hub.object_init_z[ids] = object_pose[:, 2] - origins[ids, 2]
            hub.object_scale_multiplier[ids] = env.rng.uniform(
                self._scale_range[0], self._scale_range[1], (count, 3)
            )

    def __call__(self, env: Any, env_ids, **params) -> None:
        del params
        ids = _ids(env, env_ids)
        if ids.numel() == 0:
            return
        self._reset_robot(env, ids)
        self._reset_table_and_object(env, ids)


class SimToolRealWrenchEvent(ManagerTermBase):
    """Sample and dispatch object wrenches on interval triggers."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: Any):
        super().__init__(env)
        params = cfg.params
        self._object = _entity(env.scene, str(params.get("entity_name", "object")))
        self._force_scale = float(params.get("force_scale", 20.0))
        self._torque_scale = float(params.get("torque_scale", 2.0))
        self._force_prob_range = tuple(params.get("force_prob_range", (0.001, 0.1)))
        self._torque_prob_range = tuple(params.get("torque_prob_range", (0.001, 0.1)))
        self._force_decay = float(params.get("force_decay", 0.0))
        self._torque_decay = float(params.get("torque_decay", 0.0))
        self._force_decay_interval = float(params.get("force_decay_interval", 0.08))
        self._torque_decay_interval = float(params.get("torque_decay_interval", 0.08))
        only_lifted = bool(params.get("only_when_lifted", True))
        self._force_only_when_lifted = bool(params.get("force_only_when_lifted", only_lifted))
        self._torque_only_when_lifted = bool(params.get("torque_only_when_lifted", only_lifted))
        self._object_mass = float(params.get("object_mass", 1.0))
        if not math.isfinite(self._object_mass) or self._object_mass < 0.0:
            raise ValueError("object_mass must be finite and non-negative")
        self._object_masses: torch.Tensor | None = None
        if "object_mass" not in params:
            assignments = fixed_variant_plan_assignment(
                getattr(env, "cfg", None), int(env.num_envs)
            )
            table = tuple(getattr(getattr(env, "cfg", None), "tool_pool_masses", ()) or ())
            if assignments is not None and table:
                if len(table) < int(assignments.max()) + 1:
                    raise ValueError("task-side tool pool mass table is too short")
                assignment_tensor = torch.as_tensor(
                    assignments, dtype=torch.int64, device=env.device
                )
                self._object_masses = torch.as_tensor(
                    table, dtype=torch.float32, device=env.device
                ).index_select(0, assignment_tensor)
            elif assignments is not None and self._object is not None:
                readback = getattr(self._object, "fixed_variant_mass", None)
                if readback is not None:
                    masses = torch.as_tensor(readback, dtype=torch.float32, device=env.device)
                    if masses.ndim != 1 or not bool(torch.isfinite(masses).all()):
                        raise ValueError("fixed variant mass readback is invalid")
                    if bool((masses < 0).any()) or masses.shape[0] != env.num_envs:
                        raise ValueError("fixed variant mass readback has invalid rows")
                    self._object_masses = masses

        body_names = params.get("body_names", ("object_root",))
        if self._object is not None:
            requested = (body_names,) if isinstance(body_names, str) else tuple(body_names)
            requested = tuple(
                match_entity_names(declared_names(self._object, "body"), requested, label="body")
            )
            requested_ids = self._object.find_bodies(requested, preserve_order=True)[0]
            self._local_body_ids, self._backend_body_ids = self._object.bind_body_wrench(
                requested_ids, torque=True, term_name="simtoolreal_wrench_dr"
            )
        else:
            self._local_body_ids = torch.empty(0, dtype=torch.int64, device=env.device)
            self._backend_body_ids = []
        body_count = len(self._backend_body_ids)
        self._forces = torch.zeros(
            (env.num_envs, body_count, 3), dtype=torch.float32, device=env.device
        )
        self._torques = torch.zeros_like(self._forces)
        self._force_prob = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._torque_prob = torch.zeros_like(self._force_prob)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = _ids(self._env, env_ids)
        self._forces[ids].zero_()
        self._torques[ids].zero_()
        if ids.numel():
            self._force_prob[ids] = _log_uniform(
                self._env.rng, self._force_prob_range, ids.numel(), label="force_prob_range"
            )
            self._torque_prob[ids] = _log_uniform(
                self._env.rng, self._torque_prob_range, ids.numel(), label="torque_prob_range"
            )

    def __call__(self, env: Any, env_ids, **params) -> None:
        del params
        if self._object is None or len(self._backend_body_ids) == 0:
            return
        ids = _ids(env, env_ids)
        if ids.numel() == 0:
            return
        dt = float(getattr(env, "step_dt", 0.0))
        if self._force_decay > 0.0:
            self._forces[ids] *= self._force_decay ** (dt / self._force_decay_interval)
        else:
            self._forces[ids].zero_()
        if self._torque_decay > 0.0:
            self._torques[ids] *= self._torque_decay ** (dt / self._torque_decay_interval)
        else:
            self._torques[ids].zero_()

        shape = (ids.numel(), len(self._backend_body_ids), 3)
        fire_force = env.rng.random(ids.numel()) < self._force_prob[ids]
        fire_torque = env.rng.random(ids.numel()) < self._torque_prob[ids]
        mass = (
            self._object_mass
            if self._object_masses is None
            else self._object_masses[ids][:, None, None]
        )
        if bool(fire_force.any()):
            sampled = env.rng.standard_normal(shape) * mass * self._force_scale
            self._forces[ids[fire_force]] = sampled[fire_force]
        if bool(fire_torque.any()):
            sampled = env.rng.standard_normal(shape) * mass * self._torque_scale
            self._torques[ids[fire_torque]] = sampled[fire_torque]

        try:
            hub = env.command_manager.get_term("goal")
            lifted = hub.lifted[ids]
        except (AttributeError, KeyError, ValueError):
            lifted = torch.ones(ids.numel(), dtype=torch.bool, device=env.device)
        not_lifted = ids[~lifted]
        if self._force_only_when_lifted:
            self._forces[not_lifted].zero_()
        if self._torque_only_when_lifted:
            self._torques[not_lifted].zero_()

        self._object.apply_body_wrench_to_sim(
            self._forces[ids],
            self._torques[ids],
            self._backend_body_ids,
            env_ids=ids,
            term_name="simtoolreal_wrench_dr",
        )


__all__ = ["SimToolRealReset", "SimToolRealWrenchEvent"]
