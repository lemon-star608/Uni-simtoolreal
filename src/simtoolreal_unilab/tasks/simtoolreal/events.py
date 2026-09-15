"""SimToolReal reset and interval randomization terms.

The implementations are NumPy translations of the original IsaacLab task's
``reset_utils.py`` and ``action_utils.py``.  Scene entities are resolved only
on the cold path; the terms use the public :class:`~unilab.base.entity.Entity`
facade for state writes and body-wrench dispatch.  The robot-only skeleton can
still be constructed while the multi-asset scene is being assembled: optional
table/object entities are skipped until they are declared by the owner scene.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from unilab.base.entity import Entity
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg

from .config import ARM_DEFAULT_JOINT_POS, ARM_JOINT_NAMES, HAND_JOINT_NAMES, ResetCfg
from .env_cfg import fixed_variant_plan_assignment


def _ids(env: Any, env_ids: np.ndarray | slice | None) -> np.ndarray:
    if env_ids is None:
        return np.arange(env.num_envs, dtype=np.intp)
    if isinstance(env_ids, slice):
        return np.arange(env.num_envs, dtype=np.intp)[env_ids]
    result = np.asarray(env_ids, dtype=np.intp).reshape(-1)
    if np.any(result < 0) or np.any(result >= env.num_envs):
        raise ValueError(f"environment ids are out of range: {result.tolist()}")
    if np.unique(result).size != result.size:
        raise ValueError(f"environment ids contain duplicates: {result.tolist()}")
    return result


def _entity(scene: Mapping[str, Any], name: str) -> Entity | None:
    try:
        return cast(Entity, scene[name])
    except (KeyError, TypeError):
        return None


def _unit_quaternions(rng: np.random.Generator, count: int) -> np.ndarray:
    values = rng.standard_normal((count, 4)).astype(np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    values /= np.maximum(norms, np.float32(1e-12))
    return values


def _log_uniform(
    rng: np.random.Generator,
    bounds: tuple[float, float],
    count: int,
    *,
    label: str,
) -> np.ndarray:
    lo, hi = (float(bounds[0]), float(bounds[1]))
    if not np.isfinite((lo, hi)).all() or lo <= 0.0 or lo > hi:
        raise ValueError(f"{label} must satisfy 0 < min <= max, got {bounds!r}")
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=count)).astype(np.float32)


class SimToolRealReset(ManagerTermBase):
    """Reset robot joints and, when declared, table and object roots.

    The robot position distribution follows ``reset_utils.py:200-220``:
    sample within the backend soft limits and interpolate from the canonical
    default pose by separate arm/hand factors.  Root writes are staged in the
    env-owned reset transaction, so they commit atomically with the generic
    ``reset_scene_to_default`` event and the GoalViz command reset.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: Any):
        super().__init__(env)
        self._robot = cast(Entity, env.scene["robot"])
        names = list(ARM_JOINT_NAMES) + list(HAND_JOINT_NAMES)
        self._joint_ids, _ = self._robot.find_joints(names, preserve_order=True)
        default = [ARM_DEFAULT_JOINT_POS[name] for name in ARM_JOINT_NAMES]
        default += [0.0] * len(HAND_JOINT_NAMES)
        self._default_pos = np.asarray(default, dtype=np.float32)

        defaults = ResetCfg()
        params = cfg.params
        self._pos_noise_arm = float(
            params.get("dof_pos_noise_arm", defaults.reset_dof_pos_random_interval_arm)
        )
        self._pos_noise_hand = float(
            params.get("dof_pos_noise_hand", defaults.reset_dof_pos_random_interval_fingers)
        )
        self._vel_noise = float(params.get("dof_vel_noise", defaults.reset_dof_vel_random_interval))
        self._reset_position_noise = np.asarray(
            (
                float(params.get("reset_position_noise_x", defaults.reset_position_noise_x)),
                float(params.get("reset_position_noise_y", defaults.reset_position_noise_y)),
                float(params.get("reset_position_noise_z", defaults.reset_position_noise_z)),
            ),
            dtype=np.float32,
        )
        self._table_reset_z = float(params.get("table_reset_z", defaults.table_reset_z))
        self._table_reset_z_range = float(
            params.get("table_reset_z_range", defaults.table_reset_z_range)
        )
        self._table_object_z_offset = float(
            params.get("table_object_z_offset", defaults.table_object_z_offset)
        )
        fixed = params.get("fixed_start_pose", defaults.fixed_start_pose)
        self._fixed_start_pose = None if fixed is None else np.asarray(fixed, dtype=np.float32)
        if self._fixed_start_pose is not None and self._fixed_start_pose.shape != (7,):
            raise ValueError("fixed_start_pose must contain (x, y, z, qw, qx, qy, qz)")
        multiplier_range = params.get("object_scale_noise_multiplier_range", (1.0, 1.0))
        multiplier_range = np.asarray(multiplier_range, dtype=np.float64)
        if multiplier_range.shape != (2,) or not np.isfinite(multiplier_range).all():
            raise ValueError("object_scale_noise_multiplier_range must be a finite pair")
        self._scale_range = (float(multiplier_range[0]), float(multiplier_range[1]))
        if self._scale_range[0] > self._scale_range[1]:
            raise ValueError("object_scale_noise_multiplier_range minimum exceeds maximum")

        self._table = _entity(env.scene, "table")
        self._object = _entity(env.scene, "object")

    def _reset_robot(self, env: Any, ids: np.ndarray) -> None:
        count = ids.size
        scales = np.full(self._default_pos.size, self._pos_noise_hand, dtype=np.float32)
        scales[: len(ARM_JOINT_NAMES)] = self._pos_noise_arm
        try:
            raw_limits = np.asarray(self._robot.data.soft_joint_pos_limits, dtype=np.float32)
            if raw_limits.shape == (self._default_pos.size, 2):
                # EntityData exposes model limits once; expand only the rows
                # being reset.  Lightweight IsaacLab-style fixtures may carry
                # an explicit (num_envs, joints, 2) tensor instead.
                limits = np.broadcast_to(raw_limits, (count, self._default_pos.size, 2))
            else:
                limits = raw_limits[ids]
        except (AttributeError, NotImplementedError):
            limits = None
        if limits is None:
            positions = np.broadcast_to(self._default_pos, (count, self._default_pos.size)).copy()
        else:
            expected = (count, self._default_pos.size, 2)
            if limits.shape != expected:
                raise ValueError(
                    "SimToolRealReset soft joint limits have shape "
                    f"{limits.shape}; expected {expected}"
                )
            sampled = env.rng.uniform(limits[..., 0], limits[..., 1]).astype(np.float32)
            positions = self._default_pos[None, :] + scales[None, :] * (
                sampled - self._default_pos[None, :]
            )
            positions = np.clip(positions, limits[..., 0], limits[..., 1])
        velocities = env.rng.uniform(
            -self._vel_noise, self._vel_noise, size=(count, self._default_pos.size)
        ).astype(np.float32)
        self._robot.write_joint_state_to_sim(
            positions, velocities, joint_ids=self._joint_ids, env_ids=ids
        )

    def _reset_table_and_object(self, env: Any, ids: np.ndarray) -> None:
        if self._table is None and self._object is None:
            return
        count = ids.size
        table_z = self._table_reset_z + env.rng.uniform(
            -self._table_reset_z_range, self._table_reset_z_range, size=count
        ).astype(np.float32)
        if self._table is not None:
            table_pose = np.array(self._table.read_reset_root_pose(ids), copy=True)
            table_pose[:, 2] = table_z
            self._table.write_root_link_pose_to_sim(table_pose, env_ids=ids)
            self._table.write_root_link_velocity_to_sim(
                np.zeros((count, 6), dtype=np.float32), env_ids=ids
            )

        if self._object is None:
            return
        if self._fixed_start_pose is not None:
            object_pose = np.broadcast_to(self._fixed_start_pose, (count, 7)).copy()
        else:
            noise = env.rng.uniform(-1.0, 1.0, size=(count, 3)).astype(np.float32)
            object_pose = np.empty((count, 7), dtype=np.float32)
            object_pose[:, 0] = noise[:, 0] * self._reset_position_noise[0]
            object_pose[:, 1] = noise[:, 1] * self._reset_position_noise[1]
            object_pose[:, 2] = (
                table_z + self._table_object_z_offset + noise[:, 2] * self._reset_position_noise[2]
            )
            object_pose[:, 3:] = _unit_quaternions(env.rng, count)
        object_pose[:, :3] += np.asarray(env.scene.env_origins[ids], dtype=np.float32)
        self._object.write_root_link_pose_to_sim(object_pose, env_ids=ids)
        self._object.write_root_link_velocity_to_sim(
            np.zeros((count, 6), dtype=np.float32), env_ids=ids
        )
        try:
            hub = env.command_manager.get_term("goal")
        except (AttributeError, KeyError, ValueError):
            hub = None
        if hub is not None:
            hub.object_init_z[ids] = object_pose[:, 2] - env.scene.env_origins[ids, 2]
            hub.object_scale_multiplier[ids] = env.rng.uniform(
                self._scale_range[0], self._scale_range[1], size=(count, 3)
            ).astype(np.float32)

    def __call__(self, env: Any, env_ids, **params) -> None:
        del params
        ids = _ids(env, env_ids)
        if ids.size == 0:
            return
        self._reset_robot(env, ids)
        self._reset_table_and_object(env, ids)


class SimToolRealWrenchEvent(ManagerTermBase):
    """Sample an object wrench on interval triggers.

    This mirrors ``action_utils.py:77-130``.  Force and torque are sampled as
    standard-normal vectors multiplied by object mass and the configured
    scales; trigger probabilities are log-uniform and the lift latch gates the
    active channels.  The adapter owns the actual world-frame write.
    """

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
        if not np.isfinite(self._object_mass) or self._object_mass < 0.0:
            raise ValueError("object_mass must be finite and non-negative")
        self._object_masses: np.ndarray | None = None
        # The worker measures each environment's variant mass from the baked
        # asset and exposes it through the Entity facade readback; the
        # immutable per-environment assignment comes from the scene's fixed
        # variant plan.  An explicit term-level object_mass remains available
        # for lightweight fixtures and other backends.  A scene without the
        # object entity keeps the scalar fallback: the term never dispatches
        # without it.
        if "object_mass" not in params:
            assignments = fixed_variant_plan_assignment(
                getattr(env, "cfg", None), int(env.num_envs)
            )
            if assignments is not None and self._object is not None:
                masses = np.asarray(self._object.fixed_variant_mass, dtype=np.float32)
                if masses.ndim != 1 or not np.isfinite(masses).all() or np.any(masses < 0.0):
                    raise ValueError(
                        "fixed variant mass readback must be finite non-negative 1-D values"
                    )
                if masses.shape[0] != env.num_envs:
                    raise ValueError(
                        "fixed variant mass readback must have one entry per environment; "
                        f"got {masses.shape[0]}, expected {env.num_envs}"
                    )
                self._object_masses = masses
        body_names = params.get("body_names", ("object_root",))
        self._local_body_ids = self._backend_body_ids = np.empty(0, dtype=np.int32)
        if self._object is not None:
            # Task config names bodies because they are stable across URDF
            # conversion.  Entity.bind_body_wrench consumes local integer
            # selectors, so resolve the names once on the cold path; the tiny
            # fallback keeps lightweight test doubles compatible.
            requested = (body_names,) if isinstance(body_names, str) else body_names
            resolver = getattr(self._object, "find_bodies", None)
            if resolver is not None and all(isinstance(name, str) for name in requested):
                requested = resolver(requested, preserve_order=True)[0]
            self._local_body_ids, self._backend_body_ids = self._object.bind_body_wrench(
                requested, torque=True, term_name="simtoolreal_wrench_dr"
            )
        self._forces = np.zeros((env.num_envs, self._backend_body_ids.size, 3), dtype=np.float32)
        self._torques = np.zeros_like(self._forces)
        self._force_prob = np.zeros(env.num_envs, dtype=np.float32)
        self._torque_prob = np.zeros(env.num_envs, dtype=np.float32)

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = _ids(self._env, env_ids)
        self._forces[ids] = 0.0
        self._torques[ids] = 0.0
        if ids.size:
            self._force_prob[ids] = _log_uniform(
                self._env.rng, self._force_prob_range, ids.size, label="force_prob_range"
            )
            self._torque_prob[ids] = _log_uniform(
                self._env.rng, self._torque_prob_range, ids.size, label="torque_prob_range"
            )

    def __call__(self, env: Any, env_ids, **params) -> None:
        del params
        if self._object is None or self._backend_body_ids.size == 0:
            return
        ids = _ids(env, env_ids)
        if ids.size == 0:
            return
        dt = float(getattr(env, "step_dt", 0.0))
        if self._force_decay > 0.0:
            self._forces[ids] *= self._force_decay ** (dt / self._force_decay_interval)
        else:
            self._forces[ids] = 0.0
        if self._torque_decay > 0.0:
            self._torques[ids] *= self._torque_decay ** (dt / self._torque_decay_interval)
        else:
            self._torques[ids] = 0.0

        count = ids.size
        shape = (count, self._backend_body_ids.size, 3)
        fire_force = env.rng.random(count) < self._force_prob[ids]
        fire_torque = env.rng.random(count) < self._torque_prob[ids]
        if np.any(fire_force):
            sampled = (
                env.rng.standard_normal(shape).astype(np.float32)
                * (
                    self._object_mass
                    if self._object_masses is None
                    else self._object_masses[ids][:, None, None]
                )
                * self._force_scale
            )
            self._forces[ids[fire_force]] = sampled[fire_force]
        if np.any(fire_torque):
            sampled = (
                env.rng.standard_normal(shape).astype(np.float32)
                * (
                    self._object_mass
                    if self._object_masses is None
                    else self._object_masses[ids][:, None, None]
                )
                * self._torque_scale
            )
            self._torques[ids[fire_torque]] = sampled[fire_torque]

        try:
            hub = env.command_manager.get_term("goal")
            lifted = np.asarray(hub.lifted[ids], dtype=bool)
        except (AttributeError, KeyError, ValueError):
            lifted = np.ones(count, dtype=bool)
        if self._force_only_when_lifted:
            self._forces[ids[~lifted]] = 0.0
        if self._torque_only_when_lifted:
            self._torques[ids[~lifted]] = 0.0

        self._object.apply_body_wrench_to_sim(
            self._forces[ids],
            self._torques[ids],
            self._backend_body_ids,
            env_ids=ids,
            term_name="simtoolreal_wrench_dr",
        )


__all__ = ["SimToolRealReset", "SimToolRealWrenchEvent"]
