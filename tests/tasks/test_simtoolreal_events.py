"""Pure NumPy coverage for SimToolReal reset and wrench event terms."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from unilab.managers.manager_base import ManagerTermBaseCfg

from simtoolreal_unilab.tasks.simtoolreal.config import (
    ARM_DEFAULT_JOINT_POS,
    ARM_JOINT_NAMES,
    HAND_JOINT_NAMES,
)
from simtoolreal_unilab.tasks.simtoolreal.events import (
    SimToolRealReset,
    SimToolRealWrenchEvent,
)

NUM_ENVS = 5
NUM_JOINTS = len(ARM_JOINT_NAMES) + len(HAND_JOINT_NAMES)


class _Robot:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            soft_joint_pos_limits=np.broadcast_to(
                np.asarray([-3.0, 3.0], dtype=np.float32), (NUM_ENVS, NUM_JOINTS, 2)
            ).copy()
        )
        self.writes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def find_joints(self, names, preserve_order=False):
        del preserve_order
        assert len(names) == NUM_JOINTS
        return list(range(NUM_JOINTS)), list(names)

    def write_joint_state_to_sim(self, position, velocity, *, joint_ids, env_ids):
        self.writes.append((np.asarray(position), np.asarray(velocity), np.asarray(env_ids)))
        assert len(joint_ids) == NUM_JOINTS


class _Root:
    def __init__(self, default_z: float = 0.0) -> None:
        self.default = np.tile(
            np.asarray([0.0, 0.0, default_z, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (NUM_ENVS, 1),
        )
        self.poses: list[tuple[np.ndarray, np.ndarray]] = []
        self.velocities: list[tuple[np.ndarray, np.ndarray]] = []

    def read_reset_root_pose(self, ids):
        return self.default[np.asarray(ids)].copy()

    def write_root_link_pose_to_sim(self, pose, *, env_ids):
        self.poses.append((np.asarray(pose).copy(), np.asarray(env_ids).copy()))

    def write_root_link_velocity_to_sim(self, velocity, *, env_ids):
        self.velocities.append((np.asarray(velocity).copy(), np.asarray(env_ids).copy()))


class _WrenchObject:
    def __init__(self, variant_masses: np.ndarray | None = None) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self._variant_masses = variant_masses

    @property
    def fixed_variant_mass(self) -> np.ndarray:
        if self._variant_masses is None:
            raise NotImplementedError("no fixed variant pool bound to this entity")
        return self._variant_masses

    def bind_body_wrench(self, body_names, *, torque, term_name):
        assert tuple(body_names) == ("object_root",)
        assert torque is True
        assert term_name == "simtoolreal_wrench_dr"
        return np.asarray([0], dtype=np.int32), np.asarray([17], dtype=np.int32)

    def apply_body_wrench_to_sim(self, forces, torques, body_ids, *, env_ids, term_name):
        self.calls.append(
            (
                np.asarray(forces).copy(),
                np.asarray(torques).copy(),
                np.asarray(body_ids).copy(),
                np.asarray(env_ids).copy(),
            )
        )
        assert term_name == "simtoolreal_wrench_dr"


def _plan(assignment, variants=("a.xml", "b.xml")):
    """Scene-shaped entity variant binding double for the task-side reader."""
    return SimpleNamespace(
        entity_variant=SimpleNamespace(
            plan=SimpleNamespace(
                assignment=np.asarray(assignment, dtype=np.int32),
                variants=variants,
            )
        )
    )


def _env(*, with_roots: bool = True, with_object: bool = True):
    robot = _Robot()
    scene = {"robot": robot}
    if with_roots:
        scene["table"] = _Root(default_z=0.38)
    if with_object:
        scene["object"] = _Root()
    hub = SimpleNamespace(
        lifted=np.asarray([True, False, True, False, True], dtype=bool),
        object_init_z=np.zeros(NUM_ENVS, dtype=np.float32),
        object_scale_multiplier=np.ones((NUM_ENVS, 3), dtype=np.float32),
    )
    return SimpleNamespace(
        num_envs=NUM_ENVS,
        rng=np.random.default_rng(19),
        step_dt=0.016,
        scene=SimpleNamespace(
            __getitem__=scene.__getitem__,
            env_origins=np.zeros((NUM_ENVS, 3), dtype=np.float32),
        ),
        command_manager=SimpleNamespace(get_term=lambda name: hub if name == "goal" else None),
        robot=robot,
        roots=scene,
        hub=hub,
    )


def _scene_mapping(env):
    # SimpleNamespace does not delegate special methods; replace with a tiny
    # mapping object while keeping the rest of the fixture readable.
    class _Scene(dict):
        env_origins = np.zeros((NUM_ENVS, 3), dtype=np.float32)

    scene = _Scene(env.roots)
    env.scene = scene
    return env


def test_reset_samples_joint_noise_and_multi_asset_roots():
    env = _scene_mapping(_env())
    term = SimToolRealReset(
        ManagerTermBaseCfg(
            func=SimToolRealReset,
            params={
                "dof_pos_noise_arm": 0.1,
                "dof_pos_noise_hand": 0.2,
                "dof_vel_noise": 0.5,
                "object_scale_noise_multiplier_range": (0.9, 1.1),
            },
        ),
        env,
    )
    ids = np.asarray([1, 4], dtype=np.intp)
    term(env, ids)

    position, velocity, written_ids = env.robot.writes[0]
    np.testing.assert_array_equal(written_ids, ids)
    assert position.shape == (2, NUM_JOINTS)
    assert velocity.shape == (2, NUM_JOINTS)
    assert np.all(position >= -3.0) and np.all(position <= 3.0)
    assert np.all(np.abs(velocity) <= 0.5 + 1e-6)
    np.testing.assert_allclose(
        position[:, :7], np.tile(ARM_DEFAULT_JOINT_POS_ARRAY, (2, 1)), atol=0.6
    )

    table = env.scene["table"]
    obj = env.scene["object"]
    assert len(table.poses) == 1 and len(obj.poses) == 1
    table_pose, table_ids = table.poses[0]
    object_pose, object_ids = obj.poses[0]
    np.testing.assert_array_equal(table_ids, ids)
    np.testing.assert_array_equal(object_ids, ids)
    assert np.all((table_pose[:, 2] >= 0.37) & (table_pose[:, 2] <= 0.39))
    assert np.all(object_pose[:, 2] >= table_pose[:, 2] + 0.24)
    np.testing.assert_allclose(np.linalg.norm(object_pose[:, 3:], axis=-1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(env.hub.object_init_z[ids], object_pose[:, 2])
    assert np.all(env.hub.object_scale_multiplier[ids] >= 0.9)
    assert np.all(env.hub.object_scale_multiplier[ids] <= 1.1)


def test_reset_robot_only_skeleton_keeps_optional_scene_paths_disabled():
    env = _scene_mapping(_env(with_roots=False, with_object=False))
    term = SimToolRealReset(ManagerTermBaseCfg(func=SimToolRealReset), env)
    term(env, np.asarray([0, 2]))
    assert len(env.robot.writes) == 1


def test_reset_accepts_model_level_joint_limits():
    env = _scene_mapping(_env(with_roots=False, with_object=False))
    env.robot.data.soft_joint_pos_limits = env.robot.data.soft_joint_pos_limits[0]
    term = SimToolRealReset(ManagerTermBaseCfg(func=SimToolRealReset), env)
    term(env, np.asarray([1, 3]))
    positions, velocities, called_ids = env.robot.writes[-1]
    assert positions.shape == (2, NUM_JOINTS)
    assert velocities.shape == (2, NUM_JOINTS)
    np.testing.assert_array_equal(called_ids, [1, 3])


def test_wrench_event_samples_and_lift_gates_channels():
    env = _scene_mapping(_env(with_roots=False))
    obj = _WrenchObject()
    env.scene["object"] = obj
    term = SimToolRealWrenchEvent(
        ManagerTermBaseCfg(
            func=SimToolRealWrenchEvent,
            params={
                "force_scale": 20.0,
                "torque_scale": 2.0,
                "force_prob_range": (1.0, 1.0),
                "torque_prob_range": (1.0, 1.0),
                "object_mass": 2.0,
                "only_when_lifted": True,
            },
        ),
        env,
    )
    ids = np.arange(NUM_ENVS, dtype=np.intp)
    term.reset(ids)
    term(env, ids)
    assert len(obj.calls) == 1
    forces, torques, body_ids, called_ids = obj.calls[0]
    assert forces.shape == (NUM_ENVS, 1, 3)
    assert torques.shape == (NUM_ENVS, 1, 3)
    np.testing.assert_array_equal(body_ids, [17])
    np.testing.assert_array_equal(called_ids, ids)
    np.testing.assert_array_equal(forces[~env.hub.lifted], 0.0)
    np.testing.assert_array_equal(torques[~env.hub.lifted], 0.0)
    assert np.any(forces[env.hub.lifted] != 0.0)
    assert np.any(torques[env.hub.lifted] != 0.0)


def test_wrench_probabilities_require_positive_log_uniform_bounds():
    env = _scene_mapping(_env(with_roots=False))
    env.scene["object"] = _WrenchObject()
    term = SimToolRealWrenchEvent(
        ManagerTermBaseCfg(
            func=SimToolRealWrenchEvent,
            params={"force_prob_range": (0.0, 0.1)},
        ),
        env,
    )
    with pytest.raises(ValueError, match="force_prob_range"):
        term.reset(np.asarray([0]))


def test_wrench_event_uses_backend_readback_masses_per_environment():
    env = _scene_mapping(_env(with_roots=False))
    env.scene["object"] = _WrenchObject(
        variant_masses=np.asarray([1.0, 3.0, 1.0, 3.0, 1.0], dtype=np.float32)
    )

    class _FixedRng:
        def random(self, count):
            return np.zeros(count, dtype=np.float32)

        def uniform(self, low, high, size=None):
            del low, high
            return np.ones(size, dtype=np.float32)

        def standard_normal(self, shape):
            return np.ones(shape, dtype=np.float32)

    env.rng = _FixedRng()
    # Same round-robin assignment the worker applies to the pool; the mass
    # itself comes from the entity's backend readback (already per-env).
    env.cfg = SimpleNamespace(scene=_plan([0, 1, 0, 1, 0]))
    term = SimToolRealWrenchEvent(
        ManagerTermBaseCfg(
            func=SimToolRealWrenchEvent,
            params={
                "force_scale": 1.0,
                "torque_scale": 1.0,
                "force_prob_range": (1.0, 1.0),
                "torque_prob_range": (1.0, 1.0),
                "only_when_lifted": False,
            },
        ),
        env,
    )
    ids = np.arange(NUM_ENVS, dtype=np.intp)
    term.reset(ids)
    term(env, ids)
    forces, torques, _, _ = env.scene["object"].calls[-1]
    force_norm = np.linalg.norm(forces[:, 0], axis=-1)
    torque_norm = np.linalg.norm(torques[:, 0], axis=-1)
    np.testing.assert_allclose(force_norm, np.sqrt(3.0) * np.asarray([1, 3, 1, 3, 1]))
    np.testing.assert_allclose(torque_norm, np.sqrt(3.0) * np.asarray([1, 3, 1, 3, 1]))
    # The term's cold-path vector is the backend readback, per environment.
    np.testing.assert_array_equal(term._object_masses, [1.0, 3.0, 1.0, 3.0, 1.0])


def _wrench_term(env):
    return SimToolRealWrenchEvent(
        ManagerTermBaseCfg(
            func=SimToolRealWrenchEvent,
            params={
                "force_scale": 1.0,
                "torque_scale": 1.0,
                "force_prob_range": (1.0, 1.0),
                "torque_prob_range": (1.0, 1.0),
                "only_when_lifted": False,
            },
        ),
        env,
    )


def test_wrench_event_fails_closed_on_malformed_readback():
    env = _scene_mapping(_env(with_roots=False))
    env.scene["object"] = _WrenchObject(
        variant_masses=np.asarray([1.0, -3.0, 1.0, 3.0, 1.0], dtype=np.float32)
    )
    env.cfg = SimpleNamespace(scene=_plan([0, 1, 0, 1, 0]))
    with pytest.raises(ValueError, match="finite non-negative"):
        _wrench_term(env)


def test_wrench_event_fails_closed_on_short_readback():
    env = _scene_mapping(_env(with_roots=False))
    env.scene["object"] = _WrenchObject(variant_masses=np.asarray([1.0, 3.0], dtype=np.float32))
    env.cfg = SimpleNamespace(scene=_plan([0, 1, 0, 1, 0]))
    with pytest.raises(ValueError, match="one entry per environment"):
        _wrench_term(env)


def test_wrench_event_fails_closed_on_plan_assignment_length_mismatch():
    env = _scene_mapping(_env(with_roots=False))
    env.scene["object"] = _WrenchObject(
        variant_masses=np.asarray([1.0, 3.0, 1.0, 3.0, 1.0], dtype=np.float32)
    )
    env.cfg = SimpleNamespace(scene=_plan([0, 1]))
    with pytest.raises(ValueError, match="one entry per environment"):
        _wrench_term(env)


def test_wrench_event_falls_back_to_scalar_without_plan():
    env = _scene_mapping(_env(with_roots=False))
    env.scene["object"] = _WrenchObject(
        variant_masses=np.asarray([9.0] * NUM_ENVS, dtype=np.float32)
    )
    env.cfg = SimpleNamespace(scene=SimpleNamespace(entity_variant=None))
    term = _wrench_term(env)
    # No entity variant binding: the term-level scalar default stays
    # authoritative and the backend readback is not consumed.
    assert term._object_masses is None


ARM_DEFAULT_JOINT_POS_ARRAY = np.asarray(
    [ARM_DEFAULT_JOINT_POS[name] for name in ARM_JOINT_NAMES], dtype=np.float32
)
