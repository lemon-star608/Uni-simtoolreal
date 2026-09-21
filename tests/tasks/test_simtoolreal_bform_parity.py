"""B-form parity gate (M-J5): the A-form composed pool as oracle.

Three layers, per the M-J5 contract:

- source layer: covered by ``test_simtoolreal_entity_mjcf_writer.py`` (A/B
  value and FK parity on the writers' outputs).
- composed-model layer (this file): the B-form scene composed through the
  official unisim ``compose_scene`` vs the A-form task-side composed pool,
  compared array-by-array with name-prefix mapping (B qualifies every model
  name with the physical entity).
- rollout layer (this file): both consumption paths on the real mujoco
  backend, same home state + same control sequence, trajectory comparison.

The A-form compose path (``compose_scene_mjcf``/``materialize_tool_pool_mjcf``)
exists only to serve as this oracle; it is retired once the gate is green
(roadmap M-J5).
"""

from __future__ import annotations

import pathlib

import mujoco
import numpy as np
import pytest

from simtoolreal_unilab.tasks.simtoolreal.entity_mjcf_writer import write_entity_set
from simtoolreal_unilab.tasks.simtoolreal.tool_pool_mjcf import materialize_tool_pool_mjcf

NUM_PER_TYPE = 1  # 12 variants keep the gate fast
NUM_TOOLS = 12
SIM_DT = 1 / 120


@pytest.fixture(scope="module")
def a_pool(tmp_path_factory):
    root = tmp_path_factory.mktemp("a_pool")
    paths, masses, scales = materialize_tool_pool_mjcf(
        seed=42, num_per_type=NUM_PER_TYPE, configured_root=str(root)
    )
    return root, paths, masses, scales


@pytest.fixture(scope="module")
def b_set(tmp_path_factory):
    root = tmp_path_factory.mktemp("b_set") / "entity_set"
    manifest = write_entity_set(root, seed=42, num_per_type=NUM_PER_TYPE)
    return root, manifest


def _b_scene(b_root: pathlib.Path, tools: list[pathlib.Path], num_envs: int):
    from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
    from unisim.entities import (
        EntityInitialState,
        EntityVariantBinding,
        SceneEntitySpec,
    )
    from unisim.scene import SceneCfg

    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                name="robot",
                source=ModelSourceDescriptor(str(b_root / "robot.xml")),
                kind="articulation",
                root_mode="fixed",
                initial_state=EntityInitialState(position=(0.0, 0.8, 0.0)),
            ),
            SceneEntitySpec(
                name="table",
                source=ModelSourceDescriptor(str(b_root / "table.xml")),
                kind="rigid",
                root_mode="kinematic",
                initial_state=EntityInitialState(position=(0.0, 0.0, 0.38)),
            ),
            SceneEntitySpec(
                name="floor",
                source=ModelSourceDescriptor(str(b_root / "floor.xml")),
                kind="rigid",
                root_mode="kinematic",
            ),
            SceneEntitySpec(
                name="object",
                source=ModelSourceDescriptor(str(tools[0])),
                kind="rigid",
                root_mode="floating",
                initial_state=EntityInitialState(position=(0.0, 0.0, 0.63)),
            ),
            SceneEntitySpec(
                name="goalviz",
                mirror_of="object",
                kind="rigid",
                root_mode="kinematic",
                collision_enabled=False,
                initial_state=EntityInitialState(position=(0.0, 0.0, 0.63)),
            ),
        ),
        entity_variant=EntityVariantBinding(
            target_entity="object",
            plan=FixedVariantPlan(
                variants=tuple(ModelSourceDescriptor(str(p)) for p in tools),
                assignment=np.arange(num_envs, dtype=np.int32) % len(tools),
                layout=FixedVariantLayout.SAME_LAYOUT,
            ),
        ),
        default_keyframe_name="home",
    )


def _stem(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _b_id(model, objtype, a_name: str, *, kind: str) -> int:
    """Find B's qualified counterpart of an A-form (bare) object name."""
    candidates = {
        "body": (f"robot/{a_name}", f"table/{a_name}", f"object/{a_name}", f"floor/{a_name}"),
        "joint": (f"robot/{a_name}", f"object/{a_name}"),
        "actuator": (f"robot/{a_name}",),
        "geom": (f"robot/{a_name}", f"table/{a_name}", f"object/{a_name}", f"floor/{a_name}"),
    }[kind]
    for candidate in candidates:
        idx = mujoco.mj_name2id(model, objtype, candidate)
        if idx >= 0:
            return idx
    raise KeyError(f"no B-form {kind} for A-form name {a_name!r}")


def test_composed_model_parity(a_pool, b_set):
    """B composed (official compose_scene) == A composed, array by array."""
    from unisim.mjcf_compiler import compose_scene

    _, a_paths, _, _ = a_pool
    b_root, manifest = b_set
    b_tools = sorted((b_root / "tools").glob("*.xml"))
    a_by_name = {pathlib.Path(p).name: p for p in a_paths}
    assert set(a_by_name) == {p.name for p in b_tools}

    # Compose B per variant: the composed canonical model always realizes
    # one variant, so recompose with a single-variant plan per tool.
    for b_tool in b_tools:
        variant_scene = _b_scene(b_root, [b_tool], num_envs=1)
        variant_composed = compose_scene(variant_scene, 1, SIM_DT)
        b_model = variant_composed.model
        a_model = mujoco.MjModel.from_xml_path(str(a_by_name[b_tool.name]))

        # Structure: identical dynamics shape; B carries two extra bodies
        # (the floor is an entity body where A bakes a worldbody geom, plus
        # the goalviz mirror body) and the mirror's four collision-disabled
        # geoms.  The table is a body in both forms.
        assert (b_model.nq, b_model.nv, b_model.nu) == (
            a_model.nq,
            a_model.nv,
            a_model.nu,
        )
        assert b_model.nbody == a_model.nbody + 2
        assert b_model.ngeom == a_model.ngeom + 4
        mirror_body = mujoco.mj_name2id(b_model, mujoco.mjtObj.mjOBJ_BODY, "goalviz/object_root")
        assert mirror_body > 0
        for g in range(b_model.ngeom):
            if int(b_model.geom_bodyid[g]) != mirror_body:
                continue
            assert b_model.geom_contype[g] == 0 and b_model.geom_conaffinity[g] == 0
        assert b_model.nmesh == a_model.nmesh
        assert b_model.nexclude == a_model.nexclude == 35

        # Global physics options: bit-identical except timestep, which the
        # composer's 8-digit XML round-trip quantizes (the backend
        # overwrites opt.timestep with the double sim_dt at build time —
        # mujoco backend.py:1649 — so the quantization never reaches the
        # runtime).
        skip = {"timestep"}
        for field in dir(a_model.opt):
            if field.startswith("_") or field in skip:
                continue
            va, vb = getattr(a_model.opt, field), getattr(b_model.opt, field)
            if callable(va):
                continue
            assert np.array_equal(np.asarray(va), np.asarray(vb)), f"opt.{field}"
        assert abs(a_model.opt.timestep - b_model.opt.timestep) < 1e-8

        # Joints: ranges/stiffness per canonical name.
        for j in range(a_model.njnt):
            name = mujoco.mj_id2name(a_model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name is None:
                continue
            bj = _b_id(b_model, mujoco.mjtObj.mjOBJ_JOINT, name, kind="joint")
            # 6-significant-digit XML round-trip on the compose side.
            np.testing.assert_allclose(
                b_model.jnt_range[bj], a_model.jnt_range[j], rtol=1e-5, atol=1e-7
            )
            np.testing.assert_allclose(
                b_model.jnt_stiffness[bj], a_model.jnt_stiffness[j], rtol=1e-5, atol=1e-12
            )
            np.testing.assert_allclose(
                b_model.dof_armature[b_model.jnt_dofadr[bj]],
                a_model.dof_armature[a_model.jnt_dofadr[j]],
                rtol=1e-6,
                atol=1e-12,
            )

        # Bodies: mass/inertia per mapped name (object identity per
        # variant is the point of the pool).
        for i in range(1, a_model.nbody):
            name = mujoco.mj_id2name(a_model, mujoco.mjtObj.mjOBJ_BODY, i)
            bi = _b_id(b_model, mujoco.mjtObj.mjOBJ_BODY, name, kind="body")
            np.testing.assert_allclose(
                b_model.body_mass[bi], a_model.body_mass[i], rtol=1e-5, atol=1e-12
            )
            np.testing.assert_allclose(
                b_model.body_inertia[bi], a_model.body_inertia[i], rtol=1e-5, atol=1e-12
            )

        # Geoms: contact profile per mapped name.
        for g in range(a_model.ngeom):
            name = mujoco.mj_id2name(a_model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name is None:
                continue
            bg = _b_id(b_model, mujoco.mjtObj.mjOBJ_GEOM, name, kind="geom")
            np.testing.assert_allclose(
                b_model.geom_friction[bg], a_model.geom_friction[g], rtol=0, atol=1e-12
            )
            assert b_model.geom_condim[bg] == a_model.geom_condim[g]
            np.testing.assert_allclose(
                b_model.geom_solimp[bg], a_model.geom_solimp[g], rtol=0, atol=1e-12
            )
            np.testing.assert_allclose(
                b_model.geom_solref[bg], a_model.geom_solref[g], rtol=0, atol=1e-12
            )
            assert (
                b_model.geom_contype[bg],
                b_model.geom_conaffinity[bg],
                b_model.geom_group[bg],
            ) == (
                a_model.geom_contype[g],
                a_model.geom_conaffinity[g],
                a_model.geom_group[g],
            )

        # Actuators: position gains per mapped name.
        for k in range(a_model.nu):
            name = mujoco.mj_id2name(a_model, mujoco.mjtObj.mjOBJ_ACTUATOR, k)
            bk = _b_id(b_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name, kind="actuator")
            np.testing.assert_allclose(
                b_model.actuator_gainprm[bk, 0], a_model.actuator_gainprm[k, 0], rtol=1e-5, atol=1e-12
            )
            np.testing.assert_allclose(
                b_model.actuator_biasprm[bk, 1], a_model.actuator_biasprm[k, 1], rtol=1e-5, atol=1e-12
            )

        # Excludes: same authored pairs, prefix-stripped.  MuJoCo 3.x encodes
        # the pair as a signature: body1 in the low 16 bits, body2 high.
        def exclude_pairs(model):
            pairs = set()
            for e in range(model.nexclude):
                sig = int(model.exclude_signature[e])
                b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, sig & 0xFFFF)
                b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, sig >> 16)
                pairs.add(tuple(sorted((_stem(b1), _stem(b2)))))
            return pairs

        assert exclude_pairs(a_model) == exclude_pairs(b_model)

        # Home keyframe: identical 36-wide qpos/ctrl.
        np.testing.assert_allclose(b_model.key_qpos[0], a_model.key_qpos[0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(b_model.key_ctrl[0], a_model.key_ctrl[0], rtol=0, atol=1e-12)
        variant_composed.close()


@pytest.fixture(scope="module")
def rollout_pair(a_pool, b_set):
    """Both consumption paths on the real backend, same assignment."""
    from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
    from unisim.factory import create_backend
    from unisim.scene import SceneCfg

    _, a_paths, _, _ = a_pool
    b_root, manifest = b_set
    b_tools = [b_root / "tools" / name for name in manifest["tool_files"]]
    num_envs = 4
    assignment = np.arange(num_envs, dtype=np.int32) % len(a_paths)

    scene_a = SceneCfg(
        model_file=str(a_paths[0]),
        default_keyframe_name="home",
        fixed_variant_plan=FixedVariantPlan(
            variants=tuple(ModelSourceDescriptor(str(p)) for p in a_paths),
            assignment=assignment,
            layout=FixedVariantLayout.SAME_LAYOUT,
        ),
    )
    # A and B pools share the catalog order, so the same round-robin formula
    # binds identical variants per env on both paths.
    scene_b = _b_scene(b_root, b_tools, num_envs)
    np.testing.assert_array_equal(scene_b.entity_variant.plan.assignment, assignment)
    be_a = create_backend("mujoco", scene_a, num_envs, SIM_DT)
    be_b = create_backend("mujoco", scene_b, num_envs, SIM_DT)
    be_a.materialize()
    be_b.materialize()
    yield be_a, be_b
    be_a.cleanup_scene_assets()
    be_b.cleanup_scene_assets()


def test_rollout_parity(rollout_pair):
    """Same home state + same control stream -> near-bitwise trajectories.

    Tolerance rationale: the two composed models carry byte-different XML
    (entity-qualified names, 8-digit text quantization of non-pinned doubles)
    and different geom ordering, so contact evaluation order differs; over
    200 control steps (400 sim steps, contact-rich) the measured divergence is
    dqpos <= 6.2e-6 / dqvel <= 3.9e-4.  The gate pins 25x headroom over the
    measured values — far below any task-observable scale (reset noise is
    1e-1 rad, obs noise 1e-2 m) — while still catching any structural
    regression (those show up at 1e-1+ immediately).
    """
    be_a, be_b = rollout_pair
    be_a.reset()
    be_b.reset()
    qa = np.asarray(be_a._qpos_view)
    qb = np.asarray(be_b._qpos_view)
    va = np.asarray(be_a._qvel_view)
    vb = np.asarray(be_b._qvel_view)
    # The bare backend reset ignores the default keyframe on the whole-model
    # path (keyframe consumption is a manager-layer semantic); pin both sides
    # to the identical home state explicitly.
    home = be_a._model.key_qpos[0]
    np.testing.assert_allclose(be_b._model.key_qpos[0], home, rtol=0, atol=1e-12)
    qa[:] = home
    qb[:] = home
    va[:] = 0.0
    vb[:] = 0.0

    rng = np.random.default_rng(7)
    nu = be_a._model.nu
    max_dq = max_dv = 0.0
    for _ in range(200):
        ctrl = home[:nu][None, :] + rng.uniform(-0.3, 0.3, size=(qa.shape[0], nu))
        be_a.step(ctrl, nsteps=2)  # decimation 2 -> one 1/60 s control step
        be_b.step(ctrl, nsteps=2)
        max_dq = max(max_dq, float(np.abs(qa - qb).max()))
        max_dv = max(max_dv, float(np.abs(va - vb).max()))
    assert max_dq < 2e-5, f"qpos divergence {max_dq:.3e}"
    assert max_dv < 1e-2, f"qvel divergence {max_dv:.3e}"
