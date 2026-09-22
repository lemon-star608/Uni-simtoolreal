"""Compile-contract and parity tests for the robot/scene MJCF writer.

Three anchors:
1. The vendored URDF compiled by MuJoCo itself (structure/merge authority —
   the compiler is its own format authority, roadmap M-J1 D5).
2. The old tree's training-validated ``kuka_sharpa.xml`` (ground truth for
   the compiled contract: nq/nv/nu/nmesh, joint/actuator order, contact
   profile, FK).
3. The owner YAML (``conf/rlgames_sapg/task/simtoolreal/base.yaml``) for
   actuator gains, armature, and the v1 friction mapping.

Known, user-approved divergences from the old tree, pinned explicitly below:
- sliding friction: old tree unified hand collision slide at 1.0; the v1
  mapping writes the owner YAML's frozen values (robot 0.5 everywhere except
  the five fingertip capsules at 1.5).  Torsion/roll (0.005/0.0001) match.
- hand joint frictionloss: old tree carried HAND_JOINT_FRICTION values; this
  repository's config.py retires that table as dead code, so joints compile
  with frictionloss 0.
- joint ranges: transcribed at URDF full precision; the old tree's MJCF was
  rounded to 6 significant digits by mj_saveLastXML.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from omegaconf import OmegaConf

from simtoolreal_unilab.assets import ASSET_ROOT
from simtoolreal_unilab.tasks.simtoolreal.config import (
    ARM_DEFAULT_JOINT_POS,
    JOINT_NAMES_CANONICAL,
)
from simtoolreal_unilab.tasks.simtoolreal.owner_physics import BASE_YAML
from simtoolreal_unilab.tasks.simtoolreal.robot_mjcf_writer import (
    FINGERTIP_SITE_LINKS,
    write_robot_mjcf,
)
from simtoolreal_unilab.tasks.simtoolreal.tool_catalog import build_tool_catalog

WORKSPACE = Path(__file__).resolve().parents[3]
OLD_TREE_KUKA_MJCF = (
    WORKSPACE / "simtoolreal-unilab-old/UniLab/src/unilab/assets/robots/kuka_sharpa/kuka_sharpa.xml"
)
ROBOT_URDF = ASSET_ROOT / "urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"

EXPECTED_JOINTS = JOINT_NAMES_CANONICAL
EXPECTED_ACTUATORS = tuple(f"{name}_ctrl" for name in EXPECTED_JOINTS)
FINGERTIP_BODY_NAMES = (
    "left_thumb_DP",
    "left_index_DP",
    "left_middle_DP",
    "left_ring_DP",
    "left_pinky_DP",
)


@pytest.fixture(scope="module")
def robot_xml(tmp_path_factory) -> Path:
    return write_robot_mjcf(tmp_path_factory.mktemp("robot_mjcf") / "kuka_sharpa.xml")


@pytest.fixture(scope="module")
def robot_model(robot_xml) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(robot_xml))


@pytest.fixture(scope="module")
def old_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(OLD_TREE_KUKA_MJCF))


@pytest.fixture(scope="module")
def urdf_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(ROBOT_URDF))


@pytest.fixture(scope="module")
def entity_set_dir(tmp_path_factory) -> Path:
    """The B-form entity bootstrap sources (robot/table/floor + object)."""
    from simtoolreal_unilab.tasks.simtoolreal.entity_mjcf_writer import (
        write_bootstrap_entity_set,
    )

    root = tmp_path_factory.mktemp("entity_bootstrap")
    write_bootstrap_entity_set(root)
    return root


@pytest.fixture(scope="module")
def composed_model(entity_set_dir):
    """The B-form scene through the official unisim compose_scene (M-J5).

    Replaces the retired A-form task-side composer: robot + table + floor +
    bootstrap object composed as entities, home keyframe merged by name.
    """
    from unisim.dr.types import ModelSourceDescriptor
    from unisim.entities import EntityInitialState, SceneEntitySpec
    from unisim.mjcf_compiler import compose_scene
    from unisim.scene import SceneCfg

    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                name="robot",
                source=ModelSourceDescriptor(str(entity_set_dir / "robot.xml")),
                kind="articulation",
                root_mode="fixed",
                initial_state=EntityInitialState(position=(0.0, 0.8, 0.0)),
            ),
            SceneEntitySpec(
                name="table",
                source=ModelSourceDescriptor(str(entity_set_dir / "table.xml")),
                kind="rigid",
                root_mode="kinematic",
                initial_state=EntityInitialState(position=(0.0, 0.0, 0.38)),
            ),
            SceneEntitySpec(
                name="floor",
                source=ModelSourceDescriptor(str(entity_set_dir / "floor.xml")),
                kind="rigid",
                root_mode="kinematic",
            ),
            SceneEntitySpec(
                name="object",
                source=ModelSourceDescriptor(str(entity_set_dir / "object_bootstrap.xml")),
                kind="rigid",
                root_mode="floating",
                initial_state=EntityInitialState(position=(0.0, 0.0, 0.63)),
            ),
        ),
        default_keyframe_name="home",
    )
    composed = compose_scene(scene, 1, 1 / 120)
    yield composed.model
    composed.close()


def _body_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]


# ----------------------------------------------------------------------------
# Compile contract
# ----------------------------------------------------------------------------


def test_robot_compile_contract(robot_model, old_model):
    assert (robot_model.nq, robot_model.nv, robot_model.nu, robot_model.nmesh) == (
        29,
        29,
        29,
        40,
    )
    assert (old_model.nq, old_model.nv, old_model.nu, old_model.nmesh) == (29, 29, 29, 62)
    # 31 bodies = world + 30 kept links (43 URDF links minus the 12 welded
    # fixed links + ... ); 5 fingertip frames survive as sites.
    assert robot_model.nbody == 31
    assert robot_model.nsite == 5
    # 70 geoms = 16 arm (vis+col) + 28 hand visual + 26 hand URDF collision.
    # (The retired Menagerie layer was 97; the old tree's count stays pinned
    # on the right-hand side as the historical record.)
    assert robot_model.ngeom == 70
    assert old_model.ngeom == 97
    # Total mass is conserved by the weld (parallel-axis composition).
    assert robot_model.body_mass.sum() == pytest.approx(old_model.body_mass.sum(), rel=1e-12)


def test_composed_compile_contract(composed_model):
    """The B-form composed scene (official compose_scene) keeps the contract."""
    model = composed_model
    assert (model.nq, model.nv, model.nu, model.nmesh) == (36, 35, 29, 40)
    # 72 robot+floor+table geoms + 4 object geoms (handle/head, vis/col).
    assert model.ngeom == 76
    assert model.nkey == 1
    assert model.key_qpos.shape[1] == 36
    # The composer's XML round-trip quantizes timestep to 6 significant
    # digits; the backend overwrites opt.timestep with the double sim_dt at
    # build time, so this never reaches the runtime.
    assert model.opt.timestep == pytest.approx(1.0 / 120.0, abs=1e-8)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.impratio == pytest.approx(10.0)
    assert model.opt.iterations == 100
    assert model.opt.ls_iterations == 50


def test_scene_ownership_sections(entity_set_dir, robot_xml):
    """The bare robot file carries no option/keyframe; the entity robot does."""
    import xml.etree.ElementTree as ET

    robot_root = ET.parse(robot_xml).getroot()
    assert not list(robot_root.iter("keyframe"))
    assert not list(robot_root.iter("option"))
    entity_root = ET.parse(entity_set_dir / "robot.xml").getroot()
    assert len(list(entity_root.iter("key"))) == 1
    assert len(list(entity_root.iter("option"))) == 1


def test_home_keyframe_matches_config(composed_model):
    model = composed_model
    expected = [ARM_DEFAULT_JOINT_POS[name] for name in JOINT_NAMES_CANONICAL[:7]] + [0.0] * 22
    # Robot segment exact; the object freejoint segment is the home spawn pose.
    np.testing.assert_allclose(model.key_qpos[0][:29], expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        model.key_qpos[0][29:36], [0.0, 0.0, 0.63, 1.0, 0.0, 0.0, 0.0], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(model.key_ctrl[0][:29], expected, rtol=0, atol=1e-12)


# ----------------------------------------------------------------------------
# Joint and actuator ordering + gains
# ----------------------------------------------------------------------------


def test_joint_order_is_canonical(robot_model):
    joints = [
        mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(robot_model.njnt)
    ]
    assert joints == list(EXPECTED_JOINTS)


def test_actuator_order_is_canonical(robot_model):
    actuators = [
        mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        for a in range(robot_model.nu)
    ]
    assert actuators == list(EXPECTED_ACTUATORS)
    trntype = robot_model.actuator_trntype
    assert (trntype == mujoco.mjtTrn.mjTRN_JOINT).all()


def test_actuator_gains_match_owner_yaml(robot_model):
    conf = OmegaConf.load(BASE_YAML)
    overrides = {
        entry.joint_name: entry for entry in conf.owner_physics.robot.actuator_gain_overrides
    }
    assert set(overrides) == set(EXPECTED_JOINTS)
    for a in range(robot_model.nu):
        joint_name = mujoco.mj_id2name(
            robot_model, mujoco.mjtObj.mjOBJ_JOINT, robot_model.actuator_trnid[a, 0]
        )
        entry = overrides[joint_name]
        kp = robot_model.actuator_gainprm[a, 0]
        kv = -robot_model.actuator_biasprm[a, 2]
        assert kp == pytest.approx(float(entry.stiffness), rel=1e-12)
        assert kv == pytest.approx(float(entry.damping), rel=1e-12)
        dof = robot_model.jnt_dofadr[robot_model.actuator_trnid[a, 0]]
        expected_armature = float(entry.get("armature", 0.0))
        assert robot_model.dof_armature[dof] == pytest.approx(expected_armature, rel=1e-9)


def test_joint_ranges_transcribe_urdf(robot_model, old_model):
    import xml.etree.ElementTree as ET

    root = ET.parse(ROBOT_URDF).getroot()
    limits = {
        joint.get("name"): (
            float(joint.find("limit").get("lower")),
            float(joint.find("limit").get("upper")),
        )
        for joint in root.findall("joint")
        if joint.get("type") == "revolute"
    }
    for j in range(robot_model.njnt):
        name = mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        np.testing.assert_allclose(robot_model.jnt_range[j], limits[name], rtol=0, atol=0)
        # Old tree rounded to 6 significant digits (mj_saveLastXML).
        old_j = mujoco.mj_name2id(old_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        np.testing.assert_allclose(robot_model.jnt_range[j], old_model.jnt_range[old_j], atol=1e-5)


def test_composed_scene_preserves_joint_ranges(entity_set_dir, composed_model):
    """Every entity source and the composed model must carry radian ranges.

    Regression gate for the 2026-09-17 forensic finding (a composer that
    dropped the robot file's <compiler angle="radian"> compiled every joint
    range into a ±0.05 rad cage).  On the B form the risk surface is the
    entity sources; all of them carry the declaration and the official
    compose preserves the radian-compiled ranges (up to its 6-significant-
    digit XML round-trip).
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(ROBOT_URDF).getroot()
    limits = {
        joint.get("name"): (
            float(joint.find("limit").get("lower")),
            float(joint.find("limit").get("upper")),
        )
        for joint in root.findall("joint")
        if joint.get("type") == "revolute"
    }
    for name in ("robot.xml", "table.xml", "floor.xml", "object_bootstrap.xml"):
        source_root = ET.parse(entity_set_dir / name).getroot()
        compiler = source_root.find("compiler")
        assert compiler is not None and compiler.get("angle") == "radian", name
    for label, model in (
        ("entity_robot", mujoco.MjModel.from_xml_path(str(entity_set_dir / "robot.xml"))),
        ("composed", composed_model),
    ):
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            bare = name.rsplit("/", 1)[-1]
            if bare == "object_root_joint":
                continue
            np.testing.assert_allclose(
                model.jnt_range[j], limits[bare], rtol=1e-5, atol=1e-7, err_msg=f"{label}:{name}"
            )
            # inheritrange=1: actuator ctrlrange tracks the joint range (the
            # compiler's copy can differ by 1 ulp; the joint range itself is
            # asserted to the compose quantization bound above).
            act_name = f"robot/{bare}_ctrl" if "/" in name else f"{bare}_ctrl"
            act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            assert act >= 0, f"{label}:{act_name}"
            np.testing.assert_allclose(
                model.actuator_ctrlrange[act],
                limits[bare],
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"{label}:{act_name}",
            )


def test_fingertip_sites_exist(robot_model):
    sites = {
        mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_SITE, i)
        for i in range(robot_model.nsite)
    }
    assert sites == set(FINGERTIP_SITE_LINKS)
    # Sites sit on their DP ancestor bodies (elastomer/fingertip links welded).
    for i in range(robot_model.nsite):
        name = mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_SITE, i)
        finger = name.removeprefix("left_").removesuffix("_fingertip")
        body = mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_BODY, robot_model.site_bodyid[i])
        assert body == f"left_{finger}_DP"


def test_weld_merged_bodies_absent(robot_model):
    names = set(_body_names(robot_model))
    merged = {
        "iiwa14_link_ee",
        "sharpa_mount",
        "left_hand_C_MC",
        *(f"left_{f}_elastomer" for f in ("thumb", "index", "middle", "ring", "pinky")),
        *(f"left_{f}_fingertip" for f in ("thumb", "index", "middle", "ring", "pinky")),
    }
    assert names.isdisjoint(merged)


# ----------------------------------------------------------------------------
# Mass/inertia merge validated against MuJoCo's own URDF compilation
# ----------------------------------------------------------------------------


def test_body_masses_match_urdf_compile(robot_model, urdf_model):
    """The writer's weld composition equals the URDF compiler's own merge."""
    for i in range(1, robot_model.nbody):
        name = mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_BODY, i)
        j = mujoco.mj_name2id(urdf_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if j < 1:
            # iiwa14_link_0: the URDF compiler welds the fixed base into the
            # world; ours keeps it as a mass-carrying static body.
            assert name == "iiwa14_link_0"
            assert robot_model.body_mass[i] == pytest.approx(5.0)
            continue
        assert robot_model.body_mass[i] == urdf_model.body_mass[j]
        np.testing.assert_allclose(robot_model.body_ipos[i], urdf_model.body_ipos[j], atol=1e-12)


# ----------------------------------------------------------------------------
# Collision profile parity vs the old tree (sliding friction exempt, see header)
# ----------------------------------------------------------------------------


def _collision_profiles(model: mujoco.MjModel) -> dict[str, tuple]:
    """Key collision geoms by mesh identity when present, else by geom name.

    (Our writer names every geom — the mjbatch variant pack requires it — while
    the old tree left the arm mesh geoms unnamed; the mesh identity is the
    shared key on both sides.)
    """
    profiles: dict[str, tuple] = {}
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        mesh_id = model.geom_dataid[g]
        if mesh_id >= 0:
            name = f"mesh:{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)}"
        else:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            assert name, f"collision geom {g} has neither mesh nor name"
        profiles[name] = (
            int(model.geom_contype[g]),
            int(model.geom_conaffinity[g]),
            int(model.geom_group[g]),
            int(model.geom_condim[g]),
            tuple(model.geom_solimp[g]),
            tuple(model.geom_solref[g]),
            tuple(model.geom_friction[g]),
        )
    return profiles


def test_collision_profile_urdf_mesh_layer(robot_model, urdf_model, robot_xml):
    """The hand collision layer is the URDF's own collision meshes.

    After the Menagerie retirement (2026-09-21), collision geometry authority
    is the vendored URDF's <collision> elements: identical mesh bytes as the
    URDF-compiled model, with the writer's hand contact profile (condim/
    solimp/solref/group) and the owner-YAML slide split (0.5 robot-wide, 1.5
    on the five DP weld groups incl. their elastomer pads).
    """

    # Geometry identity by vertex content: every collision geom in the
    # URDF-compiled reference must have a byte-equal mesh among OUR collision
    # geoms (both sides load the same vendored STL files; arm visual/collision
    # stems collide by name, so content is the robust key).
    def mesh_verts(model, mesh_id):
        adr = model.mesh_vertadr[mesh_id]
        num = model.mesh_vertnum[mesh_id]
        return np.asarray(model.mesh_vert[adr : adr + num])

    our_collision_verts = []
    for g in range(robot_model.ngeom):
        if robot_model.geom_contype[g] == 0 and robot_model.geom_conaffinity[g] == 0:
            continue
        mesh_id = robot_model.geom_dataid[g]
        assert mesh_id >= 0, "all collision geoms are mesh geoms after the swap"
        our_collision_verts.append(mesh_verts(robot_model, mesh_id))
    matched = 0
    for g in range(urdf_model.ngeom):
        if urdf_model.geom_contype[g] == 0 and urdf_model.geom_conaffinity[g] == 0:
            continue
        mesh_id = urdf_model.geom_dataid[g]
        if mesh_id < 0:
            continue
        ref = mesh_verts(urdf_model, mesh_id)
        assert any(
            our.shape == ref.shape and np.array_equal(our, ref) for our in our_collision_verts
        ), mujoco.mj_id2name(urdf_model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        matched += 1
    assert matched == 34  # 8 arm + 26 hand collision geoms, byte-identical meshes

    # Contact profile + slide placement, per geom.
    slide_hi = []
    for g in range(robot_model.ngeom):
        if robot_model.geom_contype[g] == 0 and robot_model.geom_conaffinity[g] == 0:
            continue
        name = mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_GEOM, g)
        body = mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_BODY, robot_model.geom_bodyid[g])
        is_hand = body.startswith("left_")
        if is_hand:
            assert tuple(robot_model.geom_solimp[g]) == (0.9, 0.95, 0.001, 0.5, 2.0), name
            assert tuple(robot_model.geom_solref[g]) == (0.02, 1.0), name
            assert robot_model.geom_group[g] == 3, name
            assert robot_model.geom_condim[g] == 3, name
        assert tuple(robot_model.geom_friction[g][1:]) == (0.005, 0.0001), name
        slide = robot_model.geom_friction[g][0]
        if abs(slide - 1.5) < 1e-9:
            assert body in {
                "left_thumb_DP",
                "left_index_DP",
                "left_middle_DP",
                "left_ring_DP",
                "left_pinky_DP",
            }, (name, body)
            slide_hi.append(name)
        else:
            assert slide == pytest.approx(0.5), name
    # DP mesh + welded elastomer pad on each of the five DP bodies.
    assert len(slide_hi) == 10


def test_exclude_pairs_exist_and_target_kept_bodies(robot_xml, robot_model, old_model):
    # 50 transcribed pairs; remapping through the weld map drops 15 (merged
    # endpoints re-point to their kept ancestor, duplicates and self-pairs
    # drop) -> 35.  The old tree keeps all 50 (it welds nothing).
    assert robot_model.nexclude == 35
    assert old_model.nexclude == 50
    import xml.etree.ElementTree as ET

    body_names = set(_body_names(robot_model))
    excludes = list(ET.parse(robot_xml).getroot().iter("exclude"))
    assert len(excludes) == 35
    for element in excludes:
        b1, b2 = element.get("body1"), element.get("body2")
        assert b1 != b2
        assert b1 in body_names and b2 in body_names


# ----------------------------------------------------------------------------
# FK parity: this writer vs compiled URDF vs old tree ground truth
# ----------------------------------------------------------------------------


def _shared_bodies(a: mujoco.MjModel, b: mujoco.MjModel) -> list[str]:
    return sorted((set(_body_names(a)) & set(_body_names(b))) - {"world"})


def _random_qpos(rng: np.random.RandomState, model: mujoco.MjModel) -> np.ndarray:
    return np.array([rng.uniform(*model.jnt_range[j]) for j in range(model.njnt)])


def _assert_fk_close(a, b, qa, qb, offset_a=(0.0, 0.0, 0.0), tol=1e-6):
    da, db = mujoco.MjData(a), mujoco.MjData(b)
    da.qpos[:] = qa
    db.qpos[:] = qb
    mujoco.mj_forward(a, da)
    mujoco.mj_forward(b, db)
    offset = np.asarray(offset_a)
    for name in _shared_bodies(a, b):
        ia = mujoco.mj_name2id(a, mujoco.mjtObj.mjOBJ_BODY, name)
        ib = mujoco.mj_name2id(b, mujoco.mjtObj.mjOBJ_BODY, name)
        np.testing.assert_allclose(
            da.xpos[ia] + offset, db.xpos[ib], atol=tol, err_msg=f"body {name}"
        )
        quat_a, quat_b = da.xquat[ia], db.xquat[ib]
        assert min(np.abs(quat_a - quat_b).max(), np.abs(quat_a + quat_b).max()) < tol, (
            f"body {name} quat"
        )


def test_fk_parity_three_way(robot_model, urdf_model, old_model):
    rng = np.random.RandomState(42)
    for _ in range(100):
        q = _random_qpos(rng, robot_model)
        # Old tree's joint ranges are rounded slightly tighter in places;
        # clip into its range so all three models see a legal qpos.
        q_old = np.array([np.clip(q[j], *old_model.jnt_range[j]) for j in range(old_model.njnt)])
        # vs compiled URDF: both at the world origin, full precision.
        _assert_fk_close(robot_model, urdf_model, q, q, tol=1e-9)
        # vs old tree: its root body carries the 0.8 m spawn offset.
        _assert_fk_close(robot_model, old_model, q, q_old, offset_a=(0.0, 0.8, 0.0), tol=1e-5)


def test_fingertip_sites_match_old_tree_fingertip_bodies(robot_model, old_model):
    rng = np.random.RandomState(7)
    da = mujoco.MjData(robot_model)
    db = mujoco.MjData(old_model)
    for _ in range(100):
        q = _random_qpos(rng, robot_model)
        q_old = np.array([np.clip(q[j], *old_model.jnt_range[j]) for j in range(old_model.njnt)])
        da.qpos[:] = q
        db.qpos[:] = q_old
        mujoco.mj_forward(robot_model, da)
        mujoco.mj_forward(old_model, db)
        for site_name in FINGERTIP_SITE_LINKS:
            sid = mujoco.mj_name2id(robot_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            bid = mujoco.mj_name2id(old_model, mujoco.mjtObj.mjOBJ_BODY, site_name)
            np.testing.assert_allclose(
                da.site_xpos[sid] + (0.0, 0.8, 0.0),
                db.xpos[bid],
                atol=1e-5,
                err_msg=site_name,
            )


def test_step_smoke_no_nan(robot_model, composed_model):
    for model in (robot_model, composed_model):
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        for _ in range(4):
            mujoco.mj_step(model, data)
            assert np.isfinite(data.qpos).all()
            assert np.isfinite(data.qvel).all()
            assert np.isfinite(data.ctrl).all()


def test_grip_hold_on_urdf_mesh_collision_layer(composed_model):
    """A closed fist retains an object under gravity on the URDF-mesh layer.

    Scripted grip probe (baseline from the 2026-09-19 Menagerie-layer probe:
    escape_dist 0.020, HELD): close the fist at home, insert the object at the
    palm center, hold 2 s.  The DP-mesh + elastomer contact surfaces (the
    owner's 1.5-slide fingertip weld groups) must keep it caged.  Runs on the
    B-form composed scene (M-J5; entity-qualified names).
    """
    model = composed_model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    )
    ctrl = np.zeros(model.nu)
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name is None or name == "object/object_root_joint":
            continue
        bare = name.rsplit("/", 1)[-1]
        act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot/{bare}_ctrl")
        lo, hi = model.jnt_range[j]
        ctrl[act] = (lo + 0.9 * (hi - lo)) if bare.startswith("left_") else data.qpos[j]
    for _ in range(60):  # close the fist
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/iiwa14_link_7")
    center = data.xpos[palm_id] + data.xmat[palm_id].reshape(3, 3) @ np.asarray((0.0, -0.02, 0.14))
    obj_adr = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object/object_root_joint")
    ]
    data.qpos[obj_adr : obj_adr + 3] = center
    data.qpos[obj_adr + 3 : obj_adr + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    contacts_at_insert = data.ncon
    for _ in range(240):  # hold 2 s
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
    escape = np.linalg.norm(data.qpos[obj_adr : obj_adr + 3] - center)
    assert contacts_at_insert > 0  # the grip is actually touching
    assert escape < 0.08, f"object escaped the closed fist by {escape:.3f} m"
