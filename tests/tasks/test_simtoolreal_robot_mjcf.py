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
from simtoolreal_unilab.tasks.simtoolreal.owner_physics import OWNER_YAML
from simtoolreal_unilab.tasks.simtoolreal.robot_mjcf_writer import (
    FINGERTIP_SITE_LINKS,
    build_base_scene_mjcf,
    write_base_scene_mjcf,
    write_robot_mjcf,
)
from simtoolreal_unilab.tasks.simtoolreal.tool_catalog import build_tool_catalog
from simtoolreal_unilab.tasks.simtoolreal.tool_pool_mjcf import compose_scene_mjcf

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
def base_scene_xml(tmp_path_factory) -> Path:
    return write_base_scene_mjcf(tmp_path_factory.mktemp("scene_mjcf") / "scene.xml")


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
        62,
    )
    assert (old_model.nq, old_model.nv, old_model.nu, old_model.nmesh) == (29, 29, 29, 62)
    # 31 bodies = world + 30 kept links (43 URDF links minus the 12 welded
    # fixed links + ... ); 5 fingertip frames survive as sites.
    assert robot_model.nbody == 31
    assert robot_model.nsite == 5
    assert robot_model.ngeom == old_model.ngeom == 97
    # Total mass is conserved by the weld (parallel-axis composition).
    assert robot_model.body_mass.sum() == pytest.approx(old_model.body_mass.sum(), rel=1e-12)


def test_base_scene_compile_contract(base_scene_xml):
    model = mujoco.MjModel.from_xml_path(str(base_scene_xml))
    assert (model.nq, model.nv, model.nu, model.nmesh) == (29, 29, 29, 62)
    assert model.ngeom == 99  # robot 97 + floor + table
    assert model.nkey == 1
    assert model.key_qpos.shape[1] == 29
    assert model.opt.timestep == pytest.approx(1.0 / 120.0)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.impratio == pytest.approx(10.0)
    assert model.opt.iterations == 100
    assert model.opt.ls_iterations == 50


def test_scene_ownership_sections(base_scene_xml, robot_xml):
    """Option/keyframe live in the scene only, never in the robot file."""
    import xml.etree.ElementTree as ET

    robot_root = ET.parse(robot_xml).getroot()
    assert not list(robot_root.iter("keyframe"))
    assert not list(robot_root.iter("option"))
    scene_root = ET.parse(base_scene_xml).getroot()
    assert len(list(scene_root.iter("key"))) == 1
    assert len(list(scene_root.iter("option"))) == 1


def test_home_keyframe_matches_config(base_scene_xml):
    model = mujoco.MjModel.from_xml_path(str(base_scene_xml))
    expected = [ARM_DEFAULT_JOINT_POS[name] for name in JOINT_NAMES_CANONICAL[:7]] + [0.0] * 22
    np.testing.assert_allclose(model.key_qpos[0], expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(model.key_ctrl[0], expected, rtol=0, atol=1e-15)


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
    conf = OmegaConf.load(OWNER_YAML)
    overrides = {
        entry.joint_name: entry for entry in conf.env.scene.entity_assets[0].actuator_gain_overrides
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


def test_collision_profile_matches_old_tree_except_slide_friction(robot_model, old_model):
    ours = _collision_profiles(robot_model)
    old = _collision_profiles(old_model)
    assert set(ours) == set(old)  # 61 collision geoms, identical identity set
    assert len(ours) == 61
    fingertip_geom_names = {
        f"left_{f}_fingertip_collision" for f in ("thumb", "index", "middle", "ring", "pinky")
    }
    slide_divergent = []
    for name in ours:
        o, t = ours[name], old[name]
        assert o[:6] == t[:6], f"contact profile drift on {name}: {o[:6]} vs {t[:6]}"
        # Torsion/roll stay at the MuJoCo defaults on both sides.
        assert o[6][1:] == t[6][1:] == (0.005, 0.0001)
        if o[6][0] != t[6][0]:
            slide_divergent.append(name)
            # Exemption (v1 mapping): old tree 1.0 uniformly -> owner YAML
            # values: 0.5 robot-wide, 1.5 on the five fingertip capsules.
            assert t[6][0] == pytest.approx(1.0)
            expected = 1.5 if name in fingertip_geom_names else 0.5
            assert o[6][0] == pytest.approx(expected)
    # 45 hand collision geoms diverge in slide (40 palm/fit/pinky + 5
    # fingertips); the 8 arm meshes agree at 0.5.
    assert len(slide_divergent) == 53


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


def test_step_smoke_no_nan(robot_model, base_scene_xml, tmp_path):
    spec = build_tool_catalog(("hammer",), 1, 42, shuffle=False)[0]
    composed = compose_scene_mjcf(base_scene_xml, spec, tmp_path / "composed.xml")
    for model in (robot_model, mujoco.MjModel.from_xml_path(str(composed))):
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        for _ in range(4):
            mujoco.mj_step(model, data)
            assert np.isfinite(data.qpos).all()
            assert np.isfinite(data.qvel).all()
            assert np.isfinite(data.ctrl).all()
