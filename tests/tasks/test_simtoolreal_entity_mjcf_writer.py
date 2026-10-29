"""Entity-split MJCF asset tests.

The entity sources are consumed through the official composition channel
(unisim ``entity_assets`` + ``EntityVariantBinding``). These tests pin the
split output's compile contracts, catalog value transcription, and MuJoCo
composition integration including the composed home pose.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from simtoolreal_unilab.tasks.simtoolreal.config import (
    ARM_DEFAULT_JOINT_POS,
    ENTITY_ASSETS_MANIFEST_SCHEMA,
    JOINT_NAMES_CANONICAL,
    TOOL_POOL_TARGET_ENTITY,
)
from simtoolreal_unilab.tasks.simtoolreal.entity_mjcf_writer import (
    write_entity_set,
)
from simtoolreal_unilab.tasks.simtoolreal.tool_catalog import build_tool_catalog

NUM_PER_TYPE = 2  # 24 tools keep the test fast; the full 1200 run is the A-form gate.


@pytest.fixture(scope="module")
def entity_set(tmp_path_factory):
    out = tmp_path_factory.mktemp("entity_set")
    manifest = write_entity_set(out, num_per_type=NUM_PER_TYPE)
    return out, manifest


def test_split_set_compiles_and_carries_home(entity_set):
    out, manifest = entity_set
    robot = mujoco.MjModel.from_xml_path(str(out / "robot.xml"))
    assert (robot.nq, robot.nv, robot.nu, robot.nmesh) == (29, 29, 29, 40)
    assert robot.nkey == 1
    expected_home = [ARM_DEFAULT_JOINT_POS[n] for n in JOINT_NAMES_CANONICAL[:7]] + [0.0] * 22
    np.testing.assert_allclose(robot.key_qpos[0], expected_home, atol=1e-15)
    np.testing.assert_allclose(robot.key_ctrl[0], expected_home, atol=1e-15)
    # Robot stands at the origin in the file; the scene declaration owns the
    # 0.8 m spawn offset (EntityInitialState).
    root = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, "iiwa14_link_0")
    np.testing.assert_allclose(robot.body_pos[root], (0.0, 0.0, 0.0), atol=0)

    table = mujoco.MjModel.from_xml_path(str(out / "table.xml"))
    assert (table.nbody, table.ngeom) == (2, 1)
    gid = mujoco.mj_name2id(table, mujoco.mjtObj.mjOBJ_GEOM, "table_box")
    np.testing.assert_allclose(table.geom_size[gid], (0.2375, 0.2, 0.15), atol=1e-12)
    np.testing.assert_allclose(table.geom_friction[gid], (0.5, 0.005, 0.0001), atol=0)
    np.testing.assert_allclose(table.body_pos[1], (0.0, 0.0, 0.0), atol=0)

    floor = mujoco.MjModel.from_xml_path(str(out / "floor.xml"))
    assert (floor.nbody, floor.ngeom) == (2, 1)
    np.testing.assert_allclose(
        floor.geom_friction[mujoco.mj_name2id(floor, mujoco.mjtObj.mjOBJ_GEOM, "floor")],
        (1.0, 0.005, 0.0001),
        atol=0,
    )

    assert manifest["tool_count"] == 12 * NUM_PER_TYPE
    assert manifest["layout"] == "same_layout"
    assert manifest["schema"] == ENTITY_ASSETS_MANIFEST_SCHEMA
    assert set(manifest["entities"]) == {"robot", "table", "floor", "object"}
    assert manifest["entities"]["robot"]["default_keyframe_name"] == "home"
    assert manifest["entities"][TOOL_POOL_TARGET_ENTITY]["variant_binding"] == {
        "target_entity": TOOL_POOL_TARGET_ENTITY,
        "layout": "same_layout",
    }


def test_diagonalized_inertials_reconstruct_the_same_tensors(entity_set):
    """diaginertia+iquat spelling == the fullinertia tensor, per body.

    The entity robot carries the eigen-decomposed form; the physical tensor
    must match the raw writer output.
    """
    out, _ = entity_set
    from simtoolreal_unilab.tasks.simtoolreal.robot_mjcf_writer import build_robot_mjcf

    raw = mujoco.MjModel.from_xml_string(
        __import__("xml.etree.ElementTree", fromlist=["ET"]).tostring(
            build_robot_mjcf(), encoding="unicode"
        )
    )
    norm = mujoco.MjModel.from_xml_path(str(out / "robot.xml"))
    assert raw.nbody == norm.nbody

    def rotmat(q):
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    for i in range(1, raw.nbody):
        name = mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_BODY, i)
        j = mujoco.mj_name2id(norm, mujoco.mjtObj.mjOBJ_BODY, name)
        assert j >= 0
        raw_tensor = (
            rotmat(raw.body_iquat[i]) @ np.diag(raw.body_inertia[i]) @ rotmat(raw.body_iquat[i]).T
        )
        norm_tensor = (
            rotmat(norm.body_iquat[j])
            @ np.diag(norm.body_inertia[j])
            @ rotmat(norm.body_iquat[j]).T
        )
        # Eigen-solver path difference (LAPACK eigh vs the compiler's
        # closed-form diagonalization): identical to ~1e-11 on 1e-2-scale
        # tensors, far below any physical significance.
        np.testing.assert_allclose(norm_tensor, raw_tensor, rtol=0, atol=1e-10, err_msg=name)
        assert norm.body_mass[j] == raw.body_mass[i]
        np.testing.assert_allclose(norm.body_ipos[j], raw.body_ipos[i], rtol=0, atol=0)


def test_headless_bform_has_no_goalviz_entity(entity_set):
    """GoalViz is intentionally absent from the headless B-form output."""
    _, manifest = entity_set
    assert "goalviz" not in manifest["entities"]
    assert not any("goalviz" in name for name in manifest["tool_files"])


def test_tools_same_layout_and_compile(entity_set):
    out, manifest = entity_set
    layouts = set()
    for name in manifest["tool_files"]:
        model = mujoco.MjModel.from_xml_path(str(out / "tools" / name))
        layouts.add(
            (
                model.ngeom,
                tuple(
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                    for g in range(model.ngeom)
                ),
            )
        )
    assert len(layouts) == 1  # SAME_LAYOUT across every tool variant


def test_tool_files_match_catalog_values(entity_set):
    """Every tool file compiles to its catalog spec's mass/inertia exactly.

    Catalog-order manifest pairing: ``manifest["tool_files"]`` shares the
    catalog order, and the writer transcribes ToolSpec mass/com/diaginertia
    verbatim (the same pure function the URDF line uses).
    """
    out, manifest = entity_set
    catalog = build_tool_catalog(num_per_type=NUM_PER_TYPE, seed=42, shuffle=True)
    assert len(manifest["tool_files"]) == len(catalog)
    for spec, name in zip(catalog, manifest["tool_files"], strict=True):
        model = mujoco.MjModel.from_xml_path(str(out / "tools" / name))
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_root")
        assert model.body_mass[body] == spec.mass
        np.testing.assert_allclose(model.body_inertia[body], spec.diaginertia, rtol=0, atol=0)


def test_composed_scene_home_pose(entity_set):
    """The officially composed B-form scene lands every entity at home."""
    out, manifest = entity_set
    tools = sorted((out / "tools").glob("*.xml"))
    from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
    from unisim.entities import (
        EntityInitialState,
        EntityVariantBinding,
        SceneEntitySpec,
    )
    from unisim.mjcf_compiler import compose_scene
    from unisim.scene import SceneCfg

    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                name="robot",
                source=ModelSourceDescriptor(str(out / "robot.xml")),
                kind="articulation",
                root_mode="fixed",
                initial_state=EntityInitialState(position=(0.0, 0.8, 0.0)),
            ),
            SceneEntitySpec(
                name="table",
                source=ModelSourceDescriptor(str(out / "table.xml")),
                kind="rigid",
                root_mode="kinematic",
                initial_state=EntityInitialState(position=(0.0, 0.0, 0.38)),
            ),
            SceneEntitySpec(
                name="floor",
                source=ModelSourceDescriptor(str(out / "floor.xml")),
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
        ),
        entity_variant=EntityVariantBinding(
            target_entity="object",
            plan=FixedVariantPlan(
                variants=tuple(ModelSourceDescriptor(str(p)) for p in tools),
                assignment=np.arange(4, dtype=np.int32) % len(tools),
                layout=FixedVariantLayout.SAME_LAYOUT,
            ),
        ),
        default_keyframe_name="home",
    )
    composed = compose_scene(scene, 4, 1 / 120)
    try:
        b_model = composed.model
        assert (b_model.nq, b_model.nv, b_model.nu) == (36, 35, 29)
        assert b_model.nexclude == 35  # source-authored excludes carried
        # The headless production shape is exactly four entities.
        goalviz_body = mujoco.mj_name2id(b_model, mujoco.mjtObj.mjOBJ_BODY, "goalviz/object_root")
        assert goalviz_body < 0

        b_data = mujoco.MjData(b_model)
        mujoco.mj_resetDataKeyframe(
            b_model, b_data, mujoco.mj_name2id(b_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        )
        mujoco.mj_forward(b_model, b_data)

        def body_pos(name):
            return b_data.xpos[mujoco.mj_name2id(b_model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()

        # Home FK on the composed scene: frozen spawn poses land exactly.
        np.testing.assert_allclose(body_pos("robot/iiwa14_link_0"), (0.0, 0.8, 0.0), atol=1e-12)
        np.testing.assert_allclose(body_pos("object/object_root"), (0.0, 0.0, 0.63), atol=1e-12)
    finally:
        composed.close()
