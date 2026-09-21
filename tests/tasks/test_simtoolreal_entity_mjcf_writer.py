"""Entity-split (B form) MJCF asset tests.

The B form is the MuJoCo line's asset path (M-J5), consumed through the
official M2 entity-composition channel (unisim ``entity_assets`` +
``EntityVariantBinding``).  These tests pin the split output's compile
contracts, the catalog value transcription, and the official-path
integration (unisim ``compose_scene``) including the composed home pose.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from simtoolreal_unilab.tasks.simtoolreal.config import (
    ARM_DEFAULT_JOINT_POS,
    JOINT_NAMES_CANONICAL,
)
from simtoolreal_unilab.tasks.simtoolreal.entity_mjcf_writer import (
    GOALVIZ_MIRROR_DECLARATION,
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
    assert manifest["entities"]["robot"]["default_keyframe_name"] == "home"


def test_goalviz_mirror_declaration_matches_m2_shape():
    """The mirror declaration validates against unisim's own rules."""
    from unisim.entities import SceneEntitySpec, validate_entity_declarations

    tool = SceneEntitySpec(
        name="tool",
        source=__import__(
            "unisim.dr.types", fromlist=["ModelSourceDescriptor"]
        ).ModelSourceDescriptor("tool_000.xml"),
        kind="rigid",
        root_mode="floating",
    )
    mirror = SceneEntitySpec(**GOALVIZ_MIRROR_DECLARATION)
    validate_entity_declarations((tool, mirror), None)


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
                name="tool",
                source=ModelSourceDescriptor(str(tools[0])),
                kind="rigid",
                root_mode="floating",
                initial_state=EntityInitialState(position=(0.0, 0.0, 0.63)),
            ),
            SceneEntitySpec(
                name="goalviz",
                mirror_of="tool",
                kind="rigid",
                root_mode="kinematic",
                collision_enabled=False,
                initial_state=EntityInitialState(position=(0.0, 0.0, 0.63)),
            ),
        ),
        entity_variant=EntityVariantBinding(
            target_entity="tool",
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
        # Mirror present, collision-free.
        mirror_body = mujoco.mj_name2id(b_model, mujoco.mjtObj.mjOBJ_BODY, "goalviz/object_root")
        assert mirror_body >= 1

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
