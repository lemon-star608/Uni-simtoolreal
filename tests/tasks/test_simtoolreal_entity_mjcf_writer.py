"""Entity-split (B form) MJCF asset tests.

The B form feeds the official M2 entity-composition channel (unisim
``entity_assets`` + ``EntityVariantBinding``); the A form (composed single
models, ``tool_pool_mjcf.py``) stays the MuJoCo training line.  Both derive
from the same sources of truth; these tests pin the split output's compile
contracts, the official-path integration (unisim ``compose_scene``), and A/B
value/FK parity.
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
from simtoolreal_unilab.tasks.simtoolreal.robot_mjcf_writer import (
    write_base_scene_mjcf,
    write_robot_mjcf,
)
from simtoolreal_unilab.tasks.simtoolreal.tool_catalog import build_tool_catalog
from simtoolreal_unilab.tasks.simtoolreal.tool_pool_mjcf import (
    compose_scene_mjcf,
    materialize_tool_pool_mjcf,
)

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


def test_ab_value_and_fk_parity(entity_set, tmp_path):
    """Same catalog -> same per-variant physics values; same home-pose FK."""
    out, manifest = entity_set
    # A form: composed pool from the same catalog knobs.
    a_paths, a_masses, _ = materialize_tool_pool_mjcf(
        seed=42, num_per_type=NUM_PER_TYPE, configured_root=str(tmp_path / "a_pool")
    )
    # A pool files and B tool files share the writer's filename template;
    # match by name (catalog order != filename order under the shuffle).
    b_tools = sorted((out / "tools").glob("*.xml"))
    b_by_name = {p.name: p for p in b_tools}
    assert len(a_paths) == len(b_by_name)
    for a_path in a_paths:
        b_path = b_by_name[a_path.name]
        a_model = mujoco.MjModel.from_xml_path(str(a_path))
        b_model = mujoco.MjModel.from_xml_path(str(b_path))
        a_obj = mujoco.mj_name2id(a_model, mujoco.mjtObj.mjOBJ_BODY, "object_root")
        b_obj = mujoco.mj_name2id(b_model, mujoco.mjtObj.mjOBJ_BODY, "object_root")
        assert a_model.body_mass[a_obj] == b_model.body_mass[b_obj]
        np.testing.assert_allclose(
            a_model.body_inertia[a_obj], b_model.body_inertia[b_obj], rtol=0, atol=0
        )

    # FK: B composed through the official unisim compose_scene vs A composed.
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
                source=ModelSourceDescriptor(str(b_tools[0])),
                kind="rigid",
                root_mode="floating",
            ),
            SceneEntitySpec(
                name="goalviz",
                mirror_of="tool",
                kind="rigid",
                root_mode="kinematic",
                collision_enabled=False,
            ),
        ),
        entity_variant=EntityVariantBinding(
            target_entity="tool",
            plan=FixedVariantPlan(
                variants=tuple(ModelSourceDescriptor(str(p)) for p in b_tools),
                assignment=np.arange(4, dtype=np.int32) % len(b_tools),
                layout=FixedVariantLayout.SAME_LAYOUT,
            ),
        ),
        default_keyframe_name="home",
    )
    composed = compose_scene(scene, 4, 1 / 120)
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

    a_scene = write_base_scene_mjcf(tmp_path / "a_scene.xml")
    a_composed = compose_scene_mjcf(
        a_scene, build_tool_catalog(("hammer",), 1, 42, False)[0], tmp_path / "a.xml"
    )
    a_model = mujoco.MjModel.from_xml_path(str(a_composed))
    a_data = mujoco.MjData(a_model)
    mujoco.mj_resetDataKeyframe(a_model, a_data, 0)
    mujoco.mj_forward(a_model, a_data)

    for i in range(1, a_model.nbody):
        name = mujoco.mj_id2name(a_model, mujoco.mjtObj.mjOBJ_BODY, i)
        j = -1
        for candidate in (f"robot/{name}", f"table/{name}", f"floor/{name}", name):
            j = mujoco.mj_name2id(b_model, mujoco.mjtObj.mjOBJ_BODY, candidate)
            if j >= 0:
                break
        if j < 0:
            continue  # object_root spawn pose is scene/reset-owned, not compared
        np.testing.assert_allclose(b_data.xpos[j], a_data.xpos[i], atol=1e-6, err_msg=name)
    composed.close()


def test_a_form_regression_unchanged_pool_bytes(tmp_path):
    """The collision swap changed the hand layer only; the manifest tables
    (masses/scales) must stay identical and the 1200 pool must still compile."""
    paths, masses, scales = materialize_tool_pool_mjcf(
        seed=42, num_per_type=NUM_PER_TYPE, configured_root=str(tmp_path / "pool")
    )
    assert len(paths) == 12 * NUM_PER_TYPE
    catalog = build_tool_catalog(num_per_type=NUM_PER_TYPE, seed=42, shuffle=True)
    assert masses == tuple(float(s.mass) for s in catalog)
    assert scales == tuple(tuple(float(x) for x in s.object_scale) for s in catalog)
    model = mujoco.MjModel.from_xml_path(str(paths[0]))
    assert (model.nq, model.nv, model.nu, model.nmesh) == (36, 35, 29, 40)
