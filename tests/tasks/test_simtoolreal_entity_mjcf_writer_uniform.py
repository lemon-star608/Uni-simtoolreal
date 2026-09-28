"""Uniform-public-layout (D8) entity-set, pool-parity, and contract gates.

Pins the layout-parameterized entity writer: the uniform pool carries the
same ToolSpec truth as the same-layout pool (mass/com/diaginertia/scale are
layout-independent), every tool compiles, erasers omit the collision head
slot, and a
uniform subset composes under ``FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT``
through unisim's official validation path.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from simtoolreal_unilab.tasks.simtoolreal.entity_mjcf_writer import (
    TOOL_POOL_LAYOUT_SAME,
    TOOL_POOL_LAYOUT_UNIFORM,
    write_bootstrap_entity_set,
    write_entity_set,
)
from simtoolreal_unilab.tasks.simtoolreal.tool_catalog import build_tool_catalog

NUM_PER_TYPE = 2  # 24 tools keep routine runs fast; the full 1200 has its own gate.


def _geom_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom))


@pytest.fixture(scope="module")
def uniform_set(tmp_path_factory):
    out = tmp_path_factory.mktemp("uniform_entity_set")
    manifest = write_entity_set(out, num_per_type=NUM_PER_TYPE, layout=TOOL_POOL_LAYOUT_UNIFORM)
    return out, manifest


def test_uniform_set_manifest_records_the_layout(uniform_set):
    out, manifest = uniform_set
    assert manifest["layout"] == TOOL_POOL_LAYOUT_UNIFORM
    assert manifest["entities"]["object"]["variant_binding"] == {
        "target_entity": "object",
        "layout": TOOL_POOL_LAYOUT_UNIFORM,
    }
    assert manifest["tool_count"] == 12 * NUM_PER_TYPE
    # Robot/table/floor are layout-independent; the same digest contract holds.
    assert set(manifest["digests"]) == {"robot.xml", "table.xml", "floor.xml"}
    # File-backed mesh companions are recorded with digests (IsaacSim cannot
    # consume inline meshes); erasers contribute no head mesh.
    assert manifest["tool_meshes"]
    meshes_dir = out / "tools" / "meshes"
    assert set(manifest["tool_meshes"]) == {p.name for p in meshes_dir.glob("*.obj")}
    handles = [n for n in manifest["tool_meshes"] if n.endswith("_handle.obj")]
    heads = [n for n in manifest["tool_meshes"] if n.endswith("_head.obj")]
    assert len(handles) == 12 * NUM_PER_TYPE
    # One head OBJ per headed tool plus the single placeholder shared by all
    # erasers (content-addressed naming dedups it).
    assert len(heads) == 11 * NUM_PER_TYPE + 1


def test_uniform_tools_compile_mesh_only_erasers_headless(uniform_set):
    out, manifest = uniform_set
    catalog = build_tool_catalog(num_per_type=NUM_PER_TYPE, seed=42, shuffle=True)
    erasers = 0
    for spec, name in zip(catalog, manifest["tool_files"], strict=True):
        model = mujoco.MjModel.from_xml_path(str(out / "tools" / name))
        names = _geom_names(model)
        assert all(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH for g in range(model.ngeom))
        if spec.head_scale is None:
            erasers += 1
            # Only the collision head slot is omitted; the visual slot stays
            # as the shared transparent placeholder (IsaacSim PhysX
            # homogeneity, 2026-09-28).
            assert names == ("object_handle_col", "object_handle_vis", "object_head_vis")
        else:
            assert names == (
                "object_handle_col",
                "object_handle_vis",
                "object_head_col",
                "object_head_vis",
            )
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_root")
        assert model.body_mass[body] == spec.mass
    assert erasers == NUM_PER_TYPE  # one eraser distribution per catalog round


def test_uniform_bootstrap_object_matches_the_pool_layout(tmp_path):
    """The uniform contract validates the base source against the canonical
    variant, so the bootstrap hammer must be uniform-meshed too — including
    its file-backed mesh companions (IsaacSim importer)."""
    paths = write_bootstrap_entity_set(tmp_path / "boot", layout=TOOL_POOL_LAYOUT_UNIFORM)
    model = mujoco.MjModel.from_xml_path(str(paths["object"]))
    assert _geom_names(model) == (
        "object_handle_col",
        "object_handle_vis",
        "object_head_col",
        "object_head_vis",
    )
    assert all(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH for g in range(model.ngeom))
    mesh_paths = [p for key, p in paths.items() if key.startswith("mesh:")]
    assert len(mesh_paths) == 2  # headed hammer: handle + head OBJs
    assert all(p.is_file() for p in mesh_paths)
    # The resolver's reuse check sees the complete set.
    from simtoolreal_unilab.tasks.simtoolreal.entity_mjcf_writer import (
        bootstrap_entity_set_paths,
    )

    assert bootstrap_entity_set_paths(tmp_path / "boot", layout=TOOL_POOL_LAYOUT_UNIFORM) == paths
    # Layout-independent entities are untouched.
    for name in ("robot", "table", "floor"):
        assert mujoco.MjModel.from_xml_path(str(paths[name])).ngeom > 0


def test_entity_set_layout_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="tool pool layout must be one of"):
        write_entity_set(tmp_path / "bad", num_per_type=1, layout="uniform_primitive")
    with pytest.raises(ValueError, match="tool pool layout must be one of"):
        write_bootstrap_entity_set(tmp_path / "bad_boot", layout="uniform_primitive")


def test_layouts_share_static_entities(tmp_path):
    """Robot/table/floor are byte-identical across layouts; only tools differ."""
    same = write_entity_set(tmp_path / "same", num_per_type=1, layout=TOOL_POOL_LAYOUT_SAME)
    uniform = write_entity_set(
        tmp_path / "uniform", num_per_type=1, layout=TOOL_POOL_LAYOUT_UNIFORM
    )
    assert same["digests"] == uniform["digests"]
    assert same["tool_files"] == uniform["tool_files"]  # same filenames, new realization
    assert same["tool_digests"] != uniform["tool_digests"]


def test_full_pool_parity_between_layouts(tmp_path):
    """Pool parity gate: 1200/1200 tools compile under both layouts, per-tool
    mass/com/diaginertia/object_scale exactly equal (one ToolSpec truth), and
    exactly 100 headless erasers in the uniform pool."""
    same_manifest = write_entity_set(tmp_path / "same", layout=TOOL_POOL_LAYOUT_SAME)
    uniform_manifest = write_entity_set(tmp_path / "uniform", layout=TOOL_POOL_LAYOUT_UNIFORM)
    catalog = build_tool_catalog(num_per_type=100, seed=42, shuffle=True)
    assert len(catalog) == 1200 == len(same_manifest["tool_files"])
    assert same_manifest["tool_files"] == uniform_manifest["tool_files"]

    headless = 0
    for spec, name in zip(catalog, uniform_manifest["tool_files"], strict=True):
        same_model = mujoco.MjModel.from_xml_path(str(tmp_path / "same" / "tools" / name))
        uniform_model = mujoco.MjModel.from_xml_path(str(tmp_path / "uniform" / "tools" / name))
        body = mujoco.mj_name2id(uniform_model, mujoco.mjtObj.mjOBJ_BODY, "object_root")
        assert uniform_model.body_mass[body] == spec.mass
        np.testing.assert_allclose(uniform_model.body_ipos[body], spec.com, rtol=0, atol=0)
        np.testing.assert_allclose(
            uniform_model.body_inertia[body], spec.diaginertia, rtol=0, atol=0
        )
        same_body = mujoco.mj_name2id(same_model, mujoco.mjtObj.mjOBJ_BODY, "object_root")
        assert same_model.body_mass[same_body] == uniform_model.body_mass[body]
        np.testing.assert_allclose(
            same_model.body_ipos[same_body], uniform_model.body_ipos[body], rtol=0, atol=0
        )
        np.testing.assert_allclose(
            same_model.body_inertia[same_body], uniform_model.body_inertia[body], rtol=0, atol=0
        )
        headless += int(uniform_model.ngeom == 3)  # erasers omit the collision head slot
    assert headless == 100


def test_uniform_subset_composes_under_uniform_public_layout(tmp_path):
    """Contract gate: unisim composes and validates the uniform pool subset
    (bootstrap base source + 12 variants, a few envs) under
    UNIFORM_PUBLIC_LAYOUT; eraser variants lack the head geoms, headed
    variants carry them, and the canonical model's object mass follows the
    catalog."""
    from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
    from unisim.entities import EntityInitialState, EntityVariantBinding, SceneEntitySpec
    from unisim.mjcf_compiler import (
        UniformVariantLayoutSummary,
        compose_scene,
        validate_uniform_entity_variant_layout_summaries,
    )
    from unisim.scene import SceneCfg

    pool = tmp_path / "pool"
    manifest = write_entity_set(pool, num_per_type=1, layout=TOOL_POOL_LAYOUT_UNIFORM)
    bootstrap = write_bootstrap_entity_set(tmp_path / "boot", layout=TOOL_POOL_LAYOUT_UNIFORM)
    tools = [pool / "tools" / name for name in manifest["tool_files"]]
    catalog = build_tool_catalog(num_per_type=1, seed=42, shuffle=True)

    def entity(name: str, source: Path, **kwargs) -> SceneEntitySpec:
        return SceneEntitySpec(name=name, source=ModelSourceDescriptor(str(source)), **kwargs)

    entities = (
        entity(
            "robot",
            bootstrap["robot"],
            kind="articulation",
            root_mode="fixed",
            initial_state=EntityInitialState(position=(0.0, 0.8, 0.0)),
        ),
        entity(
            "table",
            bootstrap["table"],
            kind="rigid",
            root_mode="kinematic",
            initial_state=EntityInitialState(position=(0.0, 0.0, 0.38)),
        ),
        entity("floor", bootstrap["floor"], kind="rigid", root_mode="kinematic"),
        entity(
            "object",
            bootstrap["object"],
            kind="rigid",
            root_mode="floating",
            initial_state=EntityInitialState(position=(0.0, 0.0, 0.63)),
        ),
    )
    scene = SceneCfg(
        entity_assets=entities,
        entity_variant=EntityVariantBinding(
            target_entity="object",
            plan=FixedVariantPlan(
                variants=tuple(ModelSourceDescriptor(str(p)) for p in tools),
                assignment=np.arange(4, dtype=np.int32) % len(tools),
                layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
            ),
        ),
        default_keyframe_name="home",
    )
    composed = compose_scene(scene, 4, 1 / 120)
    try:
        assert composed.variant_plan is not None
        assert composed.variant_plan.layout is FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT

        # Per-variant uniform summaries over the standalone tool files (the
        # entity namespace is only attached at compose time, so the tuples are
        # built directly — the same (name, type, body) triples
        # summarize_uniform_variant_layout produces for the composed scene).
        # Eraser variants omit exactly the collision head slot (the visual
        # slot stays as the shared placeholder); headed variants carry the
        # full four-slot set.
        head_slots = {"object_head_col", "object_head_vis"}
        all_slots = head_slots | {"object_handle_col", "object_handle_vis"}

        def summary_of(path: Path) -> UniformVariantLayoutSummary:
            model = mujoco.MjModel.from_xml_path(str(path))
            return UniformVariantLayoutSummary(
                ngeom=int(model.ngeom),
                geoms=tuple(
                    (
                        str(model.geom(g).name),
                        int(model.geom_type[g]),
                        str(model.body(int(model.geom_bodyid[g])).name),
                    )
                    for g in range(model.ngeom)
                ),
            )

        summaries = [summary_of(path) for path in tools]
        bootstrap_summary = summary_of(bootstrap["object"])
        canonical_index = validate_uniform_entity_variant_layout_summaries(
            bootstrap_summary, tuple(summaries)
        )
        assert set(name for name, _, _ in bootstrap_summary.geoms) == all_slots
        for spec, summary in zip(catalog, summaries, strict=True):
            present = {name for name, _, _ in summary.geoms}
            if spec.head_scale is None:
                assert present == all_slots - {"object_head_col"}
            else:
                assert present == all_slots
        # Canonical = the max-geom, lowest-index variant: the first headed
        # tool in catalog order, and its mass is the composed object mass
        # (compose serializes the canonical scene with ~6 significant digits,
        # so the composed realization is truncated; per-env executors compile
        # the full-precision variant sources directly).
        assert catalog[canonical_index].head_scale is not None
        object_id = mujoco.mj_name2id(
            composed.model, mujoco.mjtObj.mjOBJ_BODY, "object/object_root"
        )
        assert composed.model.body_mass[object_id] == pytest.approx(
            catalog[canonical_index].mass, rel=1e-5
        )
    finally:
        composed.close()


def test_uniform_contract_rejects_mismatched_bootstrap(tmp_path):
    """Fail-closed direction: a same-layout bootstrap (analytic handle) under a
    uniform variant catalog violates the canonical geom-type rule."""
    from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
    from unisim.entities import EntityVariantBinding, SceneEntitySpec
    from unisim.mjcf_compiler import compose_scene
    from unisim.scene import SceneCfg

    pool = tmp_path / "pool"
    manifest = write_entity_set(pool, num_per_type=1, layout=TOOL_POOL_LAYOUT_UNIFORM)
    bootstrap = write_bootstrap_entity_set(tmp_path / "boot")  # same_layout default
    tools = [pool / "tools" / name for name in manifest["tool_files"]]

    def entity(name: str, source: Path, **kwargs) -> SceneEntitySpec:
        return SceneEntitySpec(name=name, source=ModelSourceDescriptor(str(source)), **kwargs)

    scene = SceneCfg(
        entity_assets=(
            entity("robot", bootstrap["robot"], kind="articulation", root_mode="fixed"),
            entity("table", bootstrap["table"], kind="rigid", root_mode="kinematic"),
            entity("floor", bootstrap["floor"], kind="rigid", root_mode="kinematic"),
            entity("object", bootstrap["object"], kind="rigid", root_mode="floating"),
        ),
        entity_variant=EntityVariantBinding(
            target_entity="object",
            plan=FixedVariantPlan(
                variants=tuple(ModelSourceDescriptor(str(p)) for p in tools),
                assignment=np.arange(2, dtype=np.int32) % len(tools),
                layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
            ),
        ),
    )
    with pytest.raises(ValueError, match="changes the type of present geom"):
        compose_scene(scene, 2, 1 / 120)
