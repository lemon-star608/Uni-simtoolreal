"""Compile-level checks for the uniform-public-layout tool MJCF writer (D8).

Mirrors ``test_simtoolreal_tool_mjcf_writer.py`` against the full-mesh
realization: same ToolSpec truth (inertial verbatim, owner-YAML friction),
mesh-typed geom slots everywhere, erasers omitting the head slots entirely.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from simtoolreal_unilab.tasks.simtoolreal.tool_catalog import (
    ALL_TYPES,
    ToolSpec,
    build_tool_catalog,
)
from simtoolreal_unilab.tasks.simtoolreal.tool_mjcf_writer import write_tool_catalog_mjcfs
from simtoolreal_unilab.tasks.simtoolreal.tool_uniform_mjcf_writer import (
    box_mesh,
    capsule_mesh,
    tool_mjcf_text_uniform,
    write_tool_mjcf_uniform,
)

CATALOG = build_tool_catalog(ALL_TYPES, 1, 42, shuffle=False)
assert {spec.topology for spec in CATALOG} == {"box_box", "capsule_box", "box_only"}


def _compile(spec: ToolSpec, tmp_path: Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(write_tool_mjcf_uniform(spec, tmp_path / "tool.xml")))


def _geom_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)]


@pytest.mark.parametrize("spec", CATALOG, ids=lambda s: f"{s.type}-{s.topology}")
def test_uniform_inertial_matches_tool_spec(spec: ToolSpec, tmp_path: Path):
    model = _compile(spec, tmp_path)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_root")
    assert body_id >= 1
    assert model.body_mass[body_id] == pytest.approx(spec.mass, rel=1e-12)
    np.testing.assert_allclose(model.body_ipos[body_id], spec.com, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(model.body_inertia[body_id], spec.diaginertia, rtol=1e-9, atol=1e-18)


@pytest.mark.parametrize("spec", CATALOG, ids=lambda s: f"{s.type}-{s.topology}")
def test_uniform_slots_are_mesh_typed_with_unique_names(spec: ToolSpec, tmp_path: Path):
    model = _compile(spec, tmp_path)
    names = _geom_names(model)
    if spec.head_scale is None:
        # Erasers omit both head slots entirely (no placeholder is possible:
        # MuJoCo rejects near-zero-volume meshes; the uniform contract allows
        # mesh-typed slots to be absent).
        assert names == ["object_handle_col", "object_handle_vis"]
    else:
        assert names == [
            "object_handle_col",
            "object_handle_vis",
            "object_head_col",
            "object_head_vis",
        ]
    assert len(set(names)) == len(names) == model.ngeom
    for g in range(model.ngeom):
        assert model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH


@pytest.mark.parametrize("spec", CATALOG, ids=lambda s: f"{s.type}-{s.topology}")
def test_uniform_geom_roles_and_friction(spec: ToolSpec, tmp_path: Path):
    """Collision slots keep contact bits + owner-YAML friction; visuals don't."""
    model = _compile(spec, tmp_path)
    from simtoolreal_unilab.tasks.simtoolreal.owner_physics import load_owner_physics

    slide = load_owner_physics().object_slide_friction
    collisions = visuals = 0
    for g in range(model.ngeom):
        if model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0:
            collisions += 1
            assert "col" in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            np.testing.assert_allclose(
                model.geom_friction[g], (slide, 0.005, 0.0001), rtol=1e-12, atol=0
            )
        else:
            visuals += 1
            assert "vis" in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
    assert collisions == visuals == (1 if spec.head_scale is None else 2)


def test_capsule_mesh_vertices_lie_on_the_analytic_surface():
    """24-segment lathe: every vertex on the x-aligned analytic capsule."""
    height, diameter = 0.12, 0.02
    radius = diameter / 2
    verts, faces = capsule_mesh(height, diameter)
    assert faces and max(max(face) for face in faces) == len(verts) - 1
    for x, y, z in verts:
        axial = math.hypot(y, z)
        if abs(x) <= height / 2:
            assert axial == pytest.approx(radius, abs=1e-15)
        else:
            # Cap rows: vertices on the hemisphere of radius r around the pole.
            pole = math.copysign(height / 2, x)
            assert math.hypot(x - pole, axial) == pytest.approx(radius, abs=1e-15)
    xs = [v[0] for v in verts]
    assert (min(xs), max(xs)) == pytest.approx((-height / 2 - radius, height / 2 + radius))


def test_box_mesh_is_the_analytic_box():
    dims = (0.1, 0.04, 0.02)
    verts, faces = box_mesh(dims)
    assert len(verts) == 8 and len(faces) == 12
    for axis, half in enumerate(v / 2 for v in dims):
        coords = sorted({v[axis] for v in verts})
        assert coords == pytest.approx([-half, half])


def test_capsule_handle_mesh_extent_matches_tool_spec(tmp_path: Path):
    """Compiled mesh spans the analytic capsule's bounding extent (body frame).

    The compiler recenters mesh vertices to the principal frame and compensates
    with geom_pos/geom_quat, so the body-frame bounds come from transforming
    the stored vertices back through the geom frame.
    """
    spec = next(s for s in CATALOG if s.authored_shape == "cylinder")
    model = _compile(spec, tmp_path)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_handle_col")
    mesh_id = model.geom_dataid[geom_id]
    start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    local = model.mesh_vert[start : start + count].reshape(count, 3)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[geom_id])
    body = local @ rot.reshape(3, 3).T + model.geom_pos[geom_id]
    height, diameter = spec.handle_scale
    radius = diameter / 2
    # X-aligned capsule (no authored quat needed): the long axis is the body's.
    np.testing.assert_allclose(
        body.min(axis=0), (-height / 2 - radius, -radius, -radius), atol=1e-6
    )
    np.testing.assert_allclose(body.max(axis=0), (height / 2 + radius, radius, radius), atol=1e-6)


def test_meshes_are_inline_and_uniquely_named(tmp_path: Path):
    """One self-contained file per tool; mesh asset names unique within it."""
    spec = next(s for s in CATALOG if s.head_scale is not None)
    path = write_tool_mjcf_uniform(spec, tmp_path / "tool.xml")
    text = path.read_text()
    assert "<asset>" in text and 'vertex="' in text and 'face="' in text
    assert 'file="' not in text  # no external mesh assets
    model = mujoco.MjModel.from_xml_path(str(path))
    mesh_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, m) for m in range(model.nmesh)]
    assert mesh_names == ["object_handle_mesh", "object_head_mesh"]


@pytest.mark.parametrize("spec", CATALOG, ids=lambda s: f"{s.type}-{s.topology}")
def test_uniform_text_regeneration_is_byte_identical(spec: ToolSpec):
    assert tool_mjcf_text_uniform(spec).encode() == tool_mjcf_text_uniform(spec).encode()


def test_batch_writer_with_uniform_text_fn(tmp_path: Path):
    """The batch contract carries the uniform writer unchanged (D8 pool)."""
    catalog = build_tool_catalog(ALL_TYPES, 2, 42, shuffle=True)
    paths, scales = write_tool_catalog_mjcfs(
        catalog, tmp_path / "pool", text_fn=tool_mjcf_text_uniform
    )
    assert len(paths) == len(catalog) == len(scales)
    for path, spec in zip(paths, catalog, strict=True):
        assert Path(path).read_text() == tool_mjcf_text_uniform(spec)


def test_mismatched_head_fields_raise(tmp_path: Path):
    spec = build_tool_catalog(("eraser",), 1, 42, shuffle=False)[0]
    broken = ToolSpec(
        **{
            **{f: getattr(spec, f) for f in ToolSpec.__dataclass_fields__},
            "head_density": 800.0,
        }
    )
    with pytest.raises(ValueError, match="both be set or both None"):
        write_tool_mjcf_uniform(broken, tmp_path / "broken.xml")
