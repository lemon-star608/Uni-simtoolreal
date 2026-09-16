"""Compile-level parity checks for the SimToolReal tool MJCF writer.

Anchors: the ToolSpec values (same source as the URDF writer) and the MuJoCo
compilation of the corresponding URDF (``write_tool_urdf`` output compiled by
``MjModel.from_xml_path`` — the compiler is its own format authority, roadmap
M-J1 D5).  The cylinder->capsule collision mapping (URDF line's
replace_cylinders_with_capsules equivalence) is asserted explicitly.
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
from simtoolreal_unilab.tasks.simtoolreal.tool_mjcf_writer import (
    write_tool_catalog_mjcfs,
    write_tool_mjcf,
)
from simtoolreal_unilab.tasks.simtoolreal.tool_urdf_writer import (
    write_tool_catalog_urdfs,
    write_tool_urdf,
)

# Covers every topology (box_box, capsule_box, box_only) and both handle
# shapes, in the source table's distribution order.
CATALOG = build_tool_catalog(ALL_TYPES, 1, 42, shuffle=False)
assert {spec.topology for spec in CATALOG} == {"box_box", "capsule_box", "box_only"}

_HANDLE_CYL_QUAT = (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0)  # -pi/2 about y


def _object_body_id(model: mujoco.MjModel) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    assert body_id >= 1
    return body_id


@pytest.mark.parametrize("spec", CATALOG, ids=lambda s: f"{s.type}-{s.topology}")
def test_mjcf_inertial_matches_tool_spec(spec: ToolSpec, tmp_path: Path):
    model = mujoco.MjModel.from_xml_path(str(write_tool_mjcf(spec, tmp_path / "tool.xml")))
    body_id = _object_body_id(model)
    assert model.body_mass[body_id] == pytest.approx(spec.mass, rel=1e-12)
    np.testing.assert_allclose(model.body_ipos[body_id], spec.com, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(model.body_inertia[body_id], spec.diaginertia, rtol=1e-9, atol=1e-18)


@pytest.mark.parametrize("spec", CATALOG, ids=lambda s: f"{s.type}-{s.topology}")
def test_mjcf_geometry_matches_compiled_urdf(spec: ToolSpec, tmp_path: Path):
    """Collision geoms equal the MuJoCo-compiled URDF, cylinder->capsule aside."""
    mjcf_model = mujoco.MjModel.from_xml_path(str(write_tool_mjcf(spec, tmp_path / "tool.xml")))
    urdf_model = mujoco.MjModel.from_xml_path(str(write_tool_urdf(spec, tmp_path / "tool.urdf")))

    # The URDF compiler drops <visual> elements (discardvisual) and welds the
    # single link to the world; what remains are exactly the collision geoms.
    urdf_geoms = [
        g
        for g in range(urdf_model.ngeom)
        if urdf_model.geom_contype[g] or urdf_model.geom_conaffinity[g]
    ]
    mjcf_geoms = [
        g
        for g in range(mjcf_model.ngeom)
        if mjcf_model.geom_contype[g] or mjcf_model.geom_conaffinity[g]
    ]
    assert len(mjcf_geoms) == len(urdf_geoms)

    for g_mjcf, g_urdf in zip(mjcf_geoms, urdf_geoms):
        urdf_type = urdf_model.geom_type[g_urdf]
        mjcf_type = mjcf_model.geom_type[g_mjcf]
        if urdf_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            # The URDF line trained with replace_cylinders_with_capsules: the
            # MJCF writer emits the capsule directly, same frame and size.
            assert mjcf_type == mujoco.mjtGeom.mjGEOM_CAPSULE
        else:
            assert mjcf_type == urdf_type
        np.testing.assert_allclose(
            mjcf_model.geom_size[g_mjcf], urdf_model.geom_size[g_urdf], rtol=1e-12, atol=1e-18
        )
        np.testing.assert_allclose(
            mjcf_model.geom_pos[g_mjcf], urdf_model.geom_pos[g_urdf], rtol=1e-9, atol=1e-15
        )
        np.testing.assert_allclose(
            mjcf_model.geom_quat[g_mjcf], urdf_model.geom_quat[g_urdf], rtol=1e-9, atol=1e-12
        )


@pytest.mark.parametrize("spec", CATALOG, ids=lambda s: f"{s.type}-{s.topology}")
def test_mjcf_visual_and_collision_geom_roles(spec: ToolSpec, tmp_path: Path):
    """Visual geoms are contype/conaffinity 0; collision geoms carry friction."""
    model = mujoco.MjModel.from_xml_path(str(write_tool_mjcf(spec, tmp_path / "tool.xml")))
    collisions = visuals = 0
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            visuals += 1
        else:
            collisions += 1
            # Owner-YAML object slide (tuning item), torsion/roll defaults.
            np.testing.assert_allclose(model.geom_friction[g], (0.5, 0.005, 0.0001))
    n_primitive = 1 + (spec.head_scale is not None)
    n_visual_cyl = int(spec.authored_shape == "cylinder")
    assert collisions == n_primitive
    assert visuals == n_visual_cyl


def test_mjcf_handle_cylinder_orientation(tmp_path: Path):
    """The capsule axis rotation is the URDF cylinder's -pi/2 about y."""
    spec = next(s for s in CATALOG if s.authored_shape == "cylinder")
    model = mujoco.MjModel.from_xml_path(str(write_tool_mjcf(spec, tmp_path / "tool.xml")))
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_handle_cyl")
    assert geom_id >= 0
    np.testing.assert_allclose(model.geom_quat[geom_id], _HANDLE_CYL_QUAT, atol=1e-15)


def test_batch_writer_contract_matches_urdf_writer(tmp_path: Path):
    """Same filename template (modulo suffix), same return contract."""
    catalog = build_tool_catalog(ALL_TYPES, 3, 42, shuffle=True)
    mjcf_dir = tmp_path / "pool_mjcf"
    urdf_dir = tmp_path / "pool_urdf"
    mjcf_paths, mjcf_scales = write_tool_catalog_mjcfs(catalog, mjcf_dir)
    urdf_paths, urdf_scales = write_tool_catalog_urdfs(catalog, urdf_dir)
    assert len(mjcf_paths) == len(catalog) == len(mjcf_scales)
    assert mjcf_scales == urdf_scales
    for mjcf_path, urdf_path, spec in zip(mjcf_paths, urdf_paths, catalog):
        assert Path(mjcf_path).suffix == ".xml"
        assert Path(mjcf_path).stem == Path(urdf_path).stem
        # Batch output is byte-identical to the per-tool entry point.
        single = write_tool_mjcf(spec, tmp_path / "single.xml")
        assert Path(mjcf_path).read_bytes() == single.read_bytes()


def test_batch_writer_cleans_stale_mjcfs(tmp_path: Path):
    out_dir = tmp_path / "pool"
    out_dir.mkdir()
    stale = out_dir / "stale.xml"
    stale.write_text("stale")
    keep_xml_urdf = out_dir / "keep.urdf"  # the URDF pool's files are untouched
    keep_xml_urdf.write_text("keep")
    keep_txt = out_dir / "notes.txt"
    keep_txt.write_text("keep")
    catalog = build_tool_catalog(("eraser",), 2, 42, shuffle=False)
    paths, _ = write_tool_catalog_mjcfs(catalog, out_dir)
    assert len(paths) == 2
    assert len(set(paths)) == 2
    assert not stale.exists()
    assert keep_xml_urdf.exists()
    assert keep_txt.exists()


def test_mismatched_head_fields_raise(tmp_path: Path):
    spec = build_tool_catalog(("eraser",), 1, 42, shuffle=False)[0]
    broken = ToolSpec(
        **{
            **{f: getattr(spec, f) for f in ToolSpec.__dataclass_fields__},
            "head_density": 800.0,
        }
    )
    with pytest.raises(ValueError, match="both be set or both None"):
        write_tool_mjcf(broken, tmp_path / "broken.xml")
