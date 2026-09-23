"""Compile-level parity checks for the SimToolReal tool MJCF writer.

Anchors: the ToolSpec values sampled by ``build_tool_catalog`` and the MuJoCo
compilation of the emitted MJCF (``MjModel.from_xml_path`` — the compiler is
its own format authority, roadmap M-J1 D5).  The cylinder->capsule collision
mapping (the URDF line's replace_cylinders_with_capsules equivalence) is
asserted explicitly.
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

# Covers every topology (box_box, capsule_box, box_only) and both handle
# shapes, in the source table's distribution order.
CATALOG = build_tool_catalog(ALL_TYPES, 1, 42, shuffle=False)
assert {spec.topology for spec in CATALOG} == {"box_box", "capsule_box", "box_only"}

_HANDLE_CYL_QUAT = (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0)  # -pi/2 about y


def _object_body_id(model: mujoco.MjModel) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_root")
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
def test_mjcf_visual_and_collision_geom_roles(spec: ToolSpec, tmp_path: Path):
    """Fixed four-slot layout (M-J2 same-layout rule); roles by contact bits."""
    model = mujoco.MjModel.from_xml_path(str(write_tool_mjcf(spec, tmp_path / "tool.xml")))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)]
    assert names == [
        "object_handle_col",
        "object_handle_vis",
        "object_head_col",
        "object_head_vis",
    ]
    collisions = visuals = 0
    for g in range(model.ngeom):
        contacting = model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0
        if contacting:
            collisions += 1
            # Owner-YAML object slide (tuning item), torsion/roll defaults.
            np.testing.assert_allclose(model.geom_friction[g], (0.5, 0.005, 0.0001))
        else:
            visuals += 1
    # Both collision slots always exist (same-layout rule); the headless
    # placeholder keeps contact bits but is epsilon-sized and enclosed.
    assert collisions == 2
    assert visuals == 2
    if spec.head_scale is None:
        head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_head_col")
        assert model.geom_size[head_id].max() < 1e-5


def test_all_variants_share_public_layout(tmp_path: Path):
    """M-J2 same-layout contract: identical geom names/counts across topologies."""
    layouts = []
    for index, spec in enumerate(CATALOG):
        model = mujoco.MjModel.from_xml_path(str(write_tool_mjcf(spec, tmp_path / f"t{index}.xml")))
        layouts.append(
            (
                model.ngeom,
                tuple(
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                    for g in range(model.ngeom)
                ),
            )
        )
    assert len(set(layouts)) == 1


def test_mjcf_handle_cylinder_orientation(tmp_path: Path):
    """The capsule axis rotation is the URDF cylinder's -pi/2 about y."""
    spec = next(s for s in CATALOG if s.authored_shape == "cylinder")
    model = mujoco.MjModel.from_xml_path(str(write_tool_mjcf(spec, tmp_path / "tool.xml")))
    for name in ("object_handle_col", "object_handle_vis"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert geom_id >= 0
        np.testing.assert_allclose(model.geom_quat[geom_id], _HANDLE_CYL_QUAT, atol=1e-15)
    assert (
        model.geom_type[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_handle_col")]
        == mujoco.mjtGeom.mjGEOM_CAPSULE
    )


def test_batch_writer_cleans_stale_mjcfs(tmp_path: Path):
    out_dir = tmp_path / "pool"
    out_dir.mkdir()
    stale = out_dir / "stale.xml"
    stale.write_text("stale")
    keep_other = out_dir / "keep.usd"  # non-XML files are untouched
    keep_other.write_text("keep")
    keep_txt = out_dir / "notes.txt"
    keep_txt.write_text("keep")
    catalog = build_tool_catalog(("eraser",), 2, 42, shuffle=False)
    paths, _ = write_tool_catalog_mjcfs(catalog, out_dir)
    assert len(paths) == 2
    assert len(set(paths)) == 2
    assert not stale.exists()
    assert keep_other.exists()
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
