"""MJCF serialization for the SimToolReal tool catalog.

Counterpart of ``tool_urdf_writer.py`` for the MuJoCo line (roadmap M-J1,
decision D2): the same ``ToolSpec`` samples become single-body MJCF files
instead of URDF files.  Geometry mapping follows the MJCF knowledge recorded
in the old tree's ``unilab/envs/manipulation/simtoolreal/tool_assets.py``
(cylinders emit as capsules with a -pi/2 y-rotation, visual/collision geom
pairs, ``%.17g`` floats), re-expressed as whole-file template generation —
no string surgery on a host scene.  Mass/inertia come straight from
``ToolSpec`` (``tool_catalog._compute_mass_and_inertia``, the same pure
function the URDF writer uses), so the compiled ``body_mass``/``body_inertia``
match the URDF line value-for-value.  Inline comments cite the original
``generate_objects.py`` line numbers, as in ``tool_urdf_writer.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from .owner_physics import (
    ROLLING_FRICTION_DEFAULT,
    TORSIONAL_FRICTION_DEFAULT,
    load_owner_physics,
)
from .tool_catalog import OBJECT_BASE_SIZE, ToolSpec

# Root body/joint naming matches the old tree's composed scenes
# (tool_assets.py:51, scene.xml qpos-layout comment): one floating body
# ``object`` with a freejoint, appended to the scene worldbody.
_OBJECT_BODY = "object"
_OBJECT_FREEJOINT = "object_joint"

# Cylinder axis rotations as MJCF quaternions (wxyz).  The URDF writer rotates
# handle cylinders with rpy="0 -pi/2 0" (generate_objects.py:110) and head
# cylinders with rpy="-pi/2 0 0" (generate_objects.py:198); capsules carry the
# same orientations, which is exactly the geometry the URDF line trained with
# after replace_cylinders_with_capsules.
_QUAT_HANDLE_CYLINDER = "0.7071067811865476 0 -0.7071067811865476 0"  # -pi/2 about y
_QUAT_HEAD_CYLINDER = "0.7071067811865476 -0.7071067811865476 0 0"  # -pi/2 about x

_HANDLE_RGBA = "0.55 0.27 0.07 1.0"  # brown (generate_objects.py:78)
_HEAD_RGBA = "0.5 0.5 0.5 1.0"  # gray (generate_objects.py:232)


def _fmt(value: float) -> str:
    """17-significant-digit float, the MJCF line's lossless float format."""
    return f"{float(value):.17g}"


def _fmt3(values: Sequence[float]) -> str:
    return " ".join(_fmt(v) for v in values)


def _friction() -> str:
    """Geom friction triple for tool bodies (tuning item, D6).

    Sliding coefficient is the owner YAML's frozen object friction; torsional
    and rolling keep the MuJoCo defaults (the owner YAML's Isaac-side triples
    carry no torsion/roll term).  Written explicitly, as the old tree did.
    """
    slide = load_owner_physics().object_slide_friction
    return f"{_fmt(slide)} {_fmt(TORSIONAL_FRICTION_DEFAULT)} {_fmt(ROLLING_FRICTION_DEFAULT)}"


def _handle_geoms(spec: ToolSpec) -> str:
    """Handle geometry: one box geom, or a visual cylinder + collision capsule.

    Mirrors the handle branches of generate_objects.py:47-112 as compiled by
    the URDF line's capsule replacement (scene_utils.py:1701-1706): MJCF box
    sizes are the URDF full sizes halved; cylinders keep their visual shape and
    gain a capsule collision geom (tool_assets.py:26-38).
    """
    if len(spec.handle_scale) == 3:
        half = tuple(v / 2 for v in spec.handle_scale)
        return (
            f'      <geom name="object_handle" type="box" size="{_fmt3(half)}" '
            f'friction="{_friction()}" density="0" rgba="{_HANDLE_RGBA}"/>\n'
        )
    height, diameter = spec.handle_scale
    radius, half_length = diameter / 2, height / 2
    return (
        f'      <geom name="object_handle_cyl_visual" type="cylinder" '
        f'size="{_fmt(radius)} {_fmt(half_length)}" quat="{_QUAT_HANDLE_CYLINDER}" '
        f'contype="0" conaffinity="0" density="0" rgba="{_HANDLE_RGBA}"/>\n'
        f'      <geom name="object_handle_cyl" type="capsule" '
        f'size="{_fmt(radius)} {_fmt(half_length)}" quat="{_QUAT_HANDLE_CYLINDER}" '
        f'friction="{_friction()}" density="0"/>\n'
    )


def _head_geoms(spec: ToolSpec) -> str:
    """Head geometry at +x past the handle (generate_objects.py:183-204)."""
    assert spec.head_scale is not None
    if len(spec.head_scale) == 3:
        hlx = spec.head_scale[0]
        x_offset = spec.handle_scale[0] / 2 + hlx / 2
        half = tuple(v / 2 for v in spec.head_scale)
        return (
            f'      <geom name="object_head" type="box" size="{_fmt3(half)}" '
            f'pos="{_fmt(x_offset)} 0 0" friction="{_friction()}" density="0" '
            f'rgba="{_HEAD_RGBA}"/>\n'
        )
    hh, hd = spec.head_scale
    hr = hd / 2
    x_offset = spec.handle_scale[0] / 2 + hr
    return (
        f'      <geom name="object_head_cyl_visual" type="cylinder" '
        f'size="{_fmt(hr)} {_fmt(hh / 2)}" pos="{_fmt(x_offset)} 0 0" '
        f'quat="{_QUAT_HEAD_CYLINDER}" contype="0" conaffinity="0" density="0" '
        f'rgba="{_HEAD_RGBA}"/>\n'
        f'      <geom name="object_head_cyl" type="capsule" size="{_fmt(hr)} {_fmt(hh / 2)}" '
        f'pos="{_fmt(x_offset)} 0 0" quat="{_QUAT_HEAD_CYLINDER}" '
        f'friction="{_friction()}" density="0"/>\n'
    )


def _tool_mjcf_text(spec: ToolSpec) -> str:
    if (spec.head_scale is None) != (spec.head_density is None):
        raise ValueError(
            f"head_scale and head_density must both be set or both None (got "
            f"{spec.head_scale} and {spec.head_density})"
        )
    geoms = _handle_geoms(spec)
    if spec.head_scale is not None:
        geoms += _head_geoms(spec)
    return f"""<mujoco model="{_OBJECT_BODY}">
  <worldbody>
    <body name="{_OBJECT_BODY}">
      <freejoint name="{_OBJECT_FREEJOINT}"/>
      <inertial pos="{_fmt3(spec.com)}" mass="{_fmt(spec.mass)}" diaginertia="{_fmt3(spec.diaginertia)}"/>
{geoms}    </body>
  </worldbody>
</mujoco>
"""


def _tool_filename(spec: ToolSpec, idx: int) -> str:
    """Filename template of generate_objects.py:366-370 (``.xml`` suffix)."""
    return (
        f"{idx:03d}_{spec.type}_handle_{tuple(spec.handle_scale)}"
        f"_head_{None if spec.head_scale is None else tuple(spec.head_scale)}"
        f"_d{spec.handle_density:.1f}_{spec.head_density}".replace(".", "-")
        + ".xml"
    )


def write_tool_mjcf(spec: ToolSpec, path: Path) -> Path:
    """Serialize one ``ToolSpec`` to a standalone MJCF file.

    Same dispatch contract as ``write_tool_urdf`` (generate_objects.py:262-285):
    a spec without a head emits handle geometry only; a spec with both head
    scale and density emits the handle+head composite.  The ``<inertial>``
    fields are the ToolSpec's own mass/com/diaginertia (the composite
    parallel-axis math of generate_objects.py:206-218 lives in
    ``tool_catalog.build_tool_catalog``).
    """
    path.write_text(_tool_mjcf_text(spec))
    return path


def write_tool_catalog_mjcfs(
    catalog: Sequence[ToolSpec],
    out_dir: str | Path,
    object_base_size: float = OBJECT_BASE_SIZE,
) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Write one MJCF per catalog entry into ``out_dir``.

    Batch counterpart of ``write_tool_catalog_urdfs`` (generate_objects.py:
    302-399 minus sampling): same filename template with an ``.xml`` suffix,
    same stale-file cleanup (for ``.xml``), same return contract — file paths
    paired with handle scales normalized by ``object_base_size``.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        for p in out_dir.iterdir():
            if p.suffix == ".xml":
                p.unlink()
    else:
        os.makedirs(out_dir)

    paths: list[str] = []
    scales_norm: list[tuple[float, float, float]] = []
    type_counts: dict[str, int] = {}
    for spec in catalog:
        idx = type_counts.get(spec.type, 0)
        type_counts[spec.type] = idx + 1
        mjcf_path = out_dir / _tool_filename(spec, idx)
        write_tool_mjcf(spec, mjcf_path)
        paths.append(str(mjcf_path))
        # ToolSpec.object_scale is the handle scale already normalized by
        # OBJECT_BASE_SIZE; rescale if the caller overrides the base size.
        factor = OBJECT_BASE_SIZE / object_base_size
        sx, sy, sz = spec.object_scale
        scales_norm.append((sx * factor, sy * factor, sz * factor))

    return paths, scales_norm


__all__ = ["write_tool_catalog_mjcfs", "write_tool_mjcf"]
