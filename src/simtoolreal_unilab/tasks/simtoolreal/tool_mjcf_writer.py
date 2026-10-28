"""MJCF serialization for the SimToolReal tool catalog.

Each ``ToolSpec`` becomes one single-body MJCF entity source. Cylinders emit
as capsules with a -pi/2 y-rotation, geometry uses visual/collision pairs, and
floating values preserve full precision. Mass and inertia come directly from
``ToolSpec``.

Same-layout rule: the UniSim MuJoCo batch executor requires every fixed
variant to share one public layout (same entity names and counts; per-variant
``geom_type``/``geom_size``/``geom_quat``/inertial arrays are natively
supported).  Every tool therefore emits exactly four geom slots —
``object_handle_col``/``object_handle_vis``/``object_head_col``/
``object_head_vis`` — and headless variants (eraser) fill the head slots with
epsilon-size, non-contacting, fully transparent placeholders.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

from .owner_physics import (
    ROLLING_FRICTION_DEFAULT,
    TORSIONAL_FRICTION_DEFAULT,
    load_owner_physics,
)
from .tool_catalog import OBJECT_BASE_SIZE, ToolSpec

# Root body/joint naming follows this repository's canonical object namespace
# (``tool_urdf_writer._OBJECT_ROOT_LINK``; the owner YAML's object entity
# declares root_body_name=object_root): one floating body ``object_root`` with
# a freejoint, appended to the scene worldbody.
_OBJECT_BODY = "object_root"
_OBJECT_FREEJOINT = "object_root_joint"

# Cylinder axis rotation as an MJCF quaternion (wxyz).  The URDF writer rotates
# handle cylinders with rpy="0 -pi/2 0" (generate_objects.py:110); capsules
# carry the same orientation, which is exactly the geometry the URDF line
# trained with after replace_cylinders_with_capsules.
_QUAT_HANDLE_CYLINDER = "0.7071067811865476 0 -0.7071067811865476 0"  # -pi/2 about y

_HANDLE_RGBA = "0.55 0.27 0.07 1.0"  # brown (generate_objects.py:78)
_HEAD_RGBA = "0.5 0.5 0.5 1.0"  # gray (generate_objects.py:232)

# Headless-variant placeholder size: MuJoCo rejects zero geom sizes, so the
# head slots shrink to an invisible, non-contacting epsilon box instead.
_PLACEHOLDER_EPS = 1e-6


def _fmt(value: float) -> str:
    """17-significant-digit float, the MJCF line's lossless float format."""
    return f"{float(value):.17g}"


def _fmt3(values: Sequence[float]) -> str:
    return " ".join(_fmt(v) for v in values)


def _friction() -> str:
    """Geom friction triple for tool bodies (tuning item, D6).

    Sliding coefficient is the owner YAML's frozen object friction; torsional
    and rolling keep the MuJoCo defaults.
    """
    slide = load_owner_physics().object_slide_friction
    return f"{_fmt(slide)} {_fmt(TORSIONAL_FRICTION_DEFAULT)} {_fmt(ROLLING_FRICTION_DEFAULT)}"


def _handle_geoms(spec: ToolSpec) -> str:
    """Handle geometry slots: collision geom + visual geom.

    Mirrors the handle branches of generate_objects.py:47-112 as compiled by
    the URDF line's capsule replacement (scene_utils.py:1701-1706): MJCF box
    sizes are the URDF full sizes halved; cylinders keep their visual shape and
    gain a capsule collision geom (tool_assets.py:26-38).  Both slots exist in
    every variant (M-J2 same-layout rule, see module docstring): only the
    compiled type/size/quat arrays differ per variant, which mjbatch's variant
    pack supports natively (``geom_type``/``geom_size``/``geom_quat`` are
    per-variant fields).
    """
    if len(spec.handle_scale) == 3:
        half = tuple(v / 2 for v in spec.handle_scale)
        size = _fmt3(half)
        vis_type, col_type, quat = "box", "box", ""
    else:
        height, diameter = spec.handle_scale
        radius, half_length = diameter / 2, height / 2
        size = f"{_fmt(radius)} {_fmt(half_length)}"
        vis_type, col_type = "cylinder", "capsule"
        quat = f' quat="{_QUAT_HANDLE_CYLINDER}"'
    return (
        f'      <geom name="object_handle_col" type="{col_type}" size="{size}"{quat} '
        f'friction="{_friction()}" density="0" group="3"/>\n'
        f'      <geom name="object_handle_vis" type="{vis_type}" size="{size}"{quat} '
        f'contype="0" conaffinity="0" density="0" group="2" rgba="{_HANDLE_RGBA}"/>\n'
    )


def _head_geoms(spec: ToolSpec) -> str:
    """Head geometry slots at +x past the handle (generate_objects.py:183-204).

    Headless variants (eraser) keep both slots as epsilon-size,
    non-contacting, fully transparent placeholders so every pool variant has
    the same public layout (M-J2 same-layout rule).
    """
    if spec.head_scale is None:
        size = _fmt3((_PLACEHOLDER_EPS,) * 3)
        # The collision slot keeps contact bits enabled so the backend's
        # discardvisual physics pass does not strip it (visual-only geoms are
        # discarded; layouts must stay identical across variants).  The
        # epsilon box sits at the body origin, fully enclosed by the handle
        # collision geom, so it never generates a contact; density=0 keeps it
        # massless.  The visual slot is discarded uniformly in every variant.
        return (
            f'      <geom name="object_head_col" type="box" size="{size}" '
            f'friction="{_friction()}" density="0" group="3"/>\n'
            f'      <geom name="object_head_vis" type="box" size="{size}" '
            'contype="0" conaffinity="0" density="0" group="2" rgba="0.5 0.5 0.5 0"/>\n'
        )
    assert len(spec.head_scale) == 3, "catalog heads are always boxes (tool_catalog)"
    hlx = spec.head_scale[0]
    x_offset = spec.handle_scale[0] / 2 + hlx / 2
    half = tuple(v / 2 for v in spec.head_scale)
    return (
        f'      <geom name="object_head_col" type="box" size="{_fmt3(half)}" '
        f'pos="{_fmt(x_offset)} 0 0" friction="{_friction()}" density="0" group="3"/>\n'
        f'      <geom name="object_head_vis" type="box" size="{_fmt3(half)}" '
        f'pos="{_fmt(x_offset)} 0 0" contype="0" conaffinity="0" density="0" group="2" '
        f'rgba="{_HEAD_RGBA}"/>\n'
    )


def tool_mjcf_text(spec: ToolSpec) -> str:
    if (spec.head_scale is None) != (spec.head_density is None):
        raise ValueError(
            f"head_scale and head_density must both be set or both None (got "
            f"{spec.head_scale} and {spec.head_density})"
        )
    geoms = _handle_geoms(spec) + _head_geoms(spec)
    # Standalone files are entity-composition sources:
    # unisim's compose_scene requires identical global options across every
    # source, so each tool carries the scene option block verbatim (the
    # composer overwrites timestep with sim_dt anyway).
    from .robot_mjcf_writer import _scene_option_element

    option_xml = ET.tostring(_scene_option_element(), encoding="unicode")
    return f"""<mujoco model="{_OBJECT_BODY}">
  <compiler angle="radian"/>
  {option_xml}
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

    Headless specs emit placeholder head slots to preserve the same-layout
    rule; headed specs emit the handle+head composite. The ``<inertial>``
    fields are the ToolSpec's own mass, center-of-mass, and diagonal inertia.
    """
    path.write_text(tool_mjcf_text(spec))
    return path


def write_tool_catalog_mjcfs(
    catalog: Sequence[ToolSpec],
    out_dir: str | Path,
    object_base_size: float = OBJECT_BASE_SIZE,
) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Write one MJCF per catalog entry into ``out_dir``.

    Write one file per catalog entry, remove stale ``.xml`` files, and return
    file paths paired with handle scales normalized by ``object_base_size``.
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


__all__ = ["tool_mjcf_text", "write_tool_catalog_mjcfs", "write_tool_mjcf"]
