"""Uniform-public-layout MJCF serialization for the SimToolReal tool catalog.

Decision D8 (2026-09-28, probe report ``d8_geometry_fidelity_2026-09-28.md``):
the pool migrates from per-variant analytic primitives (SAME_LAYOUT,
``tool_mjcf_writer``) to a full-mesh uniform pool under
``FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT``.  Unisim's uniform contract
(``mjcf_compiler.validate_uniform_entity_variant_layout_summaries``) requires
unique non-empty geom names, pins the canonical geom type/body per slot, and
only permits *mesh-typed* slots to be absent from a variant — so:

  * every geom slot is mesh-typed: the handle is a 24-segment lathe capsule
    mesh (cylinder handles, authored x-aligned so no quat is needed) or an
    8-vertex box mesh (box handles), and the head is an 8-vertex box mesh;
  * headless variants (erasers) OMIT both head slots entirely — no epsilon
    placeholder (MuJoCo rejects near-zero-volume meshes, and the uniform
    contract makes the same-layout placeholder unnecessary);
  * the slot names stay ``object_handle_col``/``object_handle_vis``/
    ``object_head_col``/``object_head_vis`` (the same-layout public names,
    already unique and non-empty).

Only the geom realization changes: ``<inertial>`` is transcribed verbatim
from ``ToolSpec`` (same mass/com/diaginertia as the same-layout writer), the
collision friction comes from the owner YAML through the same
``owner_physics`` path, and the visual geoms keep the group/rgba conventions.
Meshes are inlined as ``vertex``/``face`` data so each tool MJCF stays one
self-contained file (no asset path resolution in the compose pipeline, and
the pool cache's byte-identity re-derivation stays simple).

The mesh generators and inline writers are ported from the proven D8 probe
(``probes/d8_geometry_lib.py``; itself ported from
``unisimtoolreal/scripts/prepare_assets.py``).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from .robot_mjcf_writer import _scene_option_element
from .tool_catalog import ToolSpec
from .tool_mjcf_writer import (
    _HANDLE_RGBA,
    _HEAD_RGBA,
    _OBJECT_BODY,
    _OBJECT_FREEJOINT,
    _fmt,
    _fmt3,
    _friction,
)

# Lathe discretization of the capsule handle (D8): 24 sectors, 4 hemisphere
# rows per cap — the probe's physics-validated resolution.
_CAPSULE_SEGMENTS = 24
_CAPSULE_CAP_ROWS = 4

_HANDLE_MESH = "object_handle_mesh"
_HEAD_MESH = "object_head_mesh"

Vec3 = tuple[float, float, float]
Face = tuple[int, int, int]


def box_mesh(dimensions: Sequence[float]) -> tuple[list[Vec3], list[Face]]:
    """8-vertex box mesh; faces wound so normals point outward."""
    hx, hy, hz = (float(v) / 2 for v in dimensions)
    verts: list[Vec3] = [
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    ]
    faces: list[Face] = [
        (0, 3, 2),
        (0, 2, 1),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    return verts, faces


def capsule_mesh(
    length: float,
    diameter: float,
    segments: int = _CAPSULE_SEGMENTS,
    cap_rows: int = _CAPSULE_CAP_ROWS,
) -> tuple[list[Vec3], list[Face]]:
    """Lathe capsule along +x: ring profile swept over ``segments`` sectors.

    Vertices lie on the analytic capsule surface; faces are chords, so the
    mesh is inscribed by r*(1-cos(pi/segments)) (~0.86% of r at 24 segments).
    """
    radius = diameter / 2
    half_length = length / 2
    profiles: list[tuple[float, float]] = [(-half_length - radius, 0.0)]
    for row in range(1, cap_rows):
        angle = math.pi * row / (2 * cap_rows)
        profiles.append((-half_length - radius * math.sin(angle), radius * math.cos(angle)))
    profiles.extend(((-half_length, radius), (half_length, radius)))
    for row in range(1, cap_rows):
        angle = math.pi * row / (2 * cap_rows)
        profiles.append((half_length + radius * math.sin(angle), radius * math.cos(angle)))
    profiles.append((half_length + radius, 0.0))

    verts: list[Vec3] = []
    ring_starts: list[int] = []
    for center, ring_radius in profiles:
        ring_starts.append(len(verts))
        if ring_radius == 0.0:
            verts.append((center, 0.0, 0.0))
            continue
        for segment in range(segments):
            angle = 2 * math.pi * segment / segments
            verts.append((center, ring_radius * math.cos(angle), ring_radius * math.sin(angle)))
    faces: list[Face] = []
    for ring in range(len(profiles) - 1):
        left_size = 1 if profiles[ring][1] == 0.0 else segments
        right_size = 1 if profiles[ring + 1][1] == 0.0 else segments
        left_start = ring_starts[ring]
        right_start = ring_starts[ring + 1]
        for segment in range(segments):
            left_a = left_start + (segment % left_size)
            left_b = left_start + ((segment + 1) % left_size)
            right_a = right_start + (segment % right_size)
            right_b = right_start + ((segment + 1) % right_size)
            faces.append((left_a, right_b, left_b))
            faces.append((left_a, right_a, right_b))
    return verts, faces


def _inline_mesh(name: str, verts: list[Vec3], faces: list[Face]) -> str:
    v = " ".join(_fmt(c) for vert in verts for c in vert)
    f = " ".join(str(i) for face in faces for i in face)
    return f'    <mesh name="{name}" vertex="{v}" face="{f}"/>\n'


def _mesh_geom_pair(slot_col: str, slot_vis: str, mesh: str, pos: str, rgba: str) -> str:
    """Collision + visual geom slots sharing one mesh (same conventions as the
    same-layout writer: friction on the collision slot, group 3; contactless
    visual, group 2)."""
    return (
        f'      <geom name="{slot_col}" type="mesh" mesh="{mesh}"{pos} '
        f'friction="{_friction()}" density="0" group="3"/>\n'
        f'      <geom name="{slot_vis}" type="mesh" mesh="{mesh}"{pos} '
        f'contype="0" conaffinity="0" density="0" group="2" rgba="{rgba}"/>\n'
    )


def _handle_mesh_geoms(spec: ToolSpec) -> tuple[str, str]:
    if len(spec.handle_scale) == 3:
        verts, faces = box_mesh(spec.handle_scale)
    else:
        height, diameter = spec.handle_scale
        verts, faces = capsule_mesh(height, diameter)
    return _inline_mesh(_HANDLE_MESH, verts, faces), _mesh_geom_pair(
        "object_handle_col", "object_handle_vis", _HANDLE_MESH, "", _HANDLE_RGBA
    )


def _head_mesh_geoms(spec: ToolSpec) -> tuple[str, str]:
    """Head slots for headed tools; erasers omit both slots (module docstring)."""
    if spec.head_scale is None:
        return "", ""
    assert len(spec.head_scale) == 3, "catalog heads are always boxes (tool_catalog)"
    x_offset = spec.handle_scale[0] / 2 + spec.head_scale[0] / 2
    verts, faces = box_mesh(spec.head_scale)
    return _inline_mesh(_HEAD_MESH, verts, faces), _mesh_geom_pair(
        "object_head_col",
        "object_head_vis",
        _HEAD_MESH,
        f' pos="{_fmt(x_offset)} 0 0"',
        _HEAD_RGBA,
    )


def tool_mjcf_text_uniform(spec: ToolSpec) -> str:
    """Standalone tool MJCF under the uniform-public layout (D8).

    Same validation contract and ``<inertial>`` transcription as
    ``tool_mjcf_writer.tool_mjcf_text``; only the geom realization differs
    (meshes everywhere, erasers omit the head slots).
    """
    if (spec.head_scale is None) != (spec.head_density is None):
        raise ValueError(
            f"head_scale and head_density must both be set or both None (got "
            f"{spec.head_scale} and {spec.head_density})"
        )
    import xml.etree.ElementTree as ET

    handle_asset, handle_geoms = _handle_mesh_geoms(spec)
    head_asset, head_geoms = _head_mesh_geoms(spec)
    option_xml = ET.tostring(_scene_option_element(), encoding="unicode")
    return f"""<mujoco model="{_OBJECT_BODY}">
  <compiler angle="radian"/>
  {option_xml}
  <asset>
{handle_asset}{head_asset}  </asset>
  <worldbody>
    <body name="{_OBJECT_BODY}">
      <freejoint name="{_OBJECT_FREEJOINT}"/>
      <inertial pos="{_fmt3(spec.com)}" mass="{_fmt(spec.mass)}" diaginertia="{_fmt3(spec.diaginertia)}"/>
{handle_geoms}{head_geoms}    </body>
  </worldbody>
</mujoco>
"""


def write_tool_mjcf_uniform(spec: ToolSpec, path: Path) -> Path:
    """Serialize one ``ToolSpec`` to a standalone uniform-layout MJCF file."""
    path.write_text(tool_mjcf_text_uniform(spec))
    return path


__all__ = [
    "box_mesh",
    "capsule_mesh",
    "tool_mjcf_text_uniform",
    "write_tool_mjcf_uniform",
]
