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
  * headless variants (erasers) OMIT the collision head slot — no epsilon
    placeholder (MuJoCo rejects near-zero-volume meshes, and the uniform
    contract makes the same-layout placeholder unnecessary) — but KEEP a
    visual-only, fully transparent, contactless 2 mm placeholder mesh in the
    ``object_head_vis`` slot (2026-09-28 IsaacSim finding, revision of the
    D8 "omit both head slots" note: the IsaacSim worker pads omitted slots
    with native placeholder prims for PhysX view homogeneity while present
    visual slots own no native shape, so omitting the *visual* slot makes
    the per-variant native-geometry masks disagree and INIT fails closed —
    "entity variants disagree on the native geometry layout"; omitting only
    the collision slot keeps every variant's native mask identical);
  * the slot names stay ``object_handle_col``/``object_handle_vis``/
    ``object_head_col``/``object_head_vis`` (the same-layout public names,
    already unique and non-empty).

Meshes are FILE-BACKED OBJ assets (2026-09-28 IsaacSim finding: IsaacLab's
MJCF importer treats inline ``vertex``/``face`` meshes as file paths and
dies — the dual-backend line requires real files, the same shape
unisimtoolreal ships).  Each mesh lives in the ``meshes/`` sibling directory
of the tool MJCF and is referenced by relative path; the OBJ text is
deterministic (``%.17g`` vertices, 1-based faces) and the filename is
content-addressed (``tool_<sha256[:16]>_{handle,head}.obj``), so the MJCF
text stays derivable from the ToolSpec alone and the pool cache's
byte-identity re-derivation covers both the XML and the OBJ files.  Erasers
share one content-identical placeholder head OBJ across the whole pool.

Only the geom realization changes: ``<inertial>`` is transcribed verbatim
from ``ToolSpec`` (same mass/com/diaginertia as the same-layout writer), the
collision friction comes from the owner YAML through the same
``owner_physics`` path, and the visual geoms keep the group/rgba conventions.

The mesh generators are ported from the proven D8 probe
(``probes/d8_geometry_lib.py``; itself ported from
``unisimtoolreal/scripts/prepare_assets.py``).
"""

from __future__ import annotations

import hashlib
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

# Mesh assets live in this subdirectory next to the tool MJCFs and are
# referenced by relative path (the unisimtoolreal layout).
MESHES_DIRNAME = "meshes"

_HANDLE_MESH = "object_handle_mesh"
_HEAD_MESH = "object_head_mesh"

# Eraser visual placeholder: the smallest mesh MuJoCo accepts (the D8
# probe's practical floor — near-zero-volume meshes are rejected).  It sits
# at the body origin, fully enclosed by every catalog handle (min handle
# half-extent ~5 mm), fully transparent and contactless, so it carries no
# physics and no rendering; every eraser shares one content-identical OBJ.
_PLACEHOLDER_HEAD_VIS_SIZE = 2e-3
_PLACEHOLDER_HEAD_VIS_RGBA = "0.5 0.5 0.5 0"

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


def _obj_text(comment: str, verts: list[Vec3], faces: list[Face]) -> str:
    """Deterministic OBJ: comment header, ``%.17g`` vertices, 1-based faces."""
    lines = [f"# {comment}"]
    lines.extend(f"v {_fmt(x)} {_fmt(y)} {_fmt(z)}" for x, y, z in verts)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    return "\n".join(lines) + "\n"


def _mesh_filename(obj: str, slot: str) -> str:
    """Content-addressed mesh filename (derivable from the ToolSpec alone)."""
    digest = hashlib.sha256(obj.encode()).hexdigest()[:16]
    return f"tool_{digest}_{slot}.obj"


def _head_mesh_obj(spec: ToolSpec) -> str:
    if spec.head_scale is not None:
        assert len(spec.head_scale) == 3, "catalog heads are always boxes (tool_catalog)"
        verts, faces = box_mesh(spec.head_scale)
    else:
        # Eraser visual placeholder (module docstring): the collision head
        # slot is omitted; only the visual slot carries this shared mesh.
        eps = (_PLACEHOLDER_HEAD_VIS_SIZE,) * 3
        verts, faces = box_mesh(eps)
    return _obj_text("simtoolreal_unilab uniform tool head", verts, faces)


def tool_mesh_objs(spec: ToolSpec) -> dict[str, str]:
    """Mesh files for one tool: ``{filename: OBJ text}`` — the handle mesh,
    plus the head mesh (the real head box for headed tools, the shared
    visual placeholder for erasers)."""
    if len(spec.handle_scale) == 3:
        handle_verts, handle_faces = box_mesh(spec.handle_scale)
    else:
        height, diameter = spec.handle_scale
        handle_verts, handle_faces = capsule_mesh(height, diameter)
    handle_obj = _obj_text("simtoolreal_unilab uniform tool handle", handle_verts, handle_faces)
    head_obj = _head_mesh_obj(spec)
    return {
        _mesh_filename(handle_obj, "handle"): handle_obj,
        _mesh_filename(head_obj, "head"): head_obj,
    }


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


def _handle_geoms(spec: ToolSpec) -> str:
    return _mesh_geom_pair("object_handle_col", "object_handle_vis", _HANDLE_MESH, "", _HANDLE_RGBA)


def _head_geoms(spec: ToolSpec) -> str:
    """Head slots: headed tools carry the collision+visual pair at +x past
    the handle; erasers omit the collision slot and keep only the shared
    transparent visual placeholder (module docstring's IsaacSim finding)."""
    if spec.head_scale is None:
        return (
            f'      <geom name="object_head_vis" type="mesh" mesh="{_HEAD_MESH}" '
            f'contype="0" conaffinity="0" density="0" group="2" '
            f'rgba="{_PLACEHOLDER_HEAD_VIS_RGBA}"/>\n'
        )
    x_offset = spec.handle_scale[0] / 2 + spec.head_scale[0] / 2
    return _mesh_geom_pair(
        "object_head_col", "object_head_vis", _HEAD_MESH, f' pos="{_fmt(x_offset)} 0 0"', _HEAD_RGBA
    )


def tool_mjcf_text_uniform(spec: ToolSpec) -> str:
    """Standalone tool MJCF under the uniform-public layout (D8).

    Same validation contract and ``<inertial>`` transcription as
    ``tool_mjcf_writer.tool_mjcf_text``; only the geom realization differs
    (mesh-typed slots everywhere, erasers omit the collision head slot and
    keep a shared transparent visual placeholder).  The mesh assets are
    referenced by relative path under ``meshes/``; the companion OBJ content
    comes from ``tool_mesh_objs`` (same ToolSpec), written by
    ``write_tool_mjcf_uniform`` / the batch writer.
    """
    if (spec.head_scale is None) != (spec.head_density is None):
        raise ValueError(
            f"head_scale and head_density must both be set or both None (got "
            f"{spec.head_scale} and {spec.head_density})"
        )
    import xml.etree.ElementTree as ET

    meshes = tool_mesh_objs(spec)
    handle_file = next(name for name in meshes if name.endswith("_handle.obj"))
    head_file = next(name for name in meshes if name.endswith("_head.obj"))
    assets = (
        f'    <mesh name="{_HANDLE_MESH}" file="{MESHES_DIRNAME}/{handle_file}"/>\n'
        f'    <mesh name="{_HEAD_MESH}" file="{MESHES_DIRNAME}/{head_file}"/>\n'
    )
    option_xml = ET.tostring(_scene_option_element(), encoding="unicode")
    return f"""<mujoco model="{_OBJECT_BODY}">
  <compiler angle="radian"/>
  {option_xml}
  <asset>
{assets}  </asset>
  <worldbody>
    <body name="{_OBJECT_BODY}">
      <freejoint name="{_OBJECT_FREEJOINT}"/>
      <inertial pos="{_fmt3(spec.com)}" mass="{_fmt(spec.mass)}" diaginertia="{_fmt3(spec.diaginertia)}"/>
{_handle_geoms(spec)}{_head_geoms(spec)}    </body>
  </worldbody>
</mujoco>
"""


def write_tool_mjcf_uniform(spec: ToolSpec, path: Path) -> Path:
    """Serialize one ``ToolSpec`` to a standalone uniform-layout MJCF file
    plus its mesh files under ``<path.parent>/meshes/`` (no stale cleanup —
    the batch writer owns directory-level cleanup)."""
    path.write_text(tool_mjcf_text_uniform(spec))
    meshes_dir = path.parent / MESHES_DIRNAME
    meshes_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in tool_mesh_objs(spec).items():
        (meshes_dir / name).write_text(obj)
    return path


__all__ = [
    "MESHES_DIRNAME",
    "box_mesh",
    "capsule_mesh",
    "tool_mesh_objs",
    "tool_mjcf_text_uniform",
    "write_tool_mjcf_uniform",
]
