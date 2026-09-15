"""URDF serialization for the SimToolReal tool catalog.

Literal port of the original IsaacSim task's writer at
``simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/generate_objects.py``.
Sampling stays in ``tool_catalog.py``; this module only turns ``ToolSpec``
samples into URDF files.  Inline comments cite the original line numbers.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Sequence

from .tool_catalog import OBJECT_BASE_SIZE, ToolSpec

# ----------------------------------------------------------------------------
# Primitive URDF emitters (generate_objects.py:34-125)
# ----------------------------------------------------------------------------


# All procedural object URDFs use this same link name so that spawned prims
# have an identical internal structure across envs (env_*/Object/<LINK>).
# Matches generate_objects.py:39-44.
_OBJECT_ROOT_LINK = "object_root"


def _compute_mass_and_inertia(
    scale: Sequence[float], density: float
) -> tuple[float, float, float, float]:
    """Capsule-approximation for cylinders; exact for cuboids.

    Returns ``(m, ixx, iyy, izz)`` with scale-axis = z (caller flips if
    needed).  Matches generate_objects.py:133-163.
    """
    if len(scale) == 3:
        lx, ly, lz = scale
        v = lx * ly * lz
        m = v * density
        ixx = (1 / 12) * m * (ly * ly + lz * lz)
        iyy = (1 / 12) * m * (lx * lx + lz * lz)
        izz = (1 / 12) * m * (lx * lx + ly * ly)
        return m, ixx, iyy, izz
    if len(scale) == 2:
        h, d = scale[0], scale[1]
        r = d / 2
        # Capsule mass = cylinder + two hemispheres.
        m_c = density * math.pi * r * r * h
        m_h = density * (2 / 3) * math.pi * r**3
        m = m_c + 2 * m_h
        # Cylinder inertia about centroid (axis = z).
        i_c_axis = 0.5 * m_c * r * r
        i_c_perp = (1 / 12) * m_c * (3 * r * r + h * h)
        # Hemisphere inertia about its own centroid.
        i_h_axis = (2 / 5) * m_h * r * r
        i_h_perp = (83 / 320) * m_h * r * r
        d_com = (h / 2) + (3 * r / 8)
        izz = i_c_axis + 2 * i_h_axis
        ixx = iyy = i_c_perp + 2 * (i_h_perp + m_h * d_com * d_com)
        return m, ixx, iyy, izz
    raise ValueError(f"Invalid scale: {scale}")


def _cuboid_urdf_constant_density(path: Path, scale: Sequence[float], density: float) -> Path:
    """Matches generate_objects.py:47-77."""
    lx, ly, lz = scale
    # Standard URDF requires <mass> + <inertia>; emit them explicitly so both
    # strict URDF importers and Gazebo-extension importers see identical
    # physics (generate_objects.py:49-55).
    m, ixx, iyy, izz = _compute_mass_and_inertia(scale, density)
    urdf = f"""<?xml version="1.0"?>
<robot name="cuboid">
  <link name="{_OBJECT_ROOT_LINK}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="{lx} {ly} {lz}"/></geometry>
      <material name="brown"><color rgba="0.55 0.27 0.07 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="{lx} {ly} {lz}"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{m}"/>
      <inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>
"""
    path.write_text(urdf)
    return path


def _cylinder_urdf_constant_density(
    path: Path, height: float, diameter: float, density: float
) -> Path:
    """Matches generate_objects.py:80-112."""
    radius = diameter / 2
    # The geometry is rotated by rpy="0 -pi/2 0" so the cylinder's length axis
    # aligns with link x.  _compute_mass_and_inertia returns inertia in the
    # geometry-local frame (axis = z); we apply the same rpy on the inertial
    # origin so the values stay correct after rotation.
    m, ixx, iyy, izz = _compute_mass_and_inertia((height, diameter), density)
    urdf = f"""<?xml version="1.0"?>
<robot name="cylinder">
  <link name="{_OBJECT_ROOT_LINK}">
    <visual>
      <origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>
      <geometry><cylinder length="{height}" radius="{radius}"/></geometry>
      <material name="brown"><color rgba="0.55 0.27 0.07 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>
      <geometry><cylinder length="{height}" radius="{radius}"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>
      <mass value="{m}"/>
      <inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>
"""
    path.write_text(urdf)
    return path


def _handle_urdf(path: Path, handle_scale: Sequence[float], handle_density: float) -> Path:
    """Emit a URDF for a single cuboid or cylinder handle.

    Matches ``generate_handle_urdf`` at generate_objects.py:115-125.
    """
    if len(handle_scale) == 3:
        return _cuboid_urdf_constant_density(path, handle_scale, handle_density)
    if len(handle_scale) == 2:
        return _cylinder_urdf_constant_density(
            path, handle_scale[0], handle_scale[1], handle_density
        )
    raise ValueError(f"Invalid handle_scale: {handle_scale}")


# ----------------------------------------------------------------------------
# Handle + head composite (generate_objects.py:128-259)
# ----------------------------------------------------------------------------


def _handle_head_urdf_variable_density(
    path: Path,
    handle_scale: Sequence[float],
    head_scale: Sequence[float],
    handle_density: float,
    head_density: float,
) -> Path:
    """Single-link composite with parallel-axis-adjusted inertia.

    Matches generate_objects.py:166-259.
    """
    # Handle geometry + inertia (axis along +x; for cylinders we rotate -pi/2 about y).
    if len(handle_scale) == 3:
        lx, ly, lz = handle_scale
        handle_geom = (
            f'<origin xyz="0 0 0" rpy="0 0 0"/>\n'
            f'        <geometry><box size="{lx} {ly} {lz}"/></geometry>'
        )
        handle_mass, handle_ixx, handle_iyy, handle_izz = _compute_mass_and_inertia(
            handle_scale, handle_density
        )
    else:
        h, d = handle_scale
        r = d / 2
        handle_geom = (
            f'<origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>\n'
            f'        <geometry><cylinder length="{h}" radius="{r}"/></geometry>'
        )
        # Rotated so handle axis is along +x; flip ixx <-> izz.
        handle_mass, handle_izz, handle_iyy, handle_ixx = _compute_mass_and_inertia(
            handle_scale, handle_density
        )

    # Head geometry + inertia.
    if len(head_scale) == 3:
        hlx, hly, hlz = head_scale
        x_offset = handle_scale[0] / 2 + hlx / 2
        head_geom = (
            f'<origin xyz="{x_offset} 0 0" rpy="0 0 0"/>\n'
            f'        <geometry><box size="{hlx} {hly} {hlz}"/></geometry>'
        )
        head_mass, head_ixx, head_iyy, head_izz = _compute_mass_and_inertia(
            head_scale, head_density
        )
    else:
        hh, hd = head_scale
        hr = hd / 2
        x_offset = handle_scale[0] / 2 + hr
        head_geom = (
            f'<origin xyz="{x_offset} 0 0" rpy="-1.5707963267948966 0 0"/>\n'
            f'        <geometry><cylinder length="{hh}" radius="{hr}"/></geometry>'
        )
        # Rotated so head axis is along +y; flip iyy <-> izz.
        head_mass, head_ixx, head_izz, head_iyy = _compute_mass_and_inertia(
            head_scale, head_density
        )

    # Parallel-axis shift to composite COM (handle at origin, head at +x_offset).
    total_mass = handle_mass + head_mass
    com_x = (handle_mass * 0.0 + head_mass * x_offset) / total_mass
    d_handle = -com_x
    d_head = x_offset - com_x

    ixx = handle_ixx + head_ixx
    iyy = (handle_iyy + handle_mass * d_handle * d_handle) + (
        head_iyy + head_mass * d_head * d_head
    )
    izz = (handle_izz + handle_mass * d_handle * d_handle) + (
        head_izz + head_mass * d_head * d_head
    )

    urdf = f"""<?xml version="1.0"?>
<robot name="handle_head">
  <link name="{_OBJECT_ROOT_LINK}">
    <visual>
      {handle_geom}
      <material name="brown"><color rgba="0.55 0.27 0.07 1.0"/></material>
    </visual>
    <collision>
      {handle_geom}
    </collision>
    <visual>
      {head_geom}
      <material name="gray"><color rgba="0.5 0.5 0.5 1.0"/></material>
    </visual>
    <collision>
      {head_geom}
    </collision>
    <inertial>
      <origin xyz="{com_x} 0 0" rpy="0 0 0"/>
      <mass value="{total_mass}"/>
      <inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>
"""
    path.write_text(urdf)
    return path


def write_tool_urdf(spec: ToolSpec, path: Path) -> Path:
    """Serialize one ``ToolSpec`` to a URDF file.

    Dispatch matches ``generate_handle_head_urdf`` at
    generate_objects.py:262-285: a spec without a head takes the handle-only
    branch of ``generate_handle_urdf`` (generate_objects.py:115-125); a spec
    with both head scale and density takes the single-link variable-density
    composite (the unstable 2-link variant is not ported).
    """
    if spec.head_scale is None and spec.head_density is None:
        return _handle_urdf(path, spec.handle_scale, spec.handle_density)
    if spec.head_scale is not None and spec.head_density is not None:
        return _handle_head_urdf_variable_density(
            path, spec.handle_scale, spec.head_scale, spec.handle_density, spec.head_density
        )
    raise ValueError(
        f"head_scale and head_density must both be set or both None (got "
        f"{spec.head_scale} and {spec.head_density})"
    )


# ----------------------------------------------------------------------------
# Catalog-level batch entry (generate_objects.py:288-399, minus sampling)
# ----------------------------------------------------------------------------


def write_tool_catalog_urdfs(
    catalog: Sequence[ToolSpec],
    out_dir: str | Path,
    object_base_size: float = OBJECT_BASE_SIZE,
) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Write one URDF per catalog entry into ``out_dir``.

    Serialization counterpart of ``generate_handle_head_urdfs`` at
    generate_objects.py:302-399: the sampling steps (seed, per-distribution
    draws, lockstep shuffle) live in ``tool_catalog.build_tool_catalog``;
    this function keeps the output-directory handling (stale ``.urdf``
    cleanup), the filename template (generate_objects.py:366-370), and the
    return contract — file paths paired with handle scales normalized by
    ``object_base_size`` (generate_objects.py:382-387).

    The ``{idx:03d}`` filename field is a per-type ordinal over the given
    catalog order; the original uses a per-distribution loop index.  The
    sampled values embedded in the name keep filenames unique either way.

    Returns
    -------
    urdf_paths : list[str]
        Paths to the generated URDFs, in catalog order.
    object_scales_normalized : list[tuple[float, float, float]]
        Per-entry 3-tuple handle scales normalized by ``object_base_size``.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        for p in out_dir.iterdir():
            if p.suffix == ".urdf":
                p.unlink()
    else:
        os.makedirs(out_dir)

    paths: list[str] = []
    scales_norm: list[tuple[float, float, float]] = []
    type_counts: dict[str, int] = {}
    for spec in catalog:
        idx = type_counts.get(spec.type, 0)
        type_counts[spec.type] = idx + 1
        fname = (
            f"{idx:03d}_{spec.type}_handle_{tuple(spec.handle_scale)}"
            f"_head_{None if spec.head_scale is None else tuple(spec.head_scale)}"
            f"_d{spec.handle_density:.1f}_{spec.head_density}".replace(".", "-")
            + ".urdf"
        )
        urdf_path = out_dir / fname
        write_tool_urdf(spec, urdf_path)
        paths.append(str(urdf_path))
        # ToolSpec.object_scale is the handle scale already normalized by
        # OBJECT_BASE_SIZE; rescale if the caller overrides the base size.
        factor = OBJECT_BASE_SIZE / object_base_size
        sx, sy, sz = spec.object_scale
        scales_norm.append((sx * factor, sy * factor, sz * factor))

    return paths, scales_norm


__all__ = ["write_tool_catalog_urdfs", "write_tool_urdf"]
