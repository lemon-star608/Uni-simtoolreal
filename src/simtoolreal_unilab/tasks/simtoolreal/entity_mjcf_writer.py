"""Entity-split MJCF output (B form) shared by both SimToolReal backends.

Both registered backends consume these per-entity sources through the official
M2 channel (unisim ``entity_assets`` + ``EntityVariantBinding``): IsaacSim's
mapped worker composes and converts the entity scene, while MuJoCo composes the
same sources and compiles its VariantPack.  The A form (task-side composed
single models) served as the parity oracle for the M-J5 switch and is retired.
This module writes the asset set from the single sources of truth
(``build_robot_mjcf``, the vendored table URDF, the tool catalog) — no second
copy of any number.  GoalViz is deliberately absent: the headless command hub
owns the authoritative goal and the B-form scene does not materialize a
duplicate tool mirror.

Output layout (see ``write_entity_set``):
    robot.xml            articulation, fixed base, home keyframe embedded
                         (M2's prepare_worker_scene reads keyframes from
                         sources, unisim scene_materialization.py:147-149)
    table.xml            one-body rigid source; pose belongs to the scene
                         declaration (EntityInitialState), so the body sits
                         at the origin here
    floor.xml            one-body rigid plane (M2 adds no implicit ground)
    tools/tool_*.xml     the 1200 standalone tool MJCFs (no scene)
    manifest.json        entity declaration records (the exact
                         SceneEntitySpec/EntityVariantBinding field values
                         the owner YAML will mirror) + digests
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from .config import (
    ARM_DEFAULT_JOINT_POS,
    ENTITY_ASSETS_MANIFEST_SCHEMA,
    JOINT_NAMES_CANONICAL,
    TOOL_POOL_TARGET_ENTITY,
)
from .owner_physics import load_owner_physics
from .robot_mjcf_writer import (
    TABLE_URDF,
    _fmt,
    _fmt_vec,
    _scene_option_element,
    build_robot_mjcf,
)
from .tool_catalog import build_tool_catalog
from .tool_mjcf_writer import write_tool_catalog_mjcfs


def _write_tree(root: ET.Element, path: Path) -> Path:
    ET.indent(root, space="  ")
    path.write_text(ET.tostring(root, encoding="unicode") + "\n")
    return path


def _quat_wxyz_from_rotmat(rot) -> tuple[float, float, float, float]:
    """Rotation matrix (orthonormal, det=+1) to a wxyz quaternion."""
    trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = (
            0.25 * s,
            (rot[2, 1] - rot[1, 2]) / s,
            (rot[0, 2] - rot[2, 0]) / s,
            (rot[1, 0] - rot[0, 1]) / s,
        )
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        quat = (
            (rot[2, 1] - rot[1, 2]) / s,
            0.25 * s,
            (rot[0, 1] + rot[1, 0]) / s,
            (rot[0, 2] + rot[2, 0]) / s,
        )
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        quat = (
            (rot[0, 2] - rot[2, 0]) / s,
            (rot[0, 1] + rot[1, 0]) / s,
            0.25 * s,
            (rot[1, 2] + rot[2, 1]) / s,
        )
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        quat = (
            (rot[1, 0] - rot[0, 1]) / s,
            (rot[0, 2] + rot[2, 0]) / s,
            (rot[1, 2] + rot[2, 1]) / s,
            0.25 * s,
        )
    norm = math.sqrt(sum(q * q for q in quat))
    values = tuple(q / norm for q in quat)
    # Canonical sign: w non-negative (mirrors the compiler's convention).
    return values if values[0] >= 0.0 else tuple(-q for q in values)


def _diagonalize_fullinertia(
    fullinertia: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Full inertia tensor (ixx iyy izz ixy ixz iyz) -> (diaginertia, quat wxyz).

    Eigen-decomposition of the symmetric tensor: eigenvalues are the
    principal moments (diaginertia), eigenvectors the principal-frame
    rotation.  numpy-only, so the task package keeps its no-physics-engine
    import boundary (tests/tasks/test_package_boundary.py).
    """
    import numpy as np

    ixx, iyy, izz, ixy, ixz, iyz = (float(v) for v in fullinertia)
    tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    values, vectors = np.linalg.eigh(tensor)
    if np.linalg.det(vectors) < 0.0:
        vectors[:, -1] = -vectors[:, -1]
    return tuple(float(v) for v in values), _quat_wxyz_from_rotmat(vectors)


def _normalize_inertials(root: ET.Element) -> None:
    """Rewrite every body's ``<inertial>`` in the diagonal spelling.

    The IsaacSim worker's scene staging (unisim subprocess_ipc
    scene_materialization ``prepare_worker_scene``) rewrites per-body
    inertials as ``diaginertia``+``iquat`` without clearing a source
    ``fullinertia`` attribute, and MjSpec rejects the mixed spelling
    ("fullinertia and diagonal inertia cannot both be specified") — an
    upstream staging gap for fullinertia-carrying sources.  Emitting the
    diagonal form (the eigen-decomposition of the same tensor, which is what
    the mujoco compiler reduces fullinertia to at build time, so the compiled
    model is unchanged) keeps one numerics source while satisfying the
    staging's diagonal-only contract.
    """
    for inertial in root.iter("inertial"):
        full = inertial.get("fullinertia")
        if full is None:
            continue
        diag, quat = _diagonalize_fullinertia(tuple(float(v) for v in full.split()))
        del inertial.attrib["fullinertia"]
        inertial.set("diaginertia", _fmt_vec(diag))
        inertial.set("quat", _fmt_vec(quat))


def _build_robot_entity() -> ET.Element:
    """Robot tree + the home keyframe (M2 reads keyframes from sources).

    qpos/ctrl are the 29-DoF home pose (ARM_DEFAULT_JOINT_POS + hand zeros);
    the standalone robot file has no object freejoint — the composer merges
    in the object home pose at scene level.
    """
    robot = build_robot_mjcf()
    _normalize_inertials(robot)
    # M-J5: as a composition source the robot must carry the scene's global
    # option block (unisim compose_scene requires agreeing options).
    robot.append(_scene_option_element())
    home = [ARM_DEFAULT_JOINT_POS[name] for name in JOINT_NAMES_CANONICAL[:7]] + [0.0] * 22
    keyframe = ET.SubElement(robot, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        {"name": "home", "qpos": _fmt_vec(home), "ctrl": _fmt_vec(home)},
    )
    return robot


def _build_table_entity() -> ET.Element:
    """One-body table source at the origin; pose is a declaration concern."""
    physics = load_owner_physics()
    root = ET.parse(TABLE_URDF).getroot()
    box = root.find("link/collision/geometry/box")
    if box is None:
        raise ValueError(f"{TABLE_URDF}: expected a collision box")
    half = tuple(float(v) / 2 for v in box.get("size").split())
    rgba_el = root.find("link/visual/material/color")
    rgba = rgba_el.get("rgba") if rgba_el is not None else "0.82 0.56 0.35 1.0"
    mujoco_el = ET.Element("mujoco", {"model": "table"})
    ET.SubElement(mujoco_el, "compiler", {"angle": "radian"})
    mujoco_el.append(_scene_option_element())
    worldbody = ET.SubElement(mujoco_el, "worldbody")
    body = ET.SubElement(worldbody, "body", {"name": "table", "pos": "0 0 0"})
    ET.SubElement(
        body,
        "geom",
        {
            "name": "table_box",
            "type": "box",
            "size": _fmt_vec(half),
            "friction": f"{_fmt(physics.table_slide_friction)} 0.005 0.0001",
            "rgba": rgba,
        },
    )
    return mujoco_el


def _build_floor_entity() -> ET.Element:
    """One-body floor plane (M2 adds no implicit ground; the floor is ours)."""
    mujoco_el = ET.Element("mujoco", {"model": "floor"})
    ET.SubElement(mujoco_el, "compiler", {"angle": "radian"})
    mujoco_el.append(_scene_option_element())
    worldbody = ET.SubElement(mujoco_el, "worldbody")
    body = ET.SubElement(worldbody, "body", {"name": "floor", "pos": "0 0 0"})
    ET.SubElement(
        body,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "0 0 0.05",
            # Same transcribed value as the old tree's scene.xml floor geom.
            "friction": "1.0 0.005 0.0001",
        },
    )
    return mujoco_el


def write_entity_set(
    out_dir: str | Path,
    *,
    seed: int = 42,
    num_per_type: int = 100,
) -> dict:
    """Write the entity-split MJCF asset set and return its manifest dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    robot_path = _write_tree(_build_robot_entity(), out_dir / "robot.xml")
    table_path = _write_tree(_build_table_entity(), out_dir / "table.xml")
    floor_path = _write_tree(_build_floor_entity(), out_dir / "floor.xml")

    catalog = build_tool_catalog(num_per_type=num_per_type, seed=seed, shuffle=True)
    tools_dir = out_dir / "tools"
    tool_paths, _ = write_tool_catalog_mjcfs(catalog, tools_dir)

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "schema": ENTITY_ASSETS_MANIFEST_SCHEMA,
        "seed": int(seed),
        "num_per_type": int(num_per_type),
        "tool_count": len(catalog),
        # Catalog-order tool file names (the variant catalog order the task
        # side's mass/scale tables share).
        "tool_files": [Path(p).name for p in tool_paths],
        "tool_digests": {Path(p).name: digest(Path(p)) for p in tool_paths},
        "layout": "same_layout",
        "entities": {
            "robot": {
                "source": robot_path.name,
                "asset_format": "mjcf",
                "kind": "articulation",
                "root_mode": "fixed",
                # self_collision is not declared on the mujoco leaf: the MJCF
                # compose path is fail-closed on the toggle and keeps the
                # source-authored exclusions (robot.xml carries all 35).  The
                # The IsaacSim leaf declares it explicitly; the MuJoCo compose
                # path instead keeps the source-authored excludes and omits the
                # unsupported toggle.
                "default_keyframe_name": "home",
            },
            "table": {
                "source": table_path.name,
                "asset_format": "mjcf",
                "kind": "rigid",
                "root_mode": "kinematic",
            },
            "floor": {
                "source": floor_path.name,
                "asset_format": "mjcf",
                "kind": "rigid",
                "root_mode": "kinematic",
            },
            # Use the production entity name even though the variant sources
            # are tools: this manifest describes the B-form scene consumed by
            # the owner YAML.
            TOOL_POOL_TARGET_ENTITY: {
                "asset_format": "mjcf",
                "kind": "rigid",
                "root_mode": "floating",
                "variant_binding": {
                    "target_entity": TOOL_POOL_TARGET_ENTITY,
                    "layout": "same_layout",
                },
            },
        },
        "digests": {
            robot_path.name: digest(robot_path),
            table_path.name: digest(table_path),
            floor_path.name: digest(floor_path),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def write_bootstrap_entity_set(out_dir: str | Path) -> dict[str, Path]:
    """Write the pool-independent entity sources for the scene declaration.

    The owner YAML's ``entity_assets`` needs concrete source files at Hydra
    resolution time — before the env config's pool knobs (seed/num_per_type)
    settle.  Robot/table/floor are pool-independent; the object entity's
    source is one canonical catalog entry (the deterministic bootstrap
    hammer) whose per-env identity the variant binding replaces at build time.
    """
    from .tool_mjcf_writer import write_tool_mjcf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "robot": _write_tree(_build_robot_entity(), out_dir / "robot.xml"),
        "table": _write_tree(_build_table_entity(), out_dir / "table.xml"),
        "floor": _write_tree(_build_floor_entity(), out_dir / "floor.xml"),
    }
    spec = build_tool_catalog(types=("hammer",), num_per_type=1, seed=42, shuffle=False)[0]
    paths["object"] = write_tool_mjcf(spec, out_dir / "object_bootstrap.xml")
    return paths


__all__ = ["write_bootstrap_entity_set", "write_entity_set"]
