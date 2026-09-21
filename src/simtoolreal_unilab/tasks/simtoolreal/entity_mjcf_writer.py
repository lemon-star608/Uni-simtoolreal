"""Entity-split MJCF output (B form) — the MuJoCo line's asset set (M-J5).

The MuJoCo line consumes per-entity sources through the official M2 channel
(unisim ``entity_assets`` + ``EntityVariantBinding``); the A form (task-side
composed single models) served as the parity oracle for the M-J5 switch and
is retired.  This module writes the asset set from the single sources of
truth (``build_robot_mjcf``, the vendored table URDF, the tool catalog) — no
second copy of any number.

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
import xml.etree.ElementTree as ET
from pathlib import Path

from .config import ARM_DEFAULT_JOINT_POS, JOINT_NAMES_CANONICAL
from .owner_physics import load_owner_physics
from .robot_mjcf_writer import (
    ROBOT_URDF,
    TABLE_URDF,
    _fmt,
    _fmt_vec,
    _scene_option_element,
    build_robot_mjcf,
)
from .tool_catalog import build_tool_catalog
from .tool_mjcf_writer import write_tool_catalog_mjcfs

# The mirror needs no file: M2's ``mirror_of`` inherits the target's source
# (unisim entities.py:89).  The declaration shape below was read off
# /tmp/unisim_cl @ 069e3b0 (entities.py:60-100: a mirror must be rigid,
# kinematic and collision-disabled, and must not carry a source).
GOALVIZ_MIRROR_DECLARATION = {
    "name": "goalviz",
    "mirror_of": "tool",
    "kind": "rigid",
    "root_mode": "kinematic",
    "collision_enabled": False,
}


def _write_tree(root: ET.Element, path: Path) -> Path:
    ET.indent(root, space="  ")
    path.write_text(ET.tostring(root, encoding="unicode") + "\n")
    return path


def _build_robot_entity() -> ET.Element:
    """Robot tree + the home keyframe (M2 reads keyframes from sources).

    qpos/ctrl are the 29-DoF home pose (ARM_DEFAULT_JOINT_POS + hand zeros);
    the standalone robot file has no object freejoint — the composer merges
    in the object home pose at scene level.
    """
    robot = build_robot_mjcf()
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
        "schema": "simtoolreal_unilab.entity_assets.v1",
        "seed": int(seed),
        "num_per_type": int(num_per_type),
        "tool_count": len(catalog),
        # Catalog-order tool file names (the variant catalog order the task
        # side's mass/scale tables share).
        "tool_files": [Path(p).name for p in tool_paths],
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
                # IsaacSim leaf declares it once the pin crosses #270, along
                # with gravity_disabled.
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
            "tool": {
                "asset_format": "mjcf",
                "kind": "rigid",
                "root_mode": "floating",
                "variant_binding": True,
            },
            "goalviz": dict(GOALVIZ_MIRROR_DECLARATION),
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


__all__ = ["GOALVIZ_MIRROR_DECLARATION", "write_bootstrap_entity_set", "write_entity_set"]
