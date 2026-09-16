"""Scene composition and the 1200-variant MJCF tool pool (roadmap M-J1, D3/D5).

``compose_scene_mjcf`` merges a tool body into the base scene's XML tree
(no ``<include>`` — the composed file is a complete single-model MJCF, with
mesh references already absolute from generation time).
``materialize_tool_pool_mjcf`` mirrors the URDF pool mechanism
(``env_cfg._materialize_pool``): a deterministic directory
``<root>/simtoolreal_tool_pool_mjcf/seed_{seed}_n{num_per_type}/`` holding the
base scene, one composed scene per catalog entry, and a manifest
(spec/files/masses/scales) that is reused when complete and rebuilt when
partial or stale.
"""

from __future__ import annotations

import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from .config import ResetCfg
from .robot_mjcf_writer import write_base_scene_mjcf
from .tool_catalog import ToolSpec, build_tool_catalog
from .tool_mjcf_writer import _tool_filename, _tool_mjcf_text

_POOL_DIR_NAME = "simtoolreal_tool_pool_mjcf"
_BASE_SCENE_NAME = "base_scene.xml"


def _object_spawn_z() -> float:
    """Object home height: table surface plus the task's z offset.

    ResetCfg.table_reset_z + table_object_z_offset (0.38 + 0.25 = 0.63); the
    same value the old tree's hammer_single.xml used.  The reset event owns
    the actual per-episode spawn; this is the compile-time home pose.
    """
    cfg = ResetCfg()
    return cfg.table_reset_z + cfg.table_object_z_offset


def compose_scene_mjcf(
    base_scene_path: str | Path,
    tool: ToolSpec | str | Path,
    out_path: str | Path,
) -> Path:
    """Compose base scene + tool body into one complete single-model MJCF.

    ``tool`` is either a ``ToolSpec`` (body generated in memory) or a path to
    a standalone tool MJCF written by ``write_tool_mjcf``.  The tool body is
    appended to the scene worldbody at the object home pose, and the home
    keyframe's qpos gains the object's freejoint coordinates (pos + wxyz
    quat); ctrl is unchanged (the freejoint is not actuated).
    """
    scene = ET.parse(str(base_scene_path)).getroot()
    worldbody = scene.find("worldbody")
    if worldbody is None:
        raise ValueError(f"base scene has no worldbody: {base_scene_path}")

    if isinstance(tool, ToolSpec):
        tool_root = ET.fromstring(_tool_mjcf_text(tool))
    else:
        tool_root = ET.parse(str(tool)).getroot()
    tool_body = tool_root.find("worldbody/body")
    if tool_body is None:
        raise ValueError("tool MJCF has no worldbody body")
    spawn_z = _object_spawn_z()
    tool_body.set("pos", f"0 0 {spawn_z:.17g}")
    worldbody.append(tool_body)

    key = scene.find("keyframe/key")
    if key is not None:
        qpos = key.get("qpos", "").split()
        key.set("qpos", " ".join([*qpos, f"0 0 {spawn_z:.17g}", "1 0 0 0"]))

    ET.indent(scene, space="  ")
    out_path = Path(out_path)
    out_path.write_text(ET.tostring(scene, encoding="unicode") + "\n")
    return out_path


def _pool_root(seed: int, num_per_type: int, configured: str | None) -> Path:
    """Pool directory, mirroring ``env_cfg._pool_root`` for the MJCF line."""
    if configured is not None and str(configured).strip():
        root = Path(str(configured)).expanduser()
    else:
        env_root = os.environ.get("SIMTOOLREAL_TOOL_POOL_ROOT", "").strip()
        base = Path(env_root).expanduser() if env_root else Path(tempfile.gettempdir())
        root = base / _POOL_DIR_NAME
    return root / f"seed_{seed}_n{num_per_type}"


def materialize_tool_pool_mjcf(
    *, seed: int = 42, num_per_type: int = 100, configured_root: str | None = None
) -> tuple[tuple[Path, ...], tuple[float, ...], tuple[tuple[float, float, float], ...]]:
    """Write or reuse the deterministic composed-MJCF pool + mass/scale tables.

    Same reuse contract as the URDF pool: a complete manifest matching the
    requested spec with every file present is reused as-is; anything partial
    or stale is rebuilt from scratch.
    """
    catalog = build_tool_catalog(num_per_type=num_per_type, seed=seed, shuffle=True)
    root = _pool_root(seed, num_per_type, configured_root)
    manifest_path = root / "manifest.json"
    expected = {
        "seed": int(seed),
        "num_per_type": int(num_per_type),
        "count": len(catalog),
    }
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            names = tuple(str(value) for value in manifest["files"])
            masses = tuple(float(value) for value in manifest["masses"])
            scales = tuple(
                tuple(float(component) for component in scale) for scale in manifest["scales"]
            )
            scales_complete = len(scales) == len(catalog) and all(
                len(scale) == 3 for scale in scales
            )
            if (
                manifest.get("spec") == expected
                and manifest.get("base_scene") == _BASE_SCENE_NAME
                and len(names) == len(catalog)
            ):
                paths = tuple(root / name for name in names)
                if (
                    (root / _BASE_SCENE_NAME).is_file()
                    and all(path.is_file() for path in paths)
                    and len(masses) == len(catalog)
                    and scales_complete
                ):
                    return paths, masses, scales
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            # A partial or stale pool is rebuilt below.
            pass

    root.mkdir(parents=True, exist_ok=True)
    for stale in root.iterdir():
        if stale.suffix == ".xml":
            stale.unlink()
    base_scene = write_base_scene_mjcf(root / _BASE_SCENE_NAME)
    paths: list[Path] = []
    type_counts: dict[str, int] = {}
    for spec in catalog:
        idx = type_counts.get(spec.type, 0)
        type_counts[spec.type] = idx + 1
        out = root / _tool_filename(spec, idx)
        compose_scene_mjcf(base_scene, spec, out)
        paths.append(out)
    path_tuple = tuple(path.resolve() for path in paths)
    masses = tuple(float(spec.mass) for spec in catalog)
    scales = tuple(tuple(float(x) for x in spec.object_scale) for spec in catalog)
    manifest_path.write_text(
        json.dumps(
            {
                "spec": expected,
                "base_scene": _BASE_SCENE_NAME,
                "files": [path.name for path in path_tuple],
                "masses": masses,
                "scales": scales,
            },
            indent=2,
        )
    )
    return path_tuple, masses, scales


__all__ = ["compose_scene_mjcf", "materialize_tool_pool_mjcf"]
