"""Static DexToolBench catalog and committed-asset lookup.

The upstream benchmark has 24 fixed object/task combinations.  Unlike the
SimToolReal training workload there is no runtime asset generation: every
scene is a committed MJCF selected directly by these three keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DexToolBenchTask:
    category: str
    object_name: str
    task_name: str
    # Upstream ``Object.scale``: normalized grasp-bbox input to the policy's
    # object_scales observation, not a physical mesh scale.
    object_scale: tuple[float, float, float]


_TASKS: tuple[DexToolBenchTask, ...] = (
    DexToolBenchTask("hammer", "claw_hammer", "swing_down", (2.5, 0.5625, 0.375)),
    DexToolBenchTask("hammer", "claw_hammer", "swing_side", (2.5, 0.5625, 0.375)),
    DexToolBenchTask("hammer", "mallet_hammer", "swing_down", (6.0, 0.75, 0.5)),
    DexToolBenchTask("hammer", "mallet_hammer", "swing_side", (6.0, 0.75, 0.5)),
    DexToolBenchTask("marker", "sharpie_marker", "draw_smile", (2.125, 0.55, 0.55)),
    DexToolBenchTask("marker", "sharpie_marker", "write_c", (2.125, 0.55, 0.55)),
    DexToolBenchTask("marker", "staples_marker", "draw_smile", (3.0, 0.45, 0.45)),
    DexToolBenchTask("marker", "staples_marker", "write_c", (3.0, 0.45, 0.45)),
    DexToolBenchTask("eraser", "flat_eraser", "wipe_smile", (2.5, 0.7, 1.25)),
    DexToolBenchTask("eraser", "flat_eraser", "wipe_c", (2.5, 0.7, 1.25)),
    DexToolBenchTask("eraser", "handle_eraser", "wipe_smile", (2.25, 0.8, 0.25)),
    DexToolBenchTask("eraser", "handle_eraser", "wipe_c", (2.25, 0.8, 0.25)),
    DexToolBenchTask("brush", "blue_brush", "sweep_forward", (3.0, 0.875, 0.5)),
    DexToolBenchTask("brush", "blue_brush", "sweep_right", (3.0, 0.875, 0.5)),
    DexToolBenchTask("brush", "red_brush", "sweep_forward", (2.5, 0.5, 0.375)),
    DexToolBenchTask("brush", "red_brush", "sweep_right", (2.5, 0.5, 0.375)),
    DexToolBenchTask("spatula", "flat_spatula", "serve_plate", (5.0, 0.375, 0.1875)),
    DexToolBenchTask("spatula", "flat_spatula", "flip_over", (5.0, 0.375, 0.1875)),
    DexToolBenchTask("spatula", "spoon_spatula", "serve_plate", (3.0, 0.5, 0.5)),
    DexToolBenchTask("spatula", "spoon_spatula", "flip_over", (3.0, 0.5, 0.5)),
    DexToolBenchTask("screwdriver", "long_screwdriver", "spin_vertical", (2.5, 0.75, 0.75)),
    DexToolBenchTask("screwdriver", "long_screwdriver", "spin_horizontal", (2.5, 0.75, 0.75)),
    DexToolBenchTask("screwdriver", "short_screwdriver", "spin_vertical", (1.75, 0.875, 0.875)),
    DexToolBenchTask("screwdriver", "short_screwdriver", "spin_horizontal", (1.75, 0.875, 0.875)),
)

ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets" / "dextoolbench"


def all_tasks() -> tuple[DexToolBenchTask, ...]:
    return _TASKS


def find_task(category: str, object_name: str, task_name: str) -> DexToolBenchTask:
    for task in _TASKS:
        if (task.category, task.object_name, task.task_name) == (
            category,
            object_name,
            task_name,
        ):
            return task
    valid = [(t.category, t.object_name, t.task_name) for t in _TASKS]
    raise KeyError(f"unknown DexToolBench task {category}/{object_name}/{task_name}; valid={valid}")


def scene_path(task: DexToolBenchTask) -> Path:
    path = ASSET_ROOT / "scenes" / task.category / task.object_name / f"{task.task_name}.xml"
    if not path.is_file():
        raise FileNotFoundError(f"static DexToolBench scene missing: {path}")
    return path


def _xyzw_to_wxyz(pose: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    if len(pose) != 7:
        raise ValueError("DexToolBench poses must contain xyz + xyzw quaternion")
    x, y, z, qx, qy, qz, qw = (float(value) for value in pose)
    return (x, y, z, qw, qx, qy, qz)


def load_trajectory(task: DexToolBenchTask, *, z_offset: float = 0.03) -> dict[str, Any]:
    """Load an upstream trajectory and convert poses to the task hub's wxyz."""
    path = ASSET_ROOT / "trajectories" / task.category / task.object_name / f"{task.task_name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"DexToolBench trajectory missing: {path}")
    raw = json.loads(path.read_text())
    start = _xyzw_to_wxyz(raw["start_pose"])
    start = (start[0], start[1], start[2] + z_offset, *start[3:])
    goals = tuple(_xyzw_to_wxyz(goal) for goal in raw["goals"])
    if not goals:
        raise ValueError(f"DexToolBench trajectory has no goals: {path}")
    return {"start_pose_wxyz": start, "goals_wxyz": goals}


__all__ = [
    "ASSET_ROOT",
    "DexToolBenchTask",
    "all_tasks",
    "find_task",
    "load_trajectory",
    "scene_path",
]
