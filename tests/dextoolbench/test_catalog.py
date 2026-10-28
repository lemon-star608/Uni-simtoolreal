from __future__ import annotations

import numpy as np

from simtoolreal_unilab.tasks.dextoolbench.catalog import (
    all_tasks,
    find_task,
    load_trajectory,
    scene_path,
)


def test_catalog_has_upstream_24_combinations() -> None:
    tasks = all_tasks()
    keys = {(task.category, task.object_name, task.task_name) for task in tasks}
    assert len(tasks) == 24
    assert len(keys) == 24
    assert {task.category for task in tasks} == {
        "hammer",
        "marker",
        "eraser",
        "brush",
        "spatula",
        "screwdriver",
    }
    assert all(load_trajectory(task)["goals_wxyz"] for task in tasks)


def test_trajectories_convert_xyzw_to_wxyz_and_raise_start_z() -> None:
    task = find_task("hammer", "claw_hammer", "swing_down")
    trajectory = load_trajectory(task, z_offset=0.03)
    start = trajectory["start_pose_wxyz"]
    assert len(start) == 7
    assert np.isclose(start[2], 0.5443193171844398 + 0.03)
    assert np.allclose(np.linalg.norm(start[3:]), 1.0, atol=1e-5)
    assert all(len(goal) == 7 for goal in trajectory["goals_wxyz"])
    assert all(
        np.isclose(np.linalg.norm(goal[3:]), 1.0, atol=1e-5) for goal in trajectory["goals_wxyz"]
    )


def test_find_task_rejects_unknown_keys() -> None:
    try:
        find_task("hammer", "not_a_hammer", "swing_down")
    except KeyError as exc:
        assert "unknown DexToolBench task" in str(exc)
    else:
        raise AssertionError("expected unknown task to fail")


def test_scene_path_is_committed_for_every_task() -> None:
    for task in all_tasks():
        path = scene_path(task)
        assert path.is_file()
        assert path.suffix == ".xml"
