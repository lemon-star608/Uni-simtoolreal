from __future__ import annotations

import mujoco
import numpy as np

from simtoolreal_unilab.tasks.dextoolbench.catalog import all_tasks, scene_path


def test_all_24_static_scenes_compile() -> None:
    models = []
    for task in all_tasks():
        model = mujoco.MjModel.from_xml_path(str(scene_path(task)))
        assert model.nq == 36
        assert model.nu == 29
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_root") >= 0
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "dextoolbench_home")
        assert key_id >= 0
        models.append(model)
    assert len(models) == 24


def test_robot_and_object_names_are_stable() -> None:
    task = all_tasks()[0]
    model = mujoco.MjModel.from_xml_path(str(scene_path(task)))
    for name in (
        "iiwa14_link_0",
        "iiwa14_link_7",
        "left_index_DP",
        "object_root",
        "object_root_joint",
        "table",
    ):
        object_type = (
            mujoco.mjtObj.mjOBJ_BODY
            if name in {"iiwa14_link_0", "iiwa14_link_7", "left_index_DP", "object_root", "table"}
            else mujoco.mjtObj.mjOBJ_JOINT
        )
        assert mujoco.mj_name2id(model, object_type, name) >= 0, name


def test_keyframe_uses_start_arm_higher() -> None:
    task = all_tasks()[0]
    model = mujoco.MjModel.from_xml_path(str(scene_path(task)))
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "dextoolbench_home")
    qpos = np.asarray(model.key_qpos[key_id])
    assert np.isclose(qpos[1], 1.571 - np.deg2rad(10), atol=1e-6)
    assert np.isclose(qpos[3], 1.376 + np.deg2rad(10), atol=1e-6)


def test_object_collision_layout_matches_training_channel_semantics() -> None:
    task = all_tasks()[0]
    model = mujoco.MjModel.from_xml_path(str(scene_path(task)))
    for name, expected_contype, expected_group in (
        ("object_visual_geom", 0, 2),
        ("object_collision_geom_0", 1, 3),
    ):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert geom_id >= 0, name
        assert int(model.geom_contype[geom_id]) == expected_contype
        assert int(model.geom_group[geom_id]) == expected_group
        assert int(model.geom_condim[geom_id]) == 3

    visual_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_visual_geom")
    collision_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision_geom_0")
    assert int(model.geom_conaffinity[visual_id]) == 0
    assert int(model.geom_conaffinity[collision_id]) == 1
    np.testing.assert_allclose(model.geom_friction[collision_id], [0.5, 0.005, 0.0001], atol=1e-8)


def test_floor_matches_training_mujoco_scene() -> None:
    task = all_tasks()[0]
    model = mujoco.MjModel.from_xml_path(str(scene_path(task)))
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert floor_id >= 0
    np.testing.assert_allclose(model.geom_size[floor_id], [0.0, 0.0, 0.05])
    np.testing.assert_allclose(model.geom_friction[floor_id], [1.0, 0.005, 0.0001], atol=1e-8)
