from __future__ import annotations

import mujoco
import numpy as np
import viser

from simtoolreal_unilab.dextoolbench.mjcf_viser import StaticMjcfScene
from simtoolreal_unilab.tasks.dextoolbench.catalog import find_task, scene_path


def test_goal_mesh_is_local_to_the_goal_frame() -> None:
    server = viser.ViserServer(port=0)
    try:
        scene = StaticMjcfScene(
            server, scene_path(find_task("hammer", "claw_hammer", "swing_down"))
        )
        goal_frame = server.scene.add_frame("/goal")
        goal_mesh = scene.add_goal_mirror()

        geom_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "object_visual_geom")
        np.testing.assert_allclose(goal_mesh.position, scene.model.geom_pos[geom_id], atol=1e-7)
        np.testing.assert_allclose(goal_mesh.wxyz, scene.model.geom_quat[geom_id], atol=1e-7)
        np.testing.assert_allclose(goal_frame.position, (0.0, 0.0, 0.0))
    finally:
        server.stop()
