"""A small Viser renderer for committed static DexToolBench MJCFs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import viser


@dataclass(frozen=True)
class _VisualGeom:
    geom_id: int
    handle: viser.SceneNodeHandle


def _quat_from_matrix(matrix: np.ndarray) -> tuple[float, float, float, float]:
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=np.float64).reshape(-1))
    return tuple(float(value) for value in quat)


class StaticMjcfScene:
    """Render visual geoms and update their world transforms from MJCF FK."""

    def __init__(self, server: viser.ViserServer, model_file: str | Path) -> None:
        self.server = server
        self.model = mujoco.MjModel.from_xml_path(str(model_file))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(
            self.model,
            self.data,
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "dextoolbench_home"),
        )
        mujoco.mj_forward(self.model, self.data)

        self.robot_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "iiwa14_link_0")
        self.object_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object_root")
        self.object_joint_qposadr = int(
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_root_joint")
            ]
        )
        self._visuals: list[_VisualGeom] = []
        self._goal_mesh_handle: viser.MeshHandle | None = None
        self._goal_local_pos: np.ndarray | None = None
        self._goal_local_quat: np.ndarray | None = None
        self._build()

    def _color(self, geom_id: int) -> tuple[int, int, int]:
        rgba = np.asarray(self.model.geom_rgba[geom_id], dtype=np.float32)
        material_id = int(self.model.geom_matid[geom_id])
        if material_id >= 0:
            texture_slots = np.asarray(self.model.mat_texid[material_id])
            positive_texture_ids = [int(value) for value in texture_slots if value >= 0]
            if positive_texture_ids:
                texture_id = positive_texture_ids[0]
                address = int(self.model.tex_adr[texture_id])
                count = int(
                    self.model.tex_width[texture_id]
                    * self.model.tex_height[texture_id]
                    * self.model.tex_nchannel[texture_id]
                )
                channels = int(self.model.tex_nchannel[texture_id])
                pixels = np.asarray(
                    self.model.tex_data[address : address + count], dtype=np.float32
                ).reshape(-1, channels)[:, :3]
                alpha = rgba[3] if rgba.shape == (4,) else 1.0
                rgba = np.array([*(pixels.mean(axis=0) / 255.0), alpha], dtype=np.float32)
        color = np.clip(rgba[:3] * 255.0, 0.0, 255.0).astype(np.uint8)
        return tuple(int(value) for value in color)

    def _mesh_arrays(self, mesh_id: int) -> tuple[np.ndarray, np.ndarray]:
        start = int(self.model.mesh_vertadr[mesh_id])
        count = int(self.model.mesh_vertnum[mesh_id])
        vertices = np.asarray(self.model.mesh_vert[start : start + count], dtype=np.float32)
        face_start = int(self.model.mesh_faceadr[mesh_id])
        face_count = int(self.model.mesh_facenum[mesh_id])
        faces = np.asarray(
            self.model.mesh_face[face_start : face_start + face_count], dtype=np.int32
        )
        return vertices, faces

    def _build(self) -> None:
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            group = int(self.model.geom_group[geom_id])
            name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                or f"geom_{geom_id}"
            )
            if body_id == self.robot_body and group != 2:
                continue
            if body_id == self.object_body and group != 2:
                continue
            if body_id not in {self.robot_body, self.object_body} and group not in {0, 2}:
                continue

            if int(self.model.geom_type[geom_id]) == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = int(self.model.geom_dataid[geom_id])
                vertices, faces = self._mesh_arrays(mesh_id)
                handle = self.server.scene.add_mesh_simple(
                    f"/mjcf/{name}",
                    vertices=vertices,
                    faces=faces,
                    color=self._color(geom_id),
                    wxyz=_quat_from_matrix(self.data.geom_xmat[geom_id]),
                    position=tuple(float(x) for x in self.data.geom_xpos[geom_id]),
                    side="double",
                )
                self._goal_mesh_handle = handle
            elif int(self.model.geom_type[geom_id]) == mujoco.mjtGeom.mjGEOM_BOX:
                handle = self.server.scene.add_box(
                    f"/mjcf/{name}",
                    dimensions=tuple(float(x) * 2.0 for x in self.model.geom_size[geom_id]),
                    color=self._color(geom_id),
                    wxyz=_quat_from_matrix(self.data.geom_xmat[geom_id]),
                    position=tuple(float(x) for x in self.data.geom_xpos[geom_id]),
                    side="double",
                )
            elif int(self.model.geom_type[geom_id]) == mujoco.mjtGeom.mjGEOM_PLANE:
                handle = self.server.scene.add_grid(
                    "/mjcf/floor_grid",
                    width=4.0,
                    height=4.0,
                    cell_size=0.1,
                )
            else:
                continue
            self._visuals.append(_VisualGeom(geom_id, handle))

        self._update_handles()

    def _update_handles(self) -> None:
        for visual in self._visuals:
            geom_id = visual.geom_id
            visual.handle.position = tuple(float(value) for value in self.data.geom_xpos[geom_id])
            visual.handle.wxyz = _quat_from_matrix(self.data.geom_xmat[geom_id])

    def update(
        self,
        joint_pos: np.ndarray,
        object_pose_wxyz: np.ndarray,
    ) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:29] = np.asarray(joint_pos, dtype=np.float64)
        address = self.object_joint_qposadr
        self.data.qpos[address : address + 7] = np.asarray(object_pose_wxyz, dtype=np.float64)
        mujoco.mj_kinematics(self.model, self.data)
        self._update_handles()

    def add_goal_mirror(self) -> viser.MeshHandle:
        for geom_id in range(self.model.ngeom):
            if (
                int(self.model.geom_bodyid[geom_id]) == self.object_body
                and int(self.model.geom_group[geom_id]) == 2
                and int(self.model.geom_type[geom_id]) == mujoco.mjtGeom.mjGEOM_MESH
            ):
                mesh_id = int(self.model.geom_dataid[geom_id])
                vertices, faces = self._mesh_arrays(mesh_id)
                handle = self.server.scene.add_mesh_simple(
                    "/goal/object",
                    vertices=vertices,
                    faces=faces,
                    color=(0, 255, 0),
                    opacity=0.5,
                    side="double",
                )
                self._goal_mesh_handle = handle
                # MuJoCo recenters mesh assets during compilation.  The object
                # visual geom therefore has a non-zero local pos/quat relative
                # to object_root.  This mesh is a child of the launcher's
                # /goal frame: /goal carries the authored root pose, while the
                # mesh handle must carry only this local geom transform.
                self._goal_local_pos = np.asarray(self.model.geom_pos[geom_id], dtype=np.float64)
                self._goal_local_quat = np.asarray(self.model.geom_quat[geom_id], dtype=np.float64)
                self.update_goal_mirror()
                return handle
        raise ValueError("static DexToolBench scene has no object visual mesh")

    def update_goal_mirror(self) -> None:
        """Set the green mirror's local pose under the /goal root frame."""
        if self._goal_mesh_handle is None or self._goal_local_pos is None:
            return
        self._goal_mesh_handle.position = tuple(float(value) for value in self._goal_local_pos)
        self._goal_mesh_handle.wxyz = tuple(float(value) for value in self._goal_local_quat)

    def set_table_props_visible(self, visible: bool) -> None:
        for visual in self._visuals:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, visual.geom_id)
            if name is not None and name.startswith("table_prop_"):
                visual.handle.visible = visible

    def set_object_visible(self, visible: bool) -> None:
        for visual in self._visuals:
            if int(self.model.geom_bodyid[visual.geom_id]) == self.object_body:
                visual.handle.visible = visible

    def remove(self) -> None:
        for visual in self._visuals:
            try:
                visual.handle.remove()
            except Exception:
                pass
        self._visuals.clear()


__all__ = ["StaticMjcfScene"]
