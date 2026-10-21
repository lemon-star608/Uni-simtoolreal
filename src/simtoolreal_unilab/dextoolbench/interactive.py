"""DexToolBench Interactive Policy Demo — static MJCF + MuJoCo backend."""

from __future__ import annotations

import argparse
import multiprocessing
import time
from pathlib import Path
from typing import Any

import numpy as np
import viser

from ..tasks.dextoolbench.catalog import all_tasks, scene_path
from .mjcf_viser import StaticMjcfScene
from .worker import sim_worker

_SIDEBAR_IMG_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "dextoolbench"
    / "dextoolbench_objects_sidebar.png"
)
_SIDEBAR_IMG: np.ndarray | None = None
if _SIDEBAR_IMG_PATH.exists():
    from PIL import Image as PILImage

    _SIDEBAR_IMG = np.asarray(PILImage.open(_SIDEBAR_IMG_PATH).convert("RGB"))

_TASKS = all_tasks()
_STRUCTURE: dict[str, dict[str, list[str]]] = {}
for _task in _TASKS:
    _STRUCTURE.setdefault(_task.category, {}).setdefault(_task.object_name, [])
    if _task.task_name not in _STRUCTURE[_task.category][_task.object_name]:
        _STRUCTURE[_task.category][_task.object_name].append(_task.task_name)

_CATEGORY_DESCRIPTIONS = {
    "hammer": "Swing a hammer to hit a nail.",
    "spatula": "Flip or serve food with a spatula.",
    "eraser": "Wipe a whiteboard with an eraser.",
    "screwdriver": "Drive a screw from the top or side.",
    "marker": "Write shapes on a whiteboard.",
    "brush": "Sweep debris forward across the table.",
}


def _snake_to_title(value: str) -> str:
    return value.replace("_", " ").title()


class InteractiveDemo:
    """Same launcher layout as upstream, with a static-MJCF renderer."""

    def __init__(
        self,
        *,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        port: int = 8080,
    ) -> None:
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.port = port
        self.server = viser.ViserServer(host="0.0.0.0", port=port)

        self._proc: multiprocessing.process.BaseProcess | None = None
        self._conn: Any = None
        self._env_ready = False
        self._episode_running = False
        self._is_paused = False
        self._pending: tuple[str, str, str] | None = None
        self.ep_count = 0
        self.ep_goals: list[float] = []
        self.ep_lengths: list[int] = []
        self.scene: StaticMjcfScene | None = None
        self.goal_mesh: viser.MeshHandle | None = None
        self.object_frame: viser.SceneNodeHandle | None = None
        self.goal_frame: viser.SceneNodeHandle | None = None

        self._build_gui()
        self._setup_static_scene()

    def _build_gui(self) -> None:
        self.server.gui.add_markdown("# DexToolBench\n### Interactive Policy Demo")
        if _SIDEBAR_IMG is not None:
            with self.server.gui.add_folder("DexToolBench Objects", expand_by_default=True):
                self.server.gui.add_image(
                    _SIDEBAR_IMG,
                    label="Tool objects in the benchmark",
                    format="jpeg",
                )

        placeholder = "-- Select --"
        with self.server.gui.add_folder("Dataset Selection", expand_by_default=True):
            categories = [placeholder] + [_snake_to_title(value) for value in sorted(_STRUCTURE)]
            self.dd_category = self.server.gui.add_dropdown(
                "Tool Category", options=categories, initial_value=placeholder
            )
            self.dd_object = self.server.gui.add_dropdown(
                "Object Instance", options=[placeholder], initial_value=placeholder
            )
            self.dd_task = self.server.gui.add_dropdown(
                "Task", options=[placeholder], initial_value=placeholder
            )
            self.description = self.server.gui.add_markdown("*Select a tool category to begin.*")
            button_load = self.server.gui.add_button("Load Environment")
            button_load.on_click(lambda _event: self.load_environment())
            self.status = self.server.gui.add_markdown("**Status:** Ready")
            self.dd_category.on_update(lambda _event: self.on_category_change())

        with self.server.gui.add_folder("Episode Controls", expand_by_default=True):
            button_run = self.server.gui.add_button("Run Episode")
            button_run.on_click(lambda _event: self.run_episode())
            self.pause_button = self.server.gui.add_button("Pause")
            self.pause_button.on_click(lambda _event: self.toggle_pause())
            button_stop = self.server.gui.add_button("Stop Episode")
            button_stop.on_click(lambda _event: self.stop_episode())

        with self.server.gui.add_folder("Status", expand_by_default=True):
            self.task_label = self.server.gui.add_markdown("**Task:** --")
            self.progress = self.server.gui.add_markdown("**Progress:** --")
            self.stats = self.server.gui.add_markdown("**Stats:** No episodes yet")
            self.object_position = self.server.gui.add_markdown("**Object Pos:** --")

    def _setup_static_scene(self) -> None:
        @self.server.on_client_connect
        def _client(client: viser.ClientHandle) -> None:
            client.camera.position = (0.0, -1.0, 1.0)
            client.camera.look_at = (0.0, 0.0, 0.5)

        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
        default_task = next(
            task for task in _TASKS if task.category == "hammer" and task.task_name == "swing_down"
        )
        self.scene = StaticMjcfScene(self.server, scene_path(default_task))
        self.scene.set_table_props_visible(False)
        self.scene.set_object_visible(False)

    def _selected_task(self) -> tuple[str, str, str] | None:
        category = self.dd_category.value.lower().replace(" ", "_")
        object_name = self.dd_object.value.lower().replace(" ", "_")
        task_name = self.dd_task.value.lower().replace(" ", "_")
        if category not in _STRUCTURE:
            return None
        if object_name not in _STRUCTURE[category]:
            return None
        if task_name not in _STRUCTURE[category][object_name]:
            return None
        return category, object_name, task_name

    def on_category_change(self) -> None:
        display = self.dd_category.value
        category = display.lower().replace(" ", "_")
        if category not in _STRUCTURE:
            return
        objects = list(_STRUCTURE[category])
        object_display = [_snake_to_title(value) for value in objects]
        self.dd_object.options = object_display
        self.dd_object.value = object_display[0]
        tasks = _STRUCTURE[category][objects[0]]
        task_display = [_snake_to_title(value) for value in tasks]
        self.dd_task.options = task_display
        self.dd_task.value = task_display[0]
        description = _CATEGORY_DESCRIPTIONS.get(category, "")
        self.description.content = f"*{description}*" if description else ""
        self.dd_object.on_update(lambda _event: self.on_object_change())

    def on_object_change(self) -> None:
        selected = self._selected_task()
        if selected is None:
            return
        _category, object_name, _task = selected
        tasks = next(
            task
            for category in _STRUCTURE.values()
            for name, task in category.items()
            if name == object_name
        )
        task_display = [_snake_to_title(value) for value in tasks]
        self.dd_task.options = task_display
        self.dd_task.value = task_display[0]

    def _clear_dynamic(self) -> None:
        if self.goal_mesh is not None:
            self.goal_mesh.remove()
            self.goal_mesh = None
        for handle in (self.object_frame, self.goal_frame):
            if handle is not None:
                handle.remove()
        self.object_frame = self.goal_frame = None

    def _kill_worker(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send("quit")
            except (BrokenPipeError, OSError):
                pass
            self._conn.close()
            self._conn = None
        if self._proc is not None:
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join()
            self._proc = None
        self._env_ready = False
        self._episode_running = False
        self._is_paused = False

    def load_environment(self) -> None:
        selected = self._selected_task()
        if selected is None:
            self.status.content = "**Status:** Please select a tool category first."
            return
        category, object_name, task_name = selected
        self._kill_worker()
        self._clear_dynamic()
        if self.scene is not None:
            self.scene.remove()
        task = next(
            item for item in _TASKS if (item.category, item.object_name, item.task_name) == selected
        )
        self.scene = StaticMjcfScene(self.server, scene_path(task))
        self._pending = selected

        label = (
            f"{_snake_to_title(category)} / {_snake_to_title(object_name)} / "
            f"{_snake_to_title(task_name)}"
        )
        self.status.content = f"**Status:** Loading *{label}* ..."
        self.task_label.content = f"**Task:** {label}"
        self.ep_count = 0
        self.ep_goals.clear()
        self.ep_lengths.clear()
        self.stats.content = "**Stats:** No episodes yet"

        context = multiprocessing.get_context("spawn")
        parent_conn, child_conn = context.Pipe()
        self._conn = parent_conn
        self._proc = context.Process(
            target=sim_worker,
            args=(
                child_conn,
                category,
                object_name,
                task_name,
                self.config_path,
                self.checkpoint_path,
            ),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        print(f"[launcher] Spawned MuJoCo worker pid={self._proc.pid}")

    def _send(self, message: str) -> None:
        if self._conn is not None:
            try:
                self._conn.send(message)
            except (BrokenPipeError, OSError):
                pass

    def run_episode(self) -> None:
        if not self._env_ready:
            self.status.content = "**Status:** Load an environment first."
            return
        if self._episode_running:
            return
        self._episode_running = True
        self._is_paused = False
        self.pause_button.name = "Pause"
        self.status.content = "**Status:** Running episode..."
        self._send("run")

    def toggle_pause(self) -> None:
        if not self._episode_running:
            return
        self._is_paused = not self._is_paused
        self._send("pause" if self._is_paused else "resume")
        self.pause_button.name = "Resume" if self._is_paused else "Pause"

    def stop_episode(self) -> None:
        if self._episode_running:
            self._send("stop")

    def _update_viz(self, state: dict[str, np.ndarray]) -> None:
        if self.scene is None:
            return
        self.scene.update(state["joint_pos"], state["object_pose_wxyz"])
        if self.object_frame is not None:
            pose = state["object_pose_wxyz"]
            self.object_frame.position = tuple(float(value) for value in pose[:3])
            self.object_frame.wxyz = tuple(float(value) for value in pose[3:])
        if self.goal_frame is not None:
            pose = state["goal_pose_wxyz"]
            self.goal_frame.position = tuple(float(value) for value in pose[:3])
            self.goal_frame.wxyz = tuple(float(value) for value in pose[3:])
        if self.goal_mesh is not None:
            self.scene.update_goal_mirror()

    def _setup_object_goal(self) -> None:
        if self.scene is None:
            return
        self.object_frame = self.server.scene.add_frame(
            "/object", show_axes=True, axes_length=0.1, axes_radius=0.001
        )
        self.goal_frame = self.server.scene.add_frame(
            "/goal", show_axes=True, axes_length=0.1, axes_radius=0.001
        )
        self.goal_mesh = self.scene.add_goal_mirror()

    def _handle(self, message: tuple[Any, ...]) -> None:
        tag = message[0]
        if tag == "ready":
            self._setup_object_goal()
            self._update_viz(message[1])
            self._env_ready = True
            self.status.content = "**Status:** Ready -- click **Run Episode**"
            print("[launcher] MuJoCo environment ready")
        elif tag == "state":
            state, successes, max_successes, step = message[1:]
            self._update_viz(state)
            pct = 100.0 * successes / max_successes
            self.progress.content = (
                f"**Time:** {step / 60.0:.1f}s &nbsp;|&nbsp; "
                f"**Goal:** {successes}/{max_successes} ({pct:.0f}%)"
            )
            pose = state["object_pose_wxyz"]
            self.object_position.content = (
                f"**Object Pos:** {pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}"
            )
        elif tag == "done":
            goal_pct, steps = message[1:]
            self._episode_running = False
            self.ep_goals.append(float(goal_pct))
            self.ep_lengths.append(int(steps))
            self.ep_count += 1
            average_goal = float(np.mean(self.ep_goals))
            average_time = float(np.mean(self.ep_lengths) / 60.0)
            self.stats.content = (
                f"**Episodes:** {self.ep_count} &nbsp;|&nbsp; "
                f"**Avg Goal:** {average_goal:.1f}% &nbsp;|&nbsp; "
                f"**Avg Time:** {average_time:.1f}s"
            )
            self.status.content = f"**Status:** Done -- {steps / 60.0:.1f}s, {goal_pct:.0f}% goals"
        elif tag == "stopped":
            self._episode_running = False
            self.status.content = "**Status:** Episode stopped."
        elif tag == "error":
            self._env_ready = False
            self._episode_running = False
            self.status.content = f"**Status:** Error -- {str(message[1])[:200]}"
            print(f"[launcher] worker error:\n{message[1]}")

    def _poll(self) -> None:
        if self._conn is None:
            return
        try:
            while self._conn.poll(0):
                self._handle(self._conn.recv())
        except (EOFError, ConnectionResetError, OSError):
            self._conn = None
            if self._proc is not None and not self._proc.is_alive():
                self.status.content = "**Status:** Worker exited unexpectedly."
                self._proc = None
                self._env_ready = False
                self._episode_running = False

    def run(self) -> None:
        print()
        print("  +-------------------------------------------------+")
        print("  |     DexToolBench Interactive Policy Demo         |")
        print(f"  |     http://localhost:{self.port:<26}|")
        print("  +-------------------------------------------------+")
        print()
        try:
            while True:
                self._poll()
                time.sleep(1.0 / 120.0)
        except KeyboardInterrupt:
            print("\n[launcher] Shutting down...")
            self._kill_worker()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DexToolBench Interactive Policy Demo (static MJCF + MuJoCo)"
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--config-path", default=None, help="Policy YAML")
    parser.add_argument("--checkpoint-path", default=None, help="Policy checkpoint")
    args = parser.parse_args()
    if bool(args.config_path) != bool(args.checkpoint_path):
        parser.error("provide both --config-path and --checkpoint-path, or neither")
    InteractiveDemo(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        port=args.port,
    ).run()


__all__ = ["InteractiveDemo", "main"]
