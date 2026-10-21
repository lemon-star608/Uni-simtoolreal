"""MuJoCo child worker for the DexToolBench interactive Viser demo."""

from __future__ import annotations

import time
import traceback
from multiprocessing.connection import Connection
from typing import Any

import numpy as np

CONTROL_DT = 1.0 / 60.0


def _state(env: Any) -> dict[str, np.ndarray]:
    from simtoolreal_unilab.tasks.simtoolreal.commands import goal_command

    robot = env.scene["robot"]
    object_entity = env.scene["object"]
    origin = np.asarray(env.scene.env_origins[0], dtype=np.float32)
    object_pose = np.asarray(object_entity.data.root_link_pose_w[0], dtype=np.float32)
    object_pose[:3] -= origin[:3]
    hub = goal_command(env)
    return {
        "joint_pos": np.asarray(robot.data.joint_pos[0], dtype=np.float32),
        "object_pose_wxyz": object_pose,
        "goal_pose_wxyz": np.asarray(hub.goal[0], dtype=np.float32),
    }


def _episode(
    conn: Connection,
    env: Any,
    policy: Any | None,
) -> None:
    from simtoolreal_unilab.tasks.simtoolreal.commands import goal_command

    if policy is not None:
        policy.reset()
    obs, _info = env.reset()
    state = env.step(np.zeros((1, 29), dtype=np.float32))
    obs = state.obs["obs"]
    hub = goal_command(env)
    max_goals = len(hub._fixed_goals)  # noqa: SLF001 - fixed by this worker's config

    step = 0
    done = False
    paused = False
    successes = 0
    while not done:
        while conn.poll(0):
            command = conn.recv()
            if command == "pause":
                paused = True
            elif command == "resume":
                paused = False
            elif command == "stop":
                conn.send(("stopped",))
                return

        if paused:
            time.sleep(0.05)
            continue

        started = time.perf_counter()
        current_state = _state(env)
        if policy is None:
            action = np.zeros((1, 29), dtype=np.float32)
        else:
            action = policy.get_action(np.asarray(obs[0], dtype=np.float32))[None]
        state = env.step(action)
        obs = state.obs["obs"]
        done = bool(state.terminated[0] or state.truncated[0])
        if done:
            # Auto-reset has already zeroed goal_count; the terminal count is
            # preserved in prev_episode_successes (upstream: _prev_episode_successes).
            successes = max(successes, int(hub.prev_episode_successes[0]))
        else:
            successes = max(int(hub.goal_count[0]), int(hub.pending_success[0]))
        step += 1
        conn.send(("state", current_state, successes, max_goals, step))
        remaining = CONTROL_DT - (time.perf_counter() - started)
        if remaining > 0.0:
            time.sleep(remaining)

    goal_pct = 100.0 * successes / max_goals
    conn.send(("done", goal_pct, step))


def sim_worker(
    conn: Connection,
    category: str,
    object_name: str,
    task_name: str,
    config_path: str | None,
    checkpoint_path: str | None,
) -> None:
    try:
        from unilab.base import registry

        import simtoolreal_unilab.tasks  # noqa: F401  (registry registration)
        from simtoolreal_unilab.tasks.dextoolbench.env_cfg import (
            build_dextoolbench_env_cfg_override,
        )

        env_cfg_override = build_dextoolbench_env_cfg_override(category, object_name, task_name)
        env = registry.make(
            "DexToolBench",
            sim_backend="mujoco",
            env_cfg_override=env_cfg_override,
            num_envs=1,
        )

        policy = None
        if config_path and checkpoint_path:
            from .policy import SapgPlayer

            policy = SapgPlayer(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                device="cpu",
            )

        obs, _info = env.reset()
        state = env.step(np.zeros((1, 29), dtype=np.float32))
        del obs, state
        conn.send(("ready", _state(env)))

        while True:
            command = conn.recv()
            if command == "run":
                _episode(conn, env, policy)
            elif command == "quit":
                break
    except Exception as exc:  # noqa: BLE001 - the launcher displays the traceback
        try:
            conn.send(("error", f"{exc}\n{traceback.format_exc()}"))
        except (BrokenPipeError, OSError):
            pass
    finally:
        try:
            env.close()  # type: ignore[union-attr]
        except Exception:
            pass
        try:
            conn.close()
        except OSError:
            pass


__all__ = ["sim_worker"]
