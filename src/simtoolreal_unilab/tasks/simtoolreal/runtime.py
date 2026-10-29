"""SimToolReal's Manager runtime compatibility with the frozen teacher baseline."""

from __future__ import annotations

from types import MethodType

import torch


def _zero_initial_episode_steps(self) -> torch.Tensor:
    return torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)


def retain_zero_initial_episode_steps(env):
    """Bind TorchEnv compatibility boundaries to one Manager instance."""
    env._initial_episode_steps = MethodType(_zero_initial_episode_steps, env)
    original_update_state = env.update_state

    def update_state_with_final_tracked_sensors(self, state):
        # UniSim 2.1 MuJoCo marks injected body sensors dirty after stepping,
        # but its HOST_BRIDGE sensor-name metadata does not currently identify
        # qualified ``track_<field>_<entity>/<body>`` aliases as tracked-body
        # sensors. A packed read can therefore republish the preceding sensor
        # packet while qpos/qvel are current. Refresh through the backend's
        # public-in-practice kinematics sync before the Manager read phase;
        # non-MuJoCo backends without this hook are unchanged.
        sync_tracked = getattr(getattr(env, "_backend", None), "_sync_tracked_body_state", None)
        if callable(sync_tracked):
            sync_tracked()
        return original_update_state(state)

    env.update_state = MethodType(update_state_with_final_tracked_sensors, env)
    return env


__all__ = ["retain_zero_initial_episode_steps"]
