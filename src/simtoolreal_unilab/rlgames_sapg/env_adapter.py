"""Synchronous ``NpEnv`` to native RL-Games vector-env ABI adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from unilab.base.np_env import NpEnv, NpEnvState


@dataclass(frozen=True)
class _RlDevice:
    device: torch.device


class RlGamesNpEnvAdapter:
    """Expose a created ``NpEnv`` through the Source RL-Games ``IVecEnv`` ABI.

    Dimensions are derived from the env contract instead of being hardcoded:
    the actor group (``obs`` on the RL-Games side) and critic group
    (``states``, feeding SAPG central value) come from ``obs_groups_spec``,
    and the action width comes from ``action_space``.  ``time_outs``
    (truncated) and terminal observations pass through unchanged so SAPG
    ``value_bootstrap`` keeps truncated-bootstrap semantics.
    """

    def __init__(
        self,
        np_env: NpEnv,
        *,
        device: str | torch.device,
        actor_group: str = "obs",
        critic_group: str = "critic",
        observation_clip: float = 10.0,
        action_clip: float = 1.0,
    ) -> None:
        self._np_env = np_env
        self.device = torch.device(device)
        self.env = _RlDevice(self.device)
        self.num_envs = int(np_env.num_envs)
        self._actor_group = actor_group
        self._critic_group = critic_group
        spec = dict(np_env.obs_groups_spec)
        for group in (actor_group, critic_group):
            if group not in spec:
                raise ValueError(
                    f"NpEnv observation groups must include {group!r}, got spec={spec}"
                )
        self._actor_dim = int(spec[actor_group])
        self._critic_dim = int(spec[critic_group])
        if observation_clip <= 0.0 or action_clip <= 0.0:
            raise ValueError("RL-Games clip bounds must be positive")
        action_space = np_env.action_space
        if (
            not isinstance(action_space, gym.spaces.Box)
            or len(action_space.shape) != 1
            or action_space.dtype != np.float32
        ):
            raise ValueError("NpEnv action space must be a 1-D Box with dtype float32")
        if (
            np.isfinite(action_space.low).all()
            and np.isfinite(action_space.high).all()
            and not (
                np.all(action_space.low == -action_clip)
                and np.all(action_space.high == action_clip)
            )
        ):
            raise ValueError(
                "bounded NpEnv action space must be Box(-action_clip, action_clip); "
                "unbounded spaces are driven in [-action_clip, action_clip] by RL-Games"
            )
        self._action_dim = int(action_space.shape[0])
        self._env_info = {
            "agents": 1,
            "value_size": 1,
            "observation_space": gym.spaces.Box(
                -observation_clip, observation_clip, (self._actor_dim,), dtype=np.float32
            ),
            "state_space": gym.spaces.Box(
                -observation_clip, observation_clip, (self._critic_dim,), dtype=np.float32
            ),
            "action_space": gym.spaces.Box(
                -action_clip, action_clip, (self._action_dim,), dtype=np.float32
            ),
        }

    def _validate_obs(self, obs: dict[str, np.ndarray]) -> None:
        expected_groups = {self._actor_group: self._actor_dim, self._critic_group: self._critic_dim}
        if self._np_env.obs_groups_spec != expected_groups or set(obs) != set(expected_groups):
            raise ValueError(
                f"NpEnv observation groups must be exactly {expected_groups}, "
                f"got spec={self._np_env.obs_groups_spec}, obs={sorted(obs)}"
            )
        for group, width in expected_groups.items():
            value = obs[group]
            if not isinstance(value, np.ndarray) or value.shape != (self.num_envs, width):
                raise ValueError(f"observation {group!r} must have shape {(self.num_envs, width)}")
            if value.dtype != np.float32:
                raise TypeError(f"observation {group!r} must be float32, got {value.dtype}")
            if not np.isfinite(value).all():
                raise ValueError(f"observation {group!r} contains non-finite values")

    def _obs_tensors(self, obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        self._validate_obs(obs)
        return {
            "obs": torch.from_numpy(obs[self._actor_group]).to(self.device),
            "states": torch.from_numpy(obs[self._critic_group]).to(self.device),
        }

    def reset(self) -> dict[str, torch.Tensor]:
        if self._np_env.state is None:
            state = self._np_env.init_state()
            return self._obs_tensors(state.obs)
        env_ids = np.arange(self.num_envs, dtype=np.int32)
        obs, _info = self._np_env.reset(env_ids)
        return self._obs_tensors(obs)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any]]:
        if not isinstance(actions, torch.Tensor):
            raise TypeError("RL-Games actions must be a Torch tensor")
        if actions.shape != (self.num_envs, self._action_dim):
            raise ValueError(
                f"RL-Games actions must have shape {(self.num_envs, self._action_dim)}"
            )
        if actions.dtype != torch.float32 or actions.device != self.device:
            raise TypeError(f"RL-Games actions must be float32 on {self.device}")
        if not torch.isfinite(actions).all():
            raise ValueError("RL-Games actions contain non-finite values")
        action_array = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
        state: NpEnvState = self._np_env.step(action_array)
        obs = self._obs_tensors(state.obs)
        reward = self._vector_tensor(state.reward, torch.float32, "reward")
        terminated = self._vector_tensor(state.terminated, torch.bool, "terminated")
        truncated = self._vector_tensor(state.truncated, torch.bool, "truncated")
        done = torch.logical_or(terminated, truncated)
        info: dict[str, Any] = {"time_outs": truncated}
        for key in ("log", "final_observation", "_final_observation"):
            if key in state.info:
                info[key] = state.info[key]
        return obs, reward, done, info

    def _vector_tensor(self, value: np.ndarray, dtype: torch.dtype, label: str) -> torch.Tensor:
        expected_numpy = np.dtype(np.bool_) if dtype == torch.bool else np.dtype(np.float32)
        if not isinstance(value, np.ndarray) or value.shape != (self.num_envs,):
            raise ValueError(f"{label} must have shape {(self.num_envs,)}")
        if value.dtype != expected_numpy:
            raise TypeError(f"{label} must have dtype {expected_numpy}, got {value.dtype}")
        if dtype == torch.float32 and not np.isfinite(value).all():
            raise ValueError(f"{label} contains non-finite values")
        return torch.from_numpy(value).to(self.device)

    def get_env_info(self) -> dict[str, Any]:
        return dict(self._env_info)

    def get_number_of_agents(self) -> int:
        return 1

    def set_train_info(self, frame: int, owner: object) -> None:
        del frame, owner

    def get_env_state(self) -> None:
        return None

    def set_env_state(self, state: object) -> None:
        if state is not None:
            raise ValueError("RL-Games SAPG only accepts checkpoint env_state=None")
