"""TorchEnv adapter for the vendored RL-Games vector-env ABI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import torch


@dataclass(frozen=True)
class _RlDevice:
    device: torch.device


class RlGamesTorchEnvAdapter:
    """Expose a UniLab ``TorchEnv`` without a NumPy or host-array boundary."""

    def __init__(
        self,
        env: Any,
        *,
        device: str | torch.device,
        actor_group: str = "obs",
        critic_group: str = "critic",
        observation_clip: float = 10.0,
        action_clip: float = 1.0,
    ) -> None:
        self._env = env
        self.device = torch.device(device)
        self.env = _RlDevice(self.device)
        self.num_envs = int(env.num_envs)
        self._actor_group = actor_group
        self._critic_group = critic_group
        spec = dict(env.obs_groups_spec)
        for group in (actor_group, critic_group):
            if group not in spec:
                raise ValueError(f"TorchEnv groups must include {group!r}, got {spec}")
        self._actor_dim = int(spec[actor_group])
        self._critic_dim = int(spec[critic_group])
        if observation_clip <= 0.0 or action_clip <= 0.0:
            raise ValueError("RL-Games clip bounds must be positive")
        action_space = env.action_space
        if (
            not isinstance(action_space, gym.spaces.Box)
            or len(action_space.shape) != 1
            or action_space.dtype != "float32"
        ):
            raise ValueError("TorchEnv action space must be a 1-D float32 Box")
        self._action_dim = int(action_space.shape[0])
        self._env_info = {
            "agents": 1,
            "value_size": 1,
            "observation_space": gym.spaces.Box(
                -observation_clip, observation_clip, (self._actor_dim,), dtype="float32"
            ),
            "state_space": gym.spaces.Box(
                -observation_clip, observation_clip, (self._critic_dim,), dtype="float32"
            ),
            "action_space": gym.spaces.Box(
                -action_clip, action_clip, (self._action_dim,), dtype="float32"
            ),
        }

    def _validate_obs(self, obs: dict[str, torch.Tensor]) -> None:
        expected = {
            self._actor_group: (self.num_envs, self._actor_dim),
            self._critic_group: (self.num_envs, self._critic_dim),
        }
        if set(obs) != set(expected):
            raise ValueError(f"TorchEnv observation keys must be {sorted(expected)}")
        for group, shape in expected.items():
            value = obs[group]
            if value.shape != shape or value.dtype != torch.float32:
                raise ValueError(f"observation {group!r} must have shape {shape} and float32 dtype")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"observation {group!r} contains non-finite values")

    def _obs_tensors(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self._validate_obs(obs)
        return {
            "obs": obs[self._actor_group].to(self.device, non_blocking=True),
            "states": obs[self._critic_group].to(self.device, non_blocking=True),
        }

    def _legacy_log(self, state: Any) -> dict[str, Any]:
        log = dict(state.info.get("log", {}))
        command_manager = getattr(self._env, "command_manager", None)
        term = command_manager.get_term("goal") if command_manager is not None else None
        log_extras = getattr(term, "log_extras", None)
        if callable(log_extras):
            log.update(
                log_extras(
                    reward_manager=self._env.reward_manager,
                    termination_manager=self._env.termination_manager,
                )
            )
        return log

    def reset(self) -> dict[str, torch.Tensor]:
        obs, _info = self._env.reset()
        return self._obs_tensors(obs)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any]]:
        if not isinstance(actions, torch.Tensor):
            raise TypeError("RL-Games actions must be a Torch tensor")
        if actions.shape != (self.num_envs, self._action_dim) or actions.dtype != torch.float32:
            raise TypeError(
                f"RL-Games actions must be float32 ({self.num_envs}, {self._action_dim})"
            )
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("RL-Games actions contain non-finite values")
        action = actions.to(self._env.device, non_blocking=True).contiguous()
        state = self._env.step(action)
        obs = self._obs_tensors(state.obs)
        reward = state.reward.to(self.device, non_blocking=True)
        done = (state.terminated | state.truncated).to(self.device, non_blocking=True)
        info: dict[str, Any] = {
            "time_outs": state.truncated.to(self.device, non_blocking=True),
            "log": self._legacy_log(state),
        }
        if state.final_observation is not None:
            info["final_observation"] = state.final_observation
        return obs, reward, done, info

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


__all__ = ["RlGamesTorchEnvAdapter"]
