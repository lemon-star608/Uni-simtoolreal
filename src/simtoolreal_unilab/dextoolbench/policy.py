"""Deterministic RL-Games SAPG player for DexToolBench policy playback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

from ..rlgames_sapg.dependency import require_rlgames_sapg


class _SingleEnv:
    def __init__(self, obs_dim: int, action_dim: int, device: torch.device) -> None:
        self.num_envs = 1
        self.device = device
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (action_dim,), dtype=np.float32)

    def get_env_info(self) -> dict[str, Any]:
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
        }

    def set_env_state(self, *_args: object, **_kwargs: object) -> None:
        return None


class SapgPlayer:
    """The original one-env deterministic player ABI on the vendored runtime."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        checkpoint_path: str | Path,
        obs_dim: int = 140,
        action_dim: int = 29,
        device: str | None = None,
    ) -> None:
        require_rlgames_sapg()
        from rl_games.common import env_configurations
        from rl_games.torch_runner import Runner

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.env = _SingleEnv(self.obs_dim, self.action_dim, self.device)
        env_configurations.register(
            "rlgpu",
            {"env_creator": lambda **_kwargs: self.env, "vecenv_type": "RLGPU"},
        )

        raw = OmegaConf.to_container(OmegaConf.load(str(config_path)), resolve=True)
        if not isinstance(raw, dict):
            raise TypeError(f"policy config must be a mapping: {config_path}")
        # UniLab training runs persist their resolved Hydra config under a
        # run/config envelope.  Prefer that exact snapshot when it is supplied.
        if isinstance(raw.get("run"), dict) and isinstance(raw.get("config"), dict):
            raw = raw["config"]
        if isinstance(raw.get("rl_games"), dict) and "params" in raw["rl_games"]:
            params = dict(raw["rl_games"]["params"])
        elif isinstance(raw.get("train"), dict) and "params" in raw["train"]:
            params = dict(raw["train"]["params"])
        else:
            raise ValueError(
                "policy config must use either the original train.params schema "
                "or simtoolreal_unilab's rl_games.params schema"
            )
        config = params["config"]
        config["device"] = str(self.device)
        config["device_name"] = str(self.device)
        config["num_actors"] = 1
        runner = Runner(algo_observer=None)
        runner.load({"params": params})
        player = runner.create_player()
        player.init_rnn()
        player.has_batch_dimension = True
        player.restore(str(checkpoint_path))
        self.player = player

    def reset(self) -> None:
        self.player.reset()

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"expected obs {self.obs_dim}, got {obs.shape}")
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None]
        # Original DexToolBench playback pins the first SAPG coefficient block.
        tensor = torch.cat((tensor, torch.full((1, 1), 50.0, device=self.device)), dim=1)
        action = self.player.get_action(tensor, is_deterministic=True)
        action = action.reshape(-1)
        if action.shape != (self.action_dim,):
            raise RuntimeError(f"policy returned action {action.shape}")
        return action.detach().cpu().numpy()
