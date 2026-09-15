"""Native ``PpoPlayerContinuous`` bridge for UniLab playback/video shells."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import DictConfig, OmegaConf

from unilab.utils.sim2sim import policy_load_dim_guard

from .checkpoint import validate_native_checkpoint
from .dependency import require_rlgames_sapg


@dataclass
class NativePlayerBridge:
    runner: Any
    player: Any
    adapter: Any
    last_reward: torch.Tensor | None = None
    last_done: torch.Tensor | None = None
    last_info: dict[str, Any] | None = None

    def initialize(self) -> torch.Tensor:
        obs = self.player.env_reset(self.adapter)
        self.player.get_batch_size(obs, 1)
        self.player.init_rnn()
        return obs

    def step(self, obs: torch.Tensor) -> torch.Tensor:
        action = self.player.get_action(obs, self.player.is_deterministic)
        next_obs, reward, done, info = self.player.env_step(self.adapter, action)
        all_done_indices = done.nonzero(as_tuple=False)
        if self.player.is_rnn and len(all_done_indices) > 0:
            for state in self.player.states:
                state[:, all_done_indices, :] = state[:, all_done_indices, :] * 0.0
        self.last_reward = reward
        self.last_done = done
        self.last_info = info
        return next_obs


def build_native_player_bridge(
    cfg: DictConfig,
    *,
    adapter: Any,
    checkpoint: str | Path,
    runner_factory: Callable[..., Any] | None = None,
    verify_dependency: bool = True,
    validate_checkpoint: bool = True,
) -> NativePlayerBridge:
    """Create and restore the pinned native player without entering its run loop."""
    checkpoint_path = Path(checkpoint)
    if verify_dependency:
        require_rlgames_sapg()
    if validate_checkpoint:
        validate_native_checkpoint(checkpoint_path)
    resolved = OmegaConf.to_container(cfg.rl_games.params, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("cfg.rl_games.params must resolve to a mapping")
    params = copy.deepcopy(resolved)
    native = params["config"]
    native["device"] = str(cfg.training.device)
    native["device_name"] = str(cfg.training.device)
    native["num_actors"] = int(adapter.num_envs)
    native["env_info"] = adapter.get_env_info()
    factory = runner_factory
    if factory is None:
        from rl_games.torch_runner import Runner

        factory = Runner
    runner = factory(algo_observer=None)
    runner.load({"params": params})
    runner.set_vec_env(adapter)
    player = runner.create_player()
    if (
        type(player).__module__ != "rl_games.algos_torch.players"
        or type(player).__name__ != "PpoPlayerContinuous"
    ):
        raise TypeError(f"unexpected native player owner: {type(player)!r}")
    env_info = adapter.get_env_info()
    with policy_load_dim_guard(
        env_obs_dim=int(env_info["observation_space"].shape[0]),
        env_action_dim=int(env_info["action_space"].shape[0]),
        algo_name="rlgames_sapg",
    ):
        player.restore(str(checkpoint_path))
    return NativePlayerBridge(runner=runner, player=player, adapter=adapter)
