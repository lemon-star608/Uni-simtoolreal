from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


class Adapter:
    def __init__(self, count=6):
        self.num_envs = count
        self.env = type("Device", (), {"device": torch.device("cpu")})()

    def get_env_info(self):
        return {
            "agents": 1,
            "value_size": 1,
            "observation_space": gym.spaces.Box(-10, 10, (140,), dtype=np.float32),
            "state_space": gym.spaces.Box(-10, 10, (162,), dtype=np.float32),
            "action_space": gym.spaces.Box(-1, 1, (29,), dtype=np.float32),
        }


class PpoPlayerContinuous:
    __module__ = "rl_games.algos_torch.players"

    def __init__(self, adapter):
        self.adapter = adapter
        self.is_deterministic = False
        self.is_rnn = True
        self.states = None
        self.calls = []

    def restore(self, path):
        self.calls.append(("restore", path))

    def env_reset(self, adapter):
        self.calls.append(("env_reset", adapter))
        return torch.zeros((adapter.num_envs, 140))

    def init_rnn(self):
        self.calls.append(("init_rnn",))
        self.states = [torch.ones((1, self.adapter.num_envs, 4))]

    def get_batch_size(self, obs, batch_size):
        self.calls.append(("get_batch_size", obs.shape[0]))
        self.has_batch_dimension = True
        return obs.shape[0]

    def get_action(self, obs, deterministic):
        self.calls.append(("get_action", deterministic))
        return torch.full((self.adapter.num_envs, 29), 0.25)

    def env_step(self, adapter, action):
        self.calls.append(("env_step", adapter, action))
        done = torch.zeros(adapter.num_envs, dtype=torch.bool)
        done[1] = True
        return torch.ones((adapter.num_envs, 140)), torch.ones(adapter.num_envs), done, {"log": {}}


class Runner:
    def __init__(self, *, algo_observer):
        self.calls = []
        self.adapter = None

    def load(self, value):
        self.calls.append("load")

    def set_vec_env(self, adapter):
        self.calls.append("set_vec_env")
        self.adapter = adapter

    def create_player(self):
        self.calls.append("create_player")
        return PpoPlayerContinuous(self.adapter)


def _cfg():
    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[3]
    with initialize_config_dir(
        config_dir=str(root / "src/simtoolreal_unilab/conf/rlgames_sapg"), version_base="1.3"
    ):
        return compose("config")


def test_bridge_uses_native_player_callbacks_and_exact_done_row_rnn_reset(tmp_path):
    from simtoolreal_unilab.rlgames_sapg.player import build_native_player_bridge

    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"trusted-by-resolver")
    adapter = Adapter(6)
    bridge = build_native_player_bridge(
        _cfg(),
        adapter=adapter,
        checkpoint=checkpoint,
        runner_factory=Runner,
        verify_dependency=False,
        validate_checkpoint=False,
    )
    assert bridge.runner.calls == ["load", "set_vec_env", "create_player"]
    obs = bridge.initialize()
    next_obs = bridge.step(obs)
    assert next_obs.shape == (6, 140)
    assert bridge.player.calls[0] == ("restore", str(checkpoint))
    assert ("get_batch_size", 6) in bridge.player.calls
    assert ("get_action", False) in bridge.player.calls
    assert torch.count_nonzero(bridge.player.states[0][:, 1, :]) == 0
    assert torch.all(bridge.player.states[0][:, 0, :] == 1)
    assert bridge.last_done.tolist() == [False, True, False, False, False, False]


def test_bridge_uses_native_deterministic_flag_without_policy_forward(tmp_path):
    from simtoolreal_unilab.rlgames_sapg.player import build_native_player_bridge

    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"trusted-by-resolver")
    bridge = build_native_player_bridge(
        _cfg(),
        adapter=Adapter(5),
        checkpoint=checkpoint,
        runner_factory=Runner,
        verify_dependency=False,
        validate_checkpoint=False,
    )
    bridge.player.is_deterministic = True
    bridge.step(bridge.initialize())
    assert ("get_action", True) in bridge.player.calls


def test_player_bridge_source_never_calls_base_player_run_or_model_forward():
    source = (
        Path(__file__).resolve().parents[3] / "src/simtoolreal_unilab/rlgames_sapg/player.py"
    ).read_text()
    assert "BasePlayer.run" not in source
    assert ".model(" not in source
