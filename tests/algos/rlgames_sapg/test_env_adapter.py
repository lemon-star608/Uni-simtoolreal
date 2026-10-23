from __future__ import annotations

import gymnasium as gym
import pytest
import torch
from unilab.base.torch_env import TorchEnvState


class FakeTorchEnv:
    def __init__(self, num_envs: int = 3):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.obs_groups_spec = {"obs": 140, "critic": 162}
        self.action_space = gym.spaces.Box(-float("inf"), float("inf"), (29,), dtype="float32")
        self.reset_calls = 0
        self.step_calls: list[torch.Tensor] = []
        self.command_manager = None

    def _obs(self, offset: float = 0.0):
        rows = torch.arange(self.num_envs, dtype=torch.float32)[:, None]
        return {
            "obs": (rows + offset).expand(-1, 140).contiguous(),
            "critic": (rows + offset + 10.0).expand(-1, 162).contiguous(),
        }

    def reset(self, env_ids=None):
        del env_ids
        self.reset_calls += 1
        return self._obs(20.0), {"log": {"reset": 1.0}}

    def step(self, actions):
        self.step_calls.append(actions)
        return TorchEnvState(
            obs=self._obs(1.0),
            reward=torch.arange(self.num_envs, dtype=torch.float32),
            terminated=torch.tensor([True, False, False]),
            truncated=torch.tensor([False, True, False]),
            info={"log": {"reward": 2.0}},
            final_observation=self._obs(99.0),
        )


def _adapter(env=None):
    from simtoolreal_unilab.rlgames_sapg.env_adapter import RlGamesTorchEnvAdapter

    return RlGamesTorchEnvAdapter(env or FakeTorchEnv(), device="cpu")


def test_reset_and_step_preserve_torch_carriers():
    env = FakeTorchEnv()
    adapter = _adapter(env)
    obs = adapter.reset()
    assert env.reset_calls == 1
    assert tuple(obs) == ("obs", "states")
    assert obs["obs"].shape == (3, 140)
    assert obs["states"].shape == (3, 162)
    assert obs["obs"][:, 0].tolist() == [20.0, 21.0, 22.0]

    obs, reward, done, info = adapter.step(torch.zeros((3, 29)))
    assert env.step_calls[0].dtype == torch.float32
    assert obs["obs"][:, 0].tolist() == [1.0, 2.0, 3.0]
    assert reward.dtype == torch.float32
    assert done.tolist() == [True, True, False]
    assert info["time_outs"].tolist() == [False, True, False]
    assert info["log"]["reward"] == 2.0
    assert "final_observation" in info


def test_spaces_and_env_info_are_gymnasium_contracts():
    adapter = _adapter()
    info = adapter.get_env_info()
    assert set(info) == {
        "agents",
        "value_size",
        "observation_space",
        "state_space",
        "action_space",
    }
    assert (info["agents"], info["value_size"]) == (1, 1)
    assert info["observation_space"].shape == (140,)
    assert info["state_space"].shape == (162,)
    assert info["action_space"].shape == (29,)


def test_device_train_info_and_env_state_contract():
    adapter = _adapter()
    assert adapter.env.device == torch.device("cpu")
    assert adapter.get_number_of_agents() == 1
    assert adapter.set_train_info(12, object()) is None
    assert adapter.get_env_state() is None
    assert adapter.set_env_state(None) is None
    with pytest.raises(ValueError, match="env_state=None"):
        adapter.set_env_state({})


def test_invalid_observation_or_action_fails_closed():
    env = FakeTorchEnv()
    adapter = _adapter(env)
    with pytest.raises(TypeError):
        adapter.step(torch.zeros((3, 29), dtype=torch.float64))
