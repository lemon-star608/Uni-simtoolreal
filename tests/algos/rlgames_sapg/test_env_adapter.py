from __future__ import annotations

import ast
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from unilab.base.np_env import NpEnvState


class FakeNpEnv:
    def __init__(self, num_envs: int = 3):
        self.num_envs = num_envs
        self.state = None
        self.init_calls = 0
        self.reset_calls: list[np.ndarray] = []
        self.step_calls: list[np.ndarray] = []
        self.obs_groups_spec = {"obs": 140, "critic": 162}
        self.action_space = gym.spaces.Box(-1.0, 1.0, (29,), dtype=np.float32)
        self.next_terminated = np.zeros(num_envs, dtype=bool)
        self.next_truncated = np.zeros(num_envs, dtype=bool)

    def _obs(self, offset: float = 0.0):
        rows = np.arange(self.num_envs, dtype=np.float32)[:, None]
        return {
            "obs": np.broadcast_to(rows + offset, (self.num_envs, 140)).copy(),
            "critic": np.broadcast_to(rows + offset + 10, (self.num_envs, 162)).copy(),
        }

    def init_state(self):
        self.init_calls += 1
        self.state = NpEnvState(
            self._obs(),
            np.zeros(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=bool),
            np.zeros(self.num_envs, dtype=bool),
            {"log": {"reset": 1.0}},
        )
        return self.state

    def reset(self, env_ids):
        self.reset_calls.append(env_ids.copy())
        return self._obs(20.0), {"reset": True}

    def step(self, actions):
        self.step_calls.append(actions)
        self.state = NpEnvState(
            self._obs(1.0),
            np.arange(self.num_envs, dtype=np.float32),
            self.next_terminated.copy(),
            self.next_truncated.copy(),
            {
                "log": {"reward": 2.0},
                "final_observation": self._obs(99.0),
                "_final_observation": self.next_terminated | self.next_truncated,
                "timing": {"large": np.zeros((self.num_envs, 40))},
            },
        )
        return self.state


def _adapter(env=None):
    from simtoolreal_unilab.rlgames_sapg.env_adapter import RlGamesNpEnvAdapter

    return RlGamesNpEnvAdapter(env or FakeNpEnv(), device="cpu")


def test_first_reset_reuses_init_state_and_routes_nonidentity_rows():
    env = FakeNpEnv()
    adapter = _adapter(env)
    obs = adapter.reset()
    assert env.init_calls == 1
    assert env.reset_calls == []
    assert tuple(obs) == ("obs", "states")
    assert obs["obs"].shape == (3, 140)
    assert obs["states"].shape == (3, 162)
    assert obs["obs"][:, 0].tolist() == [0.0, 1.0, 2.0]
    assert obs["states"][:, 0].tolist() == [10.0, 11.0, 12.0]
    adapter.reset()
    assert len(env.reset_calls) == 1
    assert env.reset_calls[0].tolist() == [0, 1, 2]


def test_spaces_and_env_info_are_finite_gymnasium_contracts():
    adapter = _adapter()
    info = adapter.get_env_info()
    assert set(info) == {"agents", "value_size", "observation_space", "state_space", "action_space"}
    assert (info["agents"], info["value_size"]) == (1, 1)
    for key in ("observation_space", "state_space", "action_space"):
        assert isinstance(info[key], gym.spaces.Box)
        assert np.isfinite(info[key].low).all() and np.isfinite(info[key].high).all()
    assert info["observation_space"].shape == (140,)
    assert info["state_space"].shape == (162,)
    assert info["action_space"].shape == (29,)


@pytest.mark.parametrize(
    ("terminated", "truncated", "done"),
    [
        ([True, False, False], [False, False, False], [True, False, False]),
        ([False, False, False], [False, True, False], [False, True, False]),
        ([True, False, False], [True, False, False], [True, False, False]),
    ],
)
def test_step_preserves_terminal_masks_timeout_and_log(terminated, truncated, done):
    env = FakeNpEnv()
    env.next_terminated[:] = terminated
    env.next_truncated[:] = truncated
    adapter = _adapter(env)
    adapter.reset()
    obs, reward, actual_done, info = adapter.step(torch.zeros((3, 29)))
    assert len(env.step_calls) == 1
    assert env.step_calls[0].dtype == np.float32
    assert obs["obs"][:, 0].tolist() == [1.0, 2.0, 3.0]
    assert reward.dtype == torch.float32 and reward.tolist() == [0.0, 1.0, 2.0]
    assert actual_done.dtype == torch.bool and actual_done.tolist() == done
    assert info["time_outs"].dtype == torch.bool
    assert info["time_outs"].tolist() == truncated
    assert info["log"] == {"reward": 2.0}
    assert "timing" not in info
    assert "final_observation" in info and "_final_observation" in info


def test_device_dtype_train_info_and_env_state_contract():
    adapter = _adapter()
    assert adapter.env.device == torch.device("cpu")
    assert adapter.get_number_of_agents() == 1
    assert adapter.set_train_info(12, object()) is None
    assert adapter.get_env_state() is None
    assert adapter.set_env_state(None) is None
    with pytest.raises(ValueError, match="env_state=None"):
        adapter.set_env_state({})


@pytest.mark.parametrize(
    "case", ["groups", "shape", "dtype", "finite", "action_shape", "action_dtype"]
)
def test_invalid_observation_or_action_fails_closed(case):
    env = FakeNpEnv()
    adapter = _adapter(env)
    if case == "groups":
        env.obs_groups_spec = {"obs": 140, "wrong": 162}
    elif case == "shape":
        env._obs = lambda offset=0.0: {  # type: ignore[method-assign]
            "obs": np.zeros((3, 139), np.float32),
            "critic": np.zeros((3, 162), np.float32),
        }
    elif case == "dtype":
        env._obs = lambda offset=0.0: {  # type: ignore[method-assign]
            "obs": np.zeros((3, 140), np.float64),
            "critic": np.zeros((3, 162), np.float32),
        }
    elif case == "finite":
        original = env._obs

        def bad(offset=0.0):
            obs = original(offset)
            obs["obs"][0, 0] = np.nan
            return obs

        env._obs = bad  # type: ignore[method-assign]
    if case in {"groups", "shape", "dtype", "finite"}:
        with pytest.raises((TypeError, ValueError)):
            adapter.reset()
        return
    adapter.reset()
    action = torch.zeros((3, 28) if case == "action_shape" else (3, 29))
    if case == "action_dtype":
        action = action.to(torch.float64)
    with pytest.raises((TypeError, ValueError)):
        adapter.step(action)


def test_adapter_source_has_no_backend_asset_or_concurrency_probe():
    source = (
        Path(__file__).resolve().parents[3] / "src/simtoolreal_unilab/rlgames_sapg/env_adapter.py"
    )
    tree = ast.parse(source.read_text())
    text = source.read_text()
    assert "_backend" not in text and "getattr(" not in text and "hasattr(" not in text
    assert "read_text" not in text and "read_bytes" not in text and "open(" not in text
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and any(alias.name in {"threading", "queue"} for alias in node.names)
        for node in ast.walk(tree)
    )
