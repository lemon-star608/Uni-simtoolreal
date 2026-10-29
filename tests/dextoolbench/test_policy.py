"""Policy loading from persisted SimToolReal run configurations."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from simtoolreal_unilab.dextoolbench.policy import SapgPlayer


class _Player:
    def __init__(self) -> None:
        self.restored: str | None = None
        self.has_batch_dimension = False

    def init_rnn(self) -> None:
        return None

    def restore(self, checkpoint_path: str) -> None:
        self.restored = checkpoint_path


class _Runner:
    loaded: dict[str, Any] | None = None
    player = _Player()

    def load(self, payload: dict[str, Any]) -> None:
        self.loaded = payload

    def create_player(self) -> _Player:
        return self.player


def test_policy_consumes_simtoolreal_run_config_without_backend_dispatch(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    import rl_games.torch_runner as torch_runner

    checkpoint = tmp_path / "simtoolreal.pth"
    checkpoint.write_bytes(b"checkpoint")
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(
        {
            "run": {"name": "full-scale"},
            "config": {
                "training": {"sim_backend": "mujoco"},
                "rl_games": {
                    "params": {
                        "config": {
                            "name": "SimToolReal",
                            "num_actors": 24576,
                        }
                    }
                },
            },
        },
        config_path,
    )

    runner = _Runner()
    monkeypatch.setattr(torch_runner, "Runner", lambda algo_observer=None: runner)
    player = SapgPlayer(
        config_path=config_path,
        checkpoint_path=checkpoint,
        device="cpu",
    )

    assert runner.loaded is not None
    policy_config = runner.loaded["params"]["config"]
    assert policy_config["num_actors"] == 1
    assert policy_config["device"] == "cpu"
    assert policy_config["device_name"] == "cpu"
    assert player.player.restored == str(checkpoint)
    assert player.player.has_batch_dimension is True
