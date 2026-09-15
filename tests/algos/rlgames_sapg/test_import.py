from __future__ import annotations

import inspect
import json
import os
import shutil
import sys
import types
from pathlib import Path

import gymnasium
import numpy as np
import pytest
import torch

from tests.algos.rlgames_sapg import _runtime_requirement as runtime_requirement
from tests.algos.rlgames_sapg._runtime_requirement import (
    VENDOR_PACKAGE_ROOT,
    require_simtoolreal_rl_games,
)

require_simtoolreal_rl_games()

from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch.central_value import CentralValueTrain
from rl_games.algos_torch.model_builder import ModelBuilder
from rl_games.algos_torch.players import PpoPlayerContinuous
from rl_games.common.experience import ExperienceBuffer, ReplayBuffer
from rl_games.torch_runner import Runner


def _assert_vendor_owned(symbol: object) -> None:
    source_path = Path(inspect.getfile(symbol)).resolve()
    assert source_path.is_relative_to(VENDOR_PACKAGE_ROOT), source_path


def test_required_gate_is_active() -> None:
    if os.environ.get("UNILAB_REQUIRE_SAPG") != "1":
        pytest.skip("UNILAB_REQUIRE_SAPG=1 required-mode check is a CI-only assertion")
    require_simtoolreal_rl_games()


def test_required_import_does_not_pollute_the_vendored_inventory() -> None:
    assert not list(VENDOR_PACKAGE_ROOT.rglob("__pycache__"))
    assert not list(VENDOR_PACKAGE_ROOT.rglob("*.pyc"))


def test_native_runner_agent_central_value_builder_and_player_are_vendored() -> None:
    for symbol in (Runner, A2CAgent, CentralValueTrain, ModelBuilder, PpoPlayerContinuous):
        _assert_vendor_owned(symbol)


def test_required_gate_rejects_a_wrong_distribution(monkeypatch) -> None:
    wrong_distribution = types.SimpleNamespace(
        metadata={"Name": "rl-games"},
        version=runtime_requirement.V2_DISTRIBUTION_VERSION,
    )
    monkeypatch.setattr(
        runtime_requirement.importlib.metadata,
        "distribution",
        lambda _name: wrong_distribution,
    )

    with pytest.raises(RuntimeError, match="wrong rl_games distribution name"):
        require_simtoolreal_rl_games()


def test_required_gate_rejects_a_wrong_package_path(monkeypatch, tmp_path) -> None:
    wrong_origin = tmp_path / "rl_games/__init__.py"
    monkeypatch.setattr(
        runtime_requirement.importlib.util,
        "find_spec",
        lambda _name: types.SimpleNamespace(origin=str(wrong_origin)),
    )

    with pytest.raises(RuntimeError, match="resolves outside the pinned vendor"):
        require_simtoolreal_rl_games()


def test_required_gate_rejects_an_already_loaded_module_from_another_path(
    monkeypatch, tmp_path
) -> None:
    intruder = types.ModuleType("rl_games.injected")
    intruder.__file__ = str(tmp_path / "injected.py")
    monkeypatch.setitem(sys.modules, intruder.__name__, intruder)

    with pytest.raises(RuntimeError, match=r"loaded rl_games\.injected resolves outside"):
        require_simtoolreal_rl_games()


def test_required_gate_rejects_hash_drift_in_an_already_loaded_module(
    monkeypatch, tmp_path
) -> None:
    loaded_name = "rl_games.algos_torch.models"
    assert loaded_name in sys.modules
    vendor_copy = shutil.copytree(runtime_requirement.VENDOR_ROOT, tmp_path / "vendor")
    drifted = vendor_copy / "rl_games/algos_torch/models.py"
    drifted.write_bytes(drifted.read_bytes() + b"\n")
    monkeypatch.setattr(runtime_requirement, "VENDOR_ROOT", vendor_copy.resolve())
    real_distribution = runtime_requirement.importlib.metadata.distribution(
        runtime_requirement.EXPECTED_DISTRIBUTION
    )
    copied_distribution = types.SimpleNamespace(
        metadata=real_distribution.metadata,
        version=real_distribution.version,
        read_text=lambda name: (
            json.dumps(
                {
                    "url": vendor_copy.resolve().as_uri(),
                    "dir_info": {"editable": True},
                }
            )
            if name == "direct_url.json"
            else real_distribution.read_text(name)
        ),
    )
    monkeypatch.setattr(
        runtime_requirement.importlib.metadata,
        "distribution",
        lambda _name: copied_distribution,
    )

    with pytest.raises(RuntimeError, match="vendored module hash drift"):
        require_simtoolreal_rl_games()


def test_experience_buffer_accepts_exact_gymnasium_box_spaces() -> None:
    env_info = {
        "observation_space": gymnasium.spaces.Box(
            low=-10.0, high=10.0, shape=(140,), dtype="float32"
        ),
        "state_space": gymnasium.spaces.Box(low=-10.0, high=10.0, shape=(162,), dtype="float32"),
        "action_space": gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(29,), dtype="float32"),
        "agents": 1,
        "value_size": 1,
    }
    algo_info = {
        "num_actors": 2,
        "horizon_length": 4,
        "has_central_value": True,
        "use_action_masks": False,
    }

    buffer = ExperienceBuffer(env_info, algo_info, device="cpu")

    assert buffer.is_continuous is True
    assert buffer.actions_num == 29
    assert buffer.actions_shape == (29,)
    assert buffer.tensor_dict["obses"].shape == (4, 2, 140)
    assert buffer.tensor_dict["states"].shape == (4, 2, 162)
    assert buffer.tensor_dict["actions"].shape == (4, 2, 29)
    assert buffer.tensor_dict["mus"].shape == (4, 2, 29)
    assert buffer.tensor_dict["sigmas"].shape == (4, 2, 29)
    assert buffer.tensor_dict["dones"].shape == (4, 2)
    assert buffer.tensor_dict["rewards"].shape == (4, 2, 1)
    assert buffer.tensor_dict["values"].shape == (4, 2, 1)


def test_removed_numpy_bool_aliases_keep_replay_and_action_mask_dtypes() -> None:
    observation_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    replay = ReplayBuffer(size=5, ob_space=observation_space)
    assert replay._dones.shape == (5,)
    assert replay._dones.dtype == np.bool_

    env_info = {
        "observation_space": observation_space,
        "action_space": gymnasium.spaces.Discrete(5),
        "agents": 1,
        "value_size": 1,
    }
    algo_info = {
        "num_actors": 2,
        "horizon_length": 4,
        "has_central_value": False,
        "use_action_masks": True,
    }
    buffer = ExperienceBuffer(env_info, algo_info, device="cpu")
    assert buffer.tensor_dict["action_masks"].shape == (4, 2, 5)
    assert buffer.tensor_dict["action_masks"].dtype == torch.bool
