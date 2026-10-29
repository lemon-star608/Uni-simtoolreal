from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[3]


def _cfg():
    with initialize_config_dir(
        config_dir=str(ROOT / "src/simtoolreal_unilab/conf/rlgames_sapg"), version_base="1.3"
    ):
        return compose("config")


class FakeRunner:
    def __init__(self, *, algo_observer):
        self.observer = algo_observer
        self.calls = []

    def load(self, config):
        self.calls.append(("load", config))

    def set_vec_env(self, adapter):
        self.calls.append(("set_vec_env", adapter))

    def run_train(self, args):
        self.calls.append(("run_train", args))
        return {"epoch": 1}


@pytest.mark.parametrize(
    ("checkpoint", "mode", "expected_args"),
    [
        (None, "none", {}),
        ("model.pth", "resume", {"checkpoint": "model.pth", "checkpoint_load_mode": "resume"}),
        ("model.pth", "weights", {"checkpoint": "model.pth", "checkpoint_load_mode": "weights"}),
    ],
)
def test_native_executor_deep_copies_config_and_calls_exact_runner_path(
    tmp_path, checkpoint, mode, expected_args
):
    from simtoolreal_unilab.rlgames_sapg.runtime import execute_native_train

    cfg = _cfg()
    before = cfg.copy()
    adapter = object()
    observer = object()
    result = execute_native_train(
        cfg,
        adapter=adapter,
        observer=observer,
        train_dir=tmp_path,
        run_name="0_test_mujoco",
        checkpoint=checkpoint,
        checkpoint_load_mode=mode,
        runner_factory=FakeRunner,
        verify_dependency=False,
    )
    assert [call[0] for call in result.runner.calls] == ["load", "set_vec_env", "run_train"]
    loaded = result.runner.calls[0][1]["params"]
    assert loaded["config"]["train_dir"] == str(tmp_path)
    assert loaded["config"]["full_experiment_name"] == "0_test_mujoco"
    assert loaded["config"]["device"] == "cuda:0"
    assert result.args == expected_args and result.result == {"epoch": 1}
    assert cfg == before


def test_actual_runner_and_agent_factories_are_vendored_native_owners():
    from simtoolreal_unilab.rlgames_sapg.runtime import create_native_runner

    runner = create_native_runner(object())
    assert type(runner).__module__ == "rl_games.torch_runner"
    builder = runner.algo_factory._builders["a2c_continuous"]
    agent = builder
    assert callable(agent)
    assert (
        "a2c_continuous.A2CAgent"
        in Path(ROOT / "third_party/simtoolreal_rl_games/rl_games/torch_runner.py").read_text()
    )


@pytest.mark.parametrize(
    ("checkpoint", "mode"), [(None, "resume"), ("x.pth", "none"), ("x.pth", "bad")]
)
def test_executor_rejects_checkpoint_mode_conflicts(checkpoint, mode, tmp_path):
    from simtoolreal_unilab.rlgames_sapg.runtime import execute_native_train

    with pytest.raises(ValueError, match="checkpoint"):
        execute_native_train(
            _cfg(),
            adapter=object(),
            observer=object(),
            train_dir=tmp_path,
            run_name="0_test_mujoco",
            checkpoint=checkpoint,
            checkpoint_load_mode=mode,
            runner_factory=FakeRunner,
            verify_dependency=False,
        )
