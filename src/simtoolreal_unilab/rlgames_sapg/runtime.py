"""Thin executor for the vendored native Runner/A2CAgent training path."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

from omegaconf import DictConfig, OmegaConf

from .dependency import require_rlgames_sapg


@dataclass(frozen=True)
class NativeTrainResult:
    result: Any
    runner: Any
    params: dict[str, Any]
    args: dict[str, str]


def create_native_runner(observer: object) -> Any:
    """Construct the pinned native Runner after the dependency identity gate."""
    require_rlgames_sapg()
    from rl_games.torch_runner import Runner

    return Runner(algo_observer=observer)


def _checkpoint_args(checkpoint: str | None, mode: str) -> dict[str, str]:
    if mode == "none":
        if checkpoint is not None:
            raise ValueError("checkpoint must be absent when checkpoint_load_mode=none")
        return {}
    if mode not in {"resume", "weights"}:
        raise ValueError(f"unsupported checkpoint_load_mode: {mode!r}")
    if not checkpoint:
        raise ValueError(f"checkpoint is required for checkpoint_load_mode={mode}")
    return {"checkpoint": checkpoint, "checkpoint_load_mode": mode}


def execute_native_train(
    cfg: DictConfig,
    *,
    adapter: object,
    observer: object,
    train_dir: str | Path,
    run_name: str,
    checkpoint: str | None,
    checkpoint_load_mode: str,
    runner_factory: Callable[..., Any] | None = None,
    verify_dependency: bool = True,
) -> NativeTrainResult:
    """Execute the only production rollout/update path: native ``run_train``."""
    if not run_name.startswith("0_"):
        raise ValueError("native run_name must start with '0_'")
    if verify_dependency:
        require_rlgames_sapg()
    resolved = OmegaConf.to_container(cfg.rl_games.params, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("cfg.rl_games.params must resolve to a mapping")
    if not all(isinstance(key, str) for key in resolved):
        raise TypeError("cfg.rl_games.params keys must be strings")
    params = copy.deepcopy(cast(dict[str, Any], resolved))
    native_config = params["config"]
    native_config["train_dir"] = str(Path(train_dir))
    native_config["full_experiment_name"] = run_name
    native_config["device"] = str(cfg.training.device)
    native_config["device_name"] = str(cfg.training.device)
    args = _checkpoint_args(checkpoint, checkpoint_load_mode)
    factory = runner_factory
    if factory is None:
        from rl_games.torch_runner import Runner

        factory = Runner
    runner = factory(algo_observer=observer)
    runner.load({"params": params})
    runner.set_vec_env(adapter)
    result = runner.run_train(args)
    return NativeTrainResult(result=result, runner=runner, params=params, args=args)
