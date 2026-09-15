"""Run-seed injection into the task env cfg override (reaudit R1 / fix P0).

``BackendAdapter.build_task_env_cfg_override()`` must propagate the resolved
training seed into ``ManagerBasedRlEnvCfg.seed``.  Without it the manager env
draws ``secrets.randbits(63)`` per construction (manager_based_rl_env init),
so env-side sampling — reset poses, observation noise, delay lags, wrench —
is irreproducible run-to-run.  The original repository seeds the env from the
agent seed (isaacsimenvs/train.py:131, ``env_cfg.seed = agent seed``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

from unilab.base import config_adapter
from unilab.base import registry as registry_module


@dataclass
class _SeedCfg:
    """Minimal stand-in with the ``ManagerBasedRlEnvCfg.seed`` field."""

    seed: int | None = None


@dataclass
class _NoSeedCfg:
    """Stand-in for a future env cfg without a ``seed`` field."""


def _stub_cfg(
    *,
    algo_seed: int | None = 42,
    training_seed: int | None = None,
    env: dict[str, Any] | None = None,
) -> DictConfig:
    return OmegaConf.create(
        {
            "training": {
                "task_name": "StubTask",
                "sim_backend": "mujoco",
                "seed": training_seed,
            },
            "algo": {"seed": algo_seed},
            "env": env if env is not None else {"sim_dt": 0.01},
        }
    )


@pytest.fixture
def stub_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass the real registry stack; the builder only needs four calls."""

    monkeypatch.setattr(registry_module, "ensure_registries", lambda: None)
    monkeypatch.setattr(
        registry_module, "resolve_reward_override_field", lambda name: "reward_config"
    )
    monkeypatch.setattr(registry_module, "materialize_env_config", lambda name: _SeedCfg())
    monkeypatch.setattr(
        config_adapter,
        "extract_reward_config",
        lambda cfg, *, target_field: {target_field: {}},
    )


def _build(cfg: DictConfig) -> dict[str, Any]:
    return config_adapter.BackendAdapter(cfg, root_dir=Path(".")).build_task_env_cfg_override()


def test_override_injects_algo_seed(stub_registry: None) -> None:
    override = _build(_stub_cfg(algo_seed=42))
    assert override["seed"] == 42


def test_training_seed_used_when_algo_seed_absent(stub_registry: None) -> None:
    override = _build(_stub_cfg(algo_seed=None, training_seed=7))
    assert override["seed"] == 7


def test_explicit_env_seed_wins_over_algo_seed(stub_registry: None) -> None:
    override = _build(_stub_cfg(algo_seed=42, env={"seed": 123, "sim_dt": 0.01}))
    assert override["seed"] == 123
    assert override["sim_dt"] == 0.01


def test_no_seed_configured_leaves_key_absent(stub_registry: None) -> None:
    override = _build(_stub_cfg(algo_seed=None, training_seed=None))
    assert "seed" not in override


def test_seed_not_injected_when_env_cfg_lacks_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future non-manager env cfgs without ``seed`` must keep constructing."""
    monkeypatch.setattr(registry_module, "ensure_registries", lambda: None)
    monkeypatch.setattr(
        registry_module, "resolve_reward_override_field", lambda name: "reward_config"
    )
    monkeypatch.setattr(registry_module, "materialize_env_config", lambda name: _NoSeedCfg())
    monkeypatch.setattr(
        config_adapter,
        "extract_reward_config",
        lambda cfg, *, target_field: {target_field: {}},
    )
    override = _build(_stub_cfg(algo_seed=42))
    assert "seed" not in override
