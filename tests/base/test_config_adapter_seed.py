"""The RL-Games owner seeds the Manager-owned Torch RNG explicitly."""

from __future__ import annotations

from omegaconf import OmegaConf

from simtoolreal_unilab.scripts.train_rlgames_sapg import _propagate_env_seed


def _cfg(seed):
    return OmegaConf.create({"algo": {"seed": seed}})


def test_algo_seed_is_injected_once():
    override: dict[str, int] = {}
    _propagate_env_seed(_cfg(42), override)
    assert override == {"seed": 42}
    _propagate_env_seed(_cfg(43), override)
    assert override == {"seed": 42}


def test_absent_seed_leaves_override_untouched():
    override: dict[str, int] = {}
    _propagate_env_seed(_cfg(None), override)
    assert override == {}
