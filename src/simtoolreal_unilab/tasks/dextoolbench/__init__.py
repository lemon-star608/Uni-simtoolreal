"""Registry owner for the fixed-scene DexToolBench MuJoCo task."""

from unilab.base import registry
from unilab.envs import make_manager_based_rl_env

from ..simtoolreal.runtime import retain_zero_initial_episode_steps
from .env_cfg import DexToolBenchEnvCfg


def _make_env(cfg, num_envs: int = 1, backend_type: str = "mujoco"):
    if backend_type != "mujoco":
        raise ValueError(f"DexToolBench supports backend mujoco, got {backend_type!r}")
    if num_envs != 1:
        raise ValueError("DexToolBench evaluation is a single-environment fixed scene")
    env = make_manager_based_rl_env(cfg, num_envs=1, backend_type="mujoco")
    return retain_zero_initial_episode_steps(env)


registry.register_env_config("DexToolBench", DexToolBenchEnvCfg)
registry.register_env("DexToolBench", _make_env, sim_backend="mujoco")

__all__ = []
