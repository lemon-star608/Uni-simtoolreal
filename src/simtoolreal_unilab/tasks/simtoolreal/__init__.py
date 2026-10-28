"""Registry owner for the SimToolReal MuJoCo Manager-Based task."""

from unilab.base import registry
from unilab.envs import make_manager_based_rl_env

from .env_cfg import SimToolRealEnvCfg
from .runtime import retain_zero_initial_episode_steps
from .scene import register_resolvers

register_resolvers()


def _make_env_with_entity_binding(cfg, num_envs: int = 1, backend_type: str = "mujoco"):
    """Factory wrapper: bind the entity variant plan, then the stock factory.

    The entity variant plan's assignment is sized by ``num_envs``, which only
    this call site knows. This wrapper provides that task-side hook before the
    fixed MuJoCo factory builds the environment.
    """
    if isinstance(cfg, SimToolRealEnvCfg):
        cfg.bind_entity_variant(num_envs)
    env = make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend_type)
    return retain_zero_initial_episode_steps(env)


registry.register_env_config("SimToolReal", SimToolRealEnvCfg)
registry.register_env("SimToolReal", _make_env_with_entity_binding, sim_backend="mujoco")

__all__ = []
