"""Registry owner for the SimToolReal Manager-Based task.

Both backends are registered: ``isaacsim`` (URDF asset path, the merge
anchor) and ``mujoco`` (MJCF asset path, entity-split B form since M-J5).
The repository convention is to register only verified backends; the mujoco
registration landed after the L2 smoke gate on the real backend.  See
DESIGN.md in this task package for the migration contract.
"""

from unilab.base import registry
from unilab.envs import make_manager_based_rl_env

from .env_cfg import SimToolRealEnvCfg
from .scene import register_resolvers

register_resolvers()


def _make_mujoco_env(cfg, num_envs: int = 1, backend_type: str = "mujoco"):
    """Mujoco factory: bind the entity variant plan, then the stock factory.

    The B-form entity variant plan's assignment is sized by ``num_envs``,
    which only this call site knows (the upstream guard builds the whole-model
    ``fixed_variant_plan`` the same way, but no upstream hook exists for
    ``entity_variant`` — this wrapper is that hook, task-side).
    """
    if isinstance(cfg, SimToolRealEnvCfg):
        cfg.bind_entity_variant(num_envs)
    return make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend_type)


registry.register_env_config("SimToolReal", SimToolRealEnvCfg)
registry.register_env("SimToolReal", make_manager_based_rl_env, sim_backend="isaacsim")
registry.register_env("SimToolReal", _make_mujoco_env, sim_backend="mujoco")

__all__ = []
