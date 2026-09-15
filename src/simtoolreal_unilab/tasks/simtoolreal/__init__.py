"""Registry owner for the SimToolReal Manager-Based task.

Only the ``isaacsim`` backend is registered: the repository convention is to
register only verified backends (motion_tracking precedent).  See DESIGN.md
in this task package for the migration contract.
"""

from unilab.base import registry
from unilab.envs import make_manager_based_rl_env

from .env_cfg import SimToolRealEnvCfg
from .scene import register_resolvers

register_resolvers()

registry.register_env_config("SimToolReal", SimToolRealEnvCfg)
registry.register_env("SimToolReal", make_manager_based_rl_env, sim_backend="isaacsim")

__all__ = []
