"""Registry owner for the SimToolReal Manager-Based task (step-0 skeleton).

Only the ``isaacsim`` backend is registered: the repository convention is to
register only verified backends (motion_tracking precedent), so the MuJoCo
registration line lands in step 2 after that path is verified.  See
step0discussion.md (work item 0.5) and DESIGN.md for the migration contract.
"""

from unilab.base import registry
from unilab.envs import make_manager_based_rl_env

from .env_cfg import SimToolRealEnvCfg
from .scene import register_resolvers

register_resolvers()

registry.register_env_config("SimToolReal", SimToolRealEnvCfg)
registry.register_env("SimToolReal", make_manager_based_rl_env, sim_backend="isaacsim")

__all__ = []
