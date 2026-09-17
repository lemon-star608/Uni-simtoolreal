"""Registry owner for the SimToolReal Manager-Based task.

Both backends are registered: ``isaacsim`` (URDF asset path, the merge
anchor) and ``mujoco`` (MJCF asset path, M-J2).  The repository convention is
to register only verified backends; the mujoco registration landed after the
L2 smoke gate on the real backend.  See DESIGN.md in this task package for
the migration contract.
"""

from unilab.base import registry
from unilab.envs import make_manager_based_rl_env

from .env_cfg import SimToolRealEnvCfg
from .scene import register_resolvers

register_resolvers()

registry.register_env_config("SimToolReal", SimToolRealEnvCfg)
registry.register_env("SimToolReal", make_manager_based_rl_env, sim_backend="isaacsim")
registry.register_env("SimToolReal", make_manager_based_rl_env, sim_backend="mujoco")

__all__ = []
