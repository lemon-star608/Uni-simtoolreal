"""Registry owner for the SimToolReal Manager-Based task.

Both backends are registered on the B-form entity asset path (MJCF sources,
entity composition owned by unisim): ``mujoco`` since M-J5, ``isaacsim``
since the Isaac/MJCF bring-up.  The repository convention is to register
only verified backends; each registration lands after the L2 smoke gate on
the real backend.  See DESIGN.md in this task package for the migration
contract.
"""

from unilab.base import registry
from unilab.envs import make_manager_based_rl_env

from .env_cfg import SimToolRealEnvCfg
from .scene import register_resolvers

register_resolvers()


def _make_env_with_entity_binding(cfg, num_envs: int = 1, backend_type: str = "mujoco"):
    """Factory wrapper: bind the entity variant plan, then the stock factory.

    The B-form entity variant plan's assignment is sized by ``num_envs``,
    which only this call site knows (the upstream guard builds the whole-model
    ``fixed_variant_plan`` the same way, but no upstream hook exists for
    ``entity_variant`` — this wrapper is that hook, task-side).  Both B-form
    lines (mujoco since M-J5, isaacsim since the Isaac/MJCF bring-up) register
    through it.  Backend creation (unisim compose + VariantPack + worker INIT)
    and the first reset carry cold-start phase markers.
    """
    from .cold_start import mark, phase

    if isinstance(cfg, SimToolRealEnvCfg):
        cfg.bind_entity_variant(num_envs)
    with phase("backend creation", f"backend={backend_type} num_envs={num_envs}"):
        env = make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend_type)

    original_reset = env.reset

    def _reset_with_first_marker(*args, **kwargs):
        env.reset = original_reset  # one-shot marker: restore, then delegate
        with phase("first reset"):
            return original_reset(*args, **kwargs)

    try:
        env.reset = _reset_with_first_marker
    except AttributeError:
        mark("first reset", "marker unavailable (env disallows reset wrapping)")
    return env


registry.register_env_config("SimToolReal", SimToolRealEnvCfg)
registry.register_env("SimToolReal", _make_env_with_entity_binding, sim_backend="isaacsim")
registry.register_env("SimToolReal", _make_env_with_entity_binding, sim_backend="mujoco")

__all__ = []
