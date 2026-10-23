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


def _forward_isaacsim_solver_kwargs() -> None:
    """Bridge the task-owned PhysX knobs until UniLab's generic owner grows them."""
    from unilab.envs import manager_based_rl_env as module

    if getattr(module.env_backend_kwargs, "_simtoolreal_physx_bridge", False):
        return
    original = module.env_backend_kwargs

    def adapter(cfg):
        kwargs = original(cfg)
        for name in (
            "solver_position_iteration_count",
            "solver_velocity_iteration_count",
            "bounce_threshold_velocity",
            "contact_offset",
            "rest_offset",
            "max_depenetration_velocity",
            "gpu_max_rigid_contact_count",
            "gpu_max_rigid_patch_count",
        ):
            value = getattr(cfg, name, None)
            if value is not None:
                kwargs[name] = value
        return kwargs

    adapter._simtoolreal_physx_bridge = True
    module.env_backend_kwargs = adapter


_forward_isaacsim_solver_kwargs()


def _make_env_with_entity_binding(cfg, num_envs: int = 1, backend_type: str = "mujoco"):
    """Factory wrapper: bind the entity variant plan, then the stock factory.

    The B-form entity variant plan's assignment is sized by ``num_envs``,
    which only this call site knows (the upstream guard builds the whole-model
    ``fixed_variant_plan`` the same way, but no upstream hook exists for
    ``entity_variant`` — this wrapper is that hook, task-side).  Both B-form
    lines (mujoco since M-J5, isaacsim since the Isaac/MJCF bring-up) register
    through it.
    """
    if isinstance(cfg, SimToolRealEnvCfg):
        cfg.bind_entity_variant(num_envs)
    return make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend_type)


registry.register_env_config("SimToolReal", SimToolRealEnvCfg)
registry.register_env("SimToolReal", _make_env_with_entity_binding, sim_backend="isaacsim")
registry.register_env("SimToolReal", _make_env_with_entity_binding, sim_backend="mujoco")

__all__ = []
