"""Env-side RNG seed propagation into the task env config override.

Method bodies below are ported verbatim from the SimToolReal development
branch of UniLab and applied onto the pinned official classes at import time
(see simtoolreal_unilab._compat). They exec() inside a copy of the official
module namespace so module-level helpers resolve exactly as upstream wrote
them.
"""
from __future__ import annotations

from typing import Any  # noqa: F401

from unilab.base import config_adapter as _module
from unilab.utils.seed import resolve_training_seed

_SOURCES = {
    "BackendAdapter.build_task_env_cfg_override": (
        '    def build_task_env_cfg_override(self) -> dict[str, Any]:\n'
        '        """Build env_cfg_override from the resolved reward + env sections."""\n'
        '        registry.ensure_registries()\n'
        '        task_name = str(self.cfg.training.task_name)\n'
        '        reward_target = registry.resolve_reward_override_field(task_name)\n'
        '        env_overrides = self._to_plain_dict(getattr(self.cfg, "env", None))\n'
        '        if reward_target in env_overrides:\n'
        '            raise ValueError(\n'
        '                f"Task \'{task_name}\' declares both Hydra root \'reward\' and "\n'
        '                f"\'env.{reward_target}\'; use the root reward owner only"\n'
        '            )\n'
        '\n'
        '        env_cfg_override: dict[str, Any] = extract_reward_config(\n'
        '            self.cfg, target_field=reward_target\n'
        '        )\n'
        '        # Env-side RNG seeding (training-pipeline-reaudit R1 / fix P0): the\n'
        '        # original repository propagates the agent seed into the env config\n'
        '        # (isaacsimenvs/train.py:131 ``env_cfg.seed = agent seed``).  Without\n'
        '        # it ManagerBasedRlEnv draws ``secrets.randbits(63)`` per\n'
        '        # construction, so env-side sampling — reset poses, observation\n'
        '        # noise, delay lags, wrench — is irreproducible run-to-run.  An\n'
        '        # explicit ``env.seed`` owner declaration keeps precedence; env cfgs\n'
        '        # without a ``seed`` field (non-manager tasks) are left untouched.\n'
        '        seed = resolve_training_seed(self.cfg).effective_seed\n'
        '        if seed is not None and hasattr(registry.materialize_env_config(task_name), "seed"):\n'
        '            env_cfg_override.setdefault("seed", int(seed))\n'
        '        env_cfg_override.update(env_overrides)\n'
        '\n'
        '        return env_cfg_override\n'
    ),
}


def apply() -> None:
    exec_ns = dict(vars(_module))
    exec_ns["resolve_training_seed"] = resolve_training_seed
    for name, source in _SOURCES.items():
        exec(compile(source, f"<compat.config_adapter.{{name}}>", "exec"), exec_ns)
    setattr(_module.BackendAdapter, "build_task_env_cfg_override", exec_ns["build_task_env_cfg_override"])
