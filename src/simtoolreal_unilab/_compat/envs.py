"""ManagerBasedRlEnv read-phase hook for task-owned logging views.

Ported verbatim from the SimToolReal development branch of UniLab (the +8
line ``_update_state_in_read_phase`` extension): after the command manager
post-computes, optional ``log_extras`` views on command terms merge into the
step log. This is written as a real function in a real file (not an exec'd
source string) because tests pin the behavior by inspecting the method
source.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from unilab.base.np_env import NpEnvState
from unilab.envs import manager_based_rl_env as _module


def _update_state_in_read_phase(self: Any, state: NpEnvState) -> NpEnvState:
    log: dict[str, Any] = {}
    state.info["log"] = log
    self.extras = state.info

    np.add(state.info["steps"], 1, out=self.episode_length_buf)
    self.common_step_counter = self.step_counter + 1
    self._sim_step_counter = self.common_step_counter * self._cfg.sim_substeps

    self.termination_manager.compute()
    if self._cfg.is_finite_horizon:
        np.logical_or(
            self.termination_manager.terminated,
            self.termination_manager.time_outs,
            out=self.reset_terminated,
        )
        self.reset_time_outs.fill(False)
    else:
        np.copyto(self.reset_terminated, self.termination_manager.terminated)
        np.copyto(self.reset_time_outs, self.termination_manager.time_outs)
    np.logical_or(self.reset_terminated, self.reset_time_outs, out=self.reset_buf)

    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    log.update(self.reward_manager.step_reward_extras())

    if self._cfg.sim_substeps == 1:
        self.metrics_manager.compute_substep()
    self.metrics_manager.compute()

    applied_runtime_event = False
    if "step" in self.event_manager.available_modes:
        self.event_manager.apply(mode="step", dt=self.step_dt)
        applied_runtime_event = True
    if "interval" in self.event_manager.available_modes:
        self.event_manager.apply(mode="interval", dt=self.step_dt)
        applied_runtime_event = True
    if applied_runtime_event:
        # Event terms may mutate simulation state through their formal
        # interval/step capabilities; EventManager does not expose whether a
        # particular interval fired, so this boundary stays fail-closed.
        self.scene._invalidate_state_reads()

    self._command_dt.fill(self.step_dt)
    self._command_dt[self.reset_buf] = 0.0
    with self._reset_state.scoped(self._all_env_ids):
        self.command_manager.compute(dt=self._command_dt)
    if self._reset_state.last_commit_had_writes:
        self.scene._invalidate_state_reads()
    self.command_manager.post_compute()
    log_extras = getattr(self.command_manager, "log_extras", None)
    if callable(log_extras):
        log.update(
            log_extras(
                reward_manager=self.reward_manager,
                termination_manager=self.termination_manager,
            )
        )

    manager_obs = self.observation_manager.compute(update_history=True)

    self.obs_buf = self._map_observations(manager_obs)
    self._has_transition = True

    return state.replace(
        obs=self.obs_buf,
        reward=self.reward_buf,
        terminated=self.reset_terminated,
        truncated=self.reset_time_outs,
        info=state.info,
    )


def apply() -> None:
    import inspect

    cls = _module.ManagerBasedRlEnv
    if "log_extras" in inspect.getsource(cls._update_state_in_read_phase):
        raise RuntimeError(
            "official UniLab already merges log_extras in the read phase; the "
            "pinned revision drifted away from what simtoolreal_unilab._compat "
            "expects — revisit the compatibility layer"
        )
    cls._update_state_in_read_phase = _update_state_in_read_phase
