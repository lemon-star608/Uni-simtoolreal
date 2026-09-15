"""Entity multi-root binding, variant-mass facade, ops-form interval plans.

Method bodies below are ported verbatim from the SimToolReal development
branch of UniLab and applied onto the pinned official classes at import time
(see simtoolreal_unilab._compat). They exec() inside a copy of the official
module namespace so module-level helpers resolve exactly as upstream wrote
them.
"""

from __future__ import annotations

import textwrap

import numpy as np
from unilab.base import entity as _module
from unisim.dr.types import (
    INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_FORCE,
    INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_TORQUE,
    IntervalRandomizationPlan,
    IntervalTermOp,
)


def _interval_term_ops(
    *entries: tuple[str, np.ndarray | None, np.ndarray | None],
) -> tuple[IntervalTermOp, ...]:
    """Build interval-plan ops from ``(term, payload, body_ids)`` triples.

    Entries with a ``None`` payload are skipped, mirroring how
    ``IntervalRandomizationPlan.iter_ops`` drops unset legacy fields: a plan
    built with these ops and the legacy plan built from the same arrays
    expand to byte-identical op tuples.  New constructions must use this
    ops form; the plan's legacy ``body_*`` fields are deprecated upstream.
    """
    return tuple(
        IntervalTermOp(term=term, payload=payload, body_ids=body_ids)
        for term, payload, body_ids in entries
        if payload is not None
    )


_SOURCES = {
    "Entity.fixed_variant_mass": (
        "    @property\n"
        "    def fixed_variant_mass(self) -> np.ndarray:\n"
        '        """Read-only per-environment mass of this entity\'s fixed variant pool.\n'
        "\n"
        "        Cold-path readback of the worker-measured variant masses through the\n"
        "        backend contract.  The pool assignment is immutable for the life of\n"
        "        the scene, so the readback is cached once per entity instead of\n"
        "        participating in the per-step state-read cache.  Fails closed when the\n"
        "        backend materialized no variant pool bound to this entity.\n"
        '        """\n'
        "        cached = self._fixed_variant_mass\n"
        "        if cached is None:\n"
        "            try:\n"
        "                cached = self._backend.get_entity_variant_metadata(self.name).mass\n"
        "            except (AttributeError, NotImplementedError) as exc:\n"
        '                raise self._capability_error("fixed variant mass", str(exc)) from exc\n'
        "            self._fixed_variant_mass = cached\n"
        "        return cached\n"
    ),
    "Entity.write_root_state_to_sim": (
        "    def write_root_state_to_sim(\n"
        "        self,\n"
        "        root_state: np.ndarray,\n"
        "        env_ids: np.ndarray | slice | None = None,\n"
        "    ) -> None:\n"
        '        """Stage a 13-D world-frame root state in the active reset transaction."""\n'
        "        reset_state, layout = self._require_root_state_write()\n"
        "        resolved_env_ids = self._normalize_reset_env_ids(env_ids)\n"
        "        reset_state.write_root_state(\n"
        "            resolved_env_ids,\n"
        "            layout,\n"
        "            root_state,\n"
        "            entity_name=self.name,\n"
        '            term_name=f"{self.name}.write_root_state_to_sim",\n'
        "        )\n"
    ),
    "Entity.apply_root_velocity_delta_to_sim": (
        "    def apply_root_velocity_delta_to_sim(\n"
        "        self,\n"
        "        linear_delta: np.ndarray | None,\n"
        "        angular_delta: np.ndarray | None,\n"
        "        env_ids: np.ndarray | slice | None = None,\n"
        "        *,\n"
        '        term_name: str = "push_by_setting_velocity",\n'
        "    ) -> None:\n"
        '        """Dispatch world-frame root linear/angular velocity deltas in one interval plan."""\n'
        "        if linear_delta is None and angular_delta is None:\n"
        "            return\n"
        "        if self._root_body_ids is None:\n"
        "            raise self._capability_error(\n"
        '                "interval root velocity delta",\n'
        '                "root_body_name was not declared in EntityCfg",\n'
        "            )\n"
        "        ids = self._normalize_reset_env_ids(env_ids)\n"
        "        linear_plan = self._validate_root_velocity_delta(\n"
        "            linear_delta,\n"
        "            ids,\n"
        "            term_name=term_name,\n"
        "        )\n"
        "        angular_plan = self._validate_root_velocity_delta(\n"
        "            angular_delta,\n"
        "            ids,\n"
        "            term_name=term_name,\n"
        "        )\n"
        "        try:\n"
        "            self._backend.apply_interval_randomization(\n"
        "                IntervalRandomizationPlan(\n"
        "                    ops=_interval_term_ops(\n"
        "                        (\n"
        "                            INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,\n"
        "                            linear_plan,\n"
        "                            self._root_body_ids,\n"
        "                        ),\n"
        "                        (\n"
        "                            INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,\n"
        "                            angular_plan,\n"
        "                            self._root_body_ids,\n"
        "                        ),\n"
        "                    )\n"
        "                )\n"
        "            )\n"
        "        except NotImplementedError as exc:\n"
        "            raise self._capability_error(\n"
        '                "interval root velocity delta",\n'
        "                f\"EventManager term '{term_name}': {exc}\",\n"
        "            ) from exc\n"
    ),
}


for _name in list(_SOURCES):
    _SOURCES[_name] = textwrap.dedent(_SOURCES[_name])


def apply() -> None:
    exec_ns = vars(_module)
    exec_ns.update(
        INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA=INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,
        INTERVAL_TERM_BODY_FORCE=INTERVAL_TERM_BODY_FORCE,
        INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA=INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
        INTERVAL_TERM_BODY_TORQUE=INTERVAL_TERM_BODY_TORQUE,
        IntervalRandomizationPlan=IntervalRandomizationPlan,
        IntervalTermOp=IntervalTermOp,
        _interval_term_ops=_interval_term_ops,
    )
    bound = {}
    for name, source in _SOURCES.items():
        exec(compile(source, f"<compat.entity.{name}>", "exec"), exec_ns)
        bound[name.split(".")[-1]] = exec_ns[name.split(".")[-1]]
    cls = _module.Entity
    for name, fn in bound.items():
        setattr(cls, name.split(".")[-1], fn)

    orig_init = cls.__init__

    def _init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._fixed_variant_mass = None

    cls.__init__ = _init

    # The branch bound the entity root inside __init__ right after
    # ``_materialize_root_state`` unpacked the default root state (a local
    # there, never stored). Wrapping the materializer reproduces that bind
    # at the same point without reaching into __init__ locals.
    orig_materialize_root_state = cls._materialize_root_state

    def _materialize_root_state(self, backend, root_body_name, default_qpos):
        layout, default_root_state, error = orig_materialize_root_state(
            self, backend, root_body_name, default_qpos
        )
        if self._reset_state is not None and layout is not None:
            if default_root_state is None:
                raise RuntimeError(
                    f"Entity '{self.name}' materialized a root layout without a default root state"
                )
            self._reset_state.bind_root_state(
                self.name,
                layout,
                default_root_state,
            )
        return layout, default_root_state, error

    cls._materialize_root_state = _materialize_root_state
