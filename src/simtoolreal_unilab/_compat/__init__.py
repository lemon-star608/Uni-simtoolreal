"""Compatibility layer over the pinned official UniLab revision.

The SimToolReal task needs four behaviors that upstream UniLab does not ship:
multi-rigid-root reset routing, the entity variant-mass facade with
ops-form interval plans, three manager views (reward vectors, command log
extras, row-scoped observation dispatch), and env-side RNG seed propagation.
The method bodies are ported verbatim from the development branch and applied
onto the official classes at import time (see the sibling modules).

The layer fails loudly in both directions: it refuses to apply onto a UniLab
that already provides any of the patched symbols (the pin drifted and the
layer must be revisited), and it is idempotent within one process.
"""

from __future__ import annotations

_applied = False


def apply() -> None:
    global _applied
    if _applied:
        return

    from unilab.base.entity import Entity
    from unilab.base.reset_state import ResetStateTransaction

    drifted = [
        name
        for name, owner in (
            ("bind_root_state", ResetStateTransaction),
            ("_prepare_entity_root_write", ResetStateTransaction),
            ("fixed_variant_mass", Entity),
        )
        if hasattr(owner, name)
    ]
    if drifted:
        raise RuntimeError(
            f"official UniLab already provides {drifted}; the pinned revision "
            "drifted away from what simtoolreal_unilab._compat expects — "
            "revisit the compatibility layer"
        )

    from simtoolreal_unilab._compat import (
        config_adapter as _config_adapter,
    )
    from simtoolreal_unilab._compat import (
        entity as _entity,
    )
    from simtoolreal_unilab._compat import (
        envs as _envs,
    )
    from simtoolreal_unilab._compat import (
        managers as _managers,
    )
    from simtoolreal_unilab._compat import (
        reset_state as _reset_state,
    )

    _reset_state.apply()
    _entity.apply()
    _managers.apply()
    _envs.apply()
    _config_adapter.apply()
    _applied = True
