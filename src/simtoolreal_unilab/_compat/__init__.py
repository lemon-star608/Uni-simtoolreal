"""Compatibility layer over the pinned official UniLab revision.

The SimToolReal task needs three behaviors that upstream UniLab does not
ship: three manager views (reward vectors, command log extras, row-scoped
observation dispatch) and env-side RNG seed propagation.  The method bodies
are ported verbatim from the development branch and applied onto the official
classes at import time (see the sibling modules).  The MuJoCo line removed
the IsaacSim multi-entity relics (``scene.py`` in M-J1 step 0;
``entity.py``/``reset_state.py`` in M-J2 — D3's single composite MJCF makes
multi-rigid-root routing unnecessary and D5's task-side mass table replaces
the variant-mass facade; upstream main natively covers the object freejoint
root-state write and ``Entity.write_root_state_to_sim``).

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

    from simtoolreal_unilab._compat import (
        config_adapter as _config_adapter,
    )
    from simtoolreal_unilab._compat import (
        envs as _envs,
    )
    from simtoolreal_unilab._compat import (
        managers as _managers,
    )

    _managers.apply()
    _envs.apply()
    _config_adapter.apply()
    _applied = True
