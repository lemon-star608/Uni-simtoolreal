"""SimToolReal teacher training on UniLab + UniSim (IsaacSim URDF->USD path).

Importing this package applies the compatibility layer that extends the
pinned official UniLab revision with the multi-rigid-root reset routing,
entity variant-metadata facade, and manager logging views this task needs.
The layer is additive and fails loudly if the installed UniLab drifts away
from the expected structure.
"""

from simtoolreal_unilab._compat import apply as _apply_compat

_apply_compat()

__version__ = "0.1.0"
