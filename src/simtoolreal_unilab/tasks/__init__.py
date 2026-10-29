"""Task package registered under the ``unilab.tasks`` entry-point group."""

from simtoolreal_unilab.tasks import (
    dextoolbench,  # noqa: F401  (registers on import)
    simtoolreal,  # noqa: F401  (registers on import)
)

__all__ = ["dextoolbench", "simtoolreal"]
