"""Cold-start phase timing markers for the SimToolReal launch path.

The 12288-env run (2026-09-28) showed a ~2-minute blind segment between
process launch and unisim's compose progress bar.  This module emits
timestamped single-line stderr markers around the phases the task package
controls (CLI entry, catalog build, pool materialization, bootstrap
materialization, env_cfg validate, entity binding, backend creation, first
reset), so the blind segment decomposes by phase in the launch log.

Dependency-free (``time.perf_counter`` + ``print`` to stderr), default ON,
opt out with ``SIMTOOLREAL_QUIET_COLD_START=1``.  Markers are single lines
of the form::

    [cold-start +  12.345s] <phase> — <detail>
"""

from __future__ import annotations

import os
import sys
import time

QUIET_ENV_VAR = "SIMTOOLREAL_QUIET_COLD_START"

# Module import time approximates process launch for phase-relative deltas;
# the CLI imports this module first thing (see cli.py).
_LAUNCH = time.perf_counter()

_TRUTHY = ("1", "true", "yes", "on")


def _enabled() -> bool:
    """Markers are on unless the opt-out env var is set to a truthy value."""
    return os.environ.get(QUIET_ENV_VAR, "").strip().lower() not in _TRUTHY


def mark(phase: str, detail: str = "") -> float:
    """Emit one cold-start marker line; returns seconds since launch."""
    elapsed = time.perf_counter() - _LAUNCH
    if _enabled():
        suffix = f" — {detail}" if detail else ""
        print(f"[cold-start +{elapsed:8.3f}s] {phase}{suffix}", file=sys.stderr, flush=True)
    return elapsed


class phase:
    """Context manager bracketing one cold-start phase with start/end markers.

    The end marker's detail is the phase's own duration; an opening detail is
    carried onto both lines.
    """

    def __init__(self, name: str, detail: str = "") -> None:
        self._name = name
        self._detail = detail
        self._start = 0.0

    def __enter__(self) -> phase:
        mark(f"{self._name} — start", self._detail)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        duration = time.perf_counter() - self._start
        suffix = f"{duration:.3f}s"
        if self._detail:
            suffix = f"{suffix} — {self._detail}"
        mark(f"{self._name} — done", suffix)


__all__ = ["QUIET_ENV_VAR", "mark", "phase"]
