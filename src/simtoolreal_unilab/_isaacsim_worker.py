"""Preflight for the external IsaacSim worker runtime.

The IsaacSim SDK cannot ship with this package (its license forbids
redistribution, it requires Python 3.11, and it carries its own torch
build), so the worker runtime is installed by UniLab's official installer
(vendored at ``scripts/setup_isaacsim_env.sh`` from UniLab's official
``scripts/tools/setup_isaacsim_env.sh``) into a
dedicated worker venv at the unisim-conventional root
``~/.cache/unisim/isaacsim/venv``.
The training/play entries call :func:`ensure_worker_runtime` before any
environment is constructed:

- an explicitly exported ``UNISIM_ISAACSIM_PYTHON`` (or the legacy
  ``UNILAB_ISAACSIM_PYTHON`` spelling) always wins and is validated;
- an exported ``UNISIM_ISAACSIM_HOME``/``UNILAB_ISAACSIM_HOME`` is passed
  through untouched for unisim's own home-layout discovery;
- otherwise the conventional install is auto-detected and exported
  in-process, so no shell configuration is needed after the setup script;
- when none of these exist the call fails closed with the one-command fix.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_ENV_PYTHON = "UNISIM_ISAACSIM_PYTHON"
_LEGACY_ENV_PYTHON = "UNILAB_ISAACSIM_PYTHON"
_ENV_HOME = "UNISIM_ISAACSIM_HOME"
_LEGACY_ENV_HOME = "UNILAB_ISAACSIM_HOME"

CONVENTIONAL_WORKER_PYTHON = (
    Path.home() / ".cache" / "unisim" / "isaacsim" / "venv" / "bin" / "python"
)

_SETUP_COMMAND = "bash scripts/setup_isaacsim_env.sh  # vendored from UniLab"


def _require_target_worker(interpreter: str) -> str:
    """Reject a stale IsaacSim worker venv before backend construction."""
    probe = subprocess.run(
        [
            interpreter,
            "-c",
            "import importlib.metadata as m; print(m.version('unisim-core'))",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    version = probe.stdout.strip() if probe.returncode == 0 else ""
    if version != "2.1.0":
        detail = (
            f"worker unisim-core version {version!r}"
            if version
            else "worker unisim-core is not installed"
        )
        raise RuntimeError(
            f"IsaacSim worker is incompatible with UniSim 2.1.0: {detail}. "
            "Re-run the one-time installer after switching this repository to "
            "the TorchEnv dependency line:\n"
            f"    {_SETUP_COMMAND}"
        )
    return interpreter


def ensure_worker_runtime() -> str:
    """Resolve the IsaacSim worker interpreter, failing closed with a fix.

    Returns the interpreter path that will serve as the IsaacSim worker,
    exporting ``UNISIM_ISAACSIM_PYTHON`` in-process when the conventional
    install is used.
    """
    explicit = os.environ.get(_ENV_PYTHON) or os.environ.get(_LEGACY_ENV_PYTHON)
    if explicit:
        interpreter = Path(explicit).expanduser()
        if not interpreter.is_file():
            raise FileNotFoundError(
                f"{_ENV_PYTHON} points to a missing interpreter: {interpreter}. "
                "Unset it or point it at an IsaacSim-capable Python 3.11 venv."
            )
        return _require_target_worker(str(interpreter))
    if os.environ.get(_ENV_HOME) or os.environ.get(_LEGACY_ENV_HOME):
        # Home-layout discovery belongs to unisim; nothing to prepare here.
        return os.environ.get(_ENV_HOME) or os.environ.get(_LEGACY_ENV_HOME)  # type: ignore[return-value]
    conventional = Path(CONVENTIONAL_WORKER_PYTHON)
    if conventional.is_file():
        os.environ[_ENV_PYTHON] = str(conventional)
        return _require_target_worker(str(conventional))
    raise RuntimeError(
        "IsaacSim worker runtime not found. Run the one-time setup with "
        "UniLab's official installer:\n"
        f"    {_SETUP_COMMAND}\n"
        "It installs the official IsaacSim 5.1 + IsaacLab 2.3.0 stack into a "
        "dedicated Python 3.11 venv at ~/.cache/unisim/isaacsim (~15 GB "
        "download), which simtoolreal-train/simtoolreal-play detect "
        "automatically. "
        f"Alternatively export {_ENV_PYTHON} to an existing IsaacSim-capable "
        "interpreter."
    )


__all__ = ["CONVENTIONAL_WORKER_PYTHON", "ensure_worker_runtime"]
