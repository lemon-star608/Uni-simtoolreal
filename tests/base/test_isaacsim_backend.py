"""IsaacSim worker preflight remains fail-closed under UniSim 2.x."""

from __future__ import annotations

from pathlib import Path

import pytest

from simtoolreal_unilab._isaacsim_worker import ensure_worker_runtime


def test_explicit_worker_python_wins(monkeypatch, tmp_path):
    interpreter = tmp_path / "python"
    interpreter.write_text("#!/bin/sh\necho 2.1.0\n")
    interpreter.chmod(0o755)
    monkeypatch.setenv("UNISIM_ISAACSIM_PYTHON", str(interpreter))
    assert ensure_worker_runtime() == str(interpreter)


def test_stale_worker_version_fails_with_reinstall_hint(monkeypatch, tmp_path):
    interpreter = tmp_path / "python"
    interpreter.write_text("#!/bin/sh\necho 2.0.1\n")
    interpreter.chmod(0o755)
    monkeypatch.setenv("UNISIM_ISAACSIM_PYTHON", str(interpreter))
    with pytest.raises(RuntimeError, match="Re-run the one-time installer"):
        ensure_worker_runtime()


def test_missing_worker_report_gives_one_command_fix(monkeypatch):
    for name in (
        "UNISIM_ISAACSIM_PYTHON",
        "UNILAB_ISAACSIM_PYTHON",
        "UNISIM_ISAACSIM_HOME",
        "UNILAB_ISAACSIM_HOME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "simtoolreal_unilab._isaacsim_worker.CONVENTIONAL_WORKER_PYTHON",
        Path("/nonexistent-simtoolreal-home/isaacsim/venv/bin/python"),
    )
    with pytest.raises(RuntimeError, match="setup_isaacsim_env.sh"):
        ensure_worker_runtime()
