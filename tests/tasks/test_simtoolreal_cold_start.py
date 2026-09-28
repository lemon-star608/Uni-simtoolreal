"""Cold-start phase marker format and gating tests (no timing assertions)."""

from __future__ import annotations

import re

from simtoolreal_unilab.tasks.simtoolreal import cold_start
from simtoolreal_unilab.tasks.simtoolreal.cold_start import QUIET_ENV_VAR, mark, phase

_MARKER = re.compile(r"^\[cold-start \+\s*\d+\.\d{3}s\] .+$")


def test_mark_format_with_and_without_detail(capsys):
    mark("pool materialization", "layout=same_layout (cache hit)")
    mark("env_cfg validate")
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 2
    assert all(_MARKER.match(line) for line in lines)
    assert "pool materialization — layout=same_layout (cache hit)" in lines[0]
    assert lines[1].endswith("] env_cfg validate")


def test_phase_emits_start_and_done_with_duration(capsys):
    with phase("backend creation", "backend=mujoco num_envs=8"):
        pass
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 2
    assert all(_MARKER.match(line) for line in lines)
    assert "backend creation — start — backend=mujoco num_envs=8" in lines[0]
    done = lines[1]
    assert "backend creation — done" in done
    assert re.search(r"\d+\.\d{3}s", done)
    assert "backend=mujoco num_envs=8" in done


def test_quiet_env_var_gates_markers(monkeypatch, capsys):
    monkeypatch.setenv(QUIET_ENV_VAR, "1")
    mark("hidden")
    with phase("hidden phase"):
        pass
    assert capsys.readouterr().err == ""

    monkeypatch.setenv(QUIET_ENV_VAR, "0")  # not a truthy opt-out value
    mark("shown")
    assert "shown" in capsys.readouterr().err

    monkeypatch.delenv(QUIET_ENV_VAR)
    mark("default on")
    assert "default on" in capsys.readouterr().err


def test_mark_returns_elapsed_even_when_quiet(monkeypatch):
    monkeypatch.setenv(QUIET_ENV_VAR, "1")
    elapsed = mark("hidden")
    assert elapsed >= 0.0
