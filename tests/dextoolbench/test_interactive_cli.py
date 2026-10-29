from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from simtoolreal_unilab.dextoolbench import interactive


class _Demo:
    instances: list[_Demo] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _Demo.instances.append(self)

    def run(self) -> None:
        return None


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> _Demo:
    _Demo.instances.clear()
    monkeypatch.setattr(sys, "argv", ["simtoolreal-dextoolbench-interactive", *argv])
    monkeypatch.setattr(interactive, "InteractiveDemo", _Demo)
    interactive.main()
    assert len(_Demo.instances) == 1
    return _Demo.instances[0]


def test_run_selector_resolves_config_and_last_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "13000"
    (run / "last").mkdir(parents=True)
    (run / "run_config.json").write_text("{}", encoding="utf-8")
    checkpoint = run / "last" / "model.pth"
    checkpoint.write_bytes(b"checkpoint")

    demo = _run_main(
        monkeypatch,
        ["--run", str(run), "--checkpoint", "last", "--port", "8081"],
    )

    assert demo.kwargs["config_path"] == str(run / "run_config.json")
    assert demo.kwargs["checkpoint_path"] == str(checkpoint)
    assert demo.kwargs["port"] == 8081


def test_run_selector_rejects_a_missing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "13000"
    run.mkdir()
    (run / "run_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        _run_main(monkeypatch, ["--run", str(run), "--checkpoint", "last"])

    assert exc_info.value.code == 2
    assert "last/model.pth" in capsys.readouterr().err


def test_explicit_policy_paths_remain_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "run_config.json"
    checkpoint = tmp_path / "custom.pth"
    config.write_text("{}", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")

    demo = _run_main(
        monkeypatch,
        ["--config-path", str(config), "--checkpoint-path", str(checkpoint)],
    )

    assert demo.kwargs["config_path"] == str(config)
    assert demo.kwargs["checkpoint_path"] == str(checkpoint)
