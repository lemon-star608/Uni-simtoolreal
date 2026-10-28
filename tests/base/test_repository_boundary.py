"""Repository boundary for the MuJoCo-only training product."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from unilab.base import registry

from simtoolreal_unilab import cli
from simtoolreal_unilab import config as package_config

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = (
    REPO_ROOT / "src/simtoolreal_unilab",
    REPO_ROOT / "tests/base",
    REPO_ROOT / "tests/tasks",
    REPO_ROOT / "scripts",
)
SCAN_FILES = (REPO_ROOT / "pyproject.toml",)
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".md"}
BANNED = ("isaacsim", "isaacgym", "isaac sim", "npenv", "np_env", "_compat")


def test_first_party_runtime_and_tests_have_no_removed_runtime_terms():
    paths = [path for root in SCAN_ROOTS for path in sorted(root.rglob("*"))]
    paths.extend(path for path in SCAN_FILES if path.is_file())
    violations = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for term in BANNED:
            if term in text:
                violations.append(f"{path.relative_to(REPO_ROOT)}: {term}")
    assert violations == []


def test_routing_is_fixed_to_mujoco_constants():
    assert package_config.DEFAULT_TASK == "simtoolreal/mujoco"
    assert package_config.SIM_BACKEND == "mujoco"
    assert not hasattr(package_config, "TASK_BACKENDS")
    assert not hasattr(package_config, "TASK_OWNERS")


def test_registry_exposes_only_mujoco_simtoolreal_backend():
    registry.ensure_registries()
    meta = registry._envs["SimToolReal"]
    assert set(meta.env_factory_dict) == {"mujoco"}


def test_cli_generates_fixed_mujoco_task(monkeypatch):
    captured = {}
    monkeypatch.setattr(sys, "argv", ["simtoolreal-train"])

    def run_module(module_name, run_name):
        captured["module_name"] = module_name
        captured["run_name"] = run_name
        captured["argv"] = list(sys.argv)

    monkeypatch.setattr(cli.runpy, "run_module", run_module)
    cli._run("train")
    assert captured["module_name"] == "simtoolreal_unilab.scripts.train_rlgames_sapg"
    assert "task=simtoolreal/mujoco" in captured["argv"]
    assert "--config-dir" in captured["argv"]


@pytest.mark.parametrize(("option", "value"), [("--task", "SimToolReal"), ("--backend", "mujoco")])
def test_cli_rejects_removed_routing_options(monkeypatch, option, value):
    monkeypatch.setattr(sys, "argv", ["simtoolreal-train", option, value])
    monkeypatch.setattr(
        cli.runpy,
        "run_module",
        lambda *args, **kwargs: pytest.fail("routing option should fail before dispatch"),
    )
    with pytest.raises(SystemExit):
        cli._run("train")


def test_cli_has_no_list_envs_entry_point():
    assert not hasattr(cli, "list_main")


def test_checkpoint_option_is_play_only(monkeypatch):
    captured = {}
    monkeypatch.setattr(sys, "argv", ["simtoolreal-play", "--checkpoint-file", "/tmp/run.pth"])

    def run_module(*args, **kwargs):
        captured["argv"] = list(sys.argv)

    monkeypatch.setattr(cli.runpy, "run_module", run_module)
    cli._run("play")
    assert "algo.load_run=/tmp/run.pth" in captured["argv"]
    assert "training.play_only=true" in captured["argv"]

    monkeypatch.setattr(sys, "argv", ["simtoolreal-train", "--checkpoint-file", "/tmp/run.pth"])
    with pytest.raises(SystemExit):
        cli._run("train")
