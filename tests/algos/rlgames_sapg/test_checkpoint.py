from __future__ import annotations

from pathlib import Path

import pytest
import torch


def _checkpoint(path: Path, *, valid: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {0: {"model": {"weight": torch.ones(1)}, "env_state": None}} if valid else {"bad": 1}, path
    )
    return path


def test_create_run_directory_uses_native_prefix_and_deterministic_collision_suffix(tmp_path):
    from simtoolreal_unilab.rlgames_sapg.checkpoint import (
        create_evaluation_run_dir,
        create_training_run_dir,
    )

    first = create_training_run_dir(
        tmp_path, sim_backend="isaacsim", timestamp="2026-08-21_12-00-00"
    )
    second = create_training_run_dir(
        tmp_path, sim_backend="isaacsim", timestamp="2026-08-21_12-00-00"
    )
    evaluation = create_evaluation_run_dir(
        tmp_path, sim_backend="isaacsim", timestamp="2026-08-21_12-00-00"
    )
    assert first.name == "0_2026-08-21_12-00-00_isaacsim"
    assert second.name == "0_2026-08-21_12-00-00_isaacsim_01"
    assert evaluation.name == "eval_2026-08-21_12-00-00_isaacsim"


def test_latest_run_and_checkpoint_selection_are_deterministic(tmp_path):
    from simtoolreal_unilab.rlgames_sapg.checkpoint import resolve_native_checkpoint

    old = tmp_path / "0_2026-08-20_12-00-00_mujoco"
    new = tmp_path / "0_2026-08-21_12-00-00_mujoco"
    _checkpoint(old / "nn/last_0_simtoolreal_sapg_ep_9_rew_0.pth")
    _checkpoint(new / "last/model.pth")
    selected = _checkpoint(new / "nn/last_0_simtoolreal_sapg_ep_12_rew_0.pth")
    _checkpoint(new / "nn/last_0_simtoolreal_sapg_frame_99_rew_0.pth")
    resolved, run = resolve_native_checkpoint(tmp_path, load_run="-1", checkpoint="-1")
    assert (resolved, run) == (selected.resolve(), new.resolve())


def test_latest_run_ignores_newer_evaluation_siblings(tmp_path):
    from simtoolreal_unilab.rlgames_sapg.checkpoint import resolve_native_checkpoint

    training = tmp_path / "0_2026-08-21_12-00-00_mujoco"
    selected = _checkpoint(training / "nn/last_owner_ep_1_rew_-inf.pth")
    evaluation = tmp_path / "eval_9999-12-31_23-59-59_mujoco"
    evaluation.mkdir()

    resolved, run = resolve_native_checkpoint(tmp_path, load_run="-1", checkpoint="-1")
    assert (resolved, run) == (selected.resolve(), training.resolve())


def test_latest_run_skips_newer_incomplete_training_run(tmp_path):
    from simtoolreal_unilab.rlgames_sapg.checkpoint import resolve_native_checkpoint

    completed = tmp_path / "0_2026-08-21_12-00-00_mujoco"
    selected = _checkpoint(completed / "last/model.pth")
    incomplete = tmp_path / "0_2026-08-22_12-00-00_mujoco"
    incomplete.mkdir()

    resolved, run = resolve_native_checkpoint(tmp_path, load_run="-1", checkpoint="-1")
    assert (resolved, run) == (selected.resolve(), completed.resolve())


def test_explicit_incomplete_run_still_fails_closed(tmp_path):
    from simtoolreal_unilab.rlgames_sapg.checkpoint import resolve_native_checkpoint

    completed = tmp_path / "0_2026-08-21_12-00-00_mujoco"
    _checkpoint(completed / "last/model.pth")
    incomplete = tmp_path / "0_2026-08-22_12-00-00_mujoco"
    incomplete.mkdir()

    with pytest.raises(FileNotFoundError, match="no native .* checkpoint"):
        resolve_native_checkpoint(tmp_path, load_run=incomplete.name, checkpoint="-1")


def test_explicit_run_relative_checkpoint_is_accepted(tmp_path):
    from simtoolreal_unilab.rlgames_sapg.checkpoint import resolve_native_checkpoint

    run = tmp_path / "0_named_mujoco"
    expected = _checkpoint(run / "best/model.pth")
    assert resolve_native_checkpoint(tmp_path, load_run=run.name, checkpoint="best/model.pth") == (
        expected.resolve(),
        run.resolve(),
    )


@pytest.mark.parametrize(
    ("load_run", "checkpoint", "match"),
    [
        ("../escape", "-1", "run name"),
        ("0_run", "../x.pth", "relative"),
        ("0_run", "/tmp/x.pth", "relative"),
        ("0_run", "https://x/y.pth", "URL"),
        ("0_run", "model.pt", r"\.pth"),
    ],
)
def test_resolver_rejects_traversal_absolute_url_and_wrong_suffix(
    tmp_path, load_run, checkpoint, match
):
    from simtoolreal_unilab.rlgames_sapg.checkpoint import resolve_native_checkpoint

    (tmp_path / "0_run").mkdir()
    with pytest.raises((FileNotFoundError, ValueError), match=match):
        resolve_native_checkpoint(tmp_path, load_run=load_run, checkpoint=checkpoint)


def test_resolver_rejects_symlink_escape(tmp_path):
    from simtoolreal_unilab.rlgames_sapg.checkpoint import resolve_native_checkpoint

    outside = _checkpoint(tmp_path.parent / "outside.pth")
    run = tmp_path / "0_run"
    run.mkdir()
    (run / "model.pth").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        resolve_native_checkpoint(tmp_path, load_run="0_run", checkpoint="model.pth")


def test_native_payload_validation_requires_rank_zero_model_and_env_state_none(tmp_path):
    from simtoolreal_unilab.rlgames_sapg.checkpoint import validate_native_checkpoint

    path = _checkpoint(tmp_path / "model.pth")
    metadata = validate_native_checkpoint(path)
    assert metadata.outer_rank_zero and metadata.env_state_is_none
    with pytest.raises(ValueError, match="rank-0.*model"):
        validate_native_checkpoint(_checkpoint(tmp_path / "bad.pth", valid=False))
    unsupported = tmp_path / "env-state.pth"
    torch.save({0: {"model": {}, "env_state": {"snapshot": 1}}}, unsupported)
    with pytest.raises(ValueError, match="env_state=None"):
        validate_native_checkpoint(unsupported)


@pytest.mark.parametrize(
    ("mode", "load_run", "checkpoint"),
    [("none", "-1", "-1"), ("resume", "0_run", "model.pth"), ("weights", "0_run", "model.pth")],
)
def test_training_checkpoint_modes(tmp_path, mode, load_run, checkpoint):
    from simtoolreal_unilab.rlgames_sapg.checkpoint import resolve_training_checkpoint

    if mode != "none":
        _checkpoint(tmp_path / "0_run/model.pth")
    result = resolve_training_checkpoint(
        tmp_path, mode=mode, load_run=load_run, checkpoint=checkpoint
    )
    assert (result is None) == (mode == "none")


@pytest.mark.parametrize(
    ("mode", "load_run", "checkpoint"),
    [("none", "0_run", "-1"), ("resume", "-1", "-1"), ("bad", "-1", "-1")],
)
def test_training_checkpoint_mode_conflicts_fail_closed(tmp_path, mode, load_run, checkpoint):
    from simtoolreal_unilab.rlgames_sapg.checkpoint import resolve_training_checkpoint

    with pytest.raises((FileNotFoundError, ValueError)):
        resolve_training_checkpoint(tmp_path, mode=mode, load_run=load_run, checkpoint=checkpoint)
