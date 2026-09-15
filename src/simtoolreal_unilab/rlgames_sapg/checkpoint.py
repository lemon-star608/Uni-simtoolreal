"""Trusted local native ``.pth`` run/checkpoint resolution."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from unilab.cli import RUN_ID_PATTERN

from .dependency import require_rlgames_sapg


@dataclass(frozen=True)
class NativeCheckpointMetadata:
    path: Path
    outer_rank_zero: bool
    state_keys: tuple[str, ...]
    env_state_is_none: bool


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _regular_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise FileNotFoundError(f"{label} is missing or not a regular directory: {path}")
    return path.resolve()


def _create_run_dir(
    task_root: str | Path, *, prefix: str, sim_backend: str, timestamp: str | None = None
) -> Path:
    root = Path(task_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = f"{prefix}_{timestamp}_{sim_backend}"
    for index in range(1000):
        name = base if index == 0 else f"{base}_{index:02d}"
        candidate = root / name
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate.resolve()
    raise FileExistsError(f"unable to allocate a unique native run under {root}")


def create_training_run_dir(
    task_root: str | Path, *, sim_backend: str, timestamp: str | None = None
) -> Path:
    return _create_run_dir(task_root, prefix="0", sim_backend=sim_backend, timestamp=timestamp)


def create_evaluation_run_dir(
    task_root: str | Path, *, sim_backend: str, timestamp: str | None = None
) -> Path:
    return _create_run_dir(task_root, prefix="eval", sim_backend=sim_backend, timestamp=timestamp)


def _number(path: Path, marker: str) -> int:
    match = re.search(rf"_{marker}_(\d+)(?:_|\.)", path.name)
    return int(match.group(1)) if match else -1


def _default_checkpoint(run: Path) -> Path:
    epoch = sorted((run / "nn").glob("last_*_ep_*.pth"), key=lambda p: (_number(p, "ep"), p.name))
    if epoch:
        return epoch[-1]
    frame = sorted(
        (run / "nn").glob("last_*_frame_*.pth"), key=lambda p: (_number(p, "frame"), p.name)
    )
    if frame:
        return frame[-1]
    last = run / "last/model.pth"
    if last.exists():
        return last
    raise FileNotFoundError(f"no native .pth checkpoint found under trusted run: {run}")


def _select_run(
    task_root: Path,
    load_run: str,
    *,
    require_default_checkpoint: bool = False,
) -> Path:
    root = _regular_directory(task_root, "SAPG task log root")
    if load_run == "-1":
        runs = sorted(
            path
            for path in root.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and path.name.startswith("0_")
            and RUN_ID_PATTERN.fullmatch(path.name)
        )
        if not runs:
            raise FileNotFoundError(f"no trusted SAPG training runs found under {root}")
        if require_default_checkpoint:
            for run in reversed(runs):
                try:
                    candidate = _default_checkpoint(run)
                except FileNotFoundError:
                    continue
                if (
                    candidate.suffix == ".pth"
                    and not candidate.is_symlink()
                    and candidate.is_file()
                    and _inside(candidate, run)
                    and _inside(candidate, root)
                ):
                    return run.resolve()
            raise FileNotFoundError(
                f"no trusted SAPG training runs with native .pth checkpoints found under {root}"
            )
        return runs[-1].resolve()
    if load_run in {".", ".."} or RUN_ID_PATTERN.fullmatch(load_run) is None:
        raise ValueError(f"load_run must be -1 or a trusted run name, got {load_run!r}")
    return _regular_directory(root / load_run, "requested SAPG run")


def resolve_native_checkpoint(
    task_root: str | Path, *, load_run: str | int, checkpoint: str | int
) -> tuple[Path, Path]:
    root = Path(task_root).resolve()
    selected = str(checkpoint)
    run = _select_run(
        root,
        str(load_run),
        require_default_checkpoint=(str(load_run) == "-1" and selected in {"", "-1", "None"}),
    )
    if selected in {"", "-1", "None"}:
        candidate = _default_checkpoint(run)
    else:
        parsed = urlparse(selected)
        if parsed.scheme or parsed.netloc:
            raise ValueError("checkpoint URL is not allowed")
        relative = Path(selected)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("checkpoint must be a safe run-relative path")
        if relative.suffix != ".pth":
            raise ValueError("native checkpoint must use the .pth suffix")
        candidate = run / relative
    if candidate.suffix != ".pth":
        raise ValueError("native checkpoint must use the .pth suffix")
    if candidate.is_symlink():
        raise ValueError("checkpoint symlinks are not trusted")
    if not candidate.is_file():
        raise FileNotFoundError(f"native checkpoint is missing: {candidate}")
    resolved = candidate.resolve()
    if not _inside(resolved, run) or not _inside(resolved, root):
        raise ValueError("checkpoint resolves outside the trusted SAPG task root")
    return resolved, run


def resolve_training_checkpoint(
    task_root: str | Path,
    *,
    mode: str,
    load_run: str | int,
    checkpoint: str | int,
) -> Path | None:
    if mode == "none":
        if str(load_run) != "-1" or str(checkpoint) != "-1":
            raise ValueError("fresh training mode cannot select a checkpoint")
        return None
    if mode not in {"resume", "weights"}:
        raise ValueError(f"checkpoint load mode must be none/resume/weights, got {mode!r}")
    return resolve_native_checkpoint(task_root, load_run=load_run, checkpoint=checkpoint)[0]


def validate_native_checkpoint(path: str | Path) -> NativeCheckpointMetadata:
    candidate = Path(path)
    if candidate.suffix != ".pth" or candidate.is_symlink() or not candidate.is_file():
        raise ValueError("native checkpoint must be a trusted regular .pth file")
    require_rlgames_sapg()
    from rl_games.algos_torch import torch_ext

    payload: Any = torch_ext.load_checkpoint(str(candidate))
    if not isinstance(payload, dict) or 0 not in payload or not isinstance(payload[0], dict):
        raise ValueError("native checkpoint must contain a rank-0 state with model")
    state = payload[0]
    if not isinstance(state.get("model"), dict):
        raise ValueError("native checkpoint rank-0 state must contain model weights")
    if "env_state" not in state or state["env_state"] is not None:
        raise ValueError("RL-Games SAPG checkpoints must record env_state=None")
    return NativeCheckpointMetadata(
        path=candidate.resolve(),
        outer_rank_zero=True,
        state_keys=tuple(sorted(str(key) for key in state)),
        env_state_is_none=state.get("env_state", object()) is None,
    )
