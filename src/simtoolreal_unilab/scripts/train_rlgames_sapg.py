"""Train or play with the pinned native RL-Games SAPG runtime."""

from __future__ import annotations

import sys

# The vendored RL-Games tree is hash-pinned; never write bytecode caches into it.
sys.dont_write_bytecode = True

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import hydra
import torch
from omegaconf import DictConfig
from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.training.common import ensure_registries
from unilab.training.experiment import ExperimentTracker
from unilab.training.run import get_log_root, should_run_playback
from unilab.utils.seed import apply_configured_training_seed
from unilab.utils.sim2sim import resolve_sim2sim_config
from unilab.visualization.playback import camera_cfg_from_training
from unisim.backend.base import log_playback_plan

from simtoolreal_unilab.rlgames_sapg.checkpoint import (
    create_evaluation_run_dir,
    create_training_run_dir,
    resolve_native_checkpoint,
    resolve_training_checkpoint,
    validate_native_checkpoint,
)
from simtoolreal_unilab.rlgames_sapg.config import preflight_config
from simtoolreal_unilab.rlgames_sapg.dependency import require_rlgames_sapg
from simtoolreal_unilab.rlgames_sapg.env_adapter import RlGamesNpEnvAdapter
from simtoolreal_unilab.rlgames_sapg.observer import ExperimentTrackerObserver
from simtoolreal_unilab.rlgames_sapg.player import build_native_player_bridge
from simtoolreal_unilab.rlgames_sapg.runtime import execute_native_train


@dataclass(frozen=True)
class TrainingRun:
    run_dir: Path
    checkpoint: Path | None
    native_result: Any
    video: str | None = None


@dataclass(frozen=True)
class PlaybackRun:
    run_dir: Path
    source_run: Path
    checkpoint: Path
    video: str | None


def _tracker(
    cfg: DictConfig,
    *,
    root_dir: str | Path,
    log_dir: Path,
    seed_info: Any,
    tracker_factory: Callable[..., Any],
) -> Any:
    return tracker_factory(
        root_dir=root_dir,
        log_dir=log_dir,
        algo_name="rlgames_sapg",
        task_name=str(cfg.training.task_name),
        sim_backend=str(cfg.training.sim_backend),
        training_cfg=cfg.training,
        full_cfg=cfg,
        device=str(cfg.training.device),
        seed_info=seed_info,
    )


def _cleanup_native_scratch(task_root: Path) -> None:
    batch_dir = task_root / "batches"
    if batch_dir.is_symlink():
        raise RuntimeError(f"native batch scratch path must not be a symlink: {batch_dir}")
    try:
        batch_dir.rmdir()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeError(f"native batch scratch directory must be empty: {batch_dir}") from exc


def run_playback(
    cfg: DictConfig,
    *,
    root_dir: str | Path | None = None,
    source_checkpoint: str | Path | None = None,
    source_run: str | Path | None = None,
    tracker: Any | None = None,
    env_factory: Callable[..., Any] = create_env,
    tracker_factory: Callable[..., Any] = ExperimentTracker,
    adapter_factory: Callable[..., Any] = RlGamesNpEnvAdapter,
    player_builder: Callable[..., Any] = build_native_player_bridge,
    checkpoint_validator: Callable[[str | Path], Any] = validate_native_checkpoint,
    sim2sim_resolver: Callable[..., Any] = resolve_sim2sim_config,
    verify_dependency: bool = True,
    ensure_registry: Callable[[], None] = ensure_registries,
) -> PlaybackRun:
    """Restore the native player and run it through the UniLab playback shell."""
    root_dir = Path.cwd() if root_dir is None else root_dir
    # Task imports register OmegaConf resolvers (notably SimToolReal's
    # ``simtoolreal_entity_mjcf``) as a cold-path side effect.  Playback
    # validates the saved sim2sim contract before constructing the
    # environment, so the registry must be loaded before any config subtree
    # is resolved.
    ensure_registry()
    preflight_config(cfg)
    if verify_dependency:
        require_rlgames_sapg()
    task_root = get_log_root(root_dir, cfg) / str(cfg.training.task_name)
    if (source_checkpoint is None) != (source_run is None):
        raise ValueError("source_checkpoint and source_run must be provided together")
    if source_checkpoint is None:
        checkpoint_path, source_run_path = resolve_native_checkpoint(
            task_root,
            load_run=str(cfg.algo.load_run),
            checkpoint=str(cfg.algo.checkpoint),
        )
    else:
        assert source_checkpoint is not None and source_run is not None
        checkpoint_path = Path(source_checkpoint).resolve()
        source_run_path = Path(source_run).resolve()
        if source_run_path.parent != task_root.resolve() or not checkpoint_path.is_relative_to(
            source_run_path
        ):
            raise ValueError("playback source must stay inside one trusted SAPG training run")
    checkpoint_validator(checkpoint_path)
    sim2sim_resolver(
        source_run_path,
        cfg,
        algo_name="rlgames_sapg",
        strict=bool(cfg.training.sim2sim_strict),
    )

    owns_tracker = tracker is None
    active_tracker: Any = tracker
    if active_tracker is None:
        run_dir = create_evaluation_run_dir(task_root, sim_backend=str(cfg.training.sim_backend))
        seed_info = apply_configured_training_seed(cfg)
        active_tracker = _tracker(
            cfg,
            root_dir=root_dir,
            log_dir=run_dir,
            seed_info=seed_info,
            tracker_factory=tracker_factory,
        )
        active_tracker.start()
    else:
        run_dir = Path(active_tracker.log_dir).resolve()

    env = None
    try:
        override = BackendAdapter(
            cfg, root_dir=root_dir, algo_name="rlgames_sapg"
        ).build_play_env_cfg_override()
        env = env_factory(
            cfg=cfg,
            num_envs=int(cfg.training.play_env_num),
            env_cfg_override=override,
        )
        adapter = adapter_factory(env, device=str(cfg.training.device))
        bridge = player_builder(
            cfg=cfg,
            adapter=adapter,
            checkpoint=checkpoint_path,
            verify_dependency=verify_dependency,
            validate_checkpoint=False,
        )
        output_video = run_dir / "play_video.mp4"
        with torch.inference_mode():
            video = env.run_playback_mode(
                play_render_mode=str(cfg.training.play_render_mode),
                play_steps=int(cfg.training.play_steps),
                output_video=output_video,
                render_spacing=float(
                    getattr(cfg.training, "render_spacing", getattr(env.cfg, "render_spacing", 1.0))
                ),
                initialize=bridge.initialize,
                step=bridge.step,
                camera_kwargs=camera_cfg_from_training(cfg.training),
                on_plan=log_playback_plan,
            )
        active_tracker.log_video(video)
        active_tracker.update_summary(
            {
                "source_run": str(source_run_path),
                "source_checkpoint": str(checkpoint_path),
                "checkpoint": str(checkpoint_path),
                "play_steps": int(cfg.training.play_steps),
                "play_video_path": video,
            }
        )
        return PlaybackRun(
            run_dir=run_dir,
            source_run=source_run_path,
            checkpoint=checkpoint_path,
            video=video,
        )
    finally:
        if env is not None:
            env.close()
        if owns_tracker:
            active_tracker.finish()


def run_training(
    cfg: DictConfig,
    *,
    root_dir: str | Path | None = None,
    env_factory: Callable[..., Any] = create_env,
    tracker_factory: Callable[..., Any] = ExperimentTracker,
    observer_factory: Callable[[Any], Any] = ExperimentTrackerObserver,
    adapter_factory: Callable[..., Any] = RlGamesNpEnvAdapter,
    executor: Callable[..., Any] = execute_native_train,
    checkpoint_validator: Callable[[str | Path], Any] = validate_native_checkpoint,
    playback_runner: Callable[..., PlaybackRun] = run_playback,
    verify_dependency: bool = True,
    ensure_registry: Callable[[], None] = ensure_registries,
) -> TrainingRun:
    """Compose lifecycle owners around the one native Runner training call."""
    root_dir = Path.cwd() if root_dir is None else root_dir
    preflight_config(cfg)
    if verify_dependency:
        require_rlgames_sapg()
    task_root = get_log_root(root_dir, cfg) / str(cfg.training.task_name)
    load_path = resolve_training_checkpoint(
        task_root,
        mode=str(cfg.algo.checkpoint_load_mode),
        load_run=str(cfg.algo.load_run),
        checkpoint=str(cfg.algo.checkpoint),
    )
    run_dir = create_training_run_dir(task_root, sim_backend=str(cfg.training.sim_backend))
    seed_info = apply_configured_training_seed(cfg)
    tracker = _tracker(
        cfg,
        root_dir=root_dir,
        log_dir=run_dir,
        seed_info=seed_info,
        tracker_factory=tracker_factory,
    )
    env = None
    observer = None
    # Task packages may register OmegaConf resolvers (e.g. SimToolReal's asset
    # resolver); the run-config snapshot resolves the full config tree, so
    # registries must be imported before tracker.start().
    ensure_registry()
    tracker.start()
    try:
        override = BackendAdapter(
            cfg, root_dir=root_dir, algo_name="rlgames_sapg"
        ).build_task_env_cfg_override()
        env = env_factory(cfg=cfg, num_envs=int(cfg.algo.num_envs), env_cfg_override=override)
        adapter = adapter_factory(env, device=str(cfg.training.device))
        observer = observer_factory(tracker)
        native = executor(
            cfg,
            adapter=adapter,
            observer=observer,
            train_dir=task_root,
            run_name=run_dir.name,
            checkpoint=str(load_path) if load_path is not None else None,
            checkpoint_load_mode=str(cfg.algo.checkpoint_load_mode),
            verify_dependency=verify_dependency,
        )
        _cleanup_native_scratch(task_root)
        observer.close_writer()
        observer = None
        env.close()
        env = None
        checkpoint_path, _ = resolve_native_checkpoint(
            task_root,
            load_run=run_dir.name,
            checkpoint="-1",
        )
        checkpoint_validator(checkpoint_path)
        video = None
        tracker.update_summary(
            {
                "native_result": native.result,
                "checkpoint_load_mode": str(cfg.algo.checkpoint_load_mode),
                "source_checkpoint": str(load_path) if load_path is not None else None,
                "checkpoint": str(checkpoint_path),
            }
        )
        if should_run_playback(
            play_only=False,
            no_play=bool(cfg.training.no_play),
            play_render_mode=str(cfg.training.play_render_mode),
        ):
            playback = playback_runner(
                cfg,
                root_dir=root_dir,
                source_checkpoint=checkpoint_path,
                source_run=run_dir,
                tracker=tracker,
                verify_dependency=verify_dependency,
                ensure_registry=ensure_registry,
            )
            video = playback.video
        return TrainingRun(
            run_dir=run_dir,
            checkpoint=checkpoint_path,
            native_result=native.result,
            video=video,
        )
    finally:
        try:
            if observer is not None:
                observer.close_writer()
            if env is not None:
                env.close()
            _cleanup_native_scratch(task_root)
        finally:
            tracker.finish()


@hydra.main(version_base="1.3", config_path="../conf/rlgames_sapg", config_name="config")
def main(cfg: DictConfig) -> None:
    if bool(cfg.training.play_only):
        run_playback(cfg)
    else:
        run_training(cfg)


if __name__ == "__main__":
    main()
