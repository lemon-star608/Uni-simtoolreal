"""Native algorithm observer that bridges UniLab and legacy RL-Games metrics."""

from __future__ import annotations

import numbers
from collections import deque
from typing import Any

import numpy as np
import torch


def _flatten_dict(d: dict[str, Any], prefix: str = "", separator: str = "/") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten_dict(value, name + separator, separator))
        else:
            out[name] = value
    return out


def _value_at(value: Any, index: int) -> float:
    if isinstance(value, torch.Tensor):
        if value.ndim > 0:
            return float(value[index].detach().cpu().item())
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return float(value[index].item() if value.ndim > 0 else value.item())
    if isinstance(value, (list, tuple)):
        return float(value[index])
    return float(value)


def _mean_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.float().mean().detach().cpu().item())
    return float(np.asarray(value).mean())


def _zeros_like(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return torch.zeros_like(value)
    return np.zeros_like(np.asarray(value))


def _has_env_axis(value: Any, boundary: int) -> bool:
    if isinstance(value, torch.Tensor):
        return value.ndim > 0 and value.shape[0] > boundary
    if isinstance(value, np.ndarray):
        return value.ndim > 0 and value.shape[0] > boundary
    if isinstance(value, (list, tuple)):
        return len(value) > boundary
    return False


def _slice_env_axis(value: Any, start: int) -> Any:
    return value[start:]


def _remove_env_boundary_from_info(info: dict[str, Any], boundary: int) -> dict[str, Any]:
    """Keep only the SAPG leader block, matching the original observer."""
    trimmed: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, dict):
            trimmed[key] = _remove_env_boundary_from_info(value, boundary)
            continue
        if not _has_env_axis(value, boundary):
            trimmed[key] = value
            continue
        if key == "successes":
            block_size = len(value) - boundary
            if block_size > 0 and len(value) % block_size == 0:
                for block in range(len(value) // block_size):
                    trimmed[f"{key}_per_block/block_{block}"] = value[
                        block * block_size : (block + 1) * block_size
                    ]
        trimmed[key] = _slice_env_axis(value, boundary)
    return trimmed


class ExperimentTrackerObserver:
    """Bridge UniLab logs while retaining the original TensorBoard contract.

    ``env/*`` scalar tags are kept for the UniLab tracker.  When a task emits
    ``episode_cumulative``/``episode_final`` vectors, this observer applies the
    same done-indexed accumulation as the original repository and writes the
    legacy ``episode_*`` plus success/tolerance tags.  The logging path is
    read-only and has no effect on reward, reset, or termination semantics.
    """

    def __init__(self, tracker: Any) -> None:
        self.tracker = tracker
        self.algo: Any | None = None
        self.writer: Any | None = None
        self._metrics: dict[str, float] = {}
        self._legacy_metrics: dict[str, float] = {}
        self._episode_cumulative: dict[str, Any] = {}
        self._episode_cumulative_avg: dict[str, deque[float]] = {}
        self._episode_final_avg: dict[str, deque[float]] = {}
        self._new_finished_episodes = False
        self._games_to_track = 100
        self._writer_closed = False

    def before_init(self, base_name: str, config: dict[str, Any], experiment_name: str) -> None:
        del base_name, config, experiment_name

    def after_init(self, algo: Any) -> None:
        self.algo = algo
        self.writer = algo.writer
        self._games_to_track = int(getattr(algo, "games_to_track", 100))

    @staticmethod
    def _scalar(value: Any) -> float | None:
        if isinstance(value, numbers.Real):
            return float(value)
        if isinstance(value, np.generic) and np.issubdtype(value.dtype, np.number):
            return float(value)
        if isinstance(value, torch.Tensor) and value.ndim == 0:
            return float(value.detach().cpu().item())
        return None

    def process_infos(self, infos: Any, done_indices: Any, **kwargs: Any) -> None:
        if not isinstance(infos, dict) or not isinstance(infos.get("log"), dict):
            return
        log = infos["log"]
        if isinstance(done_indices, torch.Tensor):
            raw_done = done_indices.reshape(-1).detach().cpu().tolist()
        else:
            raw_done = np.asarray(done_indices).reshape(-1).tolist()
        boundary = int(kwargs.get("ignore_env_boundary", 0))
        if boundary > 0:
            log = _remove_env_boundary_from_info(log, boundary)
            done = [index - boundary for index in raw_done if index >= boundary]
        else:
            done = raw_done

        self._process_episode_cumulative(log.get("episode_cumulative"), done)
        self._process_episode_final(log.get("episode_final"), done)

        metrics: dict[str, float] = {}
        for name, value in _flatten_dict(log).items():
            scalar = self._scalar(value)
            if scalar is not None and np.isfinite(scalar):
                metrics[f"env/{name}"] = scalar
        self._metrics = metrics

        self._legacy_metrics = {}
        tolerance = self._scalar(log.get("current_success_tolerance"))
        if tolerance is not None and np.isfinite(tolerance):
            self._legacy_metrics["current_success_tolerance"] = tolerance
        self._process_vector_summaries(log, tag="successes")

    def _process_episode_cumulative(self, terms: Any, done_indices: list[int]) -> None:
        if not isinstance(terms, dict) or not terms:
            return
        for key, value in terms.items():
            if key not in self._episode_cumulative:
                self._episode_cumulative[key] = _zeros_like(value)
            if key not in self._episode_cumulative_avg:
                self._episode_cumulative_avg[key] = deque(maxlen=self._games_to_track)
            self._episode_cumulative[key] += value

        for done_idx in done_indices:
            self._new_finished_episodes = True
            for key in terms:
                self._episode_cumulative_avg[key].append(
                    _value_at(self._episode_cumulative[key], done_idx)
                )
                self._episode_cumulative[key][done_idx] = 0

    def _process_episode_final(self, values: Any, done_indices: list[int]) -> None:
        if not isinstance(values, dict) or not done_indices:
            return
        self._new_finished_episodes = True
        for key, value in values.items():
            if key not in self._episode_final_avg:
                self._episode_final_avg[key] = deque(maxlen=self._games_to_track)
            for done_idx in done_indices:
                self._episode_final_avg[key].append(_value_at(value, done_idx))

    def _process_vector_summaries(self, log: dict[str, Any], *, tag: str) -> None:
        if tag not in log:
            return
        value = log[tag]
        self._legacy_metrics[tag] = _mean_float(value)
        if isinstance(value, torch.Tensor):
            self._legacy_metrics[f"{tag}_median"] = float(
                torch.median(value.float()).detach().cpu().item()
            )
            self._legacy_metrics[f"{tag}_max"] = float(value.max().detach().cpu().item())
        else:
            array = np.asarray(value)
            self._legacy_metrics[f"{tag}_median"] = float(np.median(array))
            self._legacy_metrics[f"{tag}_max"] = float(np.max(array))
        prefix = f"{tag}_per_block/"
        for key, block_value in _flatten_dict(log).items():
            if key.startswith(prefix):
                self._legacy_metrics[key] = _mean_float(block_value)

    def after_steps(self) -> None:
        return None

    def after_clear_stats(self) -> None:
        self._metrics.clear()
        self._legacy_metrics.clear()
        self._episode_cumulative_avg.clear()
        self._episode_final_avg.clear()
        self._new_finished_episodes = False

    def after_print_stats(self, frame: int, epoch_num: int, total_time: float) -> None:
        if self.writer is not None:
            if self._new_finished_episodes:
                for key, values in self._episode_cumulative_avg.items():
                    if values:
                        self.writer.add_scalar(
                            f"episode_cumulative/{key}", np.mean(values), frame
                        )
                        self.writer.add_scalar(
                            f"episode_cumulative_min/{key}_min", np.min(values), frame
                        )
                        self.writer.add_scalar(
                            f"episode_cumulative_max/{key}_max", np.max(values), frame
                        )
                for key, values in self._episode_final_avg.items():
                    if values:
                        self.writer.add_scalar(f"episode_final/{key}", np.mean(values), frame)
                self._new_finished_episodes = False
            for name, value in self._metrics.items():
                self.writer.add_scalar(name, value, frame)
            for name, value in self._legacy_metrics.items():
                self.writer.add_scalar(name, value, frame)

        payload = {**self._metrics, **self._legacy_metrics}
        run = self.tracker.run
        if run is not None and payload:
            run.log(payload, step=int(frame))
        self.tracker.update_summary(
            {
                "native_frame": int(frame),
                "native_epoch": int(epoch_num),
                "native_total_time_sec": float(total_time),
            }
        )

    def close_writer(self) -> None:
        if self.writer is not None and not self._writer_closed:
            self.writer.close()
            self._writer_closed = True
