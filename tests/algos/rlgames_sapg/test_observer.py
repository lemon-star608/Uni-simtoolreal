from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch


class Writer:
    def __init__(self):
        self.scalars = []
        self.close_calls = 0

    def add_scalar(self, key, value, frame):
        self.scalars.append((key, float(value), int(frame)))

    def close(self):
        self.close_calls += 1


class Run:
    def __init__(self):
        self.logs = []

    def log(self, payload, step=None):
        self.logs.append((payload, step))


class Tracker:
    def __init__(self, run=None):
        self.run = run
        self.summaries = []

    def update_summary(self, value):
        self.summaries.append(value)


def test_observer_bridges_only_env_log_scalars_to_native_writer_and_active_run():
    from simtoolreal_unilab.rlgames_sapg.observer import ExperimentTrackerObserver

    writer, run = Writer(), Run()
    tracker = Tracker(run)
    observer = ExperimentTrackerObserver(tracker)
    observer.before_init("run", {}, "0_test")
    observer.after_init(SimpleNamespace(writer=writer))
    observer.process_infos(
        {
            "log": {
                "reward": 2.5,
                "count": np.int64(3),
                "scalar": torch.tensor(4.0),
                "rows": np.ones(2),
            }
        },
        torch.tensor([0]),
    )
    observer.after_steps()
    observer.after_print_stats(frame=12, epoch_num=2, total_time=1.5)
    assert writer.scalars == [
        ("env/reward", 2.5, 12),
        ("env/count", 3.0, 12),
        ("env/scalar", 4.0, 12),
    ]
    assert run.logs == [({"env/reward": 2.5, "env/count": 3.0, "env/scalar": 4.0}, 12)]
    assert tracker.summaries[-1] == {
        "native_frame": 12,
        "native_epoch": 2,
        "native_total_time_sec": 1.5,
    }


def test_observer_works_without_wandb_and_closes_native_writer_exactly_once():
    from simtoolreal_unilab.rlgames_sapg.observer import ExperimentTrackerObserver

    writer = Writer()
    observer = ExperimentTrackerObserver(Tracker())
    observer.after_init(SimpleNamespace(writer=writer))
    observer.process_infos({"log": {"metric": 1}}, torch.tensor([], dtype=torch.int64))
    observer.after_clear_stats()
    observer.after_print_stats(1, 1, 0.1)
    observer.close_writer()
    observer.close_writer()
    assert writer.close_calls == 1


def test_observer_reproduces_legacy_success_tolerance_and_episode_tags():
    from simtoolreal_unilab.rlgames_sapg.observer import ExperimentTrackerObserver

    writer, run = Writer(), Run()
    observer = ExperimentTrackerObserver(Tracker(run))
    observer.after_init(SimpleNamespace(writer=writer, games_to_track=10))

    observer.process_infos(
        {
            "log": {
                "episode_cumulative": {
                    "bonus_rew": np.array([1.0, 2.0], dtype=np.float32),
                    "total_reward": np.array([1.0, 2.0], dtype=np.float32),
                },
                "episode_final": {
                    "successes": np.array([3.0, 4.0], dtype=np.float32),
                    "done_timeout": np.array([0.0, 1.0], dtype=np.float32),
                },
                "successes": np.array([7.0, 9.0], dtype=np.float32),
                "current_success_tolerance": 0.075,
            }
        },
        torch.tensor([1]),
    )
    observer.after_print_stats(frame=12, epoch_num=2, total_time=1.5)

    values = {(name, frame): value for name, value, frame in writer.scalars}
    assert values[("episode_cumulative/bonus_rew", 12)] == 2.0
    assert values[("episode_cumulative_min/bonus_rew_min", 12)] == 2.0
    assert values[("episode_cumulative_max/bonus_rew_max", 12)] == 2.0
    assert values[("episode_final/successes", 12)] == 4.0
    assert values[("episode_final/done_timeout", 12)] == 1.0
    assert values[("successes", 12)] == 8.0
    assert values[("successes_median", 12)] == 8.0
    assert values[("successes_max", 12)] == 9.0
    assert values[("current_success_tolerance", 12)] == 0.075
    assert run.logs[-1][0]["current_success_tolerance"] == 0.075


def test_observer_applies_sapg_good_reset_boundary_before_success_aggregation():
    from simtoolreal_unilab.rlgames_sapg.observer import ExperimentTrackerObserver

    writer = Writer()
    observer = ExperimentTrackerObserver(Tracker())
    observer.after_init(SimpleNamespace(writer=writer, games_to_track=10))
    observer.process_infos(
        {
            "log": {
                "episode_final": {
                    "successes": np.array([100.0, 3.0, 5.0, 7.0], dtype=np.float32),
                },
                "successes": np.array([100.0, 3.0, 5.0, 7.0], dtype=np.float32),
                "current_success_tolerance": 0.05,
            }
        },
        torch.tensor([2, 3]),
        ignore_env_boundary=2,
    )
    observer.after_print_stats(frame=4, epoch_num=1, total_time=0.1)
    values = {(name, frame): value for name, value, frame in writer.scalars}
    assert values[("episode_final/successes", 4)] == 6.0
    assert values[("successes", 4)] == 6.0
    assert values[("successes_per_block/block_0", 4)] == 51.5
    assert values[("successes_per_block/block_1", 4)] == 6.0
    assert values[("current_success_tolerance", 4)] == 0.05
