"""TensorBoard label-parity pin for the SimToolReal logging compatibility layer.

Upstream #1570 (09826339, "fix: remove duplicate reward episode metrics")
removed ``RewardManager.reset()``'s ``Episode_Reward/*`` extras — at the
migration tag that reset loop was the only writer of the family.  The
compatibility layer (ba9fecd4: ``SimToolRealCommand.log_extras`` +
``ExperimentTrackerObserver`` aggregation) never read those inputs: it consumes
the per-step ``RewardManager.step_reward_vectors()`` columns and
termination-frame masks, and performs the done-indexed episode aggregation
itself.

These tests pin the interaction contract (interface-migration.md section 4 L3
comparison surface):

1. End to end, driving the production fan-out (``CommandManager.log_extras``
   over the real SimToolReal command term) and the real observer must emit
   every required label family, with reward-vector inputs that contain no
   ``Episode_Reward`` key anywhere (pre-verified fact 1).
2. Every required family keeps a source-level writer symbol, so a future
   refactor cannot silently break the chain.
3. The compat layer's modules never reference ``Episode_Reward`` (the static
   side of fact 1).
4. ``Episode_Termination/*`` reset extras survive #1570 untouched
   (termination_manager.py reset) and flatten into ``env/*`` tags (fact 2).
5. The legacy name map covers exactly the production owner-YAML term set, so
   renaming a task term cannot silently orphan a label family.

Fake-backend pattern follows tests/tasks/test_simtoolreal_commands.py; observer
recording fakes follow tests/algos/rlgames_sapg/test_observer.py.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import yaml
from unilab.base.entity import EntityCfg, EntityScene
from unilab.base.reset_state import ResetStateTransaction
from unilab.managers._types import ManagerBasedRlEnv
from unilab.managers.command_manager import CommandManager
from unisim.backend.base import BackendRootStateLayout, SimBackend

from simtoolreal_unilab.tasks.simtoolreal.commands import (
    SimToolRealCommand,
    SimToolRealCommandCfg,
)
from simtoolreal_unilab.tasks.simtoolreal.config import (
    FINGERTIP_LINK_NAMES,
    PALM_BODY_NAME,
)

NUM_ENVS = 8
STEP_DT = 1.0 / 60.0
DONE_IDX = 4

ROBOT_BODY_NAMES = (PALM_BODY_NAME, *FINGERTIP_LINK_NAMES)
GOALVIZ_BODY_NAME = "goalviz_root"
ALL_BODY_NAMES = (*ROBOT_BODY_NAMES, GOALVIZ_BODY_NAME)
GOALVIZ_BODY_ID = ALL_BODY_NAMES.index(GOALVIZ_BODY_NAME)

_QPOS_WIDTH = 7
_QVEL_WIDTH = 6

_TASK_YAML = (
    Path(__file__).resolve().parents[2]
    / "src/simtoolreal_unilab/conf/rlgames_sapg/task/simtoolreal/base.yaml"
)

# The seven-term production reward set (owner YAML ``reward:`` block) and the
# legacy TensorBoard names the compatibility layer maps them onto
# (commands.py ``log_extras`` reward_name_map).
REWARD_TERM_TO_LEGACY = {
    "lift_progress": "lifting_rew",
    "lifting_bonus": "lift_bonus_rew",
    "fingertip_distance_delta": "fingertip_delta_rew",
    "keypoint_distance_delta": "keypoint_rew",
    "arm_joint_velocity_penalty": "kuka_actions_penalty",
    "hand_joint_velocity_penalty": "hand_actions_penalty",
    "reach_bonus": "bonus_rew",
}

# Termination-reason terms (owner YAML ``env.terminations`` minus the
# success-tracker snapshot term) and their legacy done_* names.
TERMINATION_REASON_TO_LEGACY = {
    "fall": "done_fall",
    "hand_far": "done_hand_far",
    "max_consecutive_successes": "done_max_successes",
    "time_out": "done_timeout",
}


# ----------------------------------------------------------------------------
# Fakes (patterns reused from the ba9fecd4 tests)
# ----------------------------------------------------------------------------


class _FakeBackend:
    backend_type = "fake"
    num_envs = NUM_ENVS
    num_actuators = 0

    def __init__(self) -> None:
        self.body_pos = np.zeros((NUM_ENVS, len(ALL_BODY_NAMES), 3), dtype=np.float32)
        self.body_quat = np.zeros((NUM_ENVS, len(ALL_BODY_NAMES), 4), dtype=np.float32)
        self.body_quat[..., 0] = 1.0
        self.set_state_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # --- bodies ---
    def get_body_ids(self, names) -> np.ndarray:
        return np.asarray([ALL_BODY_NAMES.index(n) for n in names], dtype=np.int32)

    # --- joints (none; Entity init still validates the empty layouts) ---
    def get_dof_pos(self) -> np.ndarray:
        return np.zeros((NUM_ENVS, 0), dtype=np.float32)

    def get_dof_vel(self) -> np.ndarray:
        return np.zeros((NUM_ENVS, 0), dtype=np.float32)

    def get_body_pos_w(self, ids) -> np.ndarray:
        return self.body_pos[:, np.asarray(ids)]

    def get_body_quat_w(self, ids) -> np.ndarray:
        return self.body_quat[:, np.asarray(ids)]

    def get_body_lin_vel_w(self, ids) -> np.ndarray:
        return np.zeros((NUM_ENVS, len(ids), 3), dtype=np.float32)

    def get_body_ang_vel_w(self, ids) -> np.ndarray:
        return np.zeros((NUM_ENVS, len(ids), 3), dtype=np.float32)

    def get_body_lin_vel_b(self, ids) -> np.ndarray:
        return np.zeros((NUM_ENVS, len(ids), 3), dtype=np.float32)

    def get_body_ang_vel_b(self, ids) -> np.ndarray:
        return np.zeros((NUM_ENVS, len(ids), 3), dtype=np.float32)

    # --- root-state layout / reset transaction ---
    def get_root_state_layout(self, root_body_name: str) -> BackendRootStateLayout:
        if root_body_name != GOALVIZ_BODY_NAME:
            raise ValueError(f"unknown root body {root_body_name}")
        return BackendRootStateLayout(
            qpos_indices=tuple(range(_QPOS_WIDTH)),
            qvel_indices=tuple(range(_QVEL_WIDTH)),
        )

    def get_default_qpos(self) -> np.ndarray:
        return np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_init_qvel(self) -> np.ndarray:
        return np.zeros(_QVEL_WIDTH, dtype=np.float32)

    def set_state(self, env_ids, qpos, qvel, randomization=None) -> dict:
        self.set_state_calls.append((np.asarray(env_ids).copy(), qpos.copy(), qvel.copy()))
        self.body_pos[env_ids, GOALVIZ_BODY_ID] = qpos[:, 0:3]
        self.body_quat[env_ids, GOALVIZ_BODY_ID] = qpos[:, 3:7]
        return {"timing": {"set_state_ms": 0.0}}


class _Writer:
    def __init__(self):
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, key, value, frame):
        self.scalars.append((key, float(value), int(frame)))

    def close(self):
        return None


class _Tracker:
    def __init__(self):
        self.run = None
        self.summaries: list[dict] = []

    def update_summary(self, value):
        self.summaries.append(value)


class _FakeRewardManager:
    """Post-#1570 RewardManager surface: per-step term vectors only.

    ``step_reward_vectors`` mimics the read-only weighted per-env columns and
    ``step_reward_extras`` the per-term means the env merges into the step log
    (manager_based_rl_env).  Neither input carries any ``Episode_Reward``
    aggregate — that is the point of the test.
    """

    def __init__(self, step_terms: dict[str, np.ndarray]):
        self._step_terms = step_terms

    def step_reward_vectors(self) -> dict[str, np.ndarray]:
        return self._step_terms

    def step_reward_extras(self) -> dict[str, float]:
        return {
            f"reward/{name}": float(np.mean(values)) for name, values in self._step_terms.items()
        }


class _FakeTerminationManager:
    def __init__(self, term_masks: dict[str, np.ndarray]):
        self._term_masks = term_masks

    def get_term(self, name: str) -> np.ndarray:
        return self._term_masks[name]


def _make_command_manager(seed: int = 7) -> tuple[CommandManager, SimToolRealCommand]:
    """A real ``CommandManager`` over the real SimToolReal command term."""
    backend = _FakeBackend()
    transaction = ResetStateTransaction(cast(SimBackend, backend))
    entities = {"robot": EntityCfg(body_names=ROBOT_BODY_NAMES)}
    scene = EntityScene(entities, cast(SimBackend, backend), reset_state=transaction)
    env = SimpleNamespace(
        num_envs=NUM_ENVS,
        rng=np.random.default_rng(seed),
        step_dt=STEP_DT,
        scene=scene,
        episode_length_buf=np.zeros(NUM_ENVS, dtype=np.int64),
        common_step_counter=0,
    )
    manager = CommandManager(
        {"goal": SimToolRealCommandCfg(resampling_time_range=(1e6, 1e6))},
        cast(ManagerBasedRlEnv, env),
    )
    env.command_manager = manager
    return manager, cast(SimToolRealCommand, manager.get_term("goal"))


def _step_reward_terms(step_idx: int) -> dict[str, np.ndarray]:
    """Per-term vectors (term_idx + 1) * 10 + step + env, all float32-exact."""
    env_offsets = np.arange(NUM_ENVS, dtype=np.float32)
    return {
        name: np.full(NUM_ENVS, (term_idx + 1) * 10 + step_idx, dtype=np.float32) + env_offsets
        for term_idx, name in enumerate(REWARD_TERM_TO_LEGACY)
    }


def _termination_masks() -> dict[str, np.ndarray]:
    masks = {name: np.zeros(NUM_ENVS, dtype=bool) for name in TERMINATION_REASON_TO_LEGACY}
    masks["fall"][[1, 5]] = True
    masks["hand_far"][[DONE_IDX]] = True
    masks["max_consecutive_successes"][[3, DONE_IDX]] = True
    masks["time_out"][[DONE_IDX, 6, 7]] = True
    return masks


def _required_scalar_tags() -> set[str]:
    legacy_names = sorted(REWARD_TERM_TO_LEGACY.values())
    cumulative = [*legacy_names, "total_reward"]
    tags = {f"episode_cumulative/{name}" for name in cumulative}
    tags |= {f"episode_cumulative_min/{name}_min" for name in cumulative}
    tags |= {f"episode_cumulative_max/{name}_max" for name in cumulative}
    tags |= {
        f"episode_final/{name}"
        for name in ("successes", "all_goals_hit", *TERMINATION_REASON_TO_LEGACY.values())
    }
    tags |= {"successes", "successes_median", "successes_max", "current_success_tolerance"}
    # Surviving equivalents of the removed env/Episode_Reward/* family: the
    # per-term step means (reward manager) and the reset-time termination
    # counts (termination manager, untouched by #1570).
    tags |= {f"env/reward/{name}" for name in REWARD_TERM_TO_LEGACY}
    tags |= {f"env/Episode_Termination/{name}" for name in TERMINATION_REASON_TO_LEGACY}
    return tags


# ----------------------------------------------------------------------------
# 1. End-to-end label parity without any Episode_Reward input
# ----------------------------------------------------------------------------


def test_compat_layer_emits_all_l3_label_families_without_episode_reward_inputs():
    from simtoolreal_unilab.rlgames_sapg.observer import ExperimentTrackerObserver

    command_manager, hub = _make_command_manager()
    hub.goal_count[:] = np.arange(NUM_ENVS)
    hub.prev_episode_successes[:] = np.arange(NUM_ENVS) + 10
    hub.current_success_tolerance = 0.0375
    masks = _termination_masks()

    writer = _Writer()
    observer = ExperimentTrackerObserver(_Tracker())
    observer.after_init(SimpleNamespace(writer=writer, games_to_track=10))

    for step_idx, done_indices in ((0, ()), (1, (DONE_IDX,))):
        reward_manager = _FakeRewardManager(_step_reward_terms(step_idx))
        log = {
            **command_manager.log_extras(
                reward_manager=reward_manager,
                termination_manager=_FakeTerminationManager(masks),
            ),
            # The env merges the reward step extras and, on the autoreset
            # tail, the manager reset extras into the same log dict.
            **reward_manager.step_reward_extras(),
            **{
                f"Episode_Termination/{name}": int(np.count_nonzero(mask))
                for name, mask in masks.items()
            },
        }
        assert not any("Episode_Reward" in key for key in log)
        observer.process_infos({"log": log}, np.asarray(done_indices, dtype=np.int64))
    observer.after_print_stats(frame=2, epoch_num=1, total_time=1.0)

    values = {(name, frame): value for name, value, frame in writer.scalars}
    missing = _required_scalar_tags() - {name for name, _frame in values}
    assert not missing, f"missing required TensorBoard tags: {sorted(missing)}"
    # Registered deviation guard (fact 3): nothing revives Episode_Reward.
    assert not any("Episode_Reward" in name for name, _frame in values)

    # Episode accumulation across the two driven steps for the done row:
    # sum_s [(term_idx + 1) * 10 + s + DONE_IDX] = 20 * (term_idx + 1) + 9.
    for term_idx, legacy in enumerate(REWARD_TERM_TO_LEGACY.values()):
        expected = 2 * ((term_idx + 1) * 10) + 1 + 2 * DONE_IDX
        assert values[(f"episode_cumulative/{legacy}", 2)] == expected
        assert values[(f"episode_cumulative_min/{legacy}_min", 2)] == expected
        assert values[(f"episode_cumulative_max/{legacy}_max", 2)] == expected
    total_expected = sum(2 * ((i + 1) * 10) + 1 + 2 * DONE_IDX for i in range(7))
    assert values[("episode_cumulative/total_reward", 2)] == total_expected

    # Terminal-frame facts for the done row (episode_final/* come from the
    # hub's goal counter and the termination-frame masks, never from rewards).
    assert values[("episode_final/successes", 2)] == float(DONE_IDX)
    assert values[("episode_final/all_goals_hit", 2)] == 1.0
    assert values[("episode_final/done_fall", 2)] == 0.0
    assert values[("episode_final/done_hand_far", 2)] == 1.0
    assert values[("episode_final/done_max_successes", 2)] == 1.0
    assert values[("episode_final/done_timeout", 2)] == 1.0

    # Success statistics over the full prev-episode vector (10..17).
    assert values[("successes", 2)] == 13.5
    assert values[("successes_median", 2)] == 13.5
    assert values[("successes_max", 2)] == 17.0
    assert values[("current_success_tolerance", 2)] == 0.0375

    # Surviving step-level reward family: per-term means of the last step.
    mean_env_offset = float(np.arange(NUM_ENVS).mean())
    for term_idx, name in enumerate(REWARD_TERM_TO_LEGACY):
        assert values[(f"env/reward/{name}", 2)] == (term_idx + 1) * 10 + 1 + mean_env_offset
    # Fact 2: the reset-time termination counts still flatten into env/*.
    for name, mask in masks.items():
        assert values[(f"env/Episode_Termination/{name}", 2)] == float(np.count_nonzero(mask))


# ----------------------------------------------------------------------------
# 2. Source-level writer existence for every required family
# ----------------------------------------------------------------------------


def test_required_label_families_keep_source_level_writers():
    from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from unilab.managers.command_manager import CommandManager
    from unilab.managers.reward_manager import RewardManager
    from unilab.managers.termination_manager import TerminationManager

    from simtoolreal_unilab.rlgames_sapg.observer import ExperimentTrackerObserver

    # episode_final/*, successes*, current_success_tolerance and the
    # episode_cumulative sources: the task term hook, the manager fan-out and
    # the env merge that carries it into the step log.
    assert callable(getattr(SimToolRealCommand, "log_extras", None))
    assert callable(getattr(CommandManager, "log_extras", None))
    env_merge_source = inspect.getsource(ManagerBasedRlEnv._update_state_in_read_phase)
    assert "log_extras(" in env_merge_source

    # Observer aggregation: tag construction for each family.
    print_stats_source = inspect.getsource(ExperimentTrackerObserver.after_print_stats)
    assert "episode_cumulative/" in print_stats_source
    assert "episode_final/" in print_stats_source
    process_infos_source = inspect.getsource(ExperimentTrackerObserver.process_infos)
    assert 'log.get("current_success_tolerance")' in process_infos_source
    assert '_process_vector_summaries(log, tag="successes")' in process_infos_source
    vector_summaries_source = inspect.getsource(ExperimentTrackerObserver._process_vector_summaries)
    assert "{tag}_median" in vector_summaries_source
    assert "{tag}_max" in vector_summaries_source

    # Reward-side feeds: per-env term columns (episode accumulation) and the
    # per-term step means (env/reward/*).
    assert callable(getattr(RewardManager, "step_reward_vectors", None))
    assert callable(getattr(RewardManager, "step_reward_extras", None))

    # Termination-reason distribution: the reset-extras writer #1570 left in
    # place (termination_manager.py), plus the done_* mask feed consumed by
    # log_extras above.
    assert callable(getattr(TerminationManager, "get_term", None))
    assert "Episode_Termination/" in inspect.getsource(TerminationManager.reset)


def test_compat_layer_source_never_references_episode_reward():
    """Fact 1, static side: no compat-layer module reads the removed family."""
    from unilab.envs import manager_based_rl_env as env_module
    from unilab.managers import command_manager as command_manager_module
    from unilab.managers import reward_manager as reward_manager_module
    from unilab.managers import termination_manager as termination_manager_module

    from simtoolreal_unilab.rlgames_sapg import observer as observer_module
    from simtoolreal_unilab.tasks.simtoolreal import commands as commands_module

    for module in (
        commands_module,
        command_manager_module,
        reward_manager_module,
        termination_manager_module,
        observer_module,
        env_module,
    ):
        assert "Episode_Reward" not in inspect.getsource(module), module.__name__


def test_legacy_name_map_covers_exactly_the_production_owner_yaml_terms():
    """A renamed task term cannot silently orphan a label family."""
    base = yaml.safe_load(_TASK_YAML.read_text())
    assert set(REWARD_TERM_TO_LEGACY) == set(base["reward"])
    terminations = set(base["env"]["terminations"])
    assert set(TERMINATION_REASON_TO_LEGACY) <= terminations
    # success_tracker writes the pending_success snapshot, never a done reason.
    assert terminations - set(TERMINATION_REASON_TO_LEGACY) == {"success_tracker"}
