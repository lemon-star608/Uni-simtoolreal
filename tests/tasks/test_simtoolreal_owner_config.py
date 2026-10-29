"""Default owner configuration contract for MuJoCo training."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from simtoolreal_unilab.tasks.simtoolreal import config as task_config
from simtoolreal_unilab.tasks.simtoolreal import observations as obs

ROOT = Path(__file__).resolve().parents[2]
CONF_ROOT = ROOT / "src/simtoolreal_unilab/conf/rlgames_sapg"


def compose_defaults():
    with initialize_config_dir(str(CONF_ROOT), version_base=None):
        return compose(config_name="config")


def test_default_owner_routes_to_full_scale_mujoco_training():
    cfg = compose_defaults()
    assert cfg.training.task_name == "SimToolReal"
    assert cfg.training.sim_backend == "mujoco"
    assert "smo" + "ke" not in cfg.training
    assert cfg.algo.num_envs == 24576
    assert cfg.env.tool_pool_enabled is True
    assert cfg.env.tool_pool_num_per_type == 100
    assert cfg.env.tool_pool_num_per_type * 12 == 1200
    assert cfg.rl_games.params.config.expl_coef_block_size == 4096
    assert cfg.algo.num_envs // cfg.rl_games.params.config.expl_coef_block_size == 6


def test_default_owner_observation_widths_are_pinned():
    cfg = task_config.ObsCfg()
    assert sum(obs.OBS_FIELD_SIZES[field] for field in cfg.obs_list) == 140
    assert sum(obs.OBS_FIELD_SIZES[field] for field in cfg.state_list) == 162
