"""DexToolBench fixed-scene Manager-Based task configuration.

This line intentionally uses UniSim's direct ``SceneCfg.model_file`` channel:
the 24 benchmark scenes are committed MJCFs, not runtime-generated B-form
entity sets.  Manager semantics are copied from the verified SimToolReal task
and then pinned to upstream DexToolBench evaluation values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from unilab.base.config_adapter import BackendAdapter

from simtoolreal_unilab.config import CONF_ROOT

from ..simtoolreal.config import (
    FINGERTIP_LINK_NAMES,
    JOINT_NAMES_CANONICAL,
    PALM_BODY_NAME,
)
from ..simtoolreal.env_cfg import SimToolRealEnvCfg
from .catalog import DexToolBenchTask, find_task, load_trajectory, scene_path


@dataclass
class DexToolBenchEnvCfg(SimToolRealEnvCfg):
    """Verified SimToolReal semantics with a fixed 1-env benchmark scene."""

    category: str = ""
    object_name: str = ""
    task_name: str = ""
    fixed_goal_states: tuple[tuple[float, ...], ...] | None = None
    object_scale: tuple[float, float, float] | None = None

    def validate(self) -> None:
        if not self.tool_pool_enabled:
            if self.fixed_model_variants is not None:
                raise ValueError("DexToolBench must not declare fixed_model_variants")
            scene = self.scene
            if scene is None or not scene.model_file:
                raise ValueError("DexToolBench requires a direct scene.model_file")
            if scene.entity_assets or scene.entity_variant is not None:
                raise ValueError("DexToolBench must not use entity composition or variants")
            if not self.category or not self.object_name or not self.task_name:
                raise ValueError("DexToolBench category/object/task fields are required")
            if not self.fixed_goal_states or self.object_scale is None:
                raise ValueError("DexToolBench fixed goals and object_scale are required")
        else:
            raise ValueError("DexToolBench must disable the SimToolReal tool pool")
        super().validate()


def _direct_scene_entities() -> dict[str, Any]:
    joints = list(JOINT_NAMES_CANONICAL)
    actuators = [f"{name}_ctrl" for name in joints]
    return {
        "robot": {
            "root_body_name": "iiwa14_link_0",
            "joint_names": joints,
            "actuator_names": actuators,
            "body_names": [PALM_BODY_NAME, *FINGERTIP_LINK_NAMES],
        },
        "object": {
            "root_body_name": "object_root",
            "body_names": ["object_root"],
        },
    }


def _pin_no_dr_and_fixed_task(
    override: dict[str, Any], task: DexToolBenchTask, trajectory: dict[str, Any]
) -> None:
    goals = trajectory["goals_wxyz"]
    override["category"] = task.category
    override["object_name"] = task.object_name
    override["task_name"] = task.task_name
    override["fixed_goal_states"] = goals
    override["object_scale"] = task.object_scale
    override["tool_pool_enabled"] = False

    policy = override["observations"]["policy"]["terms"]["simtoolreal"]["params"]
    policy.update(
        {
            "use_object_state_delay_noise": False,
            "object_state_delay_max": 0,
            "object_state_xyz_noise_std": 0.0,
            "object_state_rotation_noise_degrees": 0.0,
            "joint_velocity_obs_noise_std": 0.0,
            "use_obs_delay": False,
            "obs_delay_max": 0,
        }
    )
    action = override["actions"]["joint_pos"]
    action["use_action_delay"] = False
    action["action_delay_max"] = 0
    action["arm_moving_average"] = 0.1
    action["hand_moving_average"] = 0.1

    command = override["commands"]["goal"]
    command["goal_sampling_type"] = "absolute"
    command["eval_success_tolerance"] = 0.01
    command["success_steps"] = 1
    command["object_scale_noise_multiplier_range"] = (1.0, 1.0)
    command["fixed_goal_states"] = goals
    command["object_scale"] = task.object_scale

    reset = override["events"]["reset_robot"]["params"]
    reset.update(
        {
            "dof_pos_noise_arm": 0.0,
            "dof_pos_noise_hand": 0.0,
            "dof_vel_noise": 0.0,
            "reset_position_noise_x": 0.0,
            "reset_position_noise_y": 0.0,
            "reset_position_noise_z": 0.0,
            "table_reset_z_range": 0.0,
            "fixed_start_pose": trajectory["start_pose_wxyz"],
            "start_arm_higher": True,
        }
    )
    wrench = override["events"]["wrench_dr"]["params"]
    wrench["force_scale"] = 0.0
    wrench["torque_scale"] = 0.0
    wrench["force_prob_range"] = (0.0001, 0.0001)
    wrench["torque_prob_range"] = (0.0001, 0.0001)
    wrench["only_when_lifted"] = True

    success = override["terminations"]["success_tracker"]["params"]
    success["success_steps"] = 1
    success["force_consecutive"] = False
    override["terminations"]["max_consecutive_successes"]["params"]["max_consecutive_successes"] = (
        len(goals)
    )


def build_dextoolbench_env_cfg_override(
    category: str, object_name: str, task_name: str
) -> dict[str, Any]:
    """Compose verified SimToolReal semantics, then swap in a static MJCF scene."""
    task = find_task(category, object_name, task_name)
    trajectory = load_trajectory(task)
    model_file = scene_path(task)
    if not Path(model_file).is_file():
        raise FileNotFoundError(model_file)

    GlobalHydra.instance().clear()
    with initialize_config_dir(str(CONF_ROOT), version_base=None):
        raw = compose(
            config_name="config",
            overrides=[
                "task=simtoolreal/mujoco",
                "env.tool_pool_enabled=false",
                "~env.scene.entity_assets",
                f"+env.scene.model_file={model_file}",
            ],
        )
        scene = raw.env.scene
        scene.default_keyframe_name = "dextoolbench_home"
        scene.entities = OmegaConf.create(_direct_scene_entities())

    package_root = Path(__file__).resolve().parents[2]
    adapter = BackendAdapter(raw, root_dir=str(package_root))
    override = adapter.build_task_env_cfg_override()
    # BackendAdapter's root is only used for play-profile paths here, but keep
    # it at the package root so the behavior is deterministic in installed wheels.
    _pin_no_dr_and_fixed_task(override, task, trajectory)
    return override


__all__ = [
    "DexToolBenchEnvCfg",
    "build_dextoolbench_env_cfg_override",
]
