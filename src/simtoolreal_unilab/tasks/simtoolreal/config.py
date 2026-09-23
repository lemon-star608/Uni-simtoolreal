"""SimToolReal task configuration mirrors (backend-neutral, step-0 skeleton).

Field-by-field mirror of the original repository's
``isaacsimenvs/tasks/simtoolreal/simtoolreal_env_cfg.py`` (snapshot 313d5aea)
and ``isaacsimenvs/cfg/task/SimToolReal.yaml``.  Dead code from the original
repository is deliberately not ported (``HAND_JOINT_FRICTION``,
``scene_utils.py:104-120``, is unused upstream).

These dataclasses are the typed owner of the task parameters; the Hydra owner
YAML (``conf/rlgames_sapg/task/simtoolreal/base.yaml``) repeats the values at
the term level, and ``tests/tasks/test_simtoolreal_owner_yaml_parity.py``
pins the owner YAML against the original repository's frozen values
(per-value parity pinning; the values themselves come from the original repository).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

# ----------------------------------------------------------------------------
# Canonical robot layout (scene_utils.py:34-57)
#
# ``JOINT_NAMES_CANONICAL`` is the legacy 29-DoF policy order (7-DoF iiwa arm
# followed by the 22-DoF SHARPA hand, finger-grouped).  It matches the URDF
# document order of the non-fixed joints; the action/observation terms keep a
# canonical<->backend permutation resolved at term init (action_utils.py:31-34).
# ----------------------------------------------------------------------------

ARM_JOINT_NAMES: tuple[str, ...] = tuple(f"iiwa14_joint_{index}" for index in range(1, 8))

HAND_JOINT_NAMES: tuple[str, ...] = (
    "left_1_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_2_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_3_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_4_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_5_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
)

JOINT_NAMES_CANONICAL: tuple[str, ...] = ARM_JOINT_NAMES + HAND_JOINT_NAMES

NUM_JOINTS: int = 29
NUM_ARM_JOINTS: int = 7
NUM_FINGERTIPS: int = 5
NUM_KEYPOINTS: int = 4

TOOL_POOL_TARGET_ENTITY = "object"
"""Physical scene entity that consumes the tool-only fixed-variant catalog."""

ENTITY_ASSETS_MANIFEST_SCHEMA = "simtoolreal_unilab.entity_assets.v2"
"""Current B-form entity-pool manifest contract (four physical entities)."""

# Observation geometry constants (obs_utils.py:22-33).
PALM_CENTER_OFFSET: tuple[float, float, float] = (-0.0, -0.02, 0.16)
"""Policy was trained against the palm center, not the raw wrist body."""

FINGERTIP_OFFSET: tuple[float, float, float] = (0.02, 0.002, 0.0)
"""Shift fingertip body origins to the approximate pad centers."""

KEYPOINT_CORNERS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 1),
    (1, 1, -1),
    (-1, -1, 1),
    (-1, -1, -1),
)
"""Object-frame keypoint corners before scaling."""

ROOT_BODY_NAME = "iiwa14_link_0"
"""Fixed-base articulation root link (scene_utils.py:1721-1725, fix_base=True)."""

PALM_BODY_NAME = "iiwa14_link_7"
"""scene_utils.py:49."""

FINGERTIP_LINK_NAMES: tuple[str, ...] = (
    "left_index_DP",
    "left_middle_DP",
    "left_ring_DP",
    "left_thumb_DP",
    "left_pinky_DP",
)
"""scene_utils.py:52-56; merged fingertip bodies land on the DP links."""

# Per-joint PD gains and dynamics (scene_utils.py:59-101; verified upstream
# against the pretrained checkpoint).  Consumed by the owner actuator config
# in step 1; the converter itself imports zero-gain drives.
ARM_JOINT_STIFFNESS: dict[str, float] = {
    "iiwa14_joint_1": 600.0,
    "iiwa14_joint_2": 600.0,
    "iiwa14_joint_3": 500.0,
    "iiwa14_joint_4": 400.0,
    "iiwa14_joint_5": 200.0,
    "iiwa14_joint_6": 200.0,
    "iiwa14_joint_7": 200.0,
}
ARM_JOINT_DAMPING: dict[str, float] = {
    "iiwa14_joint_1": 27.027026473513512,
    "iiwa14_joint_2": 27.027026473513512,
    "iiwa14_joint_3": 24.672186769721083,
    "iiwa14_joint_4": 22.067474708266914,
    "iiwa14_joint_5": 9.752538131173853,
    "iiwa14_joint_6": 9.147747263670984,
    "iiwa14_joint_7": 9.147747263670984,
}
HAND_JOINT_STIFFNESS: dict[str, float] = {
    "left_1_thumb_CMC_FE": 6.95,
    "left_thumb_CMC_AA": 13.2,
    "left_thumb_MCP_FE": 4.76,
    "left_thumb_MCP_AA": 6.62,
    "left_thumb_IP": 0.9,
    "left_2_index_MCP_FE": 4.76,
    "left_index_MCP_AA": 6.62,
    "left_index_PIP": 0.9,
    "left_index_DIP": 0.9,
    "left_3_middle_MCP_FE": 4.76,
    "left_middle_MCP_AA": 6.62,
    "left_middle_PIP": 0.9,
    "left_middle_DIP": 0.9,
    "left_4_ring_MCP_FE": 4.76,
    "left_ring_MCP_AA": 6.62,
    "left_ring_PIP": 0.9,
    "left_ring_DIP": 0.9,
    "left_5_pinky_CMC": 1.38,
    "left_pinky_MCP_FE": 4.76,
    "left_pinky_MCP_AA": 6.62,
    "left_pinky_PIP": 0.9,
    "left_pinky_DIP": 0.9,
}
HAND_JOINT_DAMPING: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.28676845,
    "left_thumb_CMC_AA": 0.40845109,
    "left_thumb_MCP_FE": 0.20394083,
    "left_thumb_MCP_AA": 0.24044435,
    "left_thumb_IP": 0.04190723,
    "left_2_index_MCP_FE": 0.20859232,
    "left_index_MCP_AA": 0.24595532,
    "left_index_PIP": 0.04243185,
    "left_index_DIP": 0.03504461,
    "left_3_middle_MCP_FE": 0.2085923,
    "left_middle_MCP_AA": 0.24595532,
    "left_middle_PIP": 0.04243185,
    "left_middle_DIP": 0.03504461,
    "left_4_ring_MCP_FE": 0.20859226,
    "left_ring_MCP_AA": 0.24595528,
    "left_ring_PIP": 0.04243183,
    "left_ring_DIP": 0.0350446,
    "left_5_pinky_CMC": 0.02782345,
    "left_pinky_MCP_FE": 0.20859229,
    "left_pinky_MCP_AA": 0.24595528,
    "left_pinky_PIP": 0.04243183,
    "left_pinky_DIP": 0.0350446,
}
HAND_JOINT_ARMATURE: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.0032,
    "left_thumb_CMC_AA": 0.0032,
    "left_thumb_MCP_FE": 0.00265,
    "left_thumb_MCP_AA": 0.00265,
    "left_thumb_IP": 0.0006,
    "left_2_index_MCP_FE": 0.00265,
    "left_index_MCP_AA": 0.00265,
    "left_index_PIP": 0.0006,
    "left_index_DIP": 0.00042,
    "left_3_middle_MCP_FE": 0.00265,
    "left_middle_MCP_AA": 0.00265,
    "left_middle_PIP": 0.0006,
    "left_middle_DIP": 0.00042,
    "left_4_ring_MCP_FE": 0.00265,
    "left_ring_MCP_AA": 0.00265,
    "left_ring_PIP": 0.0006,
    "left_ring_DIP": 0.00042,
    "left_5_pinky_CMC": 0.00012,
    "left_pinky_MCP_FE": 0.00265,
    "left_pinky_MCP_AA": 0.00265,
    "left_pinky_PIP": 0.0006,
    "left_pinky_DIP": 0.00042,
}

# Proven-working default arm pose (scene_utils.py:123-128).
ARM_DEFAULT_JOINT_POS: dict[str, float] = {
    "iiwa14_joint_1": -1.571,
    "iiwa14_joint_2": 1.571,
    "iiwa14_joint_3": 0.0,
    "iiwa14_joint_4": 1.376,
    "iiwa14_joint_5": 0.0,
    "iiwa14_joint_6": 1.485,
    "iiwa14_joint_7": 1.308,
}


# ----------------------------------------------------------------------------
# Sectioned task cfgs (simtoolreal_env_cfg.py:56-507; SimToolReal.yaml)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetsCfg:
    """URDFs, procedural-generation knobs, static per-material frictions.

    simtoolreal_env_cfg.py:56-101 / SimToolReal.yaml ``assets:``.
    """

    robot_urdf: str = "urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    table_urdf: str = "urdf/table_narrow.urdf"
    object_name: str = "handle_head_primitives"
    handle_head_types: tuple[str, ...] = (
        "hammer",
        "screwdriver",
        "marker",
        "spatula",
        "eraser",
        "brush",
    )
    num_assets_per_type: int = 100
    shuffle_assets: bool = True
    # The procedural pool is seeded with MT19937 seed=42 (tool_catalog.py).
    object_pool_seed: int = 42
    # Static per-material frictions (adapter runtime write, not reset DR).
    modify_asset_frictions: bool = True
    robot_friction: float = 0.5
    finger_tip_friction: float = 1.5
    object_friction: float = 0.5
    table_friction: float = 0.5


@dataclass(frozen=True)
class ActionCfg:
    """Joint-position-target smoothing (simtoolreal_env_cfg.py:317-322)."""

    arm_moving_average: float = 0.1
    hand_moving_average: float = 0.1
    dof_speed_scale: float = 1.5


@dataclass(frozen=True)
class ObsCfg:
    """Asymmetric actor-critic obs layout + clamping (simtoolreal_env_cfg.py:108-145).

    Critic sees the full ``state_list``; actor sees the ``obs_list`` subset.
    Field widths are pinned by ``OBS_FIELD_SIZES`` in observations.py
    (obs_utils.py:35-54) and code-asserted at assembly time.
    """

    state_list: tuple[str, ...] = (
        "joint_pos",
        "joint_vel",
        "prev_action_targets",
        "palm_pos",
        "palm_rot",
        "palm_vel",
        "object_rot",
        "object_vel",
        "fingertip_pos_rel_palm",
        "keypoints_rel_palm",
        "keypoints_rel_goal",
        "object_scales",
        "closest_keypoint_max_dist",
        "closest_fingertip_dist",
        "lifted_object",
        "progress",
        "successes",
        "reward",
    )
    obs_list: tuple[str, ...] = (
        "joint_pos",
        "joint_vel",
        "prev_action_targets",
        "palm_pos",
        "palm_rot",
        "object_rot",
        "fingertip_pos_rel_palm",
        "keypoints_rel_palm",
        "keypoints_rel_goal",
        "object_scales",
    )
    clamp_abs_observations: float = 10.0


@dataclass(frozen=True)
class RewardCfg:
    """Seven-term reward set (simtoolreal_env_cfg.py:331-350).

    The original gym-era hand-delta and object linear/angular velocity
    penalties were computed with zero coefficients; they stay retired
    (DESIGN.md section 3.5).
    """

    keypoint_rew_scale: float = 200.0
    keypoint_scale: float = 1.5
    object_base_size: float = 0.04
    fixed_size: tuple[float, float, float] = (0.141, 0.03025, 0.0271)
    fixed_size_keypoint_reward: bool = True
    lifting_rew_scale: float = 20.0
    lifting_bonus: float = 300.0
    lifting_bonus_threshold: float = 0.15
    distance_delta_rew_scale: float = 50.0
    reach_goal_bonus: float = 1000.0
    kuka_actions_penalty_scale: float = 0.03
    hand_actions_penalty_scale: float = 0.003


@dataclass(frozen=True)
class ResetCfg:
    """Initial-state distribution + goal sampling (simtoolreal_env_cfg.py:359-394)."""

    reset_position_noise_x: float = 0.1
    reset_position_noise_y: float = 0.1
    reset_position_noise_z: float = 0.02
    fixed_start_pose: tuple[float, ...] | None = None
    reset_dof_pos_random_interval_arm: float = 0.1
    reset_dof_pos_random_interval_fingers: float = 0.1
    reset_dof_vel_random_interval: float = 0.5
    table_reset_z: float = 0.38
    table_reset_z_range: float = 0.01
    table_object_z_offset: float = 0.25
    goal_sampling_type: str = "delta"  # "delta" | "absolute"
    delta_goal_distance: float = 0.1
    delta_rotation_degrees: float = 90.0
    target_volume_mins: tuple[float, float, float] = (-0.35, -0.2, 0.6)
    target_volume_maxs: tuple[float, float, float] = (0.35, 0.2, 0.95)
    target_volume_region_scale: float = 1.0


@dataclass(frozen=True)
class TerminationCfg:
    """Episode-end conditions + tolerance curriculum (simtoolreal_env_cfg.py:421-444)."""

    episode_length: int = 600  # policy steps (600 * 2 * 1/120 = 10 s)
    success_tolerance: float = 0.075  # curriculum start
    target_success_tolerance: float = 0.01  # curriculum floor
    eval_success_tolerance: float | None = None
    success_steps: int = 10
    max_consecutive_successes: int = 50
    force_consecutive_near_goal_steps: bool = False
    tolerance_curriculum_increment: float = 0.9
    tolerance_curriculum_interval: int = 3000
    tolerance_curriculum_success_threshold: float = 3.0
    # termination_utils.py:54-55 / :62 (fall height and hand-far distance).
    fall_height: float = 0.1
    hand_far_distance: float = 1.5


@dataclass(frozen=True)
class DomainRandomizationCfg:
    """Sim2real DR set (simtoolreal_env_cfg.py:453-507); physics-param DR excluded."""

    use_obs_delay: bool = True
    obs_delay_max: int = 3
    use_action_delay: bool = True
    action_delay_max: int = 3
    use_object_state_delay_noise: bool = True
    object_state_delay_max: int = 10
    object_state_xyz_noise_std: float = 0.01
    object_state_rotation_noise_degrees: float = 5.0
    object_scale_noise_multiplier_range: tuple[float, float] = (1.0, 1.0)
    joint_velocity_obs_noise_std: float = 0.1
    force_scale: float = 20.0
    force_prob_range: tuple[float, float] = (0.001, 0.1)
    force_decay: float = 0.0
    force_decay_interval: float = 0.08
    force_only_when_lifted: bool = True
    torque_scale: float = 2.0
    torque_prob_range: tuple[float, float] = (0.001, 0.1)
    torque_decay: float = 0.0
    torque_decay_interval: float = 0.08
    torque_only_when_lifted: bool = True


@dataclass(frozen=True)
class SimToolRealTaskCfg:
    """Aggregate mirror of the original sectioned config classes."""

    assets: AssetsCfg = field(default_factory=AssetsCfg)
    obs: ObsCfg = field(default_factory=ObsCfg)
    action: ActionCfg = field(default_factory=ActionCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    reset: ResetCfg = field(default_factory=ResetCfg)
    termination: TerminationCfg = field(default_factory=TerminationCfg)
    domain_randomization: DomainRandomizationCfg = field(default_factory=DomainRandomizationCfg)


def match_entity_names(
    declared: Sequence[str] | None,
    canonical: Sequence[str],
    *,
    label: str,
) -> list[str]:
    """Map canonical truth-table names onto an entity's declared names.

    The frozen tables in this module use bare source names
    (``iiwa14_joint_1``, ``object_root``).  Entity-composed scenes qualify
    every name with the physical-entity prefix (``robot/iiwa14_joint_1``,
    unisim ``compile_scene_layout``), while the IsaacSim line declares the
    bare names — so manager terms resolve selectors against the entity's
    *declared* names, matching on the final path segment.  ``declared=None``
    (undeclared selectors, lightweight fixtures) passes the canonical names
    through unchanged.
    """
    if declared is None:
        return list(canonical)
    by_stem: dict[str, str] = {}
    for name in declared:
        stem = name.rsplit("/", 1)[-1]
        if stem in by_stem:
            raise ValueError(
                f"{label} selector stems are ambiguous in the entity declaration: "
                f"{by_stem[stem]!r} vs {name!r}"
            )
        by_stem[stem] = name
    missing = [name for name in canonical if name not in by_stem]
    if missing:
        raise ValueError(
            f"{label} selectors {missing} are not covered by the entity declaration "
            f"{list(declared)}"
        )
    return [by_stem[name] for name in canonical]


def declared_names(entity: object, kind: str) -> tuple[str, ...] | None:
    """An entity's declared namespace (``joint``/``body``/``actuator``).

    ``None`` when the namespace is undeclared: the facade raises
    ``NotImplementedError`` for undeclared namespaces, and lightweight test
    doubles may not carry the attribute at all.
    """
    try:
        value = getattr(entity, f"{kind}_names", None)
    except NotImplementedError:
        return None
    return None if value is None else tuple(value)


__all__ = [
    "ARM_DEFAULT_JOINT_POS",
    "ARM_JOINT_DAMPING",
    "ARM_JOINT_NAMES",
    "ARM_JOINT_STIFFNESS",
    "ActionCfg",
    "AssetsCfg",
    "DomainRandomizationCfg",
    "FINGERTIP_LINK_NAMES",
    "FINGERTIP_OFFSET",
    "HAND_JOINT_ARMATURE",
    "HAND_JOINT_DAMPING",
    "HAND_JOINT_NAMES",
    "HAND_JOINT_STIFFNESS",
    "JOINT_NAMES_CANONICAL",
    "KEYPOINT_CORNERS",
    "NUM_FINGERTIPS",
    "NUM_JOINTS",
    "NUM_KEYPOINTS",
    "ObsCfg",
    "PALM_BODY_NAME",
    "PALM_CENTER_OFFSET",
    "ROOT_BODY_NAME",
    "ResetCfg",
    "RewardCfg",
    "SimToolRealTaskCfg",
    "TerminationCfg",
    "declared_names",
    "match_entity_names",
]
