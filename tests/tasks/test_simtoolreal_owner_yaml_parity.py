"""Pin the SimToolReal owner YAML values against the original repository.

``pipeline-audit.md`` S01 verified every owner-YAML constant against the
original repository by hand; this test freezes that result.  Expected values
are hard-coded with original-repository line references — the upstream is a
frozen snapshot (``313d5aea``), so static anchoring is sufficient and avoids
re-implementing the configclass-default + YAML-overlay merge semantics.

Structure is intentionally NOT mirrored (the owner YAML re-organizes values
into manager terms), so the comparison is a selected-value checklist.
``${...}`` resolver strings (asset paths) are skipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = REPO_ROOT / "src/simtoolreal_unilab/conf/rlgames_sapg/task/simtoolreal"


@pytest.fixture(scope="module")
def base() -> dict:
    return yaml.safe_load((CONF_DIR / "base.yaml").read_text())


@pytest.fixture(scope="module")
def isaacsim() -> dict:
    return yaml.safe_load((CONF_DIR / "isaacsim.yaml").read_text())


# --- env timing (SimToolReal.yaml:4, :5, :15; decimation 2 = ctrl/sim) ------


def test_env_timing(base):
    assert base["env"]["sim_dt"] == 0.008333333333333333  # 1/120, SimToolReal.yaml:15
    assert base["env"]["ctrl_dt"] == 0.016666666666666666  # 1/60 = decimation 2, :4
    assert base["env"]["max_episode_seconds"] == 10.0  # :5 (600 policy steps)
    # The original DirectRLEnv applies no dt scaling (direct_rl_env.py:393
    # assigns the _get_rewards() result directly); fix P2 (reaudit R6/R5)
    # restores that magnitude parity.
    assert base["env"]["scale_rewards_by_dt"] is False


# --- observations (SimToolReal.yaml:63-71, :142-157) ------------------------


def test_observation_terms(base):
    policy = base["env"]["observations"]["policy"]["terms"]["simtoolreal"]["params"]
    # SimToolReal.yaml:144-147 (obs latency), :150-153 (object state),
    # :156-157 (joint vel), :71 (clamp).
    assert policy == {
        "use_object_state_delay_noise": True,
        "object_state_delay_max": 10,
        "object_state_xyz_noise_std": 0.01,
        "object_state_rotation_noise_degrees": 5.0,
        "joint_velocity_obs_noise_std": 0.1,
        "use_obs_delay": True,
        "obs_delay_max": 3,
        "clamp_abs_observations": 10.0,
    }
    critic = base["env"]["observations"]["critic"]["terms"]["simtoolreal"]["params"]
    assert critic == {"clamp_abs_observations": 10.0}


# --- actions (SimToolReal.yaml:76-79, :146-147) ------------------------------


def test_action_term(base):
    action = base["env"]["actions"]["joint_pos"]
    assert action["arm_moving_average"] == 0.1
    assert action["hand_moving_average"] == 0.1
    assert action["dof_speed_scale"] == 1.5  # :79
    assert action["use_action_delay"] is True  # :146
    assert action["action_delay_max"] == 3  # :147


# --- commands/goal (SimToolReal.yaml:118-123, :130-137; env_cfg.py:444) ------


def test_goal_command_term(base):
    goal = base["env"]["commands"]["goal"]
    assert goal["goal_sampling_type"] == "delta"  # :118
    assert goal["delta_goal_distance"] == 0.1  # :119
    assert goal["delta_rotation_degrees"] == 90.0  # :120
    assert goal["target_volume_mins"] == [-0.35, -0.2, 0.6]  # :121
    assert goal["target_volume_maxs"] == [0.35, 0.2, 0.95]  # :122
    assert goal["target_volume_region_scale"] == 1.0  # :123
    assert goal["success_tolerance"] == 0.075  # :130
    assert goal["target_success_tolerance"] == 0.01  # :131
    assert goal["success_steps"] == 10  # :133
    assert goal["tolerance_curriculum_increment"] == 0.9  # :136
    assert goal["tolerance_curriculum_interval"] == 3000  # :137
    # Configclass-only upstream default (simtoolreal_env_cfg.py:444).
    assert goal["tolerance_curriculum_success_threshold"] == 3.0
    assert goal["object_scale_noise_multiplier_range"] == [1.0, 1.0]  # :154


# --- events (SimToolReal.yaml:104-116, :160-170) -----------------------------


def test_reset_robot_event(base):
    params = base["env"]["events"]["reset_robot"]["params"]
    assert params["dof_pos_noise_arm"] == 0.1  # :110
    assert params["dof_pos_noise_hand"] == 0.1  # :111
    assert params["dof_vel_noise"] == 0.5  # :112


def test_wrench_dr_event(base):
    wrench = base["env"]["events"]["wrench_dr"]
    # Original applies the sample every policy step (simtoolreal_env.py:64-66);
    # 0.08 is the upstream decay time constant and must stay out of the
    # interval (pipeline-audit.md F8).
    assert wrench["interval_range_s"] == [0.016666666666666666, 0.016666666666666666]
    params = wrench["params"]
    assert params["force_scale"] == 20.0  # :160
    assert params["force_prob_range"] == [0.001, 0.1]  # :161
    assert params["torque_scale"] == 2.0  # :166
    assert params["torque_prob_range"] == [0.001, 0.1]  # :167
    assert params["only_when_lifted"] is True  # :164/:170
    # Upstream decays are 0.0 (SimToolReal.yaml:162/:168) and come from the
    # term defaults; an explicit nonzero value here would be a drift.
    assert "force_decay" not in params
    assert "torque_decay" not in params


# --- terminations (SimToolReal.yaml:129-137; termination_utils constants) ----


def test_termination_terms(base):
    terminations = base["env"]["terminations"]
    assert terminations["success_tracker"]["params"] == {
        "success_steps": 10,  # :133
        "force_consecutive": False,  # :135
        "keypoint_scale": 1.5,  # :86
        "fixed_size": [0.141, 0.03025, 0.0271],  # :88
    }
    # Constants promoted to cfg parameters (termination_utils.py:54-55, :62).
    assert terminations["fall"]["params"] == {"fall_height": 0.1}
    assert terminations["hand_far"]["params"] == {"hand_far_distance": 1.5}
    assert terminations["max_consecutive_successes"]["params"] == {
        "max_consecutive_successes": 50  # :134
    }
    assert terminations["time_out"]["time_out"] is True


# --- rewards (SimToolReal.yaml:84-99) ----------------------------------------


def test_reward_terms(base):
    reward = base["reward"]
    assert reward["lift_progress"]["weight"] == 20.0  # :91
    assert reward["lift_progress"]["params"] == {"bonus_threshold": 0.15}  # :93
    assert reward["lifting_bonus"]["weight"] == 300.0  # :92
    assert reward["lifting_bonus"]["params"] == {"bonus_threshold": 0.15}  # :93
    assert reward["fingertip_distance_delta"]["weight"] == 50.0  # :95
    assert reward["fingertip_distance_delta"]["params"] == {"bonus_threshold": 0.15}
    assert reward["keypoint_distance_delta"]["weight"] == 200.0  # :85
    assert reward["keypoint_distance_delta"]["params"] == {
        "bonus_threshold": 0.15,
        "keypoint_scale": 1.5,  # :86
        "fixed_size": [0.141, 0.03025, 0.0271],  # :88
    }
    assert reward["arm_joint_velocity_penalty"]["weight"] == -0.03  # :98
    assert reward["hand_joint_velocity_penalty"]["weight"] == -0.003  # :99
    assert reward["reach_bonus"]["weight"] == 1000.0  # :96
    assert reward["reach_bonus"]["params"] == {
        "success_steps": 10,  # :133
        "force_consecutive": False,  # :135
    }


# --- scene entity assets (SimToolReal.yaml:47-58) ----------------------------


def test_entity_assets_materialization_and_friction(base):
    assets = {spec["name"]: spec for spec in base["env"]["scene"]["entity_assets"]}
    assert set(assets) == {"robot", "table", "object", "goalviz"}
    assert assets["robot"]["materialization"] == "articulation"
    assert assets["robot"]["root_mode"] == "fixed"  # fix_base=True, scene_utils.py:1723
    assert assets["table"]["materialization"] == "rigid"
    assert assets["table"]["root_mode"] == "kinematic"  # scene_utils.py:1752-1758
    assert assets["object"]["materialization"] == "rigid"
    assert assets["object"]["root_mode"] == "floating"  # dynamic pool, :1701-1713
    assert assets["goalviz"]["materialization"] == "rigid"
    assert assets["goalviz"]["root_mode"] == "kinematic"  # scene_utils.py:1714-1719
    # Static per-material frictions (SimToolReal.yaml:55-58 as [s, s, 0.0]).
    assert assets["robot"]["contact_friction"] == [0.5, 0.5, 0.0]
    fingertip = assets["robot"]["contact_friction_by_body"]
    assert [entry["body_name"] for entry in fingertip] == [
        "left_index_DP",
        "left_middle_DP",
        "left_ring_DP",
        "left_thumb_DP",
        "left_pinky_DP",
    ]
    assert all(entry["friction"] == [1.5, 1.5, 0.0] for entry in fingertip)  # :56
    assert assets["table"]["contact_friction"] == [0.5, 0.5, 0.0]  # :58
    assert assets["object"]["contact_friction"] == [0.5, 0.5, 0.0]  # :57


# --- scene composition declarations (de-hardcoded worker contract) -----------


def test_scene_physx_declaration_matches_original_repository(base):
    """The original `_default_sim_cfg` PhysxCfg, moved verbatim to the owner.

    simtoolreal_env_cfg.py:515-535: solver TGS, iteration clamps 8/8-0/0,
    bounce threshold 0.2, and the raised GPU contact stream buffers (2^24
    contacts / 2^23 patches; Isaac Lab's defaults overflow at this task's
    env counts).  The friction offset/correlation distances equal Isaac
    Lab's defaults and stay explicit to mirror the source.  The worker keeps
    Isaac Lab defaults when the scene declares none of this.
    """
    assert base["env"]["scene"]["physx"] == {
        "solver_type": 1,
        "min_position_iteration_count": 8,
        "max_position_iteration_count": 8,
        "min_velocity_iteration_count": 0,
        "max_velocity_iteration_count": 0,
        "bounce_threshold_velocity": 0.2,
        "friction_offset_threshold": 0.04,
        "friction_correlation_distance": 0.025,
        "gpu_max_rigid_contact_count": 2**24,
        "gpu_max_rigid_patch_count": 2**23,
    }


def test_env_grid_spacing_matches_original_repository(base):
    # SimToolReal.yaml:34 / simtoolreal_env_cfg.py:572 (`env_spacing=1.2`);
    # undeclared scenes keep the worker's native 2.0 m grid.
    assert base["env"]["scene"]["env_grid_spacing"] == 1.2


def test_robot_spawn_pose_declaration(base):
    # scene_utils.py:1811-1842 (pos at :1821): the iiwa stands 0.8 m behind
    # the table with identity orientation.  The fixed-base root pose has no
    # other write channel, so this declaration is load-bearing.
    robot = {
        spec["name"]: spec for spec in base["env"]["scene"]["entity_assets"]
    }["robot"]
    assert robot["init_state"] == {
        "pos": [0.0, 0.8, 0.0],
        "rot_wxyz": [1.0, 0.0, 0.0, 0.0],
    }


def test_kinematic_role_contracts_are_declared_not_named(base):
    """Table/goalviz semantics ride declarations, not entity names.

    The table keeps its converted-USD collision state (scene_utils.py:
    1752-1758 — undeclared) and the goalviz disables collision with
    object-style capsule replacement while mirroring the pool target's
    variants (scene_utils.py:1714-1719); the object pins the capsule
    replacement it always had (scene_utils.py:1701-1706).
    """
    assets = {spec["name"]: spec for spec in base["env"]["scene"]["entity_assets"]}
    assert "collision_enabled" not in assets["table"]
    assert "replace_cylinders_with_capsules" not in assets["table"]
    assert assets["goalviz"]["collision_enabled"] is False
    assert assets["goalviz"]["replace_cylinders_with_capsules"] is True
    assert assets["goalviz"]["mirrors_fixed_variant_pool"] is True
    assert assets["object"]["replace_cylinders_with_capsules"] is True
    assert "mirrors_fixed_variant_pool" not in assets["object"]
    assert assets["object"]["consumes_fixed_variant_pool"] is True


# --- scene ground declaration (Fix-C; scene_utils.py setup_scene step 5) -----


def test_scene_declares_default_ground_plane(base):
    """The owner YAML turns on the declarative ground with pure defaults.

    An empty mapping means the ``GroundPlaneCfg`` defaults the original
    task-level scene composition spawned (scene_utils.py setup_scene step 5)
    and the migration L3 run trained against.  The isaacsim worker spawns its
    local collision ground only for declared scenes, so this declaration is
    what keeps the task's trained-identical ground; any value here would be a
    physics change.
    """
    assert base["env"]["scene"]["ground_plane"] == {}


def test_owner_yaml_materializes_declared_ground_plane(base):
    """The YAML -> SceneCfg construction path flows the declaration natively.

    Trace (Fix-C): the owner YAML ``env`` section becomes a plain dict in
    ``BackendAdapter.build_task_env_cfg_override`` (config_adapter.py) and is
    applied by ``registry.make`` through ``apply_cfg_overrides``
    (config_materialization.py).  Recursing into the existing ``SceneCfg``,
    the ``ground_plane`` mapping materializes via the single-concrete-candidate
    branch (``GroundPlaneSceneCfg | None`` annotation, existing value None),
    so an empty mapping constructs the all-defaults declaration with no
    adapter-side field-by-field wiring to patch.
    """
    from omegaconf import OmegaConf

    from unilab.base import registry
    from unilab.base.registry import apply_cfg_overrides
    from simtoolreal_unilab.tasks.simtoolreal.scene import register_resolvers

    register_resolvers()
    cfg = registry.materialize_env_config("SimToolReal")
    env = OmegaConf.to_container(OmegaConf.create(base)["env"], resolve=True)
    apply_cfg_overrides(cfg, env)

    ground = cfg.scene.ground_plane
    assert ground is not None
    # GroundPlaneCfg physics-material defaults (scene_utils.py:1807).
    assert ground.friction == (0.5, 0.5, 0.0)
    assert ground.restitution == 0.0
    assert ground.size_m == 200.0


def test_undeclared_scenes_have_no_ground_plane():
    """Scenes without a declaration keep the default None (legacy behavior)."""
    from unilab.base import registry
    from unilab.base.scene import SceneCfg

    assert SceneCfg(model_file="scene.xml").ground_plane is None

    registry.ensure_registries()
    for task_name in ("Go2FootStand", "FR3JointTarget"):
        other = registry.materialize_env_config(task_name)
        scene = getattr(other, "scene", None)
        assert getattr(scene, "ground_plane", None) is None, task_name


# --- tool pool knobs (SimToolReal.yaml:50-52; generate_objects.py:27) --------


def test_tool_pool_knobs(base):
    assert base["env"]["tool_pool_num_per_type"] == 100  # :52
    assert base["env"]["tool_pool_seed"] == 42  # generate_objects.py:27
    assert base["env"]["tool_pool_target_entity"] == "object"
    assert base["env"]["tool_pool_enabled"] is True


# --- isaacsim backend leaf (SimToolReal.yaml:33; env_cfg.py:571) -------------


def test_isaacsim_backend_leaf(isaacsim):
    assert isaacsim["training"]["task_name"] == "SimToolReal"
    assert isaacsim["training"]["sim_backend"] == "isaacsim"
    assert isaacsim["algo"]["num_envs"] == 24576  # SimToolReal.yaml:33
    assert isaacsim["env"]["isaacsim_device_id"] == 0


# --- robot actuator gain overrides (scene_utils.py:59-131) ------------------


def test_robot_actuator_gain_overrides(base):
    """Robot PD gains must reach the INIT overrides or the arm free-falls.

    The URDF scan yields no actuator gains, so without an explicit
    ``actuator_gain_overrides`` block every joint materializes with zero
    stiffness/damping and the arm sinks under gravity at constant speed
    (probe_reset_lift_latch.py, 2026-09-14: palm dropped 0.81 m -> 0.62 m with
    sustained ~0.35 rad/s joint speed under zero actions; the original holds).
    The YAML values must equal the typed config tables, which mirror the
    original scene_utils.py:59-131 dictionaries byte-for-byte (AST-diffed
    2026-09-14).
    """
    from simtoolreal_unilab.tasks.simtoolreal.config import (
        ARM_JOINT_DAMPING,
        ARM_JOINT_STIFFNESS,
        HAND_JOINT_ARMATURE,
        HAND_JOINT_DAMPING,
        HAND_JOINT_STIFFNESS,
    )

    robot_assets = [
        asset
        for asset in base["env"]["scene"]["entity_assets"]
        if asset["name"] == "robot"
    ]
    assert len(robot_assets) == 1
    overrides = robot_assets[0]["actuator_gain_overrides"]
    by_joint = {entry["joint_name"]: entry for entry in overrides}

    expected: dict[str, dict] = {}
    for name, stiffness in ARM_JOINT_STIFFNESS.items():
        expected[name] = {"stiffness": stiffness, "damping": ARM_JOINT_DAMPING[name]}
    for name, stiffness in HAND_JOINT_STIFFNESS.items():
        entry = {"stiffness": stiffness, "damping": HAND_JOINT_DAMPING[name]}
        armature = HAND_JOINT_ARMATURE.get(name)
        if armature is not None:
            entry["armature"] = armature
        expected[name] = entry

    assert set(by_joint) == set(expected)
    for name, want in expected.items():
        got = by_joint[name]
        assert got["stiffness"] == want["stiffness"], name
        assert got["damping"] == want["damping"], name
        assert got.get("armature") == want.get("armature"), name
        assert got.get("frictionloss") is None, name

    # Spot anchors against the original tables (scene_utils.py:60, :72, :98).
    assert ARM_JOINT_STIFFNESS["iiwa14_joint_1"] == 600.0
    assert ARM_JOINT_STIFFNESS["iiwa14_joint_7"] == 200.0
    assert HAND_JOINT_STIFFNESS["left_1_thumb_CMC_FE"] == 6.95
