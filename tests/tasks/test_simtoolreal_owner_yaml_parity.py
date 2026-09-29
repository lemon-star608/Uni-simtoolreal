"""Pin the SimToolReal owner YAML values against the original repository.

``pipeline-audit.md`` S01 verified every owner-YAML constant against the
original repository by hand; this test freezes that result.  Expected values
are hard-coded with original-repository line references — the upstream is a
frozen snapshot (``313d5aea``), so static anchoring is sufficient and avoids
re-implementing the configclass-default + YAML-overlay merge semantics.

The owner YAML now intentionally diverges from the upstream sim2real DR and
action-smoothing defaults for the fast-iteration profile recommended in
upstream issue #26; those selected values are pinned as no-DR expectations
rather than original-repository values.

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


# --- observations (fast-iteration no-DR profile) ------------------------------


def test_observation_terms(base):
    policy = base["env"]["observations"]["policy"]["terms"]["simtoolreal"]["params"]
    # Issue #26 fast-iteration guidance: disable object-state delay/noise,
    # whole-frame observation latency, and observation noise.
    assert policy == {
        "use_object_state_delay_noise": False,
        "object_state_delay_max": 0,
        "object_state_xyz_noise_std": 0.0,
        "object_state_rotation_noise_degrees": 0.0,
        "joint_velocity_obs_noise_std": 0.0,
        "use_obs_delay": False,
        "obs_delay_max": 0,
        "clamp_abs_observations": 10.0,
    }
    critic = base["env"]["observations"]["critic"]["terms"]["simtoolreal"]["params"]
    assert critic == {"clamp_abs_observations": 10.0}


# --- actions (fast-iteration no-DR profile) -----------------------------------


def test_action_term(base):
    action = base["env"]["actions"]["joint_pos"]
    assert action["arm_moving_average"] == 1.0
    assert action["hand_moving_average"] == 1.0
    assert action["dof_speed_scale"] == 1.5  # :79
    assert action["use_action_delay"] is False
    assert action["action_delay_max"] == 0


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
    # Keep the event and its RNG stream, but sampled wrenches are exactly zero.
    assert params["force_scale"] == 0.0
    assert params["force_prob_range"] == [0.001, 0.1]  # :161
    assert params["torque_scale"] == 0.0
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


# --- scene entity assets: B-form declaration (Isaac/MJCF line) ----------------


def test_entity_assets_bform_declaration(isaacsim):
    """The B-form scene: four headless entities, official field names."""
    assets = {spec["name"]: spec for spec in isaacsim["env"]["scene"]["entity_assets"]}
    assert set(assets) == {"robot", "table", "floor", "object"}
    assert isaacsim["env"]["asset_format"] == "mjcf"
    assert isaacsim["env"]["scene"]["default_keyframe_name"] == "home"

    robot = assets["robot"]
    assert robot["kind"] == "articulation"
    assert robot["root_mode"] == "fixed"  # fix_base=True, scene_utils.py:1723
    assert robot["self_collision"] is True  # honored by the IsaacSim worker (#262)
    assert robot["gravity_disabled"] is True  # honored by the IsaacSim worker (#270)
    assert robot["initial_state"] == {
        # scene_utils.py:1811-1842 (pos at :1821): the iiwa stands 0.8 m behind
        # the table; the resolver injects the frozen value (ROBOT_SCENE_POS).
        "position": "${simtoolreal_entity_position:robot}",
        "quaternion": [1.0, 0.0, 0.0, 0.0],
    }

    assert assets["table"]["kind"] == "rigid"
    assert assets["table"]["root_mode"] == "kinematic"  # scene_utils.py:1752-1758
    assert assets["floor"]["kind"] == "rigid"
    assert assets["floor"]["root_mode"] == "kinematic"
    assert assets["object"]["kind"] == "rigid"
    assert assets["object"]["root_mode"] == "floating"  # dynamic pool, :1701-1713

    # GoalViz is a visualization-only original-scene artifact.  The headless
    # B-form intentionally does not materialize a mirrored tool prototype.
    assert "goalviz" not in assets


def test_entity_selectors_are_entity_qualified(isaacsim):
    """Entity composition qualifies every model name (compile_scene_layout)."""
    robot = isaacsim["env"]["scene"]["entities"]["robot"]
    assert robot["root_body_name"] == "robot/iiwa14_link_0"
    assert len(robot["joint_names"]) == 29
    assert all(name.startswith("robot/") for name in robot["joint_names"])
    assert robot["actuator_names"] == [
        f"robot/{name.removeprefix('robot/')}_ctrl" for name in robot["joint_names"]
    ]
    entities = isaacsim["env"]["scene"]["entities"]
    assert entities["table"]["root_body_name"] == "table/table"
    assert entities["object"]["root_body_name"] == "object/object_root"
    assert "goalviz" not in entities


def test_scene_leaves_share_the_bform_declaration(isaacsim):
    """mujoco.yaml and isaacsim.yaml declare the same composed scene.

    The only deliberate per-backend differences: self_collision is declared
    on the IsaacSim leaf only (the mujoco compose path is fail-closed on the
    toggle), gravity_disabled is declared on the IsaacSim leaf only (the
    MJCF gravcomp attribute does not survive IsaacLab's MJCF conversion,
    while the MuJoCo line gets the same semantics natively from robot.xml),
    and each leaf's YAML anchor name differs.
    """
    mujoco = yaml.safe_load((CONF_DIR / "mujoco.yaml").read_text())
    isa_assets = {
        spec["name"]: {
            k: v for k, v in spec.items() if k not in ("self_collision", "gravity_disabled")
        }
        for spec in isaacsim["env"]["scene"]["entity_assets"]
    }
    mu_assets = {spec["name"]: spec for spec in mujoco["env"]["scene"]["entity_assets"]}
    assert isa_assets == mu_assets
    # entities selectors match too, except site_names: the IsaacSim backend
    # does not implement get_site_ids and no term reads the sites, so only
    # the mujoco leaf declares them.
    isa_entities = isaacsim["env"]["scene"]["entities"]
    mu_entities = mujoco["env"]["scene"]["entities"]
    isa_normalized = {
        name: {k: v for k, v in spec.items() if k != "site_names"}
        for name, spec in isa_entities.items()
    }
    mu_normalized = {
        name: {k: v for k, v in spec.items() if k != "site_names"}
        for name, spec in mu_entities.items()
    }
    assert isa_normalized == mu_normalized
    assert "site_names" not in isa_entities["robot"]
    assert len(mu_entities["robot"]["site_names"]) == 5
    assert mujoco["env"]["scene"]["default_keyframe_name"] == "home"
    assert mujoco["env"]["asset_format"] == "mjcf"


def test_both_leaves_default_to_the_same_scale(isaacsim):
    """Both backend leaves default to 24,576 envs x 6 x 4,096 blocks."""
    mujoco = yaml.safe_load((CONF_DIR / "mujoco.yaml").read_text())
    assert mujoco["algo"]["num_envs"] == 24576
    assert mujoco["rl_games"]["params"]["config"]["expl_coef_block_size"] == 4096
    assert isaacsim["algo"]["num_envs"] == 24576


# --- owner physics truth (frozen values the MJCF writers consume) -------------


def test_owner_physics_friction_tables(base, isaacsim):
    physics = base["owner_physics"]
    # Static per-material frictions (SimToolReal.yaml:55-58 as [s, s, 0.0]).
    assert physics["robot"]["contact_friction"] == [0.5, 0.5, 0.0]
    fingertip = physics["robot"]["contact_friction_by_body"]
    assert [entry["body_name"] for entry in fingertip] == [
        "left_index_DP",
        "left_middle_DP",
        "left_ring_DP",
        "left_thumb_DP",
        "left_pinky_DP",
    ]
    assert all(entry["friction"] == [1.5, 1.5, 0.0] for entry in fingertip)  # :56
    assert physics["table"]["contact_friction"] == [0.5, 0.5, 0.0]
    assert physics["object"]["contact_friction"] == [0.5, 0.5, 0.0]
    assert "owner_physics" not in isaacsim


# --- PhysX solver knobs (simtoolreal_env_cfg.py:515-535) ----------------------


def test_isaacsim_solver_knobs_match_original_repository(isaacsim):
    """The original training's PhysX solver table, through UniLab#1620's channel.

    simtoolreal_env_cfg.py:515-535 pins iteration clamps 8/8-0/0 and bounce
    threshold 0.2 explicitly; contact/rest offsets and the depenetration cap
    are the Isaac Lab defaults the original relied on (contact_offset 0.002,
    rest_offset 0.0, max_depenetration_velocity 1000).  The GPU contact
    buffers ride unilabsim/unisim#292's authored-report knobs at the original
    2^24/2^23 ("Lab defaults overflow: Patch buffer overflow detected kills
    training").  solver_type=TGS and the friction offsets still have no
    UniSim channel — defaults apply there.
    """
    assert isaacsim["env"]["isaacsim_solver_position_iteration_count"] == 8
    assert isaacsim["env"]["isaacsim_solver_velocity_iteration_count"] == 0
    assert isaacsim["env"]["isaacsim_bounce_threshold_velocity"] == 0.2
    assert isaacsim["env"]["isaacsim_contact_offset"] == 0.002
    assert isaacsim["env"]["isaacsim_rest_offset"] == 0.0
    assert isaacsim["env"]["isaacsim_max_depenetration_velocity"] == 1000.0
    assert isaacsim["env"]["isaacsim_gpu_max_rigid_contact_count"] == 2**24
    assert isaacsim["env"]["isaacsim_gpu_max_rigid_patch_count"] == 2**23


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
    assert isaacsim["env"]["asset_format"] == "mjcf"  # Isaac/MJCF line
    assert isaacsim["env"]["isaacsim_device_id"] == 0
    assert isaacsim["env"]["isaacsim_worker_timeout_s"] == 1800.0


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

    overrides = base["owner_physics"]["robot"]["actuator_gain_overrides"]
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
