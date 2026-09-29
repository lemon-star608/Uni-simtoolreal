"""Owner-YAML physics values consumed by the MJCF writers (D6).

On the MuJoCo/Isaac MJCF lines the MJCF carries gains and friction natively,
so the frozen values declared by the owner are parsed here and injected at
asset generation time.  The declaration lives in the backend-neutral base
task config (``conf/rlgames_sapg/task/simtoolreal/base.yaml``) — task truth
data kept out of either backend leaf and out of the B-form scene declaration
(the pinned SceneEntitySpec owns no such fields).  Nothing in this module
restates a number: every value is read from the owner YAML, and
``tests/tasks/test_simtoolreal_owner_yaml_parity.py`` pins that YAML against
the original repository's frozen tables.

The one exception is the joint velocity limits: MJCF has no
joint-velocity-limit concept, so they cannot ride the asset and are declared
on the IsaacSim leaf's robot entity (``joint_velocity_limits``,
unilabsim/unisim#323).  Their truth source is the vendored URDF's
``<limit velocity>`` table — the same table the target line's URDF importer
authors as ``physxJoint:maxJointVelocity`` — parsed here so no number is
restated in either leaf.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from omegaconf import OmegaConf

from ...assets import ASSET_ROOT

BASE_YAML = (
    Path(__file__).resolve().parents[2]
    / "conf"
    / "rlgames_sapg"
    / "task"
    / "simtoolreal"
    / "base.yaml"
)

# Vendored robot description; the joint velocity-limit truth source (the
# target line's URDF importer authors physxJoint:maxJointVelocity from it).
# Kept in sync with robot_mjcf_writer.ROBOT_URDF.
ROBOT_URDF = (
    ASSET_ROOT / "urdf" / "kuka_sharpa_description" / "iiwa14_left_sharpa_adjusted_restricted.urdf"
)

# Compatibility name for callers that only need the config containing
# ``owner_physics``.  The MJCF writers intentionally do not read a backend leaf.
OWNER_YAML = BASE_YAML

# MuJoCo torsional/rolling friction defaults.  The owner YAML only pins the
# sliding term (its Isaac-side [s, s, 0] triple has no torsion/roll
# counterpart), so these stay at the MuJoCo defaults — tuning items.
TORSIONAL_FRICTION_DEFAULT = 0.005
ROLLING_FRICTION_DEFAULT = 0.0001


@dataclass(frozen=True, slots=True)
class ActuatorGains:
    """One joint's PD gains + armature from ``actuator_gain_overrides``."""

    stiffness: float
    damping: float
    armature: float  # 0.0 when the owner entry declares none (arm joints)


@dataclass(frozen=True, slots=True)
class OwnerPhysics:
    """Frozen owner values the MJCF asset layer embeds (all tuning items)."""

    actuator_gains: dict[str, ActuatorGains]
    robot_slide_friction: float
    fingertip_slide_friction: float
    fingertip_body_names: tuple[str, ...]
    object_slide_friction: float
    table_slide_friction: float
    sim_dt: float


@lru_cache(maxsize=1)
def load_owner_physics(path: Path = BASE_YAML) -> OwnerPhysics:
    """Parse the owner YAML once and return its physics values."""
    conf = OmegaConf.load(path)
    # The frozen values live in the backend-neutral base task config.  This
    # keeps the MJCF writer usable by either backend without importing an
    # IsaacSim (or MuJoCo) leaf.
    physics = conf.owner_physics
    assets = {name: physics[name] for name in ("robot", "table", "object")}
    # sim_dt lives in the backend-neutral base.yaml, not the leaf.
    sim_dt = float(conf.env.sim_dt)

    gains: dict[str, ActuatorGains] = {}
    for entry in assets["robot"].actuator_gain_overrides:
        gains[str(entry.joint_name)] = ActuatorGains(
            stiffness=float(entry.stiffness),
            damping=float(entry.damping),
            armature=float(entry.get("armature", 0.0)),
        )

    fingertip = assets["robot"].contact_friction_by_body
    fingertip_frictions = {float(entry.friction[0]) for entry in fingertip}
    if len(fingertip_frictions) != 1:
        raise ValueError(f"owner fingertip frictions disagree across bodies: {fingertip_frictions}")

    return OwnerPhysics(
        actuator_gains=gains,
        # Isaac-side triples are [slide, slide, 0]; only the sliding term has
        # a MuJoCo counterpart (torsion/roll keep the MuJoCo defaults above).
        robot_slide_friction=float(assets["robot"].contact_friction[0]),
        fingertip_slide_friction=fingertip_frictions.pop(),
        fingertip_body_names=tuple(str(entry.body_name) for entry in fingertip),
        object_slide_friction=float(assets["object"].contact_friction[0]),
        table_slide_friction=float(assets["table"].contact_friction[0]),
        sim_dt=sim_dt,
    )


@lru_cache(maxsize=1)
def load_joint_velocity_limits(path: Path = ROBOT_URDF) -> dict[str, float]:
    """Parse every articulated joint's velocity limit (rad/s) from the URDF.

    The IsaacSim leaf declares the returned table as the robot entity's
    ``joint_velocity_limits`` (unilabsim/unisim#323); PhysX brakes joints at
    these caps, matching the target line's URDF-importer behavior.  Fail
    closed if any revolute/prismatic joint lacks a positive finite limit.
    """
    root = ET.parse(path).getroot()
    limits: dict[str, float] = {}
    for joint in root.iter("joint"):
        if joint.get("type") not in ("revolute", "prismatic"):
            continue
        name = joint.get("name")
        limit = joint.find("limit")
        velocity = None if limit is None else limit.get("velocity")
        value = float(velocity) if velocity is not None else float("nan")
        if not name or not value > 0.0:
            raise ValueError(f"robot URDF joint {name!r} has no positive velocity limit")
        limits[name] = value
    if not limits:
        raise ValueError(f"robot URDF {path} declares no articulated joints")
    return limits


__all__ = [
    "ActuatorGains",
    "BASE_YAML",
    "OWNER_YAML",
    "OwnerPhysics",
    "ROBOT_URDF",
    "ROLLING_FRICTION_DEFAULT",
    "TORSIONAL_FRICTION_DEFAULT",
    "load_joint_velocity_limits",
    "load_owner_physics",
]
