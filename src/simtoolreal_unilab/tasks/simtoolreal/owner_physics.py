"""Owner-YAML physics values consumed by the MJCF writers (D6).

The generated MJCF carries gains and friction natively. The frozen values
declared by the owner are parsed here and injected at asset generation time.
The declaration lives in the base task config
(``conf/rlgames_sapg/task/simtoolreal/base.yaml``), keeping task truth data out
of the scene declaration. Nothing in this module restates a number: every value
is read from the owner YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from omegaconf import OmegaConf

BASE_YAML = (
    Path(__file__).resolve().parents[2]
    / "conf"
    / "rlgames_sapg"
    / "task"
    / "simtoolreal"
    / "base.yaml"
)

# Compatibility name for callers that only need the config containing
# ``owner_physics``.  The MJCF writers intentionally do not read a backend leaf.
OWNER_YAML = BASE_YAML

# MuJoCo torsional/rolling friction defaults.  The owner YAML only pins the
# sliding term, so these stay at the MuJoCo defaults — tuning items.
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
    # The frozen values live in the base task config.
    physics = conf.owner_physics
    assets = {name: physics[name] for name in ("robot", "table", "object")}
    # sim_dt lives in the base owner configuration.
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
        # The owner declares the sliding term; torsion and roll keep the
        # MuJoCo defaults above.
        robot_slide_friction=float(assets["robot"].contact_friction[0]),
        fingertip_slide_friction=fingertip_frictions.pop(),
        fingertip_body_names=tuple(str(entry.body_name) for entry in fingertip),
        object_slide_friction=float(assets["object"].contact_friction[0]),
        table_slide_friction=float(assets["table"].contact_friction[0]),
        sim_dt=sim_dt,
    )


__all__ = [
    "ActuatorGains",
    "BASE_YAML",
    "OWNER_YAML",
    "OwnerPhysics",
    "ROLLING_FRICTION_DEFAULT",
    "TORSIONAL_FRICTION_DEFAULT",
    "load_owner_physics",
]
