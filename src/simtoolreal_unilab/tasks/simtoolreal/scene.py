"""Cold-path scene inputs for SimToolReal.

The manager-based factory keeps scene roles in the owner YAML; this module
owns the OmegaConf resolvers the owner YAML uses at Hydra time: the entity
MJCF bootstrap sources (``simtoolreal_entity_mjcf``) and the frozen entity
spawn positions (``simtoolreal_entity_position``).  The full 1200-file tool
pool remains an init-randomization adapter concern.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

_ENTITY_MJCF_RESOLVER_NAME = "simtoolreal_entity_mjcf"
_ENTITY_POSITION_RESOLVER_NAME = "simtoolreal_entity_position"


def _resolve_entity_bootstrap(name: str) -> str:
    """Materialize the pool-independent entity sources (B form, M-J5).

    Robot/table/floor plus one canonical bootstrap object (catalog hammer):
    the owner YAML's entity_assets
    resolve these at Hydra time, before pool knobs settle; each env's object
    identity comes from the entity variant binding, not this file.
    """
    if name not in {"robot", "table", "floor", "object"}:
        raise ValueError(
            "unknown SimToolReal entity bootstrap source; expected robot/table/floor/object, "
            f"got {name!r}"
        )
    from .entity_mjcf_writer import write_bootstrap_entity_set

    root = (
        Path(os.environ.get("SIMTOOLREAL_GENERATED_ASSETS_ROOT", "")).expanduser()
        if os.environ.get("SIMTOOLREAL_GENERATED_ASSETS_ROOT", "").strip()
        else Path(tempfile.gettempdir()) / "simtoolreal_generated_assets"
    ) / "entity_bootstrap_v3"
    paths = {
        "robot": root / "robot.xml",
        "table": root / "table.xml",
        "floor": root / "floor.xml",
        "object": root / "object_bootstrap.xml",
    }
    if not all(path.is_file() for path in paths.values()):
        write_bootstrap_entity_set(root)
    return str(paths[name])


def _resolve_entity_position(name: str) -> list[float]:
    """Entity root spawn positions, read from the task's frozen values.

    EntityInitialState positions for the B-form scene declaration
    (``mujoco.yaml``): the robot stands 0.8 m behind the table
    (``ROBOT_SCENE_POS``), the table top sits at ``table_reset_z``, and the
    object root starts at the object home pose
    (``table_reset_z + table_object_z_offset``) — kept on one source.
    """
    from .config import ResetCfg
    from .robot_mjcf_writer import ROBOT_SCENE_POS

    reset = ResetCfg()
    object_z = reset.table_reset_z + reset.table_object_z_offset
    values = {
        "robot": list(ROBOT_SCENE_POS),
        "table": [0.0, 0.0, reset.table_reset_z],
        "floor": [0.0, 0.0, 0.0],
        "object": [0.0, 0.0, object_z],
    }
    if name not in values:
        raise ValueError(
            f"unknown SimToolReal entity position {name!r}; expected one of {sorted(values)}"
        )
    return values[name]


def register_resolvers() -> None:
    """Register the task's OmegaConf resolvers once."""
    if not OmegaConf.has_resolver(_ENTITY_MJCF_RESOLVER_NAME):
        OmegaConf.register_new_resolver(_ENTITY_MJCF_RESOLVER_NAME, _resolve_entity_bootstrap)
    if not OmegaConf.has_resolver(_ENTITY_POSITION_RESOLVER_NAME):
        OmegaConf.register_new_resolver(_ENTITY_POSITION_RESOLVER_NAME, _resolve_entity_position)


__all__ = ["register_resolvers"]
