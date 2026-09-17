"""Owner configuration routing for the SimToolReal UniLab task package."""

from pathlib import Path

CONF_ROOT = Path(__file__).parent / "conf/rlgames_sapg"

TASK_OWNERS = {
    "SimToolReal": "simtoolreal",
}

# Registered backends per task, default first.  isaacsim is the merge anchor;
# mujoco is the MJCF asset line (M-J2).
TASK_BACKENDS = {
    "SimToolReal": ("isaacsim", "mujoco"),
}
