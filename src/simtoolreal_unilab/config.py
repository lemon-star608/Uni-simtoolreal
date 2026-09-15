"""Owner configuration routing for the SimToolReal UniLab task package."""

from pathlib import Path

CONF_ROOT = Path(__file__).parent / "conf/rlgames_sapg"

TASK_OWNERS = {
    "SimToolReal": "simtoolreal",
}

TASK_BACKENDS = {
    "SimToolReal": "isaacsim",
}
