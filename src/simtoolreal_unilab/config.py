"""Fixed owner configuration for the SimToolReal MuJoCo task package."""

from pathlib import Path

CONF_ROOT = Path(__file__).parent / "conf/rlgames_sapg"

SIM_BACKEND = "mujoco"
DEFAULT_TASK = f"simtoolreal/{SIM_BACKEND}"
