"""Owner configuration routing for the SimToolReal UniLab task package."""

from pathlib import Path

CONF_ROOT = Path(__file__).parent / "conf/rlgames_sapg"

TASK_OWNERS = {
    "SimToolReal": "simtoolreal",
}

# Registered backends per task, default first.  Both consume the shared
# entity-split MJCF B form: MuJoCo through entity composition plus its
# VariantPack runtime, and IsaacSim through UniSim's mapped worker.
TASK_BACKENDS = {
    "SimToolReal": ("mujoco", "isaacsim"),
}
