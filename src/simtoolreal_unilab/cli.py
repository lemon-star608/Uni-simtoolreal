"""Fixed SimToolReal MuJoCo command routing.

The CLI rejects routing overrides, generates the Hydra task override, and runs
the owner training module with this package's configuration root.
"""

import argparse
import runpy
import sys

from simtoolreal_unilab.config import CONF_ROOT, DEFAULT_TASK


def _run(mode):
    parser = argparse.ArgumentParser(prog=f"simtoolreal-{mode}")
    if mode == "play":
        parser.add_argument("--checkpoint-file")
    args, overrides = parser.parse_known_args()
    for value in overrides:
        key = value.lstrip("+~").split("=", 1)[0]
        bare_key = key.lstrip("-")
        if key in (
            "task",
            "training.sim_backend",
            "training.task_name",
            "algo.algo",
        ) or bare_key in ("task", "backend", "checkpoint-file"):
            parser.error(f"Cannot override task/backend routing via {key}")
    checkpoint_file = getattr(args, "checkpoint_file", None)
    generated = [f"task={DEFAULT_TASK}"]
    if checkpoint_file:
        # Playback resolves a run directory or checkpoint file directly.
        generated += [f"algo.load_run={checkpoint_file}"]
    if mode == "play":
        generated += ["training.play_only=true"]
    sys.argv = [
        f"simtoolreal-{mode}",
        "--config-dir",
        str(CONF_ROOT),
        *generated,
        *overrides,
    ]
    runpy.run_module("simtoolreal_unilab.scripts.train_rlgames_sapg", run_name="__main__")


def train_main():
    _run("train")


def play_main():
    _run("play")
