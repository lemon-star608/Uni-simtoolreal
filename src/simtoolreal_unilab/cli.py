"""Strict SimToolReal command routing; training stays in the owner script.

Mirrors the wuji_unilab CLI pattern: an argparse front end that owns the
task/backend routing, rejects overrides that would tamper with it, generates
the Hydra overrides, and runs the training module with ``--config-dir``
pointing at this package's conf root.
"""

import argparse
import runpy
import sys

from simtoolreal_unilab.config import CONF_ROOT, TASK_BACKENDS, TASK_OWNERS


def _run(mode):
    parser = argparse.ArgumentParser(prog=f"simtoolreal-{mode}")
    parser.add_argument("--task", choices=TASK_OWNERS, default="SimToolReal")
    parser.add_argument("--checkpoint-file")
    args, overrides = parser.parse_known_args()
    for value in overrides:
        key = value.lstrip("+~").split("=", 1)[0]
        if key in ("task", "training.sim_backend", "training.task_name", "algo.algo"):
            parser.error(f"Cannot override task/backend routing via {key}")
    backend = TASK_BACKENDS[args.task]
    generated = [f"task={TASK_OWNERS[args.task]}/{backend}"]
    if args.checkpoint_file:
        # Playback resolves a run directory or checkpoint file directly.
        generated += [f"algo.load_run={args.checkpoint_file}"]
    if mode == "play":
        generated += ["training.play_only=true"]
    from simtoolreal_unilab._isaacsim_worker import ensure_worker_runtime

    ensure_worker_runtime()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "SimToolReal training runs the RL-Games learner on CUDA and the "
            "IsaacSim worker on the same GPU; CPU fallback is not supported"
        )
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


def list_main():
    for task, backend in TASK_BACKENDS.items():
        print(f"{task}\t{backend}")
