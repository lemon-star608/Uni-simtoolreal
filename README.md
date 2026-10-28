# SimToolReal UniLab

[中文](README_zh.md)

SimToolReal UniLab trains the SimToolReal teacher policy with RL-Games SAPG on UniLab and UniSim, then evaluates checkpoints with the 24 committed DexToolBench scenes. Training uses a KUKA iiwa14 arm and SHARPA hand over a deterministic 1,200-tool MJCF pool. Evaluation loads committed static scenes directly in MuJoCo.

## Scope

- **SimToolReal teacher training:** 24,576 parallel environments, six RL-Games exploration blocks, and the full 1,200-tool pool.
- **DexToolBench evaluation:** all 24 category/object/task combinations, interactive Viser inspection, and deterministic policy playback.
- **Fixed runtime:** UniLab TorchEnv, UniSim MuJoCo, mjbatch VariantPack, and the vendored SHA-gated RL-Games SAPG runtime.

## Requirements

- Linux x86_64
- NVIDIA GPU with CUDA 13.0 support
- [uv](https://docs.astral.sh/uv/)
- Approximately 20 GB of free disk space for dependencies and generated assets

## Quick start

```bash
git clone -b mujoco-v1 https://github.com/lemon-star608/simtoolreal_unilab.git
cd simtoolreal_unilab
uv sync
uv run simtoolreal-train
```

The default command trains at full production scale. It creates the entity MJCF pool on first use and writes progress to TensorBoard.

## Playback

Select a checkpoint produced by training:

```bash
uv run simtoolreal-play \
  --checkpoint-file logs/rlgames_sapg/SimToolReal/<run>/nn/<checkpoint>.pth
```

View training curves and run configuration:

```bash
uv run tensorboard --logdir logs/rlgames_sapg
```

## DexToolBench

Open the interactive zero-action viewer:

```bash
uv run simtoolreal-dextoolbench-interactive
```

Run deterministic policy evaluation with a training run configuration and checkpoint:

```bash
uv run simtoolreal-dextoolbench-interactive \
  --config-path logs/rlgames_sapg/SimToolReal/<run>/run_config.json \
  --checkpoint-path /path/to/checkpoint.pth
```

The launcher provides the upstream category/object/task selectors, Viser controls, object and goal rendering, and evaluation lifecycle. See [DexToolBench MuJoCo evaluation](docs/dextoolbench-mujoco.md).

## Outputs

- Training runs: `logs/rlgames_sapg/SimToolReal/<run>/`
- Checkpoints: `logs/rlgames_sapg/SimToolReal/<run>/nn/`
- TensorBoard events and the resolved `run_config.json` accompany each run
- Generated tool pools default to the system temporary directory; set `SIMTOOLREAL_TOOL_POOL_ROOT` to choose another location

## Verification

```bash
make check
make test
uv run simtoolreal-assets
```

Full acceptance also trains with the default command until RL-Games saves a persistent checkpoint, then loads that checkpoint through the DexToolBench policy path. See [Verification](docs/verification.md).

## Documentation

- [Architecture](docs/architecture.md)
- [Verification](docs/verification.md)
- [DexToolBench MuJoCo evaluation](docs/dextoolbench-mujoco.md)
