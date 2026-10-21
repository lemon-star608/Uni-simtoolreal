# simtoolreal_unilab

[中文](README_zh.md)

SimToolReal teacher-policy training migrated onto [UniLab](https://github.com/Motphys/UniLab) + [UniSim](https://github.com/unilabsim/unisim): a KUKA iiwa14 arm (7 DoF) with a SHARPA dexterous hand (22 DoF) learns to lift and manipulate tools from a 1,200-variant procedurally generated tool pool, matching the [original repository](https://github.com/tylerlum/simtoolreal)'s training semantics. Both physics backends — MuJoCo (mjbatch VariantPack) and IsaacSim (mapped subprocess worker) — run on the same MJCF asset set, generated once per entity (robot/table/floor plus one file per tool); the learner is a vendored, SHA-gated RL-Games SAPG runtime.

<!-- 训练回放动图待补（simtoolreal-play 录制） -->

## Capabilities

| Capability | Status |
| --- | --- |
| `SimToolReal` teacher training (RL-Games SAPG) | Full-scale runs on both backends; task/learner semantics label-compared against the original repository |
| `DexToolBench` interactive evaluation | All 24 upstream object/task combinations as committed static MJCFs on the MuJoCo backend, with upstream no-DR eval settings and Viser launcher layout |
| Physics backends | MuJoCo (default, pure CPU) and IsaacSim (GPU worker); one shared asset/task definition, switchable via `--backend` |
| Label parity vs the original IsaacLab implementation | Key TensorBoard label families indistinguishable on a same-seed 12,288-env run |
| Playback / inspection | `simtoolreal-play`, TensorBoard overlay with the original tag set |
| Student / camera / sensor policies, deployment, sim2real | Not carried over; this repository contains no deployment code |

Paper-reported numbers (e.g. sim2real) belong to the original implementation and are not re-validated here.

## First run

Requires Linux x86_64, an NVIDIA GPU with CUDA (the learner and the IsaacSim worker share it), [uv](https://docs.astral.sh/uv/), and ~20 GB of disk for the worker runtime and caches.

```bash
git clone https://github.com/unilabsim/simtoolreal_unilab.git && cd simtoolreal_unilab
uv sync
uv run simtoolreal-assets     # verify the vendored assets
uv run simtoolreal-list-envs  # SimToolReal  mujoco / SimToolReal  isaacsim
```

All dependencies resolve from the lockfile (unilab, unisim-core and mjbatch-uni from PyPI; torch 2.14.0+cu130) — no extra checkouts. This repository is source-install only and is not published to PyPI.

One piece cannot ship inside `uv sync`: the IsaacSim worker runtime (NVIDIA's SDK is not redistributed). Install it once with `bash scripts/setup_isaacsim_env.sh` (vendored from UniLab's official installer, IsaacSim 5.1.0 + IsaacLab 2.3.0, idempotent); `simtoolreal-train`/`simtoolreal-play` detect it automatically. Escape hatch for another IsaacSim-capable interpreter: `export UNISIM_ISAACSIM_PYTHON=/path/to/venv/bin/python`.

Robot/table source URDFs ship under `assets/simtoolreal/` (upstream MIT, SHA-256 manifest) and are serialized once into entity MJCFs at first use; the 1,200-tool MJCF pool is generated procedurally and cached under `/tmp` (override: `SIMTOOLREAL_TOOL_POOL_ROOT`).

## Training and evaluation

```bash
# Production training (MuJoCo default, pure CPU; both backends default to
# 24,576 envs = 6 x 4,096 exploration blocks over the 1,200-tool pool)
uv run simtoolreal-train

# IsaacSim backend (same task definition; needs the worker runtime and a GPU)
uv run simtoolreal-train --backend isaacsim

# Acceptance smoke (64 envs, 24-tool pool, 2 epochs — minutes; never a policy baseline)
uv run simtoolreal-train \
  algo.num_envs=64 env.tool_pool_num_per_type=2 \
  rl_games.params.config.max_epochs=2 \
  rl_games.params.config.expl_coef_block_size=32 \
  rl_games.params.config.minibatch_size=256 \
  rl_games.params.config.central_value_config.minibatch_size=256 \
  training.smoke=true training.no_play=true

# Play back a checkpoint (mujoco playback by default; --backend isaacsim
# renders through the IsaacSim worker)
uv run simtoolreal-play --checkpoint-file logs/rlgames_sapg/SimToolReal/<run>/nn/<checkpoint>.pth

uv run tensorboard --logdir logs/rlgames_sapg
```

Outputs land in `logs/rlgames_sapg/SimToolReal/<run>/` (checkpoints, TensorBoard events, run configuration). The MuJoCo line needs no IsaacSim worker runtime; the IsaacSim line shares one GPU between the worker and the learner. Task and backend routing keys are locked by the CLI and cannot be overridden — scale and algorithm fields are Hydra overrides such as `algo.num_envs=12288 rl_games.params.config.expl_coef_block_size=2048` (num_actors must form exactly six exploration blocks).

## DexToolBench interactive demo

```bash
# Zero-action scene/task inspection
uv run simtoolreal-dextoolbench-interactive

# Deterministic policy demo
uv run simtoolreal-dextoolbench-interactive \
  --config-path /path/to/config.yaml \
  --checkpoint-path /path/to/model.pth
```

The launcher keeps the upstream category/object/task dropdowns, Viser controls,
sidebar, object/goal rendering, and worker lifecycle. Each task loads one
committed full-scene MJCF directly through `SceneCfg.model_file`; there is no
runtime MJCF writer or SimToolReal tool pool on this path. See
[`docs/dextoolbench-mujoco.md`](docs/dextoolbench-mujoco.md) for the alignment
matrix.

Alignment with the original repository is verified by `tests/dextoolbench/`
(`uv run pytest tests/dextoolbench`): all 24 committed MJCFs compile with the
upstream 29-joint layout, `startArmHigher` keyframe, and training contact/
friction channels, and the Viser goal mirror carries only the MJCF-local geom
transform under the `/goal` frame. The zero-action launcher above doubles as a
visual side-by-side check against the upstream Viser UI.

## How to modify

| What you want to change | Entry file |
| --- | --- |
| Reward terms and weights | `src/simtoolreal_unilab/tasks/simtoolreal/rewards.py`, `conf/rlgames_sapg/task/simtoolreal/base.yaml` |
| Observations / goal command / actions | `tasks/simtoolreal/observations.py`, `commands.py`, `actions.py` |
| Reset, wrench DR, terminations | `tasks/simtoolreal/events.py`, `terminations.py` |
| Tool pool and MJCF generation | `tasks/simtoolreal/tool_catalog.py`, `tool_mjcf_writer.py`, `entity_mjcf_writer.py` |
| Entity bootstrap and variant binding | `tasks/simtoolreal/scene.py`, `env_cfg.py` |
| Algorithm hyperparameters | `conf/rlgames_sapg/config.yaml` |
| Behaviors layered on pinned UniLab | `src/simtoolreal_unilab/_compat/` (removed as upstream grows them) |

Dependency pins: `unilab >= 1.3.3` (PyPI release carrying [#1620](https://github.com/Motphys/UniLab/pull/1620)/[#1636](https://github.com/Motphys/UniLab/pull/1636)), `unisim-core >= 1.7.10`, `mjbatch-uni >= 0.2.4`, `torch == 2.14.0+cu130` (unilab 1.3.3's own validation target). The vendored RL-Games runtime under `third_party/` is SHA-gated; never edit it. Run `make check` / `make test` before pushing.

## Credits

- **SimToolReal** — upstream task, assets, and training semantics ([tylerlum/simtoolreal](https://github.com/tylerlum/simtoolreal), MIT).
- **UniLab** / **UniSim** / **unilab-rl** — training infrastructure, physics-backend contract, and runtimes; **RL-Games** — the SAPG learner, vendored with an audited patch set (`third_party/simtoolreal_rl_games/PATCHES.md`).

Licensed under [Apache-2.0](LICENSE); the vendored runtime and upstream assets keep their own licenses. For framework-level problems (backends, contract), report to UniLab/UniSim instead.
