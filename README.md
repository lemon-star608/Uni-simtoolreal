# simtoolreal_unilab

[中文](README_zh.md)

SimToolReal teacher-policy training migrated onto [UniLab](https://github.com/Motphys/UniLab) + [UniSim](https://github.com/unilabsim/unisim): a KUKA iiwa14 arm (7 DoF) with a SHARPA hand (22 DoF) learns to lift and manipulate tools from a 1,200-variant procedurally generated tool pool, matching the [original repository](https://github.com/tylerlum/simtoolreal)'s training semantics. Both physics backends — IsaacSim (mapped subprocess worker) and MuJoCo (mjbatch VariantPack) — start from the same entity-split MJCF B-form asset set; the learner is a vendored, SHA-gated RL-Games SAPG runtime.

| Capability | Status |
| --- | --- |
| `SimToolReal` teacher training (RL-Games SAPG) | Full-scale runs on both backends; task/learner semantics label-compared against the original repository (see Evidence) |
| Physics backends | IsaacSim (default) and MuJoCo, one shared asset/task definition, switchable via `--backend` |
| Label parity vs the original IsaacLab implementation | Key TensorBoard label families indistinguishable on a same-seed 12,288-env run (validated route; see Evidence) |
| Playback / inspection | `simtoolreal-play`, TensorBoard overlay with the original tag set |
| Student / camera / sensor policies, deployment, sim2real | Not carried over; this repository contains no deployment code |

## First run

Requires Linux x86_64, an NVIDIA GPU with CUDA (the learner and the IsaacSim worker share it), [uv](https://docs.astral.sh/uv/), and ~20 GB of disk for the worker runtime and caches. Dependencies resolve from the lockfile (unisim-core and mjbatch-uni from PyPI; unilab pinned to an upstream git revision) — no extra checkouts.

```bash
git clone <this repo> && cd simtoolreal_unilab
uv sync
uv run simtoolreal-assets    # verify the vendored assets
```

One piece cannot ship inside `uv sync`: the IsaacSim worker runtime (NVIDIA's SDK is not redistributed). Install it once with UniLab's official installer (`scripts/tools/setup_isaacsim_env.sh` in a UniLab checkout, IsaacSim 5.1.0 + IsaacLab 2.3.0, idempotent); `simtoolreal-train`/`simtoolreal-play` detect it automatically. Escape hatch for another IsaacSim-capable interpreter: `export UNISIM_ISAACSIM_PYTHON=/path/to/venv/bin/python`.

Robot/table source URDFs ship under `assets/simtoolreal/` (upstream MIT, SHA-256 manifest) and are serialized once into entity MJCFs at first use; the 1,200-tool MJCF pool is generated procedurally and cached under `/tmp` (override: `SIMTOOLREAL_TOOL_POOL_ROOT`).

## Training and evaluation

```bash
# Production training (IsaacSim default; num_actors must form exactly six exploration blocks)
uv run simtoolreal-train algo.num_envs=12288 rl_games.params.config.expl_coef_block_size=2048

# MuJoCo backend (pure CPU; same B-form task)
uv run simtoolreal-train --backend mujoco

# Acceptance smoke (64 envs, 24-tool pool, 2 epochs — minutes; never a policy baseline)
uv run simtoolreal-train \
  algo.num_envs=64 env.tool_pool_num_per_type=2 \
  rl_games.params.config.max_epochs=2 \
  rl_games.params.config.expl_coef_block_size=32 \
  rl_games.params.config.minibatch_size=256 \
  rl_games.params.config.central_value_config.minibatch_size=256 \
  training.smoke=true training.no_play=true

# Play back a checkpoint (renders through the IsaacSim worker)
uv run simtoolreal-play --checkpoint-file logs/rlgames_sapg/SimToolReal/<run>/nn/<checkpoint>.pth

uv run simtoolreal-list-envs                 # SimToolReal  isaacsim / SimToolReal  mujoco
uv run tensorboard --logdir logs/rlgames_sapg
```

Outputs land in `logs/rlgames_sapg/SimToolReal/<run>/` (checkpoints, TensorBoard events, run configuration). The MuJoCo line needs no IsaacSim worker runtime.

## Evidence

- **Training parity (L3)**: on the previously validated IsaacSim route, a full 12,288-env seed-42 run (514 epochs) was compared against the original repository's first 500 epochs — all key TensorBoard label families indistinguishable. The current MJCF B-form route preserves those task and learner code paths.
- **Per-variant mass readback** vs the generated manifest: max relative deviation 5.7e-08; **world-frame wrench path** bit-identical to the frozen revalidation anchors.
- Real-backend smoke gates: 8-environment IsaacSim mapped-worker smoke; MuJoCo reset and 100-step smoke.
- Verified on Linux x86_64, one RTX 4090, IsaacSim 5.1.0 + IsaacLab 2.3.0 (official UniLab default stack) as the worker runtime.
- No robot video is included yet; a recording of a trained lift episode is the top missing piece for this README.

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

Dependency pins: `unilab` tracks upstream main @ `838cc03e` (git rev, carrying [#1620](https://github.com/Motphys/UniLab/pull/1620)/[#1636](https://github.com/Motphys/UniLab/pull/1636); switches to PyPI once a release contains them), `unisim-core >= 1.7.5`, `mjbatch-uni >= 0.2.4`. The vendored RL-Games runtime under `third_party/` is SHA-gated; never edit it. Run `make check` / `make test` before pushing.

## Credits

- **SimToolReal** — upstream task, assets, and training semantics ([tylerlum/simtoolreal](https://github.com/tylerlum/simtoolreal), MIT); paper-reported numbers (e.g. sim2real) belong to the original implementation and are not re-validated here.
- **UniLab** / **UniSim** / **unilab-rl** — training infrastructure, physics-backend contract, and runtimes; **RL-Games** — the SAPG learner, vendored with an audited patch set (`third_party/simtoolreal_rl_games/PATCHES.md`).

Licensed under [Apache-2.0](LICENSE); the vendored runtime and upstream assets keep their own licenses. For framework-level problems (backends, contract), report to UniLab/UniSim instead.
