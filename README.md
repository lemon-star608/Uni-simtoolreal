# simtoolreal_unilab

SimToolReal teacher-policy training on [UniLab](https://github.com/Motphys/UniLab) + [UniSim](https://github.com/unilabsim/unisim): a KUKA iiwa14 arm (7 DoF) with a SHARPA hand (22 DoF) learns to lift and manipulate tools from a 1,200-variant procedurally generated tool pool, matching the original repository's training semantics.

This is a downstream task package: install it and use UniLab's shared training tooling without cloning UniLab itself. Both registered physics backends start from the same entity-split MJCF B-form asset set. IsaacSim consumes it through UniSim's mapped subprocess worker; MuJoCo composes the entities and compiles a VariantPack for mjbatch. The learner is a vendored, SHA-gated RL-Games SAPG runtime.

| Capability | Status |
| --- | --- |
| `SimToolReal` teacher policy training (RL-Games SAPG) | Task/learner semantics trained and label-compared against the original repository (see Evidence) |
| Backend support | IsaacSim (default, mapped worker) and MuJoCo (compose + VariantPack), both on the shared MJCF B form |
| Label parity with the original IsaacLab `DirectRLEnv` implementation | TensorBoard key labels indistinguishable from the original same-seed run |
| Student / camera / contact-sensor policies, deployment, sim2real | Not carried over; this repository contains no deployment code |
| Asset sources | Robot/table source URDFs are vendored and serialized once into entity MJCFs; the 1,200-tool pool is generated procedurally as standalone tool MJCFs |

## Construction and backend routing

`entity_mjcf_writer.py` is the single asset truth point. It emits `robot.xml`, `table.xml`, `floor.xml`, and one MJCF per tool, with task-owner PD/friction values baked into the sources. `SimToolRealEnvCfg.bind_entity_variant` then binds those tool files to the scene's `object` entity with a fixed round-robin assignment.

- **IsaacSim (default)**: the owner leaf declares UniSim `entity_assets` plus the object `EntityVariantBinding`; the mapped IsaacSim worker composes entities, converts MJCF sources, materializes selected prototypes, and steps PhysX.
- **MuJoCo**: the same entity declarations are composed by UniSim, the same-layout tool pool is compiled as a VariantPack, and mjbatch steps the CPU simulation.
- **Headless scope**: neither teacher-training leaf materializes `goalviz`. The command hub owns the authoritative goal; a render-only marker can be added later without duplicating the 1,200-tool physics/prototype pool.

## Evidence

- **Training-parity comparison (L3)**: on the previously validated IsaacSim route, a full 12,288-environment, seed-42 run (514 epochs) was compared against the original repository's first 500 epochs — all key TensorBoard label families `INDISTINGUISHABLE`. The shared B-form route preserves those task and learner code paths; its backend-specific evidence is the smoke coverage below.
- **Per-variant mass readback** vs the generated manifest: max relative deviation 5.7e-08.
- **World-frame wrench path**: probe deltas bit-identical to the frozen kit-revalidation anchors.
- The current IsaacSim/MJCF mapped-worker route has an 8-environment real-backend smoke gate; the MuJoCo compose/VariantPack route has reset and 100-step smoke coverage. Production-scale IsaacSim cold start remains the active performance workstream.
- Verified on Linux x86_64, one NVIDIA GPU (RTX 4090), IsaacSim 5.1.0 + IsaacLab 2.3.2.post1 as the worker runtime.
- No robot video is included yet; a recording of a trained lift episode is the top missing piece for this README.

## First successful run

Requirements: Linux x86_64, an NVIDIA GPU with CUDA (the learner and the IsaacSim worker share it), [uv](https://docs.astral.sh/uv/), and roughly 20 GB of disk for the worker runtime and caches.

```bash
git clone https://github.com/unilabsim/simtoolreal_unilab.git
cd simtoolreal_unilab
uv sync
bash scripts/setup_isaacsim_env.sh   # one-time: IsaacSim worker runtime (~15 GB)
uv run simtoolreal-assets            # verify the vendored assets
```

That is the complete setup. The two pieces that cannot ship inside `uv sync`:

1. **The IsaacSim worker runtime.** UniSim's `isaacsim` extra is empty (NVIDIA's SDK is not redistributed), so the worker runs in a dedicated Python 3.11 venv. `scripts/setup_isaacsim_env.sh` installs the validated stack there (IsaacSim 5.1.0 + IsaacLab 2.3.2.post1 wheels + torch 2.7.0/cu128, from the NVIDIA PyPI index (cu128 covers Blackwell GPUs down to older architectures)), idempotently with per-step markers, under `~/.cache/unisim/isaacsim/venv`. The script also retains its sha256-pinned URDF-importer alignment step for the legacy validated URDF route; the current MJCF B-form runtime uses `MjcfConverter` and does not call that importer. `simtoolreal-train`/`simtoolreal-play` detect the install automatically — no environment variable needed. To use a different IsaacSim-capable interpreter instead, export `UNISIM_ISAACSIM_PYTHON=/path/to/venv/bin/python` (any venv layout; this is the escape hatch the validation used).

2. **Entity and tool assets.** The robot (KUKA iiwa14 + SHARPA hand) and table source URDFs ship with this repository under `simtoolreal_unilab/assets/simtoolreal/` (upstream MIT, SHA-256 manifest; `uv run simtoolreal-assets` verifies them). The task writer serializes them to entity MJCFs and generates the 1,200-tool MJCF pool at first use; it is cached under `/tmp` (override: `SIMTOOLREAL_TOOL_POOL_ROOT`), while bootstrap entity sources honor `SIMTOOLREAL_GENERATED_ASSETS_ROOT`. To test against an upstream checkout instead, set `SIMTOOLREAL_ASSETS_ROOT` to its `assets/` directory.

Environment acceptance smoke (64 envs, 24-tool pool, 2 epochs — minutes on one GPU; the flag set is required because production rollouts use exactly six exploration blocks):

```bash
uv run simtoolreal-train \
  algo.num_envs=64 env.tool_pool_num_per_type=2 \
  rl_games.params.config.max_epochs=2 \
  rl_games.params.config.expl_coef_block_size=32 \
  rl_games.params.config.minibatch_size=256 \
  rl_games.params.config.central_value_config.minibatch_size=256 \
  training.smoke=true training.no_play=true
```

Expected output: per-epoch statistics, then a checkpoint and a TensorBoard event file under `logs/rlgames_sapg/SimToolReal/<run>/`. A smoke checkpoint is an acceptance artifact, never a policy baseline.

## Train, evaluate, inspect

Production training uses the IsaacSim owner defaults (24,576 environments; `num_actors` must form exactly six `expl_coef_block_size` blocks):

```bash
uv run simtoolreal-train                 # defaults: task=simtoolreal/isaacsim, MJCF B form
uv run simtoolreal-train algo.num_envs=12288 rl_games.params.config.expl_coef_block_size=2048
```

Play back the latest checkpointed run (renders through the IsaacSim worker):

```bash
uv run simtoolreal-play
uv run simtoolreal-play --checkpoint-file logs/rlgames_sapg/SimToolReal/<run>/nn/<checkpoint>.pth
```

List registered tasks and backends:

```bash
uv run simtoolreal-list-envs            # SimToolReal   isaacsim / SimToolReal   mujoco
```

TensorBoard: `uv run tensorboard --logdir logs/rlgames_sapg`. The label names are kept compatible with the original repository's tag set, so runs from both sides can be overlaid directly.

## MuJoCo backend

The same B-form task runs on UniSim's MuJoCo backend (pure CPU, no IsaacSim worker runtime needed). UniSim composes the entity scene and compiles the same-layout tool pool into a VariantPack. Everything goes through the same CLI with `--backend mujoco`:

```bash
uv run simtoolreal-train --backend mujoco   # defaults: 12,288 envs x 1,200 tools
```

2,048-environment smoke recipe (2048 is not six-block divisible, so it goes through the explicit smoke channel, which also disables playback):

```bash
uv run simtoolreal-train --backend mujoco \
  algo.num_envs=2048 \
  rl_games.params.config.expl_coef_block_size=512 \
  rl_games.params.config.minibatch_size=8192 \
  rl_games.params.config.central_value_config.minibatch_size=8192 \
  training.smoke=true training.no_play=true
```

TensorBoard and checkpoints land in the same `logs/rlgames_sapg/SimToolReal/<run>/` layout; `uv run tensorboard --logdir logs/rlgames_sapg` overlays both backends' runs.

## Modify and extend

| I want to change… | Path |
| --- | --- |
| Reward terms and weights | `src/simtoolreal_unilab/tasks/simtoolreal/rewards.py`, values in `conf/rlgames_sapg/task/simtoolreal/base.yaml` |
| Observations (policy/critic groups, row-scoped reset path) | `tasks/simtoolreal/observations.py` |
| Goal/command logic and the success hub | `tasks/simtoolreal/commands.py` |
| Scene composition, tool pool wiring, asset resolution | `tasks/simtoolreal/scene.py`, `env_cfg.py` |
| The 1,200-tool pool definition and MJCF generation | `tasks/simtoolreal/tool_catalog.py`, `tool_mjcf_writer.py` |
| Entity-split B-form sources | `tasks/simtoolreal/entity_mjcf_writer.py` |
| Termination and events (wrench DR, resets) | `tasks/simtoolreal/terminations.py`, `events.py` |
| Algorithm hyperparameters | `conf/rlgames_sapg/config.yaml` |
| The UniLab behaviors this task layers on the pinned revision | `src/simtoolreal_unilab/_compat/` (see below) |

## Compatibility layer (developers)

The task needs a few behaviors upstream UniLab does not ship yet (multi-rigid-root reset routing, the entity variant-mass facade, manager logging views, seed propagation, entity-asset materialization). `simtoolreal_unilab._compat` ports those method bodies onto the pinned official UniLab revision at import time and fails loudly if the pin drifts. Dependency pins:

- `unilab` — temporary local checkout carrying the IsaacSim solver-knob channel ([Motphys/UniLab#1620](https://github.com/Motphys/UniLab/pull/1620)); switch back to the official pin after it merges and ships.
- `unisim-core` — temporary local checkout carrying mapped-worker visual-only geom/fullinertia fixes ([unilabsim/unisim#279](https://github.com/unilabsim/unisim/pull/279)); switch back to the PyPI release after it merges and ships.

The vendored RL-Games runtime under `third_party/simtoolreal_rl_games` is SHA-gated; never edit its files. Provenance is anchored by the per-file SHA256 manifest plus the recorded upstream source commit — no local Git objects are required. Run `make check` / `make test` before pushing.

## Credits and support

- **SimToolReal** — the upstream task, assets, and training semantics ([tylerlum/simtoolreal](https://github.com/tylerlum/simtoolreal), MIT). Numbers reported by the upstream paper (e.g. sim2real performance) belong to the original implementation and are not re-validated here.
- **UniLab** / **UniSim** / **unilab-rl** — training infrastructure, physics-backend contract, and async runtimes.
- **RL-Games** — the SAPG learner, vendored from the upstream SimToolReal fork with an audited patch set (`third_party/simtoolreal_rl_games/PATCHES.md`).

License: Apache-2.0 (this repository); the vendored runtime and upstream assets keep their own licenses. Issues and pull requests welcome — for framework-level problems (backends, contract), report to UniLab/UniSim instead.
