# simtoolreal_unilab

SimToolReal teacher-policy training on [UniLab](https://github.com/Motphys/UniLab) + [UniSim](https://github.com/unilabsim/unisim), through the IsaacSim URDF→USD asset path: a KUKA iiwa14 arm (7 DoF) with a SHARPA hand (22 DoF) learns to lift and manipulate tools from a 1,200-variant procedurally generated tool pool, matching the original repository's training semantics.

This is a downstream task package: install it and use UniLab's shared training tooling without cloning UniLab itself. The physics layer runs on UniSim's IsaacSim subprocess backend; the learner is a vendored, SHA-gated RL-Games SAPG runtime.

| Capability | Status |
| --- | --- |
| `SimToolReal` teacher policy training (RL-Games SAPG, IsaacSim backend) | Trained and label-compared against the original repository (see Evidence) |
| Backend support | IsaacSim only; the MuJoCo path is not implemented here |
| Label parity with the original IsaacLab `DirectRLEnv` implementation | TensorBoard key labels indistinguishable from the original same-seed run |
| Student / camera / contact-sensor policies, deployment, sim2real | Not carried over; this repository contains no deployment code |
| Asset sources | URDF→USD (robot/table URDFs vendored in-repo; the 1,200-tool pool is generated procedurally); MJCF sources are not supported |

## Evidence

- **Training-parity comparison (L3)**: a full 12,288-environment, seed-42 run (514 epochs) compared against the original repository's first 500 epochs — all key TensorBoard label families `INDISTINGUISHABLE`.
- **Per-variant mass readback** vs the generated manifest: max relative deviation 5.7e-08.
- **World-frame wrench path**: probe deltas bit-identical to the frozen kit-revalidation anchors.
- Verified on Linux x86_64, one NVIDIA GPU (RTX 4090), IsaacSim 5.1.0-rc.19 + IsaacLab 2.3.2.post1 as the worker runtime.
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

1. **The IsaacSim worker runtime.** UniSim's `isaacsim` extra is empty (NVIDIA's SDK is not redistributed), so the worker runs in a dedicated Python 3.11 venv. `scripts/setup_isaacsim_env.sh` installs the validated stack there (IsaacSim 5.1.0 + IsaacLab 2.3.2.post1 wheels + torch 2.7.0/cu126, from the NVIDIA PyPI index), idempotently with per-step markers, under `~/.cache/unisim/isaacsim/venv`. **URDF importer alignment:** the pip wheels bundle urdf importer extension 2.4.30, whose binding lacks the `set_merge_fixed_ignore_inertia` API IsaacLab 2.3.2+ requires (a fresh pip install crashes on the first URDF conversion); the script therefore downloads 2.4.31 from NVIDIA's public extension-registry CDN (sha256-pinned), swaps it over the bundled copy, and fails the install unless the binding exposes the merge API. `simtoolreal-train`/`simtoolreal-play` detect that install automatically — no environment variable needed. To use a different IsaacSim-capable interpreter instead, export `UNISIM_ISAACSIM_PYTHON=/path/to/venv/bin/python` (any venv layout; this is the escape hatch the validation used).

2. **Tool assets.** The robot (KUKA iiwa14 + SHARPA hand) and table URDFs ship with this repository under `simtoolreal_unilab/assets/simtoolreal/` (upstream MIT, SHA-256 manifest; `uv run simtoolreal-assets` verifies them). The 1,200-tool pool is generated procedurally at first use and cached under `/tmp` (override: `SIMTOOLREAL_TOOL_POOL_ROOT`); generated bootstrap URDFs honor `SIMTOOLREAL_GENERATED_ASSETS_ROOT`. To test against an upstream checkout instead, set `SIMTOOLREAL_ASSETS_ROOT` to its `assets/` directory.

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

Production training uses the owner defaults (24,576 environments; `num_actors` must form exactly six `expl_coef_block_size` blocks):

```bash
uv run simtoolreal-train                 # defaults: task=simtoolreal/isaacsim
uv run simtoolreal-train algo.num_envs=12288 rl_games.params.config.expl_coef_block_size=2048
```

Play back the latest checkpointed run (renders through the IsaacSim worker):

```bash
uv run simtoolreal-play
uv run simtoolreal-play --checkpoint-file logs/rlgames_sapg/SimToolReal/<run>/nn/<checkpoint>.pth
```

List registered tasks and backends:

```bash
uv run simtoolreal-list-envs            # SimToolReal   isaacsim
```

TensorBoard: `uv run tensorboard --logdir logs/rlgames_sapg`. The label names are kept compatible with the original repository's tag set, so runs from both sides can be overlaid directly.

## Modify and extend

| I want to change… | Path |
| --- | --- |
| Reward terms and weights | `src/simtoolreal_unilab/tasks/simtoolreal/rewards.py`, values in `conf/rlgames_sapg/task/simtoolreal/base.yaml` |
| Observations (policy/critic groups, row-scoped reset path) | `tasks/simtoolreal/observations.py` |
| Goal/command logic and the success hub | `tasks/simtoolreal/commands.py` |
| Scene composition, tool pool wiring, asset resolution | `tasks/simtoolreal/scene.py`, `env_cfg.py` |
| The 1,200-tool pool definition and URDF generation | `tasks/simtoolreal/tool_catalog.py`, `tool_urdf_writer.py` |
| Termination and events (wrench DR, resets) | `tasks/simtoolreal/terminations.py`, `events.py` |
| Algorithm hyperparameters | `conf/rlgames_sapg/config.yaml` |
| The UniLab behaviors this task layers on the pinned revision | `src/simtoolreal_unilab/_compat/` (see below) |

## Compatibility layer (developers)

The task needs a few behaviors upstream UniLab does not ship yet (multi-rigid-root reset routing, the entity variant-mass facade, manager logging views, seed propagation, entity-asset materialization). `simtoolreal_unilab._compat` ports those method bodies onto the pinned official UniLab revision at import time and fails loudly if the pin drifts. Dependency pins:

- `unilab` — official `main` at the exact commit this task was developed against;
- `unisim-core` — fork branch `feat/isaacsim-urdf-multi-entity` carrying the IsaacSim scene-composition contract (upstream PR [unilabsim/unisim#79](https://github.com/unilabsim/unisim/pull/79)); switch to the PyPI release when it merges.

The vendored RL-Games runtime under `third_party/simtoolreal_rl_games` is SHA-gated; never edit its files. Provenance is anchored by the per-file SHA256 manifest plus the recorded upstream source commit — no local Git objects are required. Run `make check` / `make test` (428 tests) before pushing.

## Credits and support

- **SimToolReal** — the upstream task, assets, and training semantics ([tylerlum/simtoolreal](https://github.com/tylerlum/simtoolreal), MIT). Numbers reported by the upstream paper (e.g. sim2real performance) belong to the original implementation and are not re-validated here.
- **UniLab** / **UniSim** / **unilab-rl** — training infrastructure, physics-backend contract, and async runtimes.
- **RL-Games** — the SAPG learner, vendored from the upstream SimToolReal fork with an audited patch set (`third_party/simtoolreal_rl_games/PATCHES.md`).

License: Apache-2.0 (this repository); the vendored runtime and upstream assets keep their own licenses. Issues and pull requests welcome — for framework-level problems (backends, contract), report to UniLab/UniSim instead.
