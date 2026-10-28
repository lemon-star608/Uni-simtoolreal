# AGENTS.md

## Scope

This repository contains two MuJoCo products:

1. Full-scale SimToolReal teacher training with UniLab TorchEnv, UniSim MuJoCo, mjbatch VariantPack, the 1,200-tool pool, and vendored RL-Games SAPG.
2. DexToolBench evaluation over 24 committed static MJCF scenes with interactive Viser and policy workflows.

There is one fixed task route: `simtoolreal/mujoco`.

## Layout

- `src/simtoolreal_unilab/`: package configuration, CLI, assets, RL-Games adapter, SimToolReal task, and DexToolBench product
- `src/simtoolreal_unilab/tasks/simtoolreal/`: SimToolReal task semantics and MJCF generation
- `src/simtoolreal_unilab/tasks/dextoolbench/`: DexToolBench catalog and direct scene configuration
- `src/simtoolreal_unilab/dextoolbench/`: worker, policy player, Viser scene, and interactive launcher
- `src/simtoolreal_unilab/assets/`: committed SimToolReal and DexToolBench assets with manifests
- `tests/`: repository boundary, runtime, asset, algorithm, task, and DexToolBench tests
- `third_party/simtoolreal_rl_games/`: SHA-gated vendored RL-Games runtime

## Constraints

- Use `uv` only for Python dependency and test commands.
- Exact dependency policy:
  - `unilab==1.5.0`
  - `unisim-core[mujoco]==2.1.0`
  - `mujoco==3.11.0`
  - `torch==2.14.0+cu130`
  - `mjbatch-uni==0.2.6.dev0` from Git rev `9da4dc59dccfaeedbbcd1f336553498dc983a99a`
- Never edit runtime files under `third_party/simtoolreal_rl_games`; packaging and patch records remain separately documented there.
- Preserve the SimToolReal action joint-id tuple behavior.
- Do not add another simulation route, backend switch, compatibility layer, generated asset pool, checkpoint, log, or cache to Git.
- First-party source, tests, and documentation must not use retired runtime terms or scaled-run wording.

## Verification

Run before pushing:

```bash
make check
make test
uv run simtoolreal-assets
```

Full release acceptance additionally runs `uv run simtoolreal-train` without overrides through a persistent checkpoint and loads that checkpoint through the DexToolBench policy path.

## Documentation

- README and README_zh must remain aligned and user-facing.
- Documentation covers requirements, commands, outputs, architecture, and verification only.
- Do not add development history, migration notes, blocker logs, or reduced-scale examples.
