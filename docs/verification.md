# Verification

## Repository gates

Run all three commands before pushing:

```bash
make check
make test
uv run simtoolreal-assets
```

`make check` validates formatting, lint, and whitespace. `make test` runs the repository boundary, MuJoCo runtime, task/MJCF, asset, RL-Games adapter, vendor-integrity, and DexToolBench tests. `simtoolreal-assets` verifies every committed SimToolReal and DexToolBench manifest entry.

## Default training contract

The documented default command is:

```bash
uv run simtoolreal-train
```

It must compose without scale overrides to:

- 24,576 environments
- 1,200 tool variants
- 4,096-environment exploration blocks, exactly six blocks
- 29 action dimensions
- 140 actor observation dimensions
- 162 critic observation dimensions

## Full-scale checkpoint acceptance

1. Run the default training command without overrides.
2. Continue until the run's persistent `last/model.pth` checkpoint exists.
3. Record the run directory and the checkpoint epoch from console or run evidence.
4. Launch DexToolBench with `run_config.json` and `last/model.pth`.
5. Confirm the policy loads and executes the evaluation path.

Training and evaluation artifacts remain ignored under `logs/`; do not commit them.
