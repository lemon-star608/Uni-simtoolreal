# simtoolreal_unilab

SimToolReal teacher training on UniLab + UniSim through the IsaacSim
URDF→USD asset path. See README.md for usage.

## Repository rules

- Python: `uv` only. `make test` / `make check` before pushing.
- The vendored RL-Games runtime under `third_party/simtoolreal_rl_games` is
  SHA-gated; never edit its runtime files. Packaging-level edits
  (pyproject/PATCHES.md) are documented in PATCHES.md.
- `src/simtoolreal_unilab/_compat` ports method bodies from the SimToolReal
  development branch of UniLab onto the pinned official revision. It fails
  loudly if the pin drifts; when upstream UniLab grows these behaviors,
  remove the layer instead of skipping it.
- Dependencies: `unilab` is pinned to the official commit the task was
  developed against; `unisim-core` is pinned to the fork branch carrying
  the IsaacSim scene-composition contract (upstream PR
  unilabsim/unisim#79) — switch both to PyPI releases when available.
- No co-author trailers from AI assistants in commits.
