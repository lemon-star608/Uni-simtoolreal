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
- Dependencies (`feat/mujoco-mjcf`): `unilab` is pinned to official main
  (`2e3f30ae`); `unisim-core` is pinned to the PyPI release (1.4.2);
  `mujoco` (pip) backs the MJCF asset line's compiler gates. The IsaacSim
  line on `main` still pins the unisim fork branch carrying the scene-
  composition contract (upstream PR unilabsim/unisim#79).
- No co-author trailers from AI assistants in commits.
