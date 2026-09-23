# simtoolreal_unilab

SimToolReal teacher training on UniLab + UniSim on the shared MJCF B-form
asset path (IsaacSim mapped worker + MuJoCo VariantPack backends). See
README.md for usage.

## Repository rules

- Python: `uv` only. `make test` / `make check` before pushing.
- The vendored RL-Games runtime under `third_party/simtoolreal_rl_games` is
  SHA-gated; never edit its runtime files. Packaging-level edits
  (pyproject/PATCHES.md) are documented in PATCHES.md.
- `src/simtoolreal_unilab/_compat` ports method bodies from the SimToolReal
  development branch of UniLab onto the pinned official revision. It fails
  loudly if the pin drifts; when upstream UniLab grows these behaviors,
  remove the layer instead of skipping it.
- Dependencies (`main`): `unilab`, `unisim-core`, and `mjbatch-uni` are
  pinned to upstream git revisions in `pyproject.toml`
  (`tool.uv.sources`); switch each to its PyPI release once one ships
  the carried PRs. The retired URDF runtime line is frozen at tag
  `urdf-anchor` for reference.
- No co-author trailers from AI assistants in commits.
