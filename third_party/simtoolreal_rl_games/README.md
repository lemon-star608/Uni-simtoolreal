# SimToolReal RL-Games Source-Fidelity Runtime

This directory preserves 72 pristine Source identities from the SimToolReal RL-Games fork at
the fixed commit recorded in `source_manifest.json`. V2 applies 7 reviewed compatibility patches
with separate pristine and current SHA256 records. The other 65 Python files remain byte-identical
to their fixed Source Git blobs. `PATCHES.md` documents every changed byte
sequence and its covering test.

The vendor intentionally does not include the 122 YAML files in the selected Source parent tree,
Source examples, notebooks, or the Source top-level `rl_games/tests/` test suite. The
runtime selection does retain test-named Python modules inside `rl_games/rl_games`, including
`common/test_utils.py`, `envs/test/**`, and `envs/test_network.py`; these are part of the 72
selected runtime identities. Code commit 2 does not wire this distribution into UniLab, add a root
dependency, or import `rl_games` from a production path.

Run the read-only integrity gate from the repository root:

```bash
uv run scripts/audit_simtoolreal_rlgames_vendor.py
```

Do not format files below `rl_games/`. The root Ruff configuration excludes this directory
so automated formatting cannot rewrite either pristine Source bytes or reviewed patch bytes.
