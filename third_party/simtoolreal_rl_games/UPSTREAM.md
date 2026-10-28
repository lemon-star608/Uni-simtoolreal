# Upstream provenance

The immutable Source oracle for this snapshot is:

- Repository: `https://github.com/tylerlum/simtoolreal.git`
- Source commit: `2a9917533bfea70419ed2667a511d7238e5b3abc`
- Selected Source path: `rl_games/rl_games`
- Selected parent-tree OID: `7a6a0bb090998d00565aaefa6ab9f2b3d356ace2`
- Source packaging blob (`rl_games/pyproject.toml`):
  `185e2b8f8b4b7437344026216e241562c49b698b`
- Source license blob (`rl_games/LICENSE`):
  `313ca229e6ca879466f94bff49362fb65667e22f`

The selected parent tree contains 72 Python blobs and 122 YAML blobs. V1 selects only the
72 Python blobs. It does not claim that the reduced vendored directory has the identity of
the complete parent tree.

The migration plan records the fork lineage as RL-Games v1.6.1 at upstream commit
`f5bd8f2a0022220a1109200a3da47d2e96cb9aa1`. The SimToolReal Source commit above, rather
than the public `rl-games>=1.6` distribution, is the runtime oracle for this migration.

`source_manifest.json` records the vendored relative path, Source path, Source Git blob OID,
and SHA256 for every selected Python file. The nested `LICENSE` is copied byte-for-byte from
the fixed Source license blob.

## Canonical Source selection anchor

The V1 audit independently anchors the complete 72-file selection. Starting from the fixed
Source Git objects, it sorts records by vendored path. Each record is:

```text
[path, source_path, source_blob, sha256, byte_size]
```

It serializes the records with:

```python
json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode()
```

The 72 records contain 439455 Source bytes. The SHA256 of the canonical payload is
`f0517fb198dbbf9dcc456ab6de4a5cf6e0c4b03cdc90e84f12e52f74a70fe0ca`. The audit rebuilds
the payload from the manifest plus current vendored bytes and compares it with this
independent hard-coded anchor, so coordinated file and manifest hash changes fail closed.
