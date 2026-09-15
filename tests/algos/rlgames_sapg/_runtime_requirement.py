from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import NoReturn
from urllib.parse import unquote, urlparse

import pytest

# Editable parity gates must not mutate the fail-closed vendored inventory.
sys.dont_write_bytecode = True

EXPECTED_DISTRIBUTION = "unilab-simtoolreal-rl-games"
EXPECTED_SOURCE_HEAD = "2a9917533bfea70419ed2667a511d7238e5b3abc"
EXPECTED_SOURCE_PARENT_TREE = "7a6a0bb090998d00565aaefa6ab9f2b3d356ace2"
EXPECTED_SELECTION_SHA256 = "f0517fb198dbbf9dcc456ab6de4a5cf6e0c4b03cdc90e84f12e52f74a70fe0ca"
V2_DISTRIBUTION_VERSION = "1.6.1+simtoolreal.2a991753.compat2"
PATCH_ENTRY_KEYS = {
    "path",
    "pristine_blob",
    "pristine_sha256",
    "patched_sha256",
    "reason",
    "covering_test",
}

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR_ROOT = (REPO_ROOT / "third_party/simtoolreal_rl_games").resolve()
VENDOR_PACKAGE_ROOT = (VENDOR_ROOT / "rl_games").resolve()


def _required() -> bool:
    return os.environ.get("UNILAB_REQUIRE_SAPG") == "1"


def _reject(message: str) -> NoReturn:
    if _required():
        raise RuntimeError(message)
    pytest.skip(message, allow_module_level=True)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return False
    return True


def _git_blob(blob_oid: str) -> bytes:
    result = subprocess.run(
        ["git", "cat-file", "blob", blob_oid],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        _reject(f"pristine Git object {blob_oid} is unavailable")
    return result.stdout


def _verify_manifest() -> dict[str, str]:
    manifest_path = VENDOR_ROOT / "source_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _reject(f"vendored source manifest is unavailable or invalid: {exc}")

    if manifest.get("manifest_version") != 2:
        _reject(
            "vendored source manifest version drift: "
            f"expected 2, got {manifest.get('manifest_version')!r}"
        )
    if manifest.get("source_head") != EXPECTED_SOURCE_HEAD:
        _reject("vendored source manifest HEAD drift")
    if manifest.get("source_parent_tree") != EXPECTED_SOURCE_PARENT_TREE:
        _reject("vendored source manifest parent-tree drift")

    entries = manifest.get("python_files")
    allowlist = manifest.get("compatibility_allowlist")
    if not isinstance(entries, list) or len(entries) != 72 or not isinstance(allowlist, list):
        _reject("vendored source manifest inventory drift")
    patches = {}
    for patch in allowlist:
        if not isinstance(patch, dict) or set(patch) != PATCH_ENTRY_KEYS:
            _reject("vendored compatibility allowlist schema drift")
        patch_path = patch["path"]
        if not isinstance(patch_path, str) or patch_path in patches:
            _reject("vendored compatibility allowlist schema drift")
        patches[patch_path] = patch
    if list(patches) != sorted(patches):
        _reject("vendored compatibility allowlist order drift")

    canonical_records = []
    current_hashes = {}
    for entry in sorted(entries, key=lambda item: item.get("path", "")):
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "source_path",
            "source_blob",
            "sha256",
        }:
            _reject("vendored Python manifest entry schema drift")
        relative_path = entry["path"]
        if entry["source_path"] != f"rl_games/{relative_path}":
            _reject(f"vendored Source path drift for {relative_path}")
        pristine = _git_blob(entry["source_blob"])
        if hashlib.sha256(pristine).hexdigest() != entry["sha256"]:
            _reject(f"pristine Git-object hash drift for {relative_path}")
        canonical_records.append(
            [
                relative_path,
                entry["source_path"],
                entry["source_blob"],
                entry["sha256"],
                len(pristine),
            ]
        )

        current_path = VENDOR_ROOT / relative_path
        if current_path.is_symlink() or not current_path.is_file():
            _reject(f"vendored module is missing or not regular: {relative_path}")
        patch = patches.get(relative_path)
        if patch is not None and (
            patch["pristine_blob"] != entry["source_blob"]
            or patch["pristine_sha256"] != entry["sha256"]
            or patch["patched_sha256"] == entry["sha256"]
        ):
            _reject(f"vendored compatibility provenance drift for {relative_path}")
        expected_current = patch["patched_sha256"] if patch else entry["sha256"]
        if hashlib.sha256(current_path.read_bytes()).hexdigest() != expected_current:
            _reject(f"vendored module hash drift for {relative_path}")
        current_hashes[relative_path] = expected_current

    if set(patches) - set(current_hashes):
        _reject("vendored compatibility allowlist path drift")

    payload = json.dumps(canonical_records, ensure_ascii=True, separators=(",", ":")).encode()
    if hashlib.sha256(payload).hexdigest() != EXPECTED_SELECTION_SHA256:
        _reject("vendored canonical Source selection anchor drift")
    return current_hashes


def _verify_editable_distribution_source(distribution: importlib.metadata.Distribution) -> None:
    try:
        direct_url = json.loads(distribution.read_text("direct_url.json") or "")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _reject(f"vendored distribution direct_url.json is invalid: {exc}")
    parsed = urlparse(direct_url.get("url", ""))
    if parsed.scheme != "file" or not direct_url.get("dir_info", {}).get("editable"):
        _reject("vendored distribution is not installed from an editable file URL")
    distribution_root = Path(unquote(parsed.path)).resolve()
    if distribution_root != VENDOR_ROOT:
        _reject(
            "vendored distribution metadata resolves outside the pinned vendor: "
            f"{distribution_root}"
        )


def _verify_loaded_modules(current_hashes: dict[str, str]) -> None:
    for module_name, module in sorted(sys.modules.items()):
        if module_name != "rl_games" and not module_name.startswith("rl_games."):
            continue
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            _reject(f"loaded {module_name} has no auditable __file__")
        module_path = Path(module_file)
        if not _is_within(module_path, VENDOR_PACKAGE_ROOT):
            _reject(f"loaded {module_name} resolves outside the pinned vendor: {module_file}")
        resolved_path = module_path.resolve()
        try:
            relative_path = resolved_path.relative_to(VENDOR_ROOT).as_posix()
        except ValueError:
            _reject(f"loaded {module_name} resolves outside the pinned vendor: {module_file}")
        expected_hash = current_hashes.get(relative_path)
        if expected_hash is None:
            _reject(f"loaded {module_name} is absent from the vendored manifest: {relative_path}")
        if resolved_path.is_symlink() or not resolved_path.is_file():
            _reject(f"loaded {module_name} is missing or not regular: {relative_path}")
        if hashlib.sha256(resolved_path.read_bytes()).hexdigest() != expected_hash:
            _reject(f"loaded {module_name} hash drift: {relative_path}")


def require_simtoolreal_rl_games() -> None:
    try:
        distribution = importlib.metadata.distribution(EXPECTED_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        _reject(f"required distribution {EXPECTED_DISTRIBUTION!r} is not installed")

    name = distribution.metadata.get("Name")
    if name != EXPECTED_DISTRIBUTION:
        _reject(
            f"wrong rl_games distribution name: expected {EXPECTED_DISTRIBUTION!r}, got {name!r}"
        )
    if distribution.version != V2_DISTRIBUTION_VERSION:
        _reject(f"unsupported vendored distribution version: {distribution.version!r}")
    _verify_editable_distribution_source(distribution)

    spec = importlib.util.find_spec("rl_games")
    if spec is None or spec.origin is None:
        _reject("rl_games import cannot be resolved")
    if not _is_within(Path(spec.origin), VENDOR_PACKAGE_ROOT):
        _reject(f"rl_games resolves outside the pinned vendor: {spec.origin}")

    current_hashes = _verify_manifest()
    _verify_loaded_modules(current_hashes)
