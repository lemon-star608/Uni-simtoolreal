"""Fail-closed identity guard for the checked-in RL-Games runtime."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

EXPECTED_DISTRIBUTION = "unilab-simtoolreal-rl-games"
EXPECTED_VERSION = "1.6.1+simtoolreal.2a991753.compat2"
EXPECTED_SOURCE_HEAD = "2a9917533bfea70419ed2667a511d7238e5b3abc"
EXPECTED_PARENT_TREE = "7a6a0bb090998d00565aaefa6ab9f2b3d356ace2"
EXPECTED_SELECTION_SHA256 = "f0517fb198dbbf9dcc456ab6de4a5cf6e0c4b03cdc90e84f12e52f74a70fe0ca"
EXPECTED_MANIFEST_SHA256 = "4f1170b222e4ba008b34070fad7aeaba4cf790cc6ae1917417ee40ef35573ac9"
INSTALL_HINT = "uv run --extra mujoco --extra rlgames-sapg ..."
REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR_ROOT = (REPO_ROOT / "third_party/simtoolreal_rl_games").resolve()
PACKAGE_ROOT = VENDOR_ROOT / "rl_games"


@dataclass(frozen=True)
class RlGamesSapgIdentity:
    distribution: str
    version: str
    vendor_root: Path
    python_files: int
    compatibility_patches: int


def _regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_manifest() -> tuple[int, int]:
    path = VENDOR_ROOT / "source_manifest.json"
    _regular(path, "vendored source_manifest.json")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("vendored source_manifest.json SHA256 drift")
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("vendored source_manifest.json is invalid") from exc
    entries = manifest.get("python_files")
    patches = manifest.get("compatibility_allowlist")
    selection = manifest.get("selection", {})
    if (
        manifest.get("manifest_version") != 2
        or manifest.get("source_head") != EXPECTED_SOURCE_HEAD
        or manifest.get("source_parent_tree") != EXPECTED_PARENT_TREE
        or not isinstance(entries, list)
        or len(entries) != 72
        or not isinstance(patches, list)
        or len(patches) != 7
        or selection.get("python_file_count") != 72
    ):
        raise RuntimeError("vendored source manifest identity/schema drift")
    patch_hashes = {item["path"]: item["patched_sha256"] for item in patches}
    records: list[list[object]] = []
    for entry in sorted(entries, key=lambda item: item["path"]):
        relative = entry["path"]
        current = VENDOR_ROOT / relative
        _regular(current, f"vendored runtime file {relative}")
        expected = patch_hashes.get(relative, entry["sha256"])
        if hashlib.sha256(current.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"vendored runtime hash drift: {relative}")
        try:
            pristine_size = int(
                subprocess.check_output(
                    ["git", "cat-file", "-s", entry["source_blob"]],
                    cwd=REPO_ROOT,
                    text=True,
                ).strip()
            )
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"pristine Git object is unavailable: {relative}") from exc
        records.append(
            [relative, entry["source_path"], entry["source_blob"], entry["sha256"], pristine_size]
        )
    # The outer manifest hash fixes every pristine record. Keep the independent
    # Source-selection anchor visible in production diagnostics as well.
    canonical = json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != EXPECTED_SELECTION_SHA256:
        raise RuntimeError("vendored Source selection identity drift")
    return len(entries), len(patches)


def require_rlgames_sapg() -> RlGamesSapgIdentity:
    """Verify the supported platform and exact editable runtime before import."""
    if platform.system() == "Linux" and platform.machine() == "aarch64":
        raise RuntimeError("RL-Games SAPG is unsupported on Linux/aarch64")
    try:
        distribution = metadata.distribution(EXPECTED_DISTRIBUTION)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Missing pinned RL-Games SAPG runtime; use `{INSTALL_HINT}`") from exc
    if distribution.metadata["Name"] != EXPECTED_DISTRIBUTION:
        raise RuntimeError("wrong RL-Games SAPG distribution metadata name")
    if distribution.version != EXPECTED_VERSION:
        raise RuntimeError(f"unsupported RL-Games SAPG runtime version: {distribution.version}")
    try:
        direct_url = json.loads(distribution.read_text("direct_url.json") or "")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("RL-Games SAPG editable direct_url.json is invalid") from exc
    parsed = urlparse(direct_url.get("url", ""))
    installed_root = Path(unquote(parsed.path)).resolve()
    if (
        parsed.scheme != "file"
        or direct_url.get("dir_info", {}).get("editable") is not True
        or installed_root != VENDOR_ROOT
    ):
        raise RuntimeError("RL-Games SAPG must be the exact checked-in editable vendor")
    spec = importlib.util.find_spec("rl_games")
    if spec is None or spec.origin is None or not _inside(Path(spec.origin), PACKAGE_ROOT):
        raise RuntimeError("rl_games namespace does not resolve to the checked-in vendor")
    python_files, patches = _validate_manifest()
    return RlGamesSapgIdentity(
        EXPECTED_DISTRIBUTION, EXPECTED_VERSION, VENDOR_ROOT, python_files, patches
    )
