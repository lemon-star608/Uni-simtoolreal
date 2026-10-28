"""Fail-closed integrity audit for the SimToolReal RL-Games V2 vendor."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SOURCE_REPOSITORY = "https://github.com/tylerlum/simtoolreal.git"
SOURCE_HEAD = "2a9917533bfea70419ed2667a511d7238e5b3abc"
SOURCE_PARENT_PATH = "rl_games/rl_games"
SOURCE_PARENT_TREE = "7a6a0bb090998d00565aaefa6ab9f2b3d356ace2"
SOURCE_LICENSE_BLOB = "313ca229e6ca879466f94bff49362fb65667e22f"
SOURCE_LICENSE_SHA256 = "46565837dec017dc2f8df9fbbb6904fb3e62dc9b91c9efd9bb8d0b22eacc47d5"
SOURCE_PACKAGING_BLOB = "185e2b8f8b4b7437344026216e241562c49b698b"
SOURCE_PACKAGING_SHA256 = "0f02830fbaef0dd6d040c028dc8e2c6975de9d3a3adc80dc5a521b3b0e9b9487"
SOURCE_SELECTION_SHA256 = "2f100c78073ded75cd5c7d6c5e55523c729f4b5524d61915ed7efccea1117e4c"
PYTHON_FILE_COUNT = 72
SOURCE_PYTHON_BYTES = 439562
EXCLUDED_YAML_FILE_COUNT = 122
APPROVED_METADATA_FILES = frozenset(
    {
        "LICENSE",
        "README.md",
        "UPSTREAM.md",
        "PATCHES.md",
        "pyproject.toml",
        "source_manifest.json",
    }
)
FIXED_METADATA_SHA256 = {
    "PATCHES.md": "dceba5f94e9232e1033a3209e89101b2e18612f2cdd036053584c01fcf67be4a",
    "README.md": "42f3a0413dadb7a34036054a547221769b34fd7ce8807591aac17c25175b5bb4",
    "UPSTREAM.md": "0daf6457fe21198047cf75d3b104d92fe1d4603eda1bd1dd54086f2f404c72bd",
    "pyproject.toml": "f26bab9e4118b864d7af26a3318cba80ee5a2ca08c3b0eda27926f9a40f0d7e5",
}
VENDOR_TOOL_EXCLUSION = "third_party/simtoolreal_rl_games"
GIT_ATTRIBUTES_CONTENT = b"third_party/simtoolreal_rl_games/rl_games/**/*.py -whitespace\n"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENDOR_ROOT = REPO_ROOT / "third_party/simtoolreal_rl_games"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")

MANIFEST_KEYS = {
    "manifest_version",
    "source_repository",
    "source_head",
    "source_parent_path",
    "source_parent_tree",
    "selection",
    "license",
    "source_packaging",
    "compatibility_allowlist",
    "python_files",
}
PYTHON_ENTRY_KEYS = {"path", "source_path", "source_blob", "sha256"}
PATCH_ENTRY_KEYS = {
    "path",
    "pristine_blob",
    "pristine_sha256",
    "patched_sha256",
    "reason",
    "covering_test",
}


class AuditError(RuntimeError):
    """Raised when the V2 vendor does not match fixed Source and patch provenance."""


@dataclass(frozen=True)
class AuditReport:
    source_head: str
    source_parent_tree: str
    python_file_count: int
    python_byte_count: int
    source_selection_sha256: str
    license_blob: str


def _git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()


def _require_regular_file(path: Path, description: str) -> None:
    if not path.exists():
        raise AuditError(f"{description} is missing: {path}")
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"{description} must be a regular file: {path}")


def _load_manifest(vendor_root: Path) -> dict[str, Any]:
    manifest_path = vendor_root / "source_manifest.json"
    _require_regular_file(manifest_path, "source manifest")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"source manifest cannot be decoded: {exc}") from exc
    if not isinstance(manifest, dict):
        raise AuditError("source manifest root must be an object")
    if set(manifest) != MANIFEST_KEYS:
        missing = sorted(MANIFEST_KEYS - set(manifest))
        extra = sorted(set(manifest) - MANIFEST_KEYS)
        raise AuditError(f"source manifest schema mismatch: missing={missing}, extra={extra}")
    return manifest


def _check_source_identity(manifest: dict[str, Any]) -> None:
    identities = {
        "source_repository": SOURCE_REPOSITORY,
        "source_head": SOURCE_HEAD,
        "source_parent_path": SOURCE_PARENT_PATH,
        "source_parent_tree": SOURCE_PARENT_TREE,
    }
    if manifest["manifest_version"] != 2:
        raise AuditError(f"manifest_version identity mismatch: {manifest['manifest_version']!r}")
    for field, expected in identities.items():
        actual = manifest[field]
        if actual != expected:
            raise AuditError(f"{field} identity mismatch: expected {expected}, got {actual!r}")

    expected_selection = {
        "included_glob": "**/*.py",
        "python_file_count": PYTHON_FILE_COUNT,
        "excluded_yaml_file_count": EXCLUDED_YAML_FILE_COUNT,
    }
    if manifest["selection"] != expected_selection:
        raise AuditError(
            f"Source selection identity mismatch: expected {expected_selection}, "
            f"got {manifest['selection']!r}"
        )


def _render_patches(patches: list[dict[str, Any]]) -> str:
    lines = [
        "# Compatibility patches",
        "",
        (
            "The pristine Git object and SHA256 remain the provenance anchor for each entry. "
            "The patched SHA256 is the only accepted runtime byte sequence."
        ),
    ]
    for patch in patches:
        lines.extend(
            [
                "",
                f"## `{patch['path']}`",
                "",
                f"- Pristine Git blob: `{patch['pristine_blob']}`",
                f"- Pristine SHA256: `{patch['pristine_sha256']}`",
                f"- Patched SHA256: `{patch['patched_sha256']}`",
                f"- Reason: {patch['reason']}",
                f"- Covering test: `{patch['covering_test']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _check_declared_patches(
    vendor_root: Path,
    manifest: dict[str, Any],
    entries_by_path: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    allowlist = manifest["compatibility_allowlist"]
    if not isinstance(allowlist, list):
        raise AuditError("compatibility_allowlist must be a list")

    patches: dict[str, dict[str, Any]] = {}
    for index, patch in enumerate(allowlist):
        if not isinstance(patch, dict) or set(patch) != PATCH_ENTRY_KEYS:
            raise AuditError(f"compatibility_allowlist[{index}] schema mismatch")
        relative_path = _safe_python_path(patch["path"]).as_posix()
        if relative_path in patches:
            raise AuditError(f"duplicate compatibility allowlist path: {relative_path}")
        if relative_path not in entries_by_path:
            raise AuditError(f"compatibility patch is outside Source selection: {relative_path}")
        source_entry = entries_by_path[relative_path]
        if patch["pristine_blob"] != source_entry["source_blob"]:
            raise AuditError(f"compatibility pristine blob drift for {relative_path}")
        if patch["pristine_sha256"] != source_entry["sha256"]:
            raise AuditError(f"compatibility pristine SHA256 drift for {relative_path}")
        if not isinstance(patch["patched_sha256"], str) or not HEX64.fullmatch(
            patch["patched_sha256"]
        ):
            raise AuditError(f"invalid patched SHA256 for {relative_path}")
        if patch["patched_sha256"] == patch["pristine_sha256"]:
            raise AuditError(f"redundant compatibility allowlist entry: {relative_path}")
        if not isinstance(patch["reason"], str) or not patch["reason"].strip():
            raise AuditError(f"empty compatibility reason for {relative_path}")
        covering_test = patch["covering_test"]
        if (
            not isinstance(covering_test, str)
            or not covering_test.startswith("tests/")
            or "::" not in covering_test
            or ".." in PurePosixPath(covering_test.split("::", 1)[0]).parts
        ):
            raise AuditError(f"invalid compatibility covering test for {relative_path}")
        patches[relative_path] = patch

    expected_order = sorted(patches)
    actual_order = [patch["path"] for patch in allowlist]
    if actual_order != expected_order:
        raise AuditError(
            f"compatibility_allowlist must be path-sorted: expected {expected_order}, "
            f"got {actual_order}"
        )

    patches_path = vendor_root / "PATCHES.md"
    _require_regular_file(patches_path, "PATCHES.md")
    try:
        patches_text = patches_path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise AuditError(f"PATCHES.md cannot be decoded: {exc}") from exc
    expected_text = _render_patches(allowlist)
    if patches_text != expected_text:
        raise AuditError("PATCHES.md does not exactly document compatibility_allowlist")
    return patches


def _safe_python_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str):
        raise AuditError(f"vendored Python path must be a string, got {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise AuditError(f"unsafe vendored Python path: {value!r}")
    if len(path.parts) < 2 or path.parts[0] != "rl_games" or path.suffix != ".py":
        raise AuditError(f"vendored Python path is outside rl_games/**/*.py: {value!r}")
    return path


def _check_python_files(vendor_root: Path, manifest: dict[str, Any]) -> tuple[int, str]:
    entries = manifest["python_files"]
    if not isinstance(entries, list) or len(entries) != PYTHON_FILE_COUNT:
        raise AuditError(
            f"python_files must contain exactly {PYTHON_FILE_COUNT} entries, "
            f"got {len(entries) if isinstance(entries, list) else type(entries).__name__}"
        )

    entries_by_path: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != PYTHON_ENTRY_KEYS:
            raise AuditError(f"python_files[{index}] schema mismatch")
        relative_path = _safe_python_path(entry["path"])
        path_string = relative_path.as_posix()
        if path_string in entries_by_path:
            raise AuditError(f"duplicate Python manifest path: {path_string}")
        expected_source_path = f"rl_games/{path_string}"
        if entry["source_path"] != expected_source_path:
            raise AuditError(
                f"Source path mismatch for {path_string}: expected {expected_source_path}, "
                f"got {entry['source_path']!r}"
            )
        if not isinstance(entry["source_blob"], str) or not HEX40.fullmatch(entry["source_blob"]):
            raise AuditError(f"invalid Source blob OID for {path_string}")
        if not isinstance(entry["sha256"], str) or not HEX64.fullmatch(entry["sha256"]):
            raise AuditError(f"invalid SHA256 for {path_string}")
        entries_by_path[path_string] = entry

    actual_by_path = {
        path.relative_to(vendor_root).as_posix(): path for path in vendor_root.rglob("*.py")
    }
    expected_paths = set(entries_by_path)
    actual_paths = set(actual_by_path)
    missing = sorted(expected_paths - actual_paths)
    if missing:
        raise AuditError(f"missing Python files: {missing}")
    extra = sorted(actual_paths - expected_paths)
    if extra:
        raise AuditError(f"extra Python files: {extra}")

    canonical_records: list[list[Any]] = []
    for relative_path in sorted(expected_paths):
        path = actual_by_path[relative_path]
        _require_regular_file(path, f"vendored Python file {relative_path}")
        entry = entries_by_path[relative_path]
        canonical_records.append(
            [relative_path, entry["source_path"], entry["source_blob"], entry["sha256"]]
        )

    payload = json.dumps(canonical_records, ensure_ascii=True, separators=(",", ":")).encode()
    selection_sha256 = hashlib.sha256(payload).hexdigest()
    if selection_sha256 != SOURCE_SELECTION_SHA256:
        raise AuditError(
            "Source selection anchor mismatch: "
            f"expected {SOURCE_SELECTION_SHA256}, got {selection_sha256}"
        )
    # Informational only (report field); the per-file SHA256 gate above is the
    # integrity barrier, so the total size must not reject mutated trees.
    python_byte_count = sum(path.stat().st_size for path in actual_by_path.values())

    patches = _check_declared_patches(vendor_root, manifest, entries_by_path)
    for relative_path in sorted(expected_paths):
        path = actual_by_path[relative_path]
        entry = entries_by_path[relative_path]
        actual_data = path.read_bytes()
        actual_sha256 = hashlib.sha256(actual_data).hexdigest()
        patch = patches.get(relative_path)
        if actual_sha256 == entry["sha256"]:
            if patch is not None:
                raise AuditError(f"redundant compatibility allowlist entry: {relative_path}")
            actual_blob = _git_blob_sha(actual_data)
            if actual_blob != entry["source_blob"]:
                raise AuditError(
                    f"Source Git blob drift for {relative_path}: "
                    f"expected {entry['source_blob']}, got {actual_blob}"
                )
            continue
        if patch is None:
            raise AuditError(
                f"unlisted compatibility patch SHA256 drift for {relative_path}: "
                f"expected pristine {entry['sha256']}, got {actual_sha256}"
            )
        if actual_sha256 != patch["patched_sha256"]:
            raise AuditError(
                f"patched SHA256 drift for {relative_path}: "
                f"expected {patch['patched_sha256']}, got {actual_sha256}"
            )
    return python_byte_count, selection_sha256


def _check_vendor_inventory(vendor_root: Path, manifest: dict[str, Any]) -> None:
    expected_python = {entry["path"] for entry in manifest["python_files"]}
    expected_files = expected_python | APPROVED_METADATA_FILES
    actual_files = {
        path.relative_to(vendor_root).as_posix()
        for path in vendor_root.rglob("*")
        if path.is_file() or path.is_symlink() or not path.is_dir()
    }

    missing = sorted(expected_files - actual_files)
    if missing:
        raise AuditError(f"missing vendor files: {missing}")
    extra = sorted(actual_files - expected_files)
    if extra:
        raise AuditError(f"extra vendor files: {extra}")
    for relative_path in sorted(expected_files):
        _require_regular_file(
            vendor_root / relative_path, f"vendor inventory member {relative_path}"
        )


def _check_fixed_metadata(vendor_root: Path) -> None:
    for relative_path, expected_sha256 in FIXED_METADATA_SHA256.items():
        data = (vendor_root / relative_path).read_bytes()
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if actual_sha256 != expected_sha256:
            raise AuditError(
                f"metadata SHA256 drift for {relative_path}: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )


def _check_license(vendor_root: Path, manifest: dict[str, Any]) -> None:
    expected_entry = {
        "path": "LICENSE",
        "source_path": "rl_games/LICENSE",
        "source_blob": SOURCE_LICENSE_BLOB,
        "sha256": SOURCE_LICENSE_SHA256,
    }
    if manifest["license"] != expected_entry:
        raise AuditError(
            f"license provenance mismatch: expected {expected_entry}, got {manifest['license']!r}"
        )

    license_path = vendor_root / "LICENSE"
    _require_regular_file(license_path, "license")
    data = license_path.read_bytes()
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 != SOURCE_LICENSE_SHA256:
        raise AuditError(
            f"license SHA256 drift: expected {SOURCE_LICENSE_SHA256}, got {actual_sha256}"
        )
    actual_blob = _git_blob_sha(data)
    if actual_blob != SOURCE_LICENSE_BLOB:
        raise AuditError(
            f"license Source Git blob drift: expected {SOURCE_LICENSE_BLOB}, got {actual_blob}"
        )


def _check_source_packaging(manifest: dict[str, Any]) -> None:
    expected_entry = {
        "source_path": "rl_games/pyproject.toml",
        "source_blob": SOURCE_PACKAGING_BLOB,
        "sha256": SOURCE_PACKAGING_SHA256,
    }
    if manifest["source_packaging"] != expected_entry:
        raise AuditError(
            "Source packaging provenance mismatch: "
            f"expected {expected_entry}, got {manifest['source_packaging']!r}"
        )


def audit_vendor(vendor_root: Path = DEFAULT_VENDOR_ROOT) -> AuditReport:
    """Audit pristine Source provenance and reviewed V2 compatibility bytes."""
    vendor_root = vendor_root.absolute()
    if vendor_root.is_symlink():
        raise AuditError(f"vendor root symlink is forbidden: {vendor_root}")
    if not vendor_root.is_dir():
        raise AuditError(f"vendor root is missing: {vendor_root}")

    manifest = _load_manifest(vendor_root)
    _check_source_identity(manifest)
    python_byte_count, selection_sha256 = _check_python_files(vendor_root, manifest)
    _check_vendor_inventory(vendor_root, manifest)
    _check_fixed_metadata(vendor_root)
    _check_license(vendor_root, manifest)
    _check_source_packaging(manifest)
    return AuditReport(
        source_head=SOURCE_HEAD,
        source_parent_tree=SOURCE_PARENT_TREE,
        python_file_count=PYTHON_FILE_COUNT,
        python_byte_count=python_byte_count,
        source_selection_sha256=selection_sha256,
        license_blob=SOURCE_LICENSE_BLOB,
    )


def _run_ruff(repo_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["ruff", *arguments],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise AuditError(f"Ruff could not be executed: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise AuditError(f"Ruff {' '.join(arguments)} failed: {detail}")
    return result.stdout


def audit_root_ruff(repo_root: Path = REPO_ROOT) -> None:
    """Require the vendored tree to be invisible to Ruff via the root ``.ignore``.

    The vendored bytes are hash-pinned, so formatters/linters must never
    rewrite them. This repo cannot use ``pyproject.toml`` ``extend-exclude``
    for the vendor (the root project config is shared), so exclusion lives in
    the root ``.ignore`` file, which Ruff honors like ripgrep does. The
    ``--show-files`` probe additionally proves the effective file set.
    """
    repo_root = repo_root.resolve()
    ignore_path = repo_root / ".ignore"
    _require_regular_file(ignore_path, "root .ignore")
    entries = [
        line.strip()
        for line in ignore_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if f"{VENDOR_TOOL_EXCLUSION}/" not in entries:
        raise AuditError(
            f"root .ignore is missing exact member {VENDOR_TOOL_EXCLUSION + '/'!r}: {entries}"
        )

    shown_files = _run_ruff(repo_root, "check", "--show-files", ".")
    vendor_root = (repo_root / VENDOR_TOOL_EXCLUSION).resolve()
    leaked = []
    for line in shown_files.splitlines():
        shown_path = Path(line.strip())
        if not shown_path.is_absolute():
            shown_path = repo_root / shown_path
        shown_path = shown_path.resolve()
        if shown_path == vendor_root or vendor_root in shown_path.parents:
            leaked.append(shown_path.relative_to(repo_root).as_posix())
    if leaked:
        raise AuditError(f"Ruff --show-files includes pristine vendor files: {sorted(leaked)}")


def audit_root_git_attributes(repo_root: Path = REPO_ROOT) -> None:
    _require_regular_file(repo_root / ".gitattributes", "Git whitespace attribute file")
    prefix = "third_party/simtoolreal_rl_games/rl_games/"
    probes = (
        prefix + "torch_runner.py",
        "scripts/audit_simtoolreal_rlgames_vendor.py",
    )
    values = ("unset", "unspecified")
    command = ["git", "-c", "core.attributesFile=", "check-attr", "whitespace", "--", *probes]
    try:
        if (repo_root / ".gitattributes").read_bytes() != GIT_ATTRIBUTES_CONTENT:
            raise AuditError("Git whitespace attribute file content mismatch")
        output = subprocess.check_output(command, cwd=repo_root, text=True)
    except (OSError, UnicodeDecodeError, subprocess.CalledProcessError) as exc:
        raise AuditError(f"Git whitespace attribute check failed: {exc}") from exc
    if output.splitlines() != [f"{p}: whitespace: {v}" for p, v in zip(probes, values)]:
        raise AuditError(f"Git whitespace attribute output/semantics mismatch: {output!r}")


def main() -> int:
    try:
        report = audit_vendor()
        audit_root_ruff()
        audit_root_git_attributes()
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Source HEAD: {report.source_head}")
    print(f"Source parent tree: {report.source_parent_tree}")
    print(f"License blob: {report.license_blob}")
    print(f"{report.python_file_count} selected Python blobs verified")
    print(f"Source Python bytes: {report.python_byte_count}")
    print(f"Source selection anchor: {report.source_selection_sha256}")
    print("Compatibility allowlist: 7 reviewed patches verified (V2)")
    print("Root Ruff formatter isolation: verified")
    print("Root Git whitespace isolation: verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
