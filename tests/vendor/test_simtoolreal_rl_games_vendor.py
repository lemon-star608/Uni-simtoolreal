import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

SOURCE_HEAD = "2a9917533bfea70419ed2667a511d7238e5b3abc"
SOURCE_PARENT_TREE = "7a6a0bb090998d00565aaefa6ab9f2b3d356ace2"
SOURCE_LICENSE_BLOB = "313ca229e6ca879466f94bff49362fb65667e22f"
SOURCE_LICENSE_SHA256 = "46565837dec017dc2f8df9fbbb6904fb3e62dc9b91c9efd9bb8d0b22eacc47d5"
SOURCE_SELECTION_SHA256 = "2f100c78073ded75cd5c7d6c5e55523c729f4b5524d61915ed7efccea1117e4c"
SOURCE_PYTHON_BYTES = 439562
EXPECTED_DISTRIBUTION = "unilab-simtoolreal-rl-games"
EXPECTED_VERSION = "1.6.1+simtoolreal.2a991753.compat2"
EXPECTED_PATCH_PATHS = {
    "rl_games/algos_torch/players.py",
    "rl_games/common/a2c_common.py",
    "rl_games/common/env_configurations.py",
    "rl_games/common/experience.py",
    "rl_games/common/player.py",
    "rl_games/common/vecenv.py",
    "rl_games/common/wrappers.py",
}
VENDOR_ROOT = Path(__file__).resolve().parents[2] / "third_party/simtoolreal_rl_games"
GIT_TEST_CONFIG = ("-c", "core.whitespace=trailing-space", "-c", "core.attributesFile=")
PATCH_ALLOWLIST_KEY = "compatibility" + "_allowlist"


def _git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()


def _git(repo_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    command = ["git", *GIT_TEST_CONFIG, "-C", repo_root, *arguments]
    return subprocess.run(command, capture_output=True, text=True)


def test_vendor_manifest_pins_pristine_objects_and_current_python_hashes():
    manifest_path = VENDOR_ROOT / "source_manifest.json"
    assert manifest_path.is_file(), f"vendor manifest does not exist: {manifest_path}"

    manifest = json.loads(manifest_path.read_text())
    assert manifest["manifest_version"] == 2
    assert manifest["source_head"] == SOURCE_HEAD
    assert manifest["source_parent_tree"] == SOURCE_PARENT_TREE
    assert len(manifest["python_files"]) == 72

    manifest_paths = [entry["path"] for entry in manifest["python_files"]]
    actual_paths = [path.relative_to(VENDOR_ROOT).as_posix() for path in VENDOR_ROOT.rglob("*.py")]
    assert len(manifest_paths) == len(set(manifest_paths))
    assert sorted(manifest_paths) == sorted(actual_paths)

    patches = {entry["path"]: entry for entry in manifest[PATCH_ALLOWLIST_KEY]}
    assert set(patches) == EXPECTED_PATCH_PATHS
    assert [entry["path"] for entry in manifest[PATCH_ALLOWLIST_KEY]] == sorted(patches)

    for patch in patches.values():
        assert set(patch) == {
            "path",
            "pristine_blob",
            "pristine_sha256",
            "patched_sha256",
            "reason",
            "covering_test",
        }
        assert patch["reason"].strip()
        assert patch["covering_test"].startswith("tests/")

    canonical_records = []
    for entry in sorted(manifest["python_files"], key=lambda item: item["path"]):
        assert set(entry) == {"path", "source_path", "source_blob", "sha256"}
        assert entry["source_path"] == f"rl_games/{entry['path']}"
        canonical_records.append(
            [entry["path"], entry["source_path"], entry["source_blob"], entry["sha256"]]
        )

        vendored_file = VENDOR_ROOT / entry["path"]
        data = vendored_file.read_bytes()
        patch = patches.get(entry["path"])
        expected_current = patch["patched_sha256"] if patch else entry["sha256"]
        assert hashlib.sha256(data).hexdigest() == expected_current
        if patch:
            assert patch["pristine_blob"] == entry["source_blob"]
            assert patch["pristine_sha256"] == entry["sha256"]
            assert expected_current != entry["sha256"]
        else:
            assert _git_blob_sha(data) == entry["source_blob"]

    payload = json.dumps(canonical_records, ensure_ascii=True, separators=(",", ":")).encode()
    assert hashlib.sha256(payload).hexdigest() == SOURCE_SELECTION_SHA256


def test_vendor_license_is_the_exact_source_blob():
    manifest = json.loads((VENDOR_ROOT / "source_manifest.json").read_text())
    license_entry = manifest["license"]
    data = (VENDOR_ROOT / "LICENSE").read_bytes()

    assert license_entry == {
        "path": "LICENSE",
        "source_path": "rl_games/LICENSE",
        "source_blob": SOURCE_LICENSE_BLOB,
        "sha256": SOURCE_LICENSE_SHA256,
    }
    assert _git_blob_sha(data) == SOURCE_LICENSE_BLOB
    assert hashlib.sha256(data).hexdigest() == SOURCE_LICENSE_SHA256


def test_declared_patches_are_documented_one_for_one():
    manifest = json.loads((VENDOR_ROOT / "source_manifest.json").read_text())
    patches = manifest[PATCH_ALLOWLIST_KEY]
    patches_text = (VENDOR_ROOT / "PATCHES.md").read_text()

    assert {entry["path"] for entry in patches} == EXPECTED_PATCH_PATHS
    for entry in patches:
        assert patches_text.count(f"`{entry['path']}`") == 1
        assert entry["pristine_blob"] in patches_text
        assert entry["pristine_sha256"] in patches_text
        assert entry["patched_sha256"] in patches_text
        assert entry["reason"] in patches_text
        assert entry["covering_test"] in patches_text


def test_nested_distribution_metadata_is_minimal_and_python_310_to_313():
    metadata = tomllib.loads((VENDOR_ROOT / "pyproject.toml").read_text())
    project = metadata["project"]

    assert project["name"] == EXPECTED_DISTRIBUTION
    assert project["version"] == EXPECTED_VERSION
    assert project["requires-python"] == ">=3.10,<3.14"
    assert project["dependencies"] == [
        "gym==0.26.2",
        "gymnasium>=1.0,<2",
        "numpy>=1.24",
        "omegaconf>=2.3",
        "tensorboard>=2.8",
        "tensorboardX>=2.5",
        "torch==2.7.0",
    ]
    assert "optional-dependencies" not in project


def test_editable_install_does_not_pollute_vendor_inventory(tmp_path):
    vendor_copy = shutil.copytree(VENDOR_ROOT, tmp_path / "vendor")
    before = {
        path.relative_to(vendor_copy).as_posix()
        for path in vendor_copy.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    venv = tmp_path / "venv"
    create = subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(venv)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert create.returncode == 0, create.stderr or create.stdout
    interpreter = venv / "bin/python"
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        str(interpreter),
        "--no-deps",
        "--editable",
        str(vendor_copy),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
    installed = subprocess.run(
        ["uv", "pip", "show", "--python", str(interpreter), EXPECTED_DISTRIBUTION],
        check=False,
        capture_output=True,
        text=True,
    )
    assert installed.returncode == 0, installed.stderr or installed.stdout
    assert f"Name: {EXPECTED_DISTRIBUTION}" in installed.stdout
    assert f"Version: {EXPECTED_VERSION}" in installed.stdout
    assert f"Editable project location: {vendor_copy}" in installed.stdout
    after = {
        path.relative_to(vendor_copy).as_posix()
        for path in vendor_copy.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    assert after == before


def test_readme_distinguishes_excluded_source_tests_from_runtime_test_modules():
    readme = (VENDOR_ROOT / "README.md").read_text()
    assert "72 pristine Source identities" in readme
    assert "7 reviewed compatibility patches" in readme
    assert "65 Python files remain byte-identical" in readme
    assert "Source top-level `rl_games/tests/` test suite" in readme
    assert "`common/test_utils.py`" in readme
    assert "`envs/test/**`" in readme
    assert "`envs/test_network.py`" in readme


def test_audit_root_ruff_accepts_effective_vendor_exclusion():
    _audit_module().audit_root_ruff(VENDOR_ROOT.parents[1])


def test_root_git_attributes_are_exact_and_scoped(tmp_path):
    _audit_module().audit_root_git_attributes(repo_root := VENDOR_ROOT.parents[1])
    subprocess.run(["git", "init", "-q", isolated := tmp_path / "repo"], check=True)
    shutil.copy2(repo_root / ".gitattributes", isolated)
    vendor = isolated / "third_party/simtoolreal_rl_games/rl_games/vendor.py"
    vendor.parent.mkdir(parents=True)
    vendor.write_text("vendor = True \n")
    _git(isolated, "add", ".").check_returncode()
    assert _git(isolated, "diff", "--cached", "--check").returncode == 0
    (isolated / "ordinary.py").write_text("ordinary = True \n")
    _git(isolated, "add", "ordinary.py").check_returncode()
    result = _git(isolated, "diff", "--cached", "--check")
    assert result.returncode
    assert "ordinary.py" in result.stdout
    assert "vendor.py" not in result.stdout


def test_audit_root_git_attributes_rejects_crlf_similar_prefix_or_content_drift(tmp_path):
    audit = _audit_module()
    _git(tmp_path, "init", "-q").check_returncode()
    rule = (VENDOR_ROOT.parents[1] / ".gitattributes").read_bytes()
    for drifted in (rule[:-1] + b"\r\n", rule.replace(b"real", b"realx"), rule + b"\n"):
        (tmp_path / ".gitattributes").write_bytes(drifted)
        with pytest.raises(audit.AuditError, match="Git whitespace.*content mismatch"):
            audit.audit_root_git_attributes(tmp_path)


@pytest.mark.parametrize(
    "ignore_content",
    [
        "# third_party/simtoolreal_rl_games/\n",
        "third_party/simtoolreal_rl_games\n",
        "third_party/simtoolreal_rl_games_not/\n",
        "\n",
    ],
    ids=["comment-only", "missing-trailing-slash", "prefix-only", "empty"],
)
def test_audit_root_ruff_rejects_ineffective_vendor_exclusion(tmp_path, ignore_content):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".ignore").write_text(ignore_content)
    (repo_root / "probe.py").write_text("value = 1\n")

    audit = _audit_module()
    with pytest.raises(audit.AuditError, match="ignore"):
        audit.audit_root_ruff(repo_root)


def _audit_module():
    return importlib.import_module("scripts.audit_simtoolreal_rlgames_vendor")


def _copy_vendor(tmp_path: Path) -> Path:
    return shutil.copytree(VENDOR_ROOT, tmp_path / "simtoolreal_rl_games")


def _rewrite_manifest(vendor_root: Path, mutate) -> None:
    manifest_path = vendor_root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def test_audit_accepts_the_dual_hash_snapshot():
    report = _audit_module().audit_vendor(VENDOR_ROOT)
    assert (report.python_file_count, report.python_byte_count) == (72, SOURCE_PYTHON_BYTES)
    assert report.source_selection_sha256 == SOURCE_SELECTION_SHA256
    assert (report.source_head, report.source_parent_tree) == (SOURCE_HEAD, SOURCE_PARENT_TREE)


def test_audit_rejects_vendor_root_symlinks(tmp_path):
    real = _copy_vendor(tmp_path)
    audit = _audit_module()
    for root, target in ((tmp_path / "vendor", real), (tmp_path / "broken", tmp_path / "missing")):
        root.symlink_to(target, target_is_directory=True)
        with pytest.raises(audit.AuditError, match=r"vendor root.*symlink"):
            audit.audit_vendor(root)


def test_audit_rejects_a_missing_python_file(tmp_path):
    vendor_root = _copy_vendor(tmp_path)
    (vendor_root / "rl_games/torch_runner.py").unlink()

    audit = _audit_module()
    with pytest.raises(audit.AuditError, match="missing Python files"):
        audit.audit_vendor(vendor_root)


def test_audit_rejects_an_extra_python_file(tmp_path):
    vendor_root = _copy_vendor(tmp_path)
    (vendor_root / "rl_games/unreviewed.py").write_text("unreviewed = True\n")

    audit = _audit_module()
    with pytest.raises(audit.AuditError, match="extra Python files"):
        audit.audit_vendor(vendor_root)


@pytest.mark.parametrize(
    "relative_path",
    [
        "rl_games/configs/forbidden.yaml",
        "rl_games/torch_runner.so",
        "injected.pth",
    ],
    ids=["extra-yaml", "same-module-so", "top-level-pth"],
)
def test_audit_rejects_extra_vendor_inventory(tmp_path, relative_path):
    vendor_root = _copy_vendor(tmp_path)
    extra = vendor_root / relative_path
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"unapproved\n")

    audit = _audit_module()
    with pytest.raises(audit.AuditError, match="extra vendor files"):
        audit.audit_vendor(vendor_root)


@pytest.mark.parametrize("relative_path", ["README.md", "UPSTREAM.md", "pyproject.toml"])
def test_audit_rejects_missing_required_metadata(tmp_path, relative_path):
    vendor_root = _copy_vendor(tmp_path)
    (vendor_root / relative_path).unlink()

    audit = _audit_module()
    with pytest.raises(audit.AuditError, match="missing vendor files"):
        audit.audit_vendor(vendor_root)


def test_audit_rejects_symlinked_required_metadata(tmp_path):
    vendor_root = _copy_vendor(tmp_path)
    readme = vendor_root / "README.md"
    readme.unlink()
    readme.symlink_to("UPSTREAM.md")

    audit = _audit_module()
    with pytest.raises(audit.AuditError, match="must be a regular file"):
        audit.audit_vendor(vendor_root)


@pytest.mark.parametrize("relative_path", ["README.md", "UPSTREAM.md", "pyproject.toml"])
def test_audit_rejects_metadata_content_drift(tmp_path, relative_path):
    vendor_root = _copy_vendor(tmp_path)
    metadata = vendor_root / relative_path
    metadata.write_bytes(metadata.read_bytes() + b"\n")

    audit = _audit_module()
    with pytest.raises(audit.AuditError, match="metadata SHA256 drift"):
        audit.audit_vendor(vendor_root)


def test_audit_rejects_python_hash_drift(tmp_path):
    vendor_root = _copy_vendor(tmp_path)
    drifted = vendor_root / "rl_games/torch_runner.py"
    drifted.write_bytes(drifted.read_bytes() + b"\n")

    audit = _audit_module()
    with pytest.raises(audit.AuditError, match="SHA256 drift"):
        audit.audit_vendor(vendor_root)


def test_audit_rejects_an_unlisted_patch(tmp_path):
    vendor_root = _copy_vendor(tmp_path)
    unlisted = vendor_root / "rl_games/torch_runner.py"
    unlisted.write_bytes(unlisted.read_bytes() + b"\n")

    audit = _audit_module()
    with pytest.raises(audit.AuditError, match="unlisted compatibility patch"):
        audit.audit_vendor(vendor_root)


def test_audit_rejects_a_redundant_patch_allowlist_entry(tmp_path):
    vendor_root = _copy_vendor(tmp_path)

    def make_redundant(manifest: dict) -> None:
        patch = manifest[PATCH_ALLOWLIST_KEY][0]
        patch["patched_sha256"] = patch["pristine_sha256"]

    _rewrite_manifest(vendor_root, make_redundant)

    audit = _audit_module()
    with pytest.raises(audit.AuditError, match="redundant patch allowlist"):
        audit.audit_vendor(vendor_root)


def test_audit_rejects_a_patched_hash_mismatch(tmp_path):
    vendor_root = _copy_vendor(tmp_path)
    manifest = json.loads((vendor_root / "source_manifest.json").read_text())
    patch = manifest[PATCH_ALLOWLIST_KEY][0]
    patched = vendor_root / patch["path"]
    patched.write_bytes(patched.read_bytes() + b"\n")

    audit = _audit_module()
    with pytest.raises(audit.AuditError, match="patched SHA256 drift"):
        audit.audit_vendor(vendor_root)


def test_audit_rejects_coordinated_python_and_manifest_hash_drift(tmp_path):
    vendor_root = _copy_vendor(tmp_path)
    drifted = vendor_root / "rl_games/torch_runner.py"
    drifted_data = drifted.read_bytes() + b"\n"
    drifted.write_bytes(drifted_data)

    def update_hashes(manifest):
        entry = next(
            item for item in manifest["python_files"] if item["path"] == "rl_games/torch_runner.py"
        )
        entry["source_blob"] = _git_blob_sha(drifted_data)
        entry["sha256"] = hashlib.sha256(drifted_data).hexdigest()

    _rewrite_manifest(vendor_root, update_hashes)

    audit = _audit_module()
    with pytest.raises(audit.AuditError, match="Source selection anchor mismatch"):
        audit.audit_vendor(vendor_root)


def test_audit_rejects_source_identity_changes(tmp_path):
    vendor_root = _copy_vendor(tmp_path)
    _rewrite_manifest(vendor_root, lambda manifest: manifest.update(source_head="0" * 40))

    audit = _audit_module()
    with pytest.raises(audit.AuditError, match="source_head identity mismatch"):
        audit.audit_vendor(vendor_root)


def test_audit_rejects_a_missing_license(tmp_path):
    vendor_root = _copy_vendor(tmp_path)
    (vendor_root / "LICENSE").unlink()

    audit = _audit_module()
    with pytest.raises(audit.AuditError, match=r"missing vendor files.*LICENSE"):
        audit.audit_vendor(vendor_root)


def test_audit_rejects_an_allowlist_entry_for_an_unchanged_file(tmp_path):
    vendor_root = _copy_vendor(tmp_path)
    manifest = json.loads((vendor_root / "source_manifest.json").read_text())
    source_entry = next(
        entry for entry in manifest["python_files"] if entry["path"] == "rl_games/torch_runner.py"
    )

    def add_redundant_entry(current_manifest):
        current_manifest[PATCH_ALLOWLIST_KEY].append(
            {
                "path": source_entry["path"],
                "pristine_blob": source_entry["source_blob"],
                "pristine_sha256": source_entry["sha256"],
                "patched_sha256": source_entry["sha256"],
                "reason": "not a real patch",
                "covering_test": "tests/vendor/test_simtoolreal_rl_games_vendor.py",
            }
        )

    _rewrite_manifest(vendor_root, add_redundant_entry)

    audit = _audit_module()
    with pytest.raises(audit.AuditError, match="redundant patch allowlist"):
        audit.audit_vendor(vendor_root)


def test_audit_rejects_patch_documentation_drift(tmp_path):
    vendor_root = _copy_vendor(tmp_path)
    patches = vendor_root / "PATCHES.md"
    patches.write_text(patches.read_text().replace("Gymnasium", "unreviewed rewrite", 1))

    audit = _audit_module()
    with pytest.raises(audit.AuditError, match="PATCHES.md"):
        audit.audit_vendor(vendor_root)
