"""Task bootstrap and package dependency boundary tests (sub-repo scope)."""

from __future__ import annotations

import ast
from pathlib import Path

from unilab.base import registry

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TASK_PACKAGE = _REPO_ROOT / "src" / "simtoolreal_unilab" / "tasks"

# The task package declares the manager-based env and its terms only: it must
# stay learner-agnostic (no RL-Games imports) and must not reach into other
# task packages or unilab's private internals.
_ALLOWED_TASK_IMPORT_ROOTS = (
    "simtoolreal_unilab",
    "unilab.base",
    "unilab.envs",
    "unilab.managers",
    "unilab.utils",
    "unilab.training",
    "unisim",
    "numpy",
    "omegaconf",
    "hydra",
)

_ALLOWED_STDLIB = {
    "__future__",
    "dataclasses",
    "enum",
    "functools",
    "hashlib",
    "math",
    "os",
    "pathlib",
    "re",
    "typing",
    "collections",
    "itertools",
    "json",
    "random",
    "tempfile",
    "warnings",
    "xml",  # ElementTree: the MJCF writers' transcription parser (no physics engine)
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: stays inside the task package
                continue
            if node.module:
                modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def _is_allowed(module: str) -> bool:
    if module.split(".")[0] in _ALLOWED_STDLIB:
        return True
    if module == "unilab" or module.startswith("unilab."):
        return module == "unilab" or any(
            module == root or module.startswith(root + ".")
            for root in _ALLOWED_TASK_IMPORT_ROOTS
            if root.startswith("unilab")
        )
    return any(
        module == root or module.startswith(root + ".")
        for root in _ALLOWED_TASK_IMPORT_ROOTS
        if not root.startswith("unilab")
    )


def test_task_package_registers_simtoolreal_on_import() -> None:
    import simtoolreal_unilab.tasks  # noqa: F401

    registered = registry.registered_envs() if hasattr(registry, "registered_envs") else None
    if registered is None:
        # Registry API varies across revisions; fall back to the entry point.
        import importlib.metadata as md

        eps = md.entry_points(group="unilab.tasks")
        assert any(ep.name == "simtoolreal_unilab" for ep in eps)
    else:
        assert any("SimToolReal" in str(name) for name in registered)


def test_task_package_boundary() -> None:
    violations = [
        (path.relative_to(_REPO_ROOT).as_posix(), module)
        for path in sorted(_TASK_PACKAGE.rglob("*.py"))
        for module in sorted(_imports(path))
        if not _is_allowed(module)
    ]
    assert violations == [], f"task package boundary violations: {violations}"


def test_compat_layer_is_applied() -> None:
    from unilab.base.entity import Entity
    from unilab.base.reset_state import ResetStateTransaction

    assert hasattr(ResetStateTransaction, "bind_root_state")
    assert hasattr(Entity, "fixed_variant_mass")
