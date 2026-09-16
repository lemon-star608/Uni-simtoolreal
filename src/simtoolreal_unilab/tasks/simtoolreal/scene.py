"""Cold-path scene inputs for SimToolReal.

The manager-based factory keeps scene roles in the owner YAML; this module owns
asset-path resolution and deterministic bootstrap object assets so the cold
path can scan robot/table/object/goalviz without hardcoding absolute paths.
The full 1200-file tool pool remains an init-randomization adapter concern.

Resolution order for the assets root:
1. ``SIMTOOLREAL_ASSETS_ROOT`` environment variable (explicit override);
2. the vendored assets distributed with this package
   (``simtoolreal_unilab/assets/simtoolreal``, provenance-checked by
   ``simtoolreal-assets``) — the default, so a fresh clone needs nothing else;
3. the ``simtoolreal/assets`` directory of a sibling checkout of the original
   repository (the historical workspace layout, kept for existing setups).

Anything else fails closed with an explanatory error instead of silently
producing a dangling scene path.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from ...assets import ASSET_ROOT as _VENDORED_ASSET_ROOT
from .config import AssetsCfg
from .tool_catalog import build_tool_catalog
from .tool_urdf_writer import write_tool_urdf

_ASSETS_ROOT_ENV = "SIMTOOLREAL_ASSETS_ROOT"
_RESOLVER_NAME = "simtoolreal_asset"


def resolve_assets_root() -> Path:
    """Return the SimToolReal assets root, failing closed when absent."""
    override = os.environ.get(_ASSETS_ROOT_ENV, "").strip()
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    # Vendored assets ship with the package: urdf/<robot>, urdf/table_narrow.urdf.
    candidates.append(_VENDORED_ASSET_ROOT)
    # <workspace>/simtoolreal/assets for a checkout layout with the original
    # repository cloned next to this repository.
    candidates.append(Path(__file__).resolve().parents[4].parent / "simtoolreal" / "assets")
    for candidate in candidates:
        if (candidate / AssetsCfg().robot_urdf).is_file():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "SimToolReal assets root not found (looked for "
        f"{AssetsCfg().robot_urdf} in: {searched}). The vendored assets ship "
        "with this package; set SIMTOOLREAL_ASSETS_ROOT only to override them "
        "with a checkout of the original simtoolreal repository."
    )


def resolve_asset(relative_path: str) -> str:
    """Resolve one asset path below the assets root to an absolute path."""
    if relative_path.startswith("generated/"):
        return _resolve_generated_asset(relative_path.removeprefix("generated/"))
    path = resolve_assets_root() / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"SimToolReal asset does not exist: {path}")
    return str(path)


def _resolve_generated_asset(name: str) -> str:
    """Materialize the deterministic bootstrap object/goal asset on demand.

    The full 1200-file variant pool is an init-randomization concern and is
    intentionally kept out of Hydra resolution.  The multi-asset scene still
    needs concrete source files for cold-path metadata scans, so use one
    canonical catalog entry for the scene declaration and reserve pool
    assignment for the IsaacSim worker integration step.
    """
    if name not in {"object.urdf", "goalviz.urdf"}:
        raise ValueError(
            "unknown generated SimToolReal asset; expected object.urdf or goalviz.urdf, "
            f"got {name!r}"
        )
    root = (
        Path(os.environ.get("SIMTOOLREAL_GENERATED_ASSETS_ROOT", "")).expanduser()
        if os.environ.get("SIMTOOLREAL_GENERATED_ASSETS_ROOT", "").strip()
        else Path(tempfile.gettempdir()) / "simtoolreal_generated_assets"
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    if path.is_file():
        return str(path)

    spec = build_tool_catalog(types=("hammer",), num_per_type=1, seed=42, shuffle=False)[0]
    if name == "object.urdf":
        write_tool_urdf(spec, path)
    else:
        # GoalViz has the same geometry and inertial values as the bootstrap
        # object but a distinct root-link namespace, which is required by the
        # public multi-asset body map.
        object_path = root / "object.urdf"
        write_tool_urdf(spec, object_path)
        path.write_text(object_path.read_text().replace("object_root", "goalviz_root"))
    return str(path)


def register_resolvers() -> None:
    """Register the ``${simtoolreal_asset:...}`` OmegaConf resolver once."""
    if not OmegaConf.has_resolver(_RESOLVER_NAME):
        OmegaConf.register_new_resolver(_RESOLVER_NAME, resolve_asset)


__all__ = ["register_resolvers", "resolve_asset", "resolve_assets_root"]
