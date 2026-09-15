"""SimToolReal owner configuration and the fixed model variant catalog.

The UniSim adapter owns the wire contract for whole-file model variants.  This
module is the task-owner side of that contract: it builds the deterministic
1200-tool catalog, materializes the URDF pool, and declares the pool as
``EnvCfg.fixed_model_variants``.  The upstream ``make_manager_based_rl_env``
guard turns that catalog into ``cfg.scene.fixed_variant_plan`` (an int32
round-robin assignment over the catalog order, identical to this task's
historical formula) and negotiates the backend capability before the worker
materializes the pool.  Variant masses are worker-measured and read back
through the backend contract; only the object-scale table stays on this
config for the command hub's expansion.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from unilab.base.variants import FixedModelVariantCatalogCfg, FixedModelVariantCfg
from unilab.envs import ManagerBasedRlEnvCfg

from .tool_catalog import build_tool_catalog
from .tool_urdf_writer import write_tool_catalog_urdfs

_DEFAULT_NUM_PER_TYPE = 100
_DEFAULT_SEED = 42
_DEFAULT_TARGET_ENTITY = "object"


def _pool_root(seed: int, num_per_type: int, configured: str | None) -> Path:
    if configured is not None and str(configured).strip():
        root = Path(str(configured)).expanduser()
    else:
        env_root = os.environ.get("SIMTOOLREAL_TOOL_POOL_ROOT", "").strip()
        base = Path(env_root).expanduser() if env_root else Path(tempfile.gettempdir())
        root = base / "simtoolreal_tool_pool"
    return root / f"seed_{seed}_n{num_per_type}"


def _materialize_pool(
    *, seed: int, num_per_type: int, configured_root: str | None
) -> tuple[tuple[Path, ...], tuple[float, ...], tuple[tuple[float, float, float], ...]]:
    """Write or reuse the deterministic URDF pool with its mass/scale tables."""

    catalog = build_tool_catalog(
        num_per_type=num_per_type,
        seed=seed,
        shuffle=True,
    )
    root = _pool_root(seed, num_per_type, configured_root)
    manifest_path = root / "manifest.json"
    expected = {
        "seed": int(seed),
        "num_per_type": int(num_per_type),
        "count": len(catalog),
    }
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            names = tuple(str(value) for value in manifest["files"])
            masses = tuple(float(value) for value in manifest["masses"])
            scales = tuple(
                tuple(float(component) for component in scale) for scale in manifest["scales"]
            )
            scales_complete = len(scales) == len(catalog) and all(
                len(scale) == 3 for scale in scales
            )
            if manifest.get("spec") == expected and len(names) == len(catalog):
                paths = tuple(root / name for name in names)
                if (
                    all(path.is_file() for path in paths)
                    and len(masses) == len(catalog)
                    and scales_complete
                ):
                    return paths, masses, scales
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            # A partial or stale pool is rebuilt below.  The worker never sees
            # this directory until the manifest and every source file exist.
            pass

    root.mkdir(parents=True, exist_ok=True)
    paths, _ = write_tool_catalog_urdfs(catalog, root)
    path_tuple = tuple(Path(path).resolve() for path in paths)
    masses = tuple(float(spec.mass) for spec in catalog)
    scales = tuple(tuple(float(x) for x in spec.object_scale) for spec in catalog)
    manifest_path.write_text(
        json.dumps(
            {
                "spec": expected,
                "files": [path.name for path in path_tuple],
                "masses": masses,
                "scales": scales,
            },
            indent=2,
        )
    )
    return path_tuple, masses, scales


def fixed_variant_plan_assignment(cfg: Any, num_envs: int) -> np.ndarray | None:
    """Return the scene plan's validated per-environment variant assignment.

    Single task-side reader for the immutable assignment the upstream factory
    guard builds into ``cfg.scene.fixed_variant_plan``.  Returns ``None`` when
    the scene carries no plan (pool disabled, lightweight fixtures); otherwise
    fails closed unless the assignment carries exactly one entry per
    environment within the plan's variant count.
    """

    scene = getattr(cfg, "scene", None)
    plan = getattr(scene, "fixed_variant_plan", None)
    if plan is None:
        return None
    assignment = np.asarray(plan.assignment)
    if assignment.ndim != 1 or assignment.shape[0] != num_envs:
        raise ValueError(
            "fixed_variant_plan assignment must have one entry per environment; got "
            f"{assignment.shape}, expected ({num_envs},)"
        )
    variant_count = len(plan.variants)
    if np.any(assignment < 0) or np.any(assignment >= variant_count):
        raise ValueError("fixed_variant_plan assignment contains an out-of-range variant id")
    return assignment.astype(np.int64, copy=False)


@dataclass
class SimToolRealEnvCfg(ManagerBasedRlEnvCfg):
    """Manager config with the task-owned 1200-tool cold-path contract."""

    tool_pool_num_per_type: int = _DEFAULT_NUM_PER_TYPE
    tool_pool_seed: int = _DEFAULT_SEED
    # Name of the scene entity the pool binds to; the owner YAML must declare
    # ``consumes_fixed_variant_pool: true`` on exactly that entity asset so
    # the backend binds the plan symmetrically.
    tool_pool_target_entity: str = _DEFAULT_TARGET_ENTITY
    tool_pool_root: str | None = None
    tool_pool_enabled: bool = True

    # Filled by ``validate()`` — after Hydra/registry overlays settle the pool
    # knobs and before the upstream factory guard reads the catalog — and read
    # by the command hub's object-scale expansion (scale deliberately stays
    # task-side) plus diagnostics.  Deliberately kept off the Hydra surface.
    tool_pool_scales: tuple[tuple[float, float, float], ...] = field(
        default_factory=tuple, init=False, repr=False
    )
    tool_pool_files: tuple[str, ...] = field(default_factory=tuple, init=False, repr=False)
    _tool_pool_materialized_from: tuple[int, int, str | None] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def validate(self) -> None:
        super().validate()
        if (
            isinstance(self.tool_pool_num_per_type, bool)
            or not isinstance(self.tool_pool_num_per_type, int)
            or self.tool_pool_num_per_type <= 0
        ):
            raise ValueError(
                "SimToolReal tool_pool_num_per_type must be a positive integer, got "
                f"{self.tool_pool_num_per_type!r}"
            )
        if (
            isinstance(self.tool_pool_seed, bool)
            or not isinstance(self.tool_pool_seed, int)
            or self.tool_pool_seed < 0
        ):
            raise ValueError(
                f"SimToolReal tool_pool_seed must be a non-negative integer, got {self.tool_pool_seed!r}"
            )
        if not isinstance(self.tool_pool_target_entity, str) or not self.tool_pool_target_entity:
            raise ValueError("SimToolReal tool_pool_target_entity must be a non-empty string")
        if self.tool_pool_root is not None and (
            not isinstance(self.tool_pool_root, str) or not self.tool_pool_root.strip()
        ):
            raise ValueError("SimToolReal tool_pool_root must be a non-empty string or None")
        if not isinstance(self.tool_pool_enabled, bool):
            raise ValueError("SimToolReal tool_pool_enabled must be bool")
        self._refresh_fixed_model_variants()

    def _refresh_fixed_model_variants(self) -> None:
        """Materialize the pool and refresh the task-owned variant catalog.

        Idempotent per pool-knob set: repeated ``validate()`` calls (registry,
        factory, env construction) reuse the already materialized catalog, and
        knob changes applied before the factory guard builds the scene plan
        rebuild from the new values.  Once ``cfg.scene.fixed_variant_plan``
        exists the plan is immutable: a knob change that would rebuild the
        catalog — or disabling the pool — fails closed instead of silently
        desynchronizing catalog and plan (final review M-6; e.g. a larger pool
        would leave the command hub's object-scale expansion indexing a stale
        table, and a disable would leave the backend materializing the pool
        the task just dropped).  With the pool disabled the catalog is
        ``None`` so the upstream guard leaves ``cfg.scene.fixed_variant_plan``
        unset.
        """

        plan = getattr(getattr(self, "scene", None), "fixed_variant_plan", None)
        if not self.tool_pool_enabled:
            if plan is not None:
                raise ValueError(
                    "SimToolReal tool pool disabled after the factory built the "
                    "fixed variant plan: the plan was built from knobs "
                    f"{self._tool_pool_materialized_from!r} but validate now sees "
                    "the pool disabled; the fixed variant plan is immutable once "
                    "built, so construct a fresh config for a different pool"
                )
            self.fixed_model_variants = None
            self.tool_pool_scales = ()
            self.tool_pool_files = ()
            self._tool_pool_materialized_from = None
            return
        knobs = (int(self.tool_pool_seed), int(self.tool_pool_num_per_type), self.tool_pool_root)
        if self.fixed_model_variants is not None and self._tool_pool_materialized_from == knobs:
            return
        if plan is not None:
            raise ValueError(
                "SimToolReal pool knobs changed after the factory built the fixed "
                "variant plan: the plan was built from knobs "
                f"{self._tool_pool_materialized_from!r} but validate now sees "
                f"{knobs!r}; the fixed variant plan is immutable once built, so "
                "construct a fresh config for a different pool"
            )

        paths, _masses, scales = _materialize_pool(
            seed=int(self.tool_pool_seed),
            num_per_type=int(self.tool_pool_num_per_type),
            configured_root=self.tool_pool_root,
        )
        if not paths:
            raise RuntimeError("SimToolReal tool pool materialized no URDF files")
        # Variant names are the deterministic pool file ids (the manifest file
        # names without suffix, unique by their per-type ordinal); the default
        # round-robin assignment needs no explicit names.
        self.fixed_model_variants = FixedModelVariantCatalogCfg(
            variants=tuple(
                FixedModelVariantCfg(name=path.stem, source_model_file=str(path)) for path in paths
            )
        )
        self.tool_pool_scales = tuple(scales)
        self.tool_pool_files = tuple(str(path) for path in paths)
        self._tool_pool_materialized_from = knobs


__all__ = ["SimToolRealEnvCfg", "fixed_variant_plan_assignment"]
