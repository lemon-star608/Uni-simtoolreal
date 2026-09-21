"""SimToolReal owner configuration and the fixed model variant catalog.

The UniSim adapter owns the wire contract for model variants.  This module is
the task-owner side of that contract: it builds the deterministic 1200-tool
catalog, materializes the pool in the configured ``asset_format`` (``urdf``
for the IsaacSim line — whole-file variants via ``fixed_model_variants``;
``mjcf`` for the MuJoCo line — entity-split B form, M-J5: the catalog binds
to the scene's ``object`` entity through ``bind_entity_variant`` once the
factory's ``num_envs`` is known).  Variant masses live in the task-side
``tool_pool_masses`` table (D5, catalog order); the object-scale table stays
on this config for the command hub's expansion.
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
_DEFAULT_ASSET_FORMAT = "urdf"
_VALID_ASSET_FORMATS = ("urdf", "mjcf")


def _entity_pool_root(seed: int, num_per_type: int, configured: str | None) -> Path:
    if configured is not None and str(configured).strip():
        root = Path(str(configured)).expanduser()
    else:
        env_root = os.environ.get("SIMTOOLREAL_TOOL_POOL_ROOT", "").strip()
        base = Path(env_root).expanduser() if env_root else Path(tempfile.gettempdir())
        root = base / "simtoolreal_entity_pool"
    return root / f"seed_{seed}_n{num_per_type}"


def _materialize_mjcf_pool(
    *, seed: int, num_per_type: int, configured_root: str | None
) -> tuple[tuple[Path, ...], tuple[float, ...], tuple[tuple[float, float, float], ...]]:
    """MJCF pool branch (M-J5, B form): entity-split sources, tool variants only.

    The pool directory carries the full entity set (``write_entity_set``:
    robot/table/floor + tools/); the variant catalog binds to the ``object``
    entity through ``bind_entity_variant`` instead of the whole-model
    ``fixed_model_variants`` channel.  Masses/scales stay task-side tables in
    catalog order (D5), exactly like the URDF branch.
    """
    from .entity_mjcf_writer import write_entity_set

    catalog = build_tool_catalog(num_per_type=num_per_type, seed=seed, shuffle=True)
    root = _entity_pool_root(seed, num_per_type, configured_root)
    manifest_path = root / "manifest.json"
    expected = {
        "schema": "simtoolreal_unilab.entity_assets.v1",
        "seed": int(seed),
        "num_per_type": int(num_per_type),
        "tool_count": len(catalog),
    }
    tool_names: tuple[str, ...] | None = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            names = tuple(str(value) for value in manifest["tool_files"])
            static = ("robot.xml", "table.xml", "floor.xml")
            if (
                all(manifest.get(key) == value for key, value in expected.items())
                and len(names) == len(catalog)
                and all((root / name).is_file() for name in static)
                and all((root / "tools" / name).is_file() for name in names)
            ):
                tool_names = names
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            # A partial or stale pool is rebuilt below.
            tool_names = None
    if tool_names is None:
        manifest = write_entity_set(root, seed=seed, num_per_type=num_per_type)
        tool_names = tuple(str(name) for name in manifest["tool_files"])
    paths = tuple((root / "tools" / name).resolve() for name in tool_names)
    masses = tuple(float(spec.mass) for spec in catalog)
    scales = tuple(tuple(float(x) for x in spec.object_scale) for spec in catalog)
    return paths, masses, scales


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
        # B form (M-J5): the pool binds to one entity instead of the whole
        # scene model; the plan shape/semantics are identical.
        binding = getattr(scene, "entity_variant", None)
        plan = binding.plan if binding is not None else None
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
    # ``urdf`` (IsaacSim line, default — unchanged behavior) or ``mjcf``
    # (MuJoCo line, D3): picks which writer materializes the variant pool.
    asset_format: str = _DEFAULT_ASSET_FORMAT

    # Filled by ``validate()`` — after Hydra/registry overlays settle the pool
    # knobs and before the upstream factory guard reads the catalog — and read
    # by the command hub's object-scale expansion (scale deliberately stays
    # task-side) plus diagnostics.  ``tool_pool_masses`` is the task-side
    # variant mass table (D5) feeding the wrench event; all three tables share
    # the catalog order.  Deliberately kept off the Hydra surface.
    tool_pool_scales: tuple[tuple[float, float, float], ...] = field(
        default_factory=tuple, init=False, repr=False
    )
    tool_pool_files: tuple[str, ...] = field(default_factory=tuple, init=False, repr=False)
    tool_pool_masses: tuple[float, ...] = field(default_factory=tuple, init=False, repr=False)
    _tool_pool_materialized_from: tuple[int, int, str, str | None] | None = field(
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
        if self.asset_format not in _VALID_ASSET_FORMATS:
            raise ValueError(
                f"SimToolReal asset_format must be one of {_VALID_ASSET_FORMATS}, "
                f"got {self.asset_format!r}"
            )
        self._refresh_fixed_model_variants()

    def _refresh_fixed_model_variants(self) -> None:
        """Materialize the pool and refresh the task-owned variant catalog.

        Idempotent per pool-knob set: repeated ``validate()`` calls (registry,
        factory, env construction) reuse the already materialized catalog, and
        knob changes applied before the factory guard builds the scene plan
        rebuild from the new values.  Once the variant plan exists —
        ``cfg.scene.fixed_variant_plan`` (URDF branch, factory guard) or
        ``cfg.scene.entity_variant`` (MJCF B form, ``bind_entity_variant``) —
        it is immutable: a knob change that would rebuild the catalog — or
        disabling the pool — fails closed instead of silently
        desynchronizing catalog and plan (final review M-6; e.g. a larger pool
        would leave the command hub's object-scale expansion indexing a stale
        table, and a disable would leave the backend materializing the pool
        the task just dropped).  With the pool disabled the catalog is
        ``None`` so the upstream guard leaves ``cfg.scene.fixed_variant_plan``
        unset.
        """

        scene = getattr(self, "scene", None)
        plan_built = getattr(scene, "fixed_variant_plan", None) is not None or (
            getattr(scene, "entity_variant", None) is not None
        )
        if not self.tool_pool_enabled:
            if plan_built:
                raise ValueError(
                    "SimToolReal tool pool disabled after the factory built the "
                    "variant plan: the plan was built from knobs "
                    f"{self._tool_pool_materialized_from!r} but validate now sees "
                    "the pool disabled; the variant plan is immutable once "
                    "built, so construct a fresh config for a different pool"
                )
            self.fixed_model_variants = None
            self.tool_pool_scales = ()
            self.tool_pool_files = ()
            self.tool_pool_masses = ()
            self._tool_pool_materialized_from = None
            return
        knobs = (
            int(self.tool_pool_seed),
            int(self.tool_pool_num_per_type),
            str(self.asset_format),
            self.tool_pool_root,
        )
        if self.tool_pool_files and self._tool_pool_materialized_from == knobs:
            return
        if plan_built:
            raise ValueError(
                "SimToolReal pool knobs changed after the factory built the "
                "variant plan: the plan was built from knobs "
                f"{self._tool_pool_materialized_from!r} but validate now sees "
                f"{knobs!r}; the variant plan is immutable once built, so "
                "construct a fresh config for a different pool"
            )

        materialize = _materialize_mjcf_pool if self.asset_format == "mjcf" else _materialize_pool
        paths, masses, scales = materialize(
            seed=int(self.tool_pool_seed),
            num_per_type=int(self.tool_pool_num_per_type),
            configured_root=self.tool_pool_root,
        )
        if not paths:
            raise RuntimeError(f"SimToolReal tool pool materialized no {self.asset_format} files")
        if self.asset_format == "mjcf":
            # B form (M-J5): the catalog binds to the scene's object entity
            # via ``bind_entity_variant`` once the factory's num_envs is
            # known; the whole-model variant channel stays empty.
            self.fixed_model_variants = None
        else:
            # Variant names are the deterministic pool file ids (the manifest
            # file names without suffix, unique by their per-type ordinal);
            # the default round-robin assignment needs no explicit names.
            self.fixed_model_variants = FixedModelVariantCatalogCfg(
                variants=tuple(
                    FixedModelVariantCfg(name=path.stem, source_model_file=str(path))
                    for path in paths
                )
            )
        self.tool_pool_scales = tuple(scales)
        self.tool_pool_files = tuple(str(path) for path in paths)
        # Task-side variant mass table (D5): same catalog order as the files.
        self.tool_pool_masses = tuple(float(mass) for mass in masses)
        self._tool_pool_materialized_from = knobs

    def bind_entity_variant(self, num_envs: int) -> None:
        """Attach the tool catalog to the scene's object entity (MJCF B form).

        The entity variant plan's assignment needs ``num_envs``, which only
        the env factory call site knows — so the task's mujoco factory
        wrapper calls this right before ``make_manager_based_rl_env``.  The
        assignment is the same round-robin the whole-model guard would build
        (``arange(num_envs) % len(catalog)``).  No-op for the URDF branch and
        for a disabled pool; idempotent per num_envs; a conflicting rebind
        fails closed (the plan is immutable once built).
        """
        if not self.tool_pool_enabled or self.asset_format != "mjcf":
            return
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
        scene = getattr(self, "scene", None)
        if scene is None:
            raise ValueError("SimToolReal bind_entity_variant requires a materialized scene")
        existing = getattr(scene, "entity_variant", None)
        if existing is not None:
            if existing.plan.assignment.shape == (num_envs,):
                return
            raise ValueError(
                "SimToolReal entity variant plan is already bound with "
                f"num_envs={existing.plan.assignment.shape[0]}; the plan is immutable "
                "once built, so construct a fresh config for a different num_envs"
            )
        if not self.tool_pool_files:
            raise RuntimeError(
                "SimToolReal tool pool is not materialized; validate() must run before "
                "bind_entity_variant"
            )
        from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
        from unisim.entities import EntityVariantBinding

        assignment = np.arange(num_envs, dtype=np.int32) % np.int32(len(self.tool_pool_files))
        plan = FixedVariantPlan(
            assignment=assignment,
            variants=tuple(ModelSourceDescriptor(model_file=path) for path in self.tool_pool_files),
            layout=FixedVariantLayout.SAME_LAYOUT,
        )
        scene.entity_variant = EntityVariantBinding(
            target_entity=self.tool_pool_target_entity, plan=plan
        )


__all__ = ["SimToolRealEnvCfg", "fixed_variant_plan_assignment"]
