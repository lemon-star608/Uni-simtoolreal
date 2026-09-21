"""Coverage for the task-owned fixed model variant catalog builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from unilab.base.scene import SceneCfg
from unilab.base.variants import FixedModelVariantCatalogCfg, _build_fixed_variant_plan

from simtoolreal_unilab.tasks.simtoolreal.env_cfg import SimToolRealEnvCfg
from simtoolreal_unilab.tasks.simtoolreal.tool_catalog import build_tool_catalog


def _cfg(tmp_path, **kwargs) -> SimToolRealEnvCfg:
    """Minimal valid manager config with a tiny pool rooted at ``tmp_path``."""

    kwargs.setdefault("tool_pool_num_per_type", 1)
    kwargs.setdefault("tool_pool_root", str(tmp_path))
    return SimToolRealEnvCfg(
        scene=SceneCfg(model_file="scene.xml"),
        max_episode_seconds=10.0,
        **kwargs,
    )


def test_tool_pool_catalog_is_deterministic_and_manifest_reusable(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.validate()

    catalog = cfg.fixed_model_variants
    assert isinstance(catalog, FixedModelVariantCatalogCfg)
    assert len(catalog.variants) == 12
    # No explicit names: the upstream guard's default round-robin applies.
    assert catalog.explicit_variant_names == ()
    assert len(cfg.tool_pool_files) == 12
    assert all(path.endswith(".urdf") for path in cfg.tool_pool_files)
    # Variant names are the manifest file ids; sources follow the pool order.
    pool_stems = [Path(path).stem for path in cfg.tool_pool_files]
    assert [variant.name for variant in catalog.variants] == pool_stems
    assert [variant.source_model_file for variant in catalog.variants] == list(cfg.tool_pool_files)
    assert len(set(variant.name for variant in catalog.variants)) == 12

    manifest = json.loads((Path(cfg.tool_pool_files[0]).parent / "manifest.json").read_text())
    assert manifest["files"] == [Path(path).name for path in cfg.tool_pool_files]
    # The manifest keeps the mass table for the worker-side measurement
    # chain (mass reads back through the backend now); its shape is unchanged.
    assert len(manifest["masses"]) == 12
    assert all(np.isfinite(mass) and mass >= 0.0 for mass in manifest["masses"])
    assert len(cfg.tool_pool_scales) == 12
    assert all(
        len(scale) == 3 and all(np.isfinite(scale)) and all(x > 0.0 for x in scale)
        for scale in cfg.tool_pool_scales
    )
    catalog_specs = build_tool_catalog(num_per_type=1, seed=cfg.tool_pool_seed, shuffle=True)
    assert cfg.tool_pool_scales == tuple(
        tuple(float(x) for x in spec.object_scale) for spec in catalog_specs
    )

    # A second owner instance must reuse the exact catalog order and scales
    # from the manifest rather than rewriting a differently ordered pool.
    second = _cfg(tmp_path)
    second.validate()
    assert second.tool_pool_files == cfg.tool_pool_files
    assert second.tool_pool_scales == cfg.tool_pool_scales
    assert second.fixed_model_variants == catalog


def test_plan_assignment_stays_round_robin_over_pool_order(tmp_path):
    """Same pool + seed: the upstream guard's plan assignment equals the
    historical ``arange(num_envs) % pool_size`` round-robin, value-for-value."""
    cfg = _cfg(tmp_path)
    cfg.validate()
    assert cfg.fixed_model_variants is not None

    plan = _build_fixed_variant_plan(cfg.fixed_model_variants, num_envs=13)

    pool_size = len(cfg.tool_pool_files)
    np.testing.assert_array_equal(plan.assignment, np.arange(13, dtype=np.int64) % pool_size)
    assert plan.assignment.dtype == np.int32
    assert tuple(variant.model_file for variant in plan.variants) == cfg.tool_pool_files


def test_tool_pool_disabled_clears_catalog_and_caches(tmp_path):
    cfg = _cfg(tmp_path, tool_pool_enabled=False)
    cfg.validate()
    assert cfg.fixed_model_variants is None
    assert cfg.tool_pool_scales == ()
    assert cfg.tool_pool_files == ()
    assert not any(tmp_path.iterdir())


def test_catalog_follows_pool_knob_changes_after_construction(tmp_path):
    """Registry/Hydra overlays mutate pool knobs after construction; the
    catalog must follow the final knob values at validate time."""
    cfg = _cfg(tmp_path)
    cfg.validate()
    first = cfg.tool_pool_files

    cfg.tool_pool_seed = 7
    cfg.validate()

    assert cfg.tool_pool_files != first
    assert all("seed_7_n1" in path for path in cfg.tool_pool_files)
    assert len(cfg.fixed_model_variants.variants) == 12


def test_pool_knob_change_after_plan_build_fails_closed(tmp_path):
    """M-6 hardening: the scene plan is immutable once the factory guard has
    built it.  A later validate that would rebuild the catalog from changed
    knobs — or that disables the pool — raises instead of silently
    desynchronizing catalog and plan (a larger pool would leave object-scale
    lookups indexing a stale table; a disable would leave the backend
    materializing the pool the task just dropped)."""
    cfg = _cfg(tmp_path)
    cfg.validate()
    catalog = cfg.fixed_model_variants
    assert catalog is not None
    # Mirror the upstream guard (make_manager_based_rl_env): the plan is built
    # from the validated catalog before backend materialization, and the env
    # constructor then re-validates with the plan already on the scene.
    cfg.scene.fixed_variant_plan = _build_fixed_variant_plan(catalog, num_envs=7)

    cfg.tool_pool_seed = 7
    with pytest.raises(ValueError, match="pool knobs changed after the factory built"):
        cfg.validate()

    # Failing closed leaves the catalog and plan exactly as the guard built them.
    assert cfg.fixed_model_variants is catalog
    assert cfg.tool_pool_seed == 7

    # Disabling the pool with the plan built is the same desynchronization:
    # clearing the task-side caches would not stop the backend from
    # materializing the pool through the plan, so validate fails closed too.
    cfg.tool_pool_enabled = False
    with pytest.raises(ValueError, match="tool pool disabled after the factory built"):
        cfg.validate()

    # The disabled-with-no-plan path stays silent and still clears (see
    # test_tool_pool_disabled_clears_catalog_and_caches); the raise above must
    # not have mutated anything either.
    assert cfg.fixed_model_variants is catalog
    assert len(cfg.tool_pool_files) == 12
    assert len(cfg.tool_pool_scales) == 12
    assert cfg.tool_pool_enabled is False
    assert cfg.scene.fixed_variant_plan.assignment.shape == (7,)


def test_repeated_validate_with_plan_built_is_side_effect_free(tmp_path):
    """Idempotent path: the post-plan re-validate during env construction
    must keep reusing the materialized catalog with no rebuild."""
    cfg = _cfg(tmp_path)
    cfg.validate()
    catalog = cfg.fixed_model_variants
    files = cfg.tool_pool_files
    scales = cfg.tool_pool_scales
    cfg.scene.fixed_variant_plan = _build_fixed_variant_plan(catalog, num_envs=7)

    cfg.validate()
    cfg.validate()

    assert cfg.fixed_model_variants is catalog
    assert cfg.tool_pool_files == files
    assert cfg.tool_pool_scales == scales
    assert cfg.scene.fixed_variant_plan.assignment.shape == (7,)


def _mjcf_cfg(tmp_path, **kwargs) -> SimToolRealEnvCfg:
    """B-form (mjcf) owner config with a tiny entity pool at ``tmp_path``."""
    from unisim.dr.types import ModelSourceDescriptor
    from unisim.entities import SceneEntitySpec

    kwargs.setdefault("asset_format", "mjcf")
    kwargs.setdefault("tool_pool_num_per_type", 1)
    kwargs.setdefault("tool_pool_root", str(tmp_path))
    return SimToolRealEnvCfg(
        scene=SceneCfg(
            entity_assets=(
                SceneEntitySpec(
                    name="object",
                    source=ModelSourceDescriptor("object.xml"),
                    kind="rigid",
                    root_mode="floating",
                ),
            )
        ),
        max_episode_seconds=10.0,
        **kwargs,
    )


def test_mjcf_branch_materializes_entity_pool_without_model_variants(tmp_path):
    cfg = _mjcf_cfg(tmp_path)
    cfg.validate()

    # B form: the whole-model channel stays empty; the catalog waits for
    # bind_entity_variant (num_envs is only known at the factory call site).
    assert cfg.fixed_model_variants is None
    assert len(cfg.tool_pool_files) == 12
    assert all(path.endswith(".xml") for path in cfg.tool_pool_files)

    root = Path(cfg.tool_pool_files[0]).parent.parent
    assert (root / "robot.xml").is_file()
    assert (root / "table.xml").is_file()
    assert (root / "floor.xml").is_file()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema"] == "simtoolreal_unilab.entity_assets.v1"
    assert manifest["tool_files"] == [Path(path).name for path in cfg.tool_pool_files]

    catalog = build_tool_catalog(num_per_type=1, seed=cfg.tool_pool_seed, shuffle=True)
    assert cfg.tool_pool_masses == tuple(float(spec.mass) for spec in catalog)
    assert cfg.tool_pool_scales == tuple(
        tuple(float(x) for x in spec.object_scale) for spec in catalog
    )

    # Reuse: a second owner over the same root must not rewrite the pool.
    second = _mjcf_cfg(tmp_path)
    second.validate()
    assert second.tool_pool_files == cfg.tool_pool_files
    assert second.tool_pool_masses == cfg.tool_pool_masses


def test_bind_entity_variant_builds_round_robin_plan(tmp_path):
    cfg = _mjcf_cfg(tmp_path)
    cfg.validate()
    cfg.bind_entity_variant(num_envs=13)

    binding = cfg.scene.entity_variant
    assert binding is not None
    assert binding.target_entity == "object"
    np.testing.assert_array_equal(binding.plan.assignment, np.arange(13, dtype=np.int64) % 12)
    assert binding.plan.assignment.dtype == np.int32
    assert tuple(variant.model_file for variant in binding.plan.variants) == cfg.tool_pool_files

    # The task-side reader sees the same assignment through the entity binding.
    from simtoolreal_unilab.tasks.simtoolreal.env_cfg import fixed_variant_plan_assignment

    np.testing.assert_array_equal(fixed_variant_plan_assignment(cfg, 13), binding.plan.assignment)

    # Idempotent for the same num_envs; a conflicting rebind fails closed.
    cfg.bind_entity_variant(num_envs=13)
    with pytest.raises(ValueError, match="entity variant plan is already bound"):
        cfg.bind_entity_variant(num_envs=7)


def test_bind_entity_variant_is_noop_for_urdf_and_disabled_pool(tmp_path):
    urdf_cfg = _cfg(tmp_path / "urdf")
    urdf_cfg.validate()
    urdf_cfg.bind_entity_variant(num_envs=4)
    assert urdf_cfg.scene.entity_variant is None

    disabled = _mjcf_cfg(tmp_path / "disabled", tool_pool_enabled=False)
    disabled.validate()
    disabled.bind_entity_variant(num_envs=4)
    assert disabled.scene.entity_variant is None


def test_mjcf_knob_change_after_binding_fails_closed(tmp_path):
    """Same immutability as the whole-model plan, on the entity channel."""
    cfg = _mjcf_cfg(tmp_path)
    cfg.validate()
    cfg.bind_entity_variant(num_envs=7)

    cfg.tool_pool_seed = 7
    with pytest.raises(ValueError, match="pool knobs changed after the factory built"):
        cfg.validate()
    cfg.tool_pool_enabled = False
    with pytest.raises(ValueError, match="tool pool disabled after the factory built"):
        cfg.validate()
    assert cfg.scene.entity_variant.plan.assignment.shape == (7,)
