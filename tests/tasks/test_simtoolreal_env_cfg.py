"""Coverage for the task-owned fixed model variant catalog builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from unilab.base.scene import SceneCfg

from simtoolreal_unilab.tasks.simtoolreal.env_cfg import SimToolRealEnvCfg
from simtoolreal_unilab.tasks.simtoolreal.tool_catalog import build_tool_catalog


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
    assert manifest["schema"] == "simtoolreal_unilab.entity_assets.v2"
    assert set(manifest["entities"]) == {"robot", "table", "floor", "object"}
    assert manifest["tool_files"] == [Path(path).name for path in cfg.tool_pool_files]
    assert len(set(manifest["tool_files"])) == len(manifest["tool_files"])

    catalog = build_tool_catalog(num_per_type=1, seed=cfg.tool_pool_seed, shuffle=True)
    assert cfg.tool_pool_masses == tuple(float(spec.mass) for spec in catalog)
    assert cfg.tool_pool_scales == tuple(
        tuple(float(x) for x in spec.object_scale) for spec in catalog
    )

    # Reuse: a v1 manifest is stale and must be rebuilt as the current
    # four-entity/object-binding shape rather than silently accepted.
    manifest["schema"] = "simtoolreal_unilab.entity_assets.v1"
    (root / "manifest.json").write_text(json.dumps(manifest))
    second = _mjcf_cfg(tmp_path)
    second.validate()
    rebuilt = json.loads((root / "manifest.json").read_text())
    assert rebuilt["schema"] == "simtoolreal_unilab.entity_assets.v2"
    assert second.tool_pool_files == cfg.tool_pool_files
    assert second.tool_pool_masses == cfg.tool_pool_masses

    # Catalog order is part of the cache contract: masses/scales are paired
    # with tool_files by index. A reordered manifest must be rebuilt, not
    # silently reused against the current task-side tables.
    rebuilt["tool_files"] = list(reversed(rebuilt["tool_files"]))
    (root / "manifest.json").write_text(json.dumps(rebuilt))
    third = _mjcf_cfg(tmp_path)
    third.validate()
    assert third.tool_pool_files == cfg.tool_pool_files


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


def test_bind_entity_variant_is_noop_for_disabled_pool(tmp_path):
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


def test_tool_pool_layout_defaults_to_same_layout(tmp_path):
    from unisim.dr.types import FixedVariantLayout

    cfg = _mjcf_cfg(tmp_path)
    assert cfg.tool_pool_layout == "same_layout"
    cfg.validate()
    cfg.bind_entity_variant(num_envs=3)
    assert cfg.scene.entity_variant.plan.layout == FixedVariantLayout.SAME_LAYOUT
    manifest = json.loads(
        (Path(cfg.tool_pool_files[0]).parent.parent / "manifest.json").read_text()
    )
    assert manifest["layout"] == "same_layout"
    assert manifest["entities"]["object"]["variant_binding"]["layout"] == "same_layout"


def test_tool_pool_layout_uniform_binds_uniform_layout(tmp_path):
    from unisim.dr.types import FixedVariantLayout

    cfg = _mjcf_cfg(tmp_path, tool_pool_layout="uniform_public_layout")
    cfg.validate()
    cfg.bind_entity_variant(num_envs=3)

    binding = cfg.scene.entity_variant
    assert binding.plan.layout == FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT
    assert tuple(variant.model_file for variant in binding.plan.variants) == cfg.tool_pool_files

    root = Path(cfg.tool_pool_files[0]).parent.parent
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["layout"] == "uniform_public_layout"
    assert manifest["entities"]["object"]["variant_binding"] == {
        "target_entity": "object",
        "layout": "uniform_public_layout",
    }
    # Masses/scales are layout-independent ToolSpec tables (D5).
    catalog = build_tool_catalog(num_per_type=1, seed=cfg.tool_pool_seed, shuffle=True)
    assert cfg.tool_pool_masses == tuple(float(spec.mass) for spec in catalog)


def test_uniform_and_same_layout_pools_coexist(tmp_path):
    """Sibling pool directories: neither cache invalidates the other (D8)."""
    same = _mjcf_cfg(tmp_path)
    same.validate()
    uniform = _mjcf_cfg(tmp_path, tool_pool_layout="uniform_public_layout")
    uniform.validate()

    same_root = Path(same.tool_pool_files[0]).parent.parent
    uniform_root = Path(uniform.tool_pool_files[0]).parent.parent
    assert same_root != uniform_root
    assert same_root.name.endswith("_n1") and uniform_root.name.endswith("_n1_uniform")
    assert same.tool_pool_masses == uniform.tool_pool_masses
    assert same.tool_pool_scales == uniform.tool_pool_scales

    # Re-validating either layout reuses its own cache rather than rebuilding
    # or cross-accepting the sibling pool.
    again_same = _mjcf_cfg(tmp_path)
    again_same.validate()
    assert again_same.tool_pool_files == same.tool_pool_files
    again_uniform = _mjcf_cfg(tmp_path, tool_pool_layout="uniform_public_layout")
    again_uniform.validate()
    assert again_uniform.tool_pool_files == uniform.tool_pool_files


def test_tool_pool_layout_invalid_value_fails_closed(tmp_path):
    cfg = _mjcf_cfg(tmp_path, tool_pool_layout="uniform_primitive")
    with pytest.raises(ValueError, match="tool_pool_layout must be one of"):
        cfg.validate()


def test_tool_pool_layout_change_after_binding_fails_closed(tmp_path):
    cfg = _mjcf_cfg(tmp_path)
    cfg.validate()
    cfg.bind_entity_variant(num_envs=5)

    cfg.tool_pool_layout = "uniform_public_layout"
    with pytest.raises(ValueError, match="pool knobs changed after the factory built"):
        cfg.validate()
    cfg.bind_entity_variant(num_envs=5)
    from unisim.dr.types import FixedVariantLayout

    assert cfg.scene.entity_variant.plan.layout == FixedVariantLayout.SAME_LAYOUT


def test_uniform_pool_cache_covers_mesh_files(tmp_path):
    """The uniform pool's file-backed meshes join the cache contract: correct
    content is silently reused; a modified or missing mesh forces a rebuild."""
    cfg = _mjcf_cfg(tmp_path, tool_pool_layout="uniform_public_layout")
    cfg.validate()
    root = Path(cfg.tool_pool_files[0]).parent.parent
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["tool_meshes"]  # manifest accounts for the mesh files
    meshes_dir = root / "tools" / "meshes"
    assert set(manifest["tool_meshes"]) == {p.name for p in meshes_dir.glob("*.obj")}
    victim = meshes_dir / sorted(manifest["tool_meshes"])[0]
    original = victim.read_bytes()

    again = _mjcf_cfg(tmp_path, tool_pool_layout="uniform_public_layout")
    again.validate()
    assert again.tool_pool_files == cfg.tool_pool_files
    assert victim.read_bytes() == original

    victim.write_text("stale")
    _mjcf_cfg(tmp_path, tool_pool_layout="uniform_public_layout").validate()
    assert victim.read_bytes() == original

    victim.unlink()
    _mjcf_cfg(tmp_path, tool_pool_layout="uniform_public_layout").validate()
    assert victim.read_bytes() == original
