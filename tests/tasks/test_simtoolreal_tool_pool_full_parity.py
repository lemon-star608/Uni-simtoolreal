"""Full-pool (12 x 100) byte parity between the UniLab tool pipeline and the
original repository generator (pipeline-audit.md S02).

The writer test (``test_simtoolreal_tool_urdf_writer.py``) compares sampled
URDFs given a shared ToolSpec sequence, and the catalog test anchors the
sampler against the old MuJoCo implementation at ``num_per_type=50``.  Neither
closes the ``num_per_type=100`` chain end to end: pools of different sizes have
no prefix relationship (step-0 review of the shared-seed stream position), so
the 1,200-file training pool is compared here against the original generator
run natively at the same size — file bytes in pool order, plus the
lockstep-shuffled normalized scales the reward math consumes.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from simtoolreal_unilab.tasks.simtoolreal.tool_catalog import (
    ALL_TYPES,
    build_tool_catalog,
)
from simtoolreal_unilab.tasks.simtoolreal.tool_urdf_writer import (
    write_tool_catalog_urdfs,
)

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_UTILS = ROOT / "simtoolreal/isaacsimenvs/tasks/simtoolreal/utils"

NUM_PER_TYPE = 100
# 12 distributions (6 type names x 2 topologies each) x 100 samples = 1,200.
EXPECTED_POOL_SIZE = 1200


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def upstream_writer():
    pkg_name = "simtoolreal_upstream_utils_full_pool"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(UPSTREAM_UTILS)]
        sys.modules[pkg_name] = pkg
    for short in ("object_size_distributions", "generate_objects"):
        name = f"{pkg_name}.{short}"
        if name not in sys.modules:
            _load_module(UPSTREAM_UTILS / f"{short}.py", name)
    return sys.modules[f"{pkg_name}.generate_objects"]


@pytest.fixture(scope="module")
def pools(tmp_path_factory, upstream_writer):
    """Both 1,200-file pools plus the original generator's scale list."""
    ref_dir = tmp_path_factory.mktemp("ref_pool")
    ours_dir = tmp_path_factory.mktemp("ours_pool")

    # The original generator seeds the *global* NumPy RNG; save/restore the
    # surrounding state so the suite's other RNG consumers are unaffected.
    rng_state = np.random.get_state()
    try:
        ref_paths, ref_scales = upstream_writer.generate_handle_head_urdfs(
            tuple(ALL_TYPES),
            num_per_type=NUM_PER_TYPE,
            out_dir=ref_dir,
            seed=42,
            shuffle=True,
        )
    finally:
        np.random.set_state(rng_state)

    catalog = build_tool_catalog(ALL_TYPES, NUM_PER_TYPE, 42, shuffle=True)
    our_paths, our_scales = write_tool_catalog_urdfs(catalog, ours_dir)
    return ref_paths, ref_scales, our_paths, our_scales


def test_full_pool_size(pools):
    ref_paths, _, our_paths, _ = pools
    assert len(ref_paths) == EXPECTED_POOL_SIZE
    assert len(our_paths) == EXPECTED_POOL_SIZE


def test_full_pool_byte_parity_in_pool_order(pools):
    ref_paths, _, our_paths, _ = pools
    mismatched = [
        index
        for index, (ref_path, our_path) in enumerate(zip(ref_paths, our_paths))
        if Path(ref_path).read_bytes() != Path(our_path).read_bytes()
    ]
    # Filenames are a registered deviation (per-distribution vs per-type
    # index); the comparison is content-only, index-aligned by pool order.
    assert mismatched == []


def test_full_pool_normalized_scales_match_exactly(pools):
    _, ref_scales, _, our_scales = pools
    assert len(ref_scales) == len(our_scales)
    for index, (ref_scale, our_scale) in enumerate(zip(ref_scales, our_scales)):
        assert tuple(ref_scale) == tuple(our_scale), f"scale mismatch at pool index {index}"
