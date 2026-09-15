"""Parity checks for the SimToolReal tool URDF writer.

The upstream anchor is the original writer module itself, loaded by path from
``simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/generate_objects.py`` (a
pure NumPy/stdlib module; a stub parent package is registered so its relative
import of ``object_size_distributions`` resolves without pulling in the
``isaacsimenvs`` package).  Byte-level equality is expected: both sides format
the same Python floats through identical f-string templates.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from simtoolreal_unilab.tasks.simtoolreal.tool_catalog import (
    ALL_TYPES,
    ToolSpec,
    build_tool_catalog,
)
from simtoolreal_unilab.tasks.simtoolreal.tool_urdf_writer import (
    write_tool_catalog_urdfs,
    write_tool_urdf,
)

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_UTILS = ROOT / "simtoolreal/isaacsimenvs/tasks/simtoolreal/utils"

# Deterministic per-distribution blocks: 50 samples per distribution, in the
# source table order.  Selection covers every distribution and every topology
# (first/middle/last of each block), including the handle-only eraser.
CATALOG = build_tool_catalog(ALL_TYPES, 50, 42, shuffle=False)
SELECTED_INDICES = [block * 50 + i for block in range(12) for i in (0, 1, 25, 49)]


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
    pkg_name = "simtoolreal_upstream_utils"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(UPSTREAM_UTILS)]
        sys.modules[pkg_name] = pkg
    for short in ("object_size_distributions", "generate_objects"):
        name = f"{pkg_name}.{short}"
        if name not in sys.modules:
            _load_module(UPSTREAM_UTILS / f"{short}.py", name)
    return sys.modules[f"{pkg_name}.generate_objects"]


def _write_upstream(upstream, spec: ToolSpec, path: Path) -> Path:
    return upstream.generate_handle_head_urdf(
        path=path,
        handle_scale=spec.handle_scale,
        head_scale=spec.head_scale,
        handle_density=spec.handle_density,
        head_density=spec.head_density,
    )


def _inertial(path: Path):
    root = ET.parse(path).getroot()
    link = root.find("link")
    assert link is not None
    inertial = link.find("inertial")
    assert inertial is not None
    origin = inertial.find("origin")
    assert origin is not None
    mass = inertial.find("mass")
    assert mass is not None
    inertia = inertial.find("inertia")
    assert inertia is not None
    com = tuple(float(v) for v in origin.get("xyz").split())
    return link, float(mass.get("value")), com, inertia


@pytest.mark.parametrize("spec_index", SELECTED_INDICES)
def test_urdf_byte_parity_with_upstream(spec_index: int, tmp_path: Path, upstream_writer):
    spec = CATALOG[spec_index]
    ours = write_tool_urdf(spec, tmp_path / "ours.urdf")
    upstream = _write_upstream(upstream_writer, spec, tmp_path / "upstream.urdf")
    assert ours.read_bytes() == upstream.read_bytes()


@pytest.mark.parametrize("spec_index", SELECTED_INDICES)
def test_urdf_inertial_matches_tool_spec(spec_index: int, tmp_path: Path):
    spec = CATALOG[spec_index]
    urdf = write_tool_urdf(spec, tmp_path / "tool.urdf")
    link, mass, com, inertia = _inertial(urdf)
    assert link.get("name") == "object_root"
    assert mass == spec.mass
    assert com == spec.com
    assert float(inertia.get("ixx")) == spec.diaginertia[0]
    assert float(inertia.get("iyy")) == spec.diaginertia[1]
    assert float(inertia.get("izz")) == spec.diaginertia[2]
    assert inertia.get("ixy") == inertia.get("ixz") == inertia.get("iyz") == "0"


def test_batch_writer_matches_per_tool_writer(tmp_path: Path, upstream_writer):
    catalog = build_tool_catalog(ALL_TYPES, 3, 42, shuffle=True)
    out_dir = tmp_path / "pool"
    paths, scales = write_tool_catalog_urdfs(catalog, out_dir)
    assert len(paths) == len(catalog) == len(scales) == 36
    for spec, path_str, scale in zip(catalog, paths, scales):
        path = Path(path_str)
        assert path.parent == out_dir and path.suffix == ".urdf"
        # Same normalization contract as upstream generate_handle_head_urdfs.
        assert scale == spec.object_scale
        # Batch output is byte-identical to the per-tool entry and to upstream.
        single = write_tool_urdf(spec, tmp_path / "single.urdf")
        assert path.read_bytes() == single.read_bytes()
        upstream = _write_upstream(upstream_writer, spec, tmp_path / "up.urdf")
        assert path.read_bytes() == upstream.read_bytes()
        link, _, _, _ = _inertial(path)
        assert link.get("name") == "object_root"


def test_batch_writer_cleans_stale_urdfs(tmp_path: Path):
    out_dir = tmp_path / "pool"
    out_dir.mkdir()
    stale = out_dir / "stale.urdf"
    stale.write_text("stale")
    keep = out_dir / "notes.txt"
    keep.write_text("keep")
    catalog = build_tool_catalog(("eraser",), 2, 42, shuffle=False)
    paths, _ = write_tool_catalog_urdfs(catalog, out_dir)
    assert len(paths) == 2
    assert len(set(paths)) == 2  # filenames unique within the shared template
    assert not stale.exists()
    assert keep.exists()
    for path_str in paths:
        assert Path(path_str).exists()


def test_mismatched_head_fields_raise(tmp_path: Path):
    spec = build_tool_catalog(("eraser",), 1, 42, shuffle=False)[0]
    broken = ToolSpec(
        **{
            **{f: getattr(spec, f) for f in ToolSpec.__dataclass_fields__},
            "head_density": 800.0,
        }
    )
    with pytest.raises(ValueError, match="both be set or both None"):
        write_tool_urdf(broken, tmp_path / "broken.urdf")
