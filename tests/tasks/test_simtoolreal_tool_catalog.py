"""Bit-exact checks for the manager-based SimToolReal catalog.

The original generator imports Isaac Gym through its package, so the upstream
anchor below is a deliberately minimal static translation of the two pure
NumPy routines in ``generate_objects.py``.  The distribution table itself is
loaded from the original repository file.  The old MuJoCo catalog is loaded
by path and compared only through common semantic fields; its half geom sizes
and head positions are intentionally not part of the new API.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from simtoolreal_unilab.tasks.simtoolreal.tool_catalog import (
    ALL_TYPES,
    build_tool_catalog,
)

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_DIST = (
    ROOT / "simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/object_size_distributions.py"
)
OLD_CATALOG = (
    ROOT / "simtoolreal-unilab-old/UniLab/src/unilab/envs/manipulation/simtoolreal/tool_catalog.py"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _mass_inertia(scale: tuple[float, ...], density: float):
    if len(scale) == 3:
        lx, ly, lz = scale
        v = lx * ly * lz
        mass = v * density
        return (
            mass,
            (1 / 12) * mass * (ly * ly + lz * lz),
            (1 / 12) * mass * (lx * lx + lz * lz),
            (1 / 12) * mass * (lx * lx + ly * ly),
        )
    h, d = scale
    r = d / 2
    m_c = density * math.pi * r * r * h
    m_h = density * (2 / 3) * math.pi * r**3
    mass = m_c + 2 * m_h
    i_c_axis = 0.5 * m_c * r * r
    i_c_perp = (1 / 12) * m_c * (3 * r * r + h * h)
    i_h_axis = (2 / 5) * m_h * r * r
    i_h_perp = (83 / 320) * m_h * r * r
    d_com = (h / 2) + (3 * r / 8)
    return (
        mass,
        i_c_perp + 2 * (i_h_perp + m_h * d_com * d_com),
        i_c_perp + 2 * (i_h_perp + m_h * d_com * d_com),
        i_c_axis + 2 * i_h_axis,
    )


def _upstream_reference(num_per_type: int, seed: int, shuffle: bool):
    """Static, import-light transcription of upstream sampling + physics."""
    dist_mod = _load_module(UPSTREAM_DIST, "simtoolreal_upstream_distributions")
    rng = np.random.RandomState(seed)  # equivalent MT19937 stream to np.random.seed
    out = []
    matching = list(dist_mod.OBJECT_SIZE_DISTRIBUTIONS)
    for dist in matching:
        hd = rng.uniform(dist.handle_min_density, dist.handle_max_density, size=num_per_type)
        head_d = (
            rng.uniform(dist.head_min_density, dist.head_max_density, size=num_per_type)
            if dist.head_min_density is not None
            else None
        )
        hs = rng.uniform(
            dist.handle_min_lengths,
            dist.handle_max_lengths,
            size=(num_per_type, len(dist.handle_min_lengths)),
        )
        heads = (
            rng.uniform(
                dist.head_min_lengths,
                dist.head_max_lengths,
                size=(num_per_type, len(dist.head_min_lengths)),
            )
            if dist.head_min_lengths is not None
            else None
        )
        for i in range(num_per_type):
            h_raw = hs[i]
            head_raw = heads[i] if heads is not None else None
            h = tuple(float(x) for x in h_raw)
            head = tuple(float(x) for x in head_raw) if head_raw is not None else None
            den = float(hd[i])
            den_head = float(head_d[i]) if head_d is not None else None
            is_cyl = len(h) == 2
            hm, hixx, hiyy, hizz = _mass_inertia(h_raw, hd[i])
            h3 = (h_raw[0], h_raw[1], h_raw[1]) if is_cyl else h_raw
            if is_cyl:
                hixx, hiyy, hizz = hizz, hiyy, hixx
            if head is None:
                total, com, inertia = hm, (0.0, 0.0, 0.0), (hixx, hiyy, hizz)
                topology = "box_only"
            else:
                m2, i2x, i2y, i2z = _mass_inertia(head_raw, head_d[i])
                xoff = h3[0] / 2 + head_raw[0] / 2
                total = hm + m2
                cx = (hm * 0.0 + m2 * xoff) / total
                d_handle, d_head = -cx, xoff - cx
                ixx = hixx + i2x
                iyy = (hiyy + hm * d_handle * d_handle) + (i2y + m2 * d_head * d_head)
                izz = (hizz + hm * d_handle * d_handle) + (i2z + m2 * d_head * d_head)
                inertia = (ixx, iyy, izz)
                com = (cx, 0.0, 0.0)
                topology = "capsule_box" if is_cyl else "box_box"
            out.append(
                (
                    dist.type,
                    topology,
                    "cylinder" if is_cyl else "box",
                    h,
                    head,
                    den,
                    den_head,
                    total,
                    com,
                    inertia,
                    tuple(float(x) / 0.04 for x in h3),
                )
            )
    if shuffle:
        idx = np.arange(len(out))
        rng.shuffle(idx)
        out = [out[int(i)] for i in idx]
    return out


def _new_row(spec):
    return (
        spec.type,
        spec.topology,
        spec.authored_shape,
        spec.handle_scale,
        spec.head_scale,
        spec.handle_density,
        spec.head_density,
        spec.mass,
        spec.com,
        spec.diaginertia,
        spec.object_scale,
    )


@pytest.mark.parametrize("shuffle", [False, True])
def test_catalog_matches_upstream_static_anchor(shuffle: bool):
    expected = _upstream_reference(50, 42, shuffle)
    actual = [_new_row(s) for s in build_tool_catalog(ALL_TYPES, 50, 42, shuffle)]
    assert len(actual) == 600
    np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("shuffle", [False, True])
def test_catalog_matches_old_tool_spec_common_semantics(shuffle: bool):
    old = _load_module(OLD_CATALOG, "simtoolreal_old_tool_catalog")
    old_specs = old.build_tool_catalog(old.ALL_TYPES, 50, 42, shuffle)
    new_specs = build_tool_catalog(ALL_TYPES, 50, 42, shuffle)
    assert len(old_specs) == len(new_specs) == 600
    for o, n in zip(old_specs, new_specs):
        assert (o.type, o.topology, o.authored_shape) == (n.type, n.topology, n.authored_shape)
        assert o.mass == n.mass
        assert o.com == n.com
        assert o.diaginertia == n.diaginertia
        assert o.object_scale == n.object_scale
        # Reconstruct original URDF full dimensions solely for comparison.
        if o.authored_shape == "cylinder":
            old_handle = (2 * o.handle_size[1], 2 * o.handle_size[0])
        else:
            old_handle = tuple(2 * x for x in o.handle_size)
        assert old_handle == n.handle_scale
        old_head = None if o.topology == "box_only" else tuple(2 * x for x in o.head_size)
        assert old_head == n.head_scale


def test_catalog_does_not_mutate_global_numpy_rng():
    np.random.seed(123)
    before = np.random.random(5)
    np.random.seed(123)
    build_tool_catalog(num_per_type=50)
    after = np.random.random(5)
    np.testing.assert_equal(after, before)


def test_defaults_and_immutability():
    catalog = build_tool_catalog()
    assert len(catalog) == 1200
    with pytest.raises((AttributeError, TypeError)):
        catalog[0].mass = 1.0
