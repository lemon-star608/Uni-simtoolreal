"""Deterministic catalog checks for the SimToolReal tool pool."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from simtoolreal_unilab.tasks.simtoolreal.tool_catalog import (
    ALL_TYPES,
    build_tool_catalog,
)


@pytest.mark.parametrize("shuffle", [False, True])
def test_catalog_is_deterministic_and_preserves_its_pool(shuffle: bool):
    first = build_tool_catalog(ALL_TYPES, 50, 42, shuffle)
    second = build_tool_catalog(ALL_TYPES, 50, 42, shuffle)
    other_order = build_tool_catalog(ALL_TYPES, 50, 42, not shuffle)

    assert len(first) == len(second) == 600
    assert first == second
    assert Counter(map(repr, first)) == Counter(map(repr, other_order))


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
