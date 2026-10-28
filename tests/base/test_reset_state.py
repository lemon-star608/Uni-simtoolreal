"""Target-version reset contract invariants used by the SimToolReal owner."""

from __future__ import annotations

import pytest
from unisim.backend.base import RootStateLayout


def test_root_state_layout_rejects_duplicate_or_malformed_columns():
    valid = RootStateLayout(
        qpos_indices=(0, 1, 2, 3, 4, 5, 6),
        qvel_indices=(0, 1, 2, 4, 5, 6),
    )
    assert len(valid.qpos_indices) == 7
    assert len(valid.qvel_indices) == 6

    with pytest.raises(ValueError, match="qpos_indices"):
        RootStateLayout(
            qpos_indices=(0, 0, 2, 3, 4, 5, 6),
            qvel_indices=(0, 1, 2, 4, 5, 6),
        )
