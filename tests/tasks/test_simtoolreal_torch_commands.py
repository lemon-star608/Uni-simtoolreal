from __future__ import annotations

import torch
from unilab.managers.rng import ManagerRng

from simtoolreal_unilab.tasks.simtoolreal.commands import sample_delta_goal_pose


def test_delta_goal_pose_returns_position_then_quaternion():
    rng = ManagerRng.seeded(42, device="cpu")
    prev_pos = torch.zeros((6, 3), dtype=torch.float32)
    prev_quat = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]] * 6,
        dtype=torch.float32,
    )

    new_pos, new_quat = sample_delta_goal_pose(
        rng,
        prev_pos=prev_pos,
        prev_quat_wxyz=prev_quat,
        delta_distance=0.1,
        delta_rotation_degrees=90.0,
        mins=(-0.35, -0.2, 0.6),
        maxs=(0.35, 0.2, 0.95),
        scale=1.0,
    )

    assert new_pos.shape == (6, 3)
    assert new_quat.shape == (6, 4)
    assert torch.all(new_pos[:, 0] >= -0.35 - 1e-6) and torch.all(new_pos[:, 0] <= 0.35 + 1e-6)
    assert torch.all(new_pos[:, 1] >= -0.2 - 1e-6) and torch.all(new_pos[:, 1] <= 0.2 + 1e-6)
    assert torch.all(new_pos[:, 2] >= 0.6 - 1e-6) and torch.all(new_pos[:, 2] <= 0.95 + 1e-6)
    assert torch.allclose(
        torch.linalg.vector_norm(new_quat, dim=-1),
        torch.ones(6),
        atol=1e-6,
    )
    # The position/quaternion convention is intentionally checked through the
    # same goal-table assignment that runs on success-driven goal advancement.
    goal = torch.zeros((6, 7), dtype=torch.float32)
    goal[:, 3] = 1.0
    env_ids = torch.arange(6)
    goal[env_ids, :3] = new_pos
    goal[env_ids, 3:] = new_quat
    assert torch.isfinite(goal).all()
