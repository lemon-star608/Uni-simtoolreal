"""Torch-native helpers shared by SimToolReal manager terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

import torch

if TYPE_CHECKING:
    from unilab.base.entity import Entity, EntityTensorBodyStateView, EntityTensorStateView


T = TypeVar("T", bound=torch.Tensor)


def resolve_entity_names(
    entity: object,
    declared: Sequence[str] | None,
    canonical: Sequence[str],
    *,
    kind: str,
) -> tuple[str, ...]:
    """Resolve canonical names while preserving the canonical term order."""
    if declared is None:
        return tuple(canonical)
    by_stem = {value.rsplit("/", 1)[-1]: value for value in declared}
    if len(by_stem) != len(declared):
        raise ValueError(f"{kind} selector stems are ambiguous for entity {entity!r}")
    missing = [name for name in canonical if name not in by_stem]
    if missing:
        raise ValueError(f"entity is missing {kind} selectors {missing}")
    return tuple(by_stem[name] for name in canonical)


def index_tensor(values: Sequence[int], device: torch.device) -> torch.Tensor:
    """Materialize a cold-path selector as a Torch index tensor."""
    return torch.tensor(tuple(values), dtype=torch.int64, device=device)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply broadcast-compatible wxyz quaternions."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    """Conjugate broadcast-compatible wxyz quaternions."""
    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by broadcast-compatible wxyz quaternions."""
    quat_xyz = quat[..., 1:]
    if vector.ndim < quat_xyz.ndim:
        vector = vector.expand(quat_xyz.shape)
    elif quat_xyz.ndim < vector.ndim:
        quat = quat.expand(vector.shape[:-1] + (4,))
        quat_xyz = quat[..., 1:]
    cross = torch.linalg.cross(quat_xyz, vector, dim=-1)
    second_cross = torch.linalg.cross(quat_xyz, cross, dim=-1)
    return vector + 2.0 * quat[..., :1] * cross + 2.0 * second_cross


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Convert per-row angle/axis values to wxyz quaternions."""
    if axis.ndim != 2 or axis.shape[1] != 3 or angle.shape != (axis.shape[0],):
        raise ValueError(
            f"expected angle ({axis.shape[0]},) and axis ({axis.shape[0]}, 3); "
            f"got {tuple(angle.shape)} and {tuple(axis.shape)}"
        )
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-12)
    half = 0.5 * angle[:, None]
    return torch.cat((torch.cos(half), axis * torch.sin(half)), dim=-1)


def normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    """Normalize the final quaternion dimension."""
    return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)


def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Return the non-negative rotation angle between two wxyz quaternions."""
    relative = quat_mul(q2, quat_conjugate(q1))
    return 2.0 * torch.acos(relative[:, 0].clamp(-1.0, 1.0))


def apply_local_offset(
    pos_w: torch.Tensor,
    rot_wxyz: torch.Tensor,
    offset: tuple[float, float, float],
) -> torch.Tensor:
    """Apply one local-frame offset to batched world positions."""
    batch_shape = pos_w.shape[:-1]
    offset_w = torch.tensor(offset, dtype=torch.float32, device=pos_w.device).expand(
        (*batch_shape, 3)
    )
    return pos_w + quat_apply(rot_wxyz, offset_w)


def keypoints_world(
    center_pos: torch.Tensor,
    center_rot: torch.Tensor,
    kp_offsets: torch.Tensor,
) -> torch.Tensor:
    """Rotate and translate per-environment object-frame keypoints."""
    quat = center_rot[:, None, :].expand(-1, kp_offsets.shape[1], -1)
    return center_pos[:, None, :] + quat_apply(quat, kp_offsets)


def require_joint_view(env: object, entity: Entity) -> EntityTensorStateView:
    """Return the phase-scoped packed joint state for one entity."""
    plan = getattr(getattr(env, "scene", None), "read_plan", None)
    if plan is None or not plan.ready:
        raise RuntimeError(
            f"SimToolReal term requires a packed joint read plan for entity {entity.name!r}"
        )
    return plan.joint_tensor_view(entity)


def require_body_view(
    env: object, entity: Entity, body_names: Sequence[str]
) -> EntityTensorBodyStateView:
    """Return the phase-scoped packed body state for one entity."""
    plan = getattr(getattr(env, "scene", None), "read_plan", None)
    if plan is None or not plan.ready:
        raise RuntimeError(
            f"SimToolReal term requires a packed body read plan for entity {entity.name!r}"
        )
    return plan.body_tensor_view(entity, tuple(body_names))


def all_ids(env: object, device: torch.device) -> torch.Tensor:
    """Return the full environment row selector."""
    return torch.arange(int(getattr(env, "num_envs")), device=device)


def scalar(value: float, *, device: torch.device) -> torch.Tensor:
    """Return a float32 scalar on the target device."""
    return torch.tensor(value, dtype=torch.float32, device=device)


__all__ = [
    "all_ids",
    "apply_local_offset",
    "index_tensor",
    "keypoints_world",
    "normalize_quat",
    "quat_apply",
    "quat_conjugate",
    "quat_error_magnitude",
    "quat_from_angle_axis",
    "quat_mul",
    "require_body_view",
    "require_joint_view",
    "resolve_entity_names",
    "scalar",
]
