"""Deterministic SimToolReal tool catalog.

The catalog samples twelve handle/head distributions in a fixed order and then
optionally shuffles the resulting pool in lockstep. It carries the generated
physical quantities, while deliberately omitting writer-only geometry fields
(half ``geom_size`` values, ``head_pos``, and similar details). It performs no
file I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, cast

import numpy as np

LOW_DENSITY_MIN, LOW_DENSITY_MAX = 300.0, 600.0
HIGH_DENSITY_MIN, HIGH_DENSITY_MAX = 800.0, 2000.0
OBJECT_BASE_SIZE = 0.04


@dataclass(frozen=True, slots=True)
class _ObjectSizeDistribution:
    type: str
    handle_min_lengths: tuple[float, ...]
    handle_max_lengths: tuple[float, ...]
    head_min_lengths: tuple[float, ...] | None
    head_max_lengths: tuple[float, ...] | None
    handle_min_density: float
    handle_max_density: float
    head_min_density: float | None
    head_max_density: float | None

    @property
    def shape(self) -> str:
        return "cuboid" if len(self.handle_min_lengths) == 3 else "cylinder"

    def sample_handle_densities(self, n: int, rng: np.random.RandomState) -> np.ndarray:
        return rng.uniform(self.handle_min_density, self.handle_max_density, size=n)

    def sample_head_densities(self, n: int, rng: np.random.RandomState) -> np.ndarray | None:
        if self.head_min_density is None or self.head_max_density is None:
            return None
        return rng.uniform(self.head_min_density, self.head_max_density, size=n)

    def sample_handle_scales(self, n: int, rng: np.random.RandomState) -> np.ndarray:
        return rng.uniform(
            self.handle_min_lengths,
            self.handle_max_lengths,
            size=(n, len(self.handle_min_lengths)),
        )

    def sample_head_scales(self, n: int, rng: np.random.RandomState) -> np.ndarray | None:
        if self.head_min_lengths is None or self.head_max_lengths is None:
            return None
        return rng.uniform(
            self.head_min_lengths,
            self.head_max_lengths,
            size=(n, len(self.head_min_lengths)),
        )


# This table is an intentionally literal transcription of
# simtoolreal/.../object_size_distributions.py.  Keep its order: it is part of
# the source RNG contract (two hammer, two screwdriver, one marker, two
# spatula, one eraser, four brush variants).
_OBJECT_SIZE_DISTRIBUTIONS = [
    _ObjectSizeDistribution(
        "hammer",
        (0.15, 0.02, 0.015),
        (0.30, 0.04, 0.03),
        (0.02, 0.05, 0.02),
        (0.06, 0.12, 0.06),
        LOW_DENSITY_MIN,
        LOW_DENSITY_MAX,
        HIGH_DENSITY_MIN,
        HIGH_DENSITY_MAX,
    ),
    _ObjectSizeDistribution(
        "hammer",
        (0.15, 0.015),
        (0.30, 0.03),
        (0.02, 0.05, 0.02),
        (0.06, 0.12, 0.06),
        LOW_DENSITY_MIN,
        LOW_DENSITY_MAX,
        HIGH_DENSITY_MIN,
        HIGH_DENSITY_MAX,
    ),
    _ObjectSizeDistribution(
        "screwdriver",
        (0.07, 0.025, 0.025),
        (0.12, 0.04, 0.04),
        (0.07, 0.01, 0.01),
        (0.15, 0.015, 0.015),
        LOW_DENSITY_MIN,
        LOW_DENSITY_MAX,
        HIGH_DENSITY_MIN,
        HIGH_DENSITY_MAX,
    ),
    _ObjectSizeDistribution(
        "screwdriver",
        (0.07, 0.025),
        (0.12, 0.04),
        (0.07, 0.01, 0.01),
        (0.15, 0.015, 0.015),
        LOW_DENSITY_MIN,
        LOW_DENSITY_MAX,
        HIGH_DENSITY_MIN,
        HIGH_DENSITY_MAX,
    ),
    _ObjectSizeDistribution(
        "marker",
        (0.075, 0.015),
        (0.15, 0.03),
        (0.01, 0.005, 0.005),
        (0.03, 0.01, 0.01),
        LOW_DENSITY_MIN,
        LOW_DENSITY_MAX,
        LOW_DENSITY_MIN,
        LOW_DENSITY_MAX,
    ),
    _ObjectSizeDistribution(
        "spatula",
        (0.1, 0.0125, 0.006),
        (0.2, 0.025, 0.025),
        (0.05, 0.03, 0.01),
        (0.15, 0.07, 0.03),
        LOW_DENSITY_MIN,
        LOW_DENSITY_MAX,
        LOW_DENSITY_MIN,
        LOW_DENSITY_MAX,
    ),
    _ObjectSizeDistribution(
        "spatula",
        (0.1, 0.0125),
        (0.2, 0.025),
        (0.05, 0.03, 0.01),
        (0.15, 0.07, 0.03),
        LOW_DENSITY_MIN,
        LOW_DENSITY_MAX,
        LOW_DENSITY_MIN,
        LOW_DENSITY_MAX,
    ),
    _ObjectSizeDistribution(
        "eraser",
        (0.07, 0.02, 0.02),
        (0.15, 0.07, 0.07),
        None,
        None,
        LOW_DENSITY_MIN,
        LOW_DENSITY_MAX,
        None,
        None,
    ),
    _ObjectSizeDistribution(
        "brush",
        (0.05, 0.01, 0.01),
        (0.2, 0.04, 0.03),
        (0.05, 0.03, 0.03),
        (0.12, 0.05, 0.08),
        LOW_DENSITY_MIN,
        LOW_DENSITY_MAX,
        LOW_DENSITY_MIN,
        LOW_DENSITY_MAX,
    ),
    _ObjectSizeDistribution(
        "brush",
        (0.05, 0.01),
        (0.2, 0.03),
        (0.05, 0.03, 0.03),
        (0.12, 0.05, 0.08),
        LOW_DENSITY_MIN,
        LOW_DENSITY_MAX,
        LOW_DENSITY_MIN,
        LOW_DENSITY_MAX,
    ),
    _ObjectSizeDistribution(
        "brush",
        (0.05, 0.01, 0.01),
        (0.2, 0.04, 0.03),
        (0.05, 0.05, 0.02),
        (0.12, 0.12, 0.04),
        LOW_DENSITY_MIN,
        LOW_DENSITY_MAX,
        LOW_DENSITY_MIN,
        LOW_DENSITY_MAX,
    ),
    _ObjectSizeDistribution(
        "brush",
        (0.05, 0.01),
        (0.2, 0.03),
        (0.05, 0.05, 0.02),
        (0.12, 0.12, 0.04),
        LOW_DENSITY_MIN,
        LOW_DENSITY_MAX,
        LOW_DENSITY_MIN,
        LOW_DENSITY_MAX,
    ),
]


def _compute_mass_and_inertia(
    scale: Sequence[float], density: float
) -> tuple[float, float, float, float]:
    """Return ``(mass, ixx, iyy, izz)`` in the geometry-local frame.

    The two-component form is the original capsule approximation for an
    authored cylinder (cylinder plus two hemispheres).
    """
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
    if len(scale) == 2:
        height, diameter = scale
        radius = diameter / 2
        m_c = density * math.pi * radius * radius * height
        m_h = density * (2 / 3) * math.pi * radius**3
        mass = m_c + 2 * m_h
        i_c_axis = 0.5 * m_c * radius * radius
        i_c_perp = (1 / 12) * m_c * (3 * radius * radius + height * height)
        i_h_axis = (2 / 5) * m_h * radius * radius
        i_h_perp = (83 / 320) * m_h * radius * radius
        d_com = (height / 2) + (3 * radius / 8)
        izz = i_c_axis + 2 * i_h_axis
        ixx = iyy = i_c_perp + 2 * (i_h_perp + m_h * d_com * d_com)
        return mass, ixx, iyy, izz
    raise ValueError(f"Invalid scale: {scale!r}")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Immutable backend-neutral sample.

    ``handle_scale`` and ``head_scale`` are the original URDF full dimensions
    (cylinders use ``(height, diameter)``).  ``object_scale`` is the source
    reward-space normalization, and ``diaginertia`` is in the authored body
    frame after the cylinder's -pi/2-y orientation.  No file is written.
    """

    type: str
    topology: str  # ``box_box``, ``capsule_box`` or ``box_only``
    authored_shape: str  # ``box`` or ``cylinder``
    handle_scale: tuple[float, ...]
    head_scale: tuple[float, ...] | None
    handle_density: float
    head_density: float | None
    mass: float
    com: tuple[float, float, float]
    diaginertia: tuple[float, float, float]
    object_scale: tuple[float, float, float]

    @property
    def inertia(self) -> tuple[float, float, float]:
        """Readable alias for callers that use the generic term ``inertia``."""
        return self.diaginertia


ALL_TYPES: tuple[str, ...] = tuple(dict.fromkeys(d.type for d in _OBJECT_SIZE_DISTRIBUTIONS))


def build_tool_catalog(
    types: Sequence[str] = ALL_TYPES,
    num_per_type: int = 100,
    seed: int = 42,
    shuffle: bool = True,
) -> tuple[ToolSpec, ...]:
    """Sample the source distributions into an immutable catalog.

    Draw order is exactly ``handle_density, head_density, handle_scale,
    head_scale`` for every distribution, followed by one lockstep shuffle.
    A legacy ``RandomState`` is intentionally local so catalog construction
    cannot alter a caller's global NumPy RNG state.
    """
    if num_per_type < 0:
        raise ValueError("num_per_type must be non-negative")
    rng = np.random.RandomState(seed)
    type_set = set(types)
    matching = [d for d in _OBJECT_SIZE_DISTRIBUTIONS if d.type in type_set]
    if not matching:
        raise ValueError(
            f"No matching distribution for types={tuple(types)!r}; valid={ALL_TYPES!r}"
        )

    specs: list[ToolSpec] = []
    for dist in matching:
        handle_densities = dist.sample_handle_densities(num_per_type, rng)
        head_densities = dist.sample_head_densities(num_per_type, rng)
        handle_scales = dist.sample_handle_scales(num_per_type, rng)
        head_scales = dist.sample_head_scales(num_per_type, rng)
        for i in range(num_per_type):
            # Preserve NumPy scalar arithmetic through physical-property
            # calculations, just as the upstream generator does.  Casting
            # before the calculation can alter a last bit on some versions.
            handle_raw: Sequence[float] = cast(Sequence[float], handle_scales[i])
            head_raw: Sequence[float] | None = (
                cast(Sequence[float], head_scales[i]) if head_scales is not None else None
            )
            handle_scale = tuple(float(x) for x in handle_raw)
            head_scale = tuple(float(x) for x in head_raw) if head_raw is not None else None
            handle_density = float(handle_densities[i])
            head_density = float(head_densities[i]) if head_densities is not None else None

            is_cylinder = len(handle_scale) == 2
            hm, hixx, hiyy, hizz = _compute_mass_and_inertia(
                handle_raw, cast(float, handle_densities[i])
            )
            if is_cylinder:
                # The URDF rotates the cylinder -pi/2 around y, so x and z
                # inertia components exchange in the authored body frame.
                hixx, hiyy, hizz = hizz, hiyy, hixx
                handle_scale_3d = (handle_raw[0], handle_raw[1], handle_raw[1])
            else:
                handle_scale_3d = (handle_raw[0], handle_raw[1], handle_raw[2])

            if head_scale is None:
                mass = hm
                com = (0.0, 0.0, 0.0)
                inertia = (hixx, hiyy, hizz)
                topology = "box_only"
            else:
                if head_density is None:  # defensive: source always pairs these
                    raise ValueError("head density missing for a distribution with a head")
                assert head_raw is not None and head_densities is not None
                head_mass, head_ixx, head_iyy, head_izz = _compute_mass_and_inertia(
                    head_raw, cast(float, head_densities[i])
                )
                x_offset = handle_scale_3d[0] / 2 + head_raw[0] / 2
                mass = hm + head_mass
                com_x = (hm * 0.0 + head_mass * x_offset) / mass
                d_handle = -com_x
                d_head = x_offset - com_x
                ixx = hixx + head_ixx
                iyy = (hiyy + hm * d_handle * d_handle) + (head_iyy + head_mass * d_head * d_head)
                izz = (hizz + hm * d_handle * d_handle) + (head_izz + head_mass * d_head * d_head)
                inertia = (ixx, iyy, izz)
                com = (com_x, 0.0, 0.0)
                topology = "capsule_box" if is_cylinder else "box_box"

            specs.append(
                ToolSpec(
                    type=dist.type,
                    topology=topology,
                    authored_shape="cylinder" if is_cylinder else "box",
                    handle_scale=handle_scale,
                    head_scale=head_scale,
                    handle_density=handle_density,
                    head_density=head_density,
                    mass=float(mass),
                    com=(float(com[0]), float(com[1]), float(com[2])),
                    diaginertia=(float(inertia[0]), float(inertia[1]), float(inertia[2])),
                    object_scale=(
                        float(handle_scale_3d[0]) / OBJECT_BASE_SIZE,
                        float(handle_scale_3d[1]) / OBJECT_BASE_SIZE,
                        float(handle_scale_3d[2]) / OBJECT_BASE_SIZE,
                    ),
                )
            )

    if shuffle:
        indices = np.arange(len(specs))
        rng.shuffle(indices)
        specs = [specs[int(i)] for i in indices]
    return tuple(specs)


__all__ = ["ALL_TYPES", "ToolSpec", "build_tool_catalog"]
