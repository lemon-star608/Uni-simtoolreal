"""Transcription-style MJCF writer for the KUKA iiwa14 + SHARPA left hand.

MuJoCo line (roadmap M-J1, decision D2): the vendored URDF
(``assets/simtoolreal/urdf/kuka_sharpa_description/
iiwa14_left_sharpa_adjusted_restricted.urdf``) is parsed with plain
ElementTree — no physics engine involved — and transcribed into MJCF.  The
explicit semantic decisions, all marked as tuning items where they are not
pure transcriptions:

* Fixed-joint subtrees are welded into their nearest articulating ancestor
  (parallel-axis inertia composition, the same math as
  ``tool_catalog._compute_mass_and_inertia``'s composite branch).  Twelve
  mass-carrying fixed links merge: ``iiwa14_link_ee`` (massless frame),
  ``sharpa_mount``, ``left_hand_C_MC``, the five ``*_elastomer`` pads and the
  five ``*_fingertip`` links.  The five fingertip frames are additionally
  preserved as ``<site>`` elements on their DP ancestor bodies.
* The hand collision layer is the URDF's own collision meshes (the Isaac
  line's training anchor): per-phalanx ``*_collision`` STLs transcribed from
  the vendored URDF, with welded elastomer pads composed onto their DP
  ancestors through the same frame machinery.  (History: this writer briefly
  carried the old tree's Menagerie Sharpa Wave transplant; it was retired in
  favor of the URDF collision geometry.  The contact *profile* below —
  friction slide split, condim/solimp/solref — is unchanged by that swap.)
* Actuator kp/kv and hand joint armature are parsed from the owner YAML
  (``owner_physics.py``) — no numbers are restated here.
* Friction v1 mapping (tuning items; user-approved): robot geoms get the
  owner YAML's robot slide (0.5), geoms on the five fingertip DP bodies (DP
  mesh + welded elastomer pad) get the fingertip slide (1.5), torsion/roll
  keep the MuJoCo defaults.  Known divergence from the old tree, which
  unified hand collision slide at 1.0; the parity test exempts the
  sliding-friction field explicitly.
* Hand joint ``frictionloss`` is 0: the old tree's values came from
  ``HAND_JOINT_FRICTION``, which the new repository's ``config.py``
  deliberately retires as dead code (unused by the original repository).
* ``gravcomp="1"`` on every robot body: the original task loads the robot
  with ``disable_gravity=True`` (scene_utils.py:1733-1734).
* The 50 contact excludes of the old tree are transcribed and remapped
  through the weld map (merged bodies re-point to their kept ancestor).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...assets import ASSET_ROOT
from .config import ARM_DEFAULT_JOINT_POS, JOINT_NAMES_CANONICAL
from .owner_physics import (
    ROLLING_FRICTION_DEFAULT,
    TORSIONAL_FRICTION_DEFAULT,
    load_owner_physics,
)

ROBOT_URDF = (
    ASSET_ROOT / "urdf" / "kuka_sharpa_description" / "iiwa14_left_sharpa_adjusted_restricted.urdf"
)
TABLE_URDF = ASSET_ROOT / "urdf" / "table_narrow.urdf"
_URDF_MESH_ROOT = ASSET_ROOT / "urdf" / "kuka_sharpa_description"

ROBOT_ROOT_LINK = "iiwa14_link_0"
FINGERTIP_SITE_LINKS: tuple[str, ...] = (
    "left_thumb_fingertip",
    "left_index_fingertip",
    "left_middle_fingertip",
    "left_ring_fingertip",
    "left_pinky_fingertip",
)

# Robot spawn pose (owner YAML robot init_state; scene_utils.py:1811-1842).
# Owned by the scene declaration, not the robot file: write_robot_mjcf leaves
# the root at the origin and the entity initial_state applies this offset.
ROBOT_SCENE_POS = (0.0, 0.8, 0.0)

_HAND_COLLISION_RGB = "0.1 0.8 1 0.35"
# Contact-solver profile of the Menagerie transplant layer, transcribed from
# the old tree's kuka_sharpa.xml geom attributes (tuning items).
_HAND_SOLIMP = "0.9 0.95 0.001 0.5 2.0"
_HAND_SOLREF = "0.02 1.0"

# ----------------------------------------------------------------------------
# Mesh asset table: arm visual+collision meshes, SHARPA visual meshes, and the
# SHARPA URDF collision meshes (the hand collision authority — the Menagerie
# transplant layer was retired in favor of the URDF collision geometry, the
# Isaac line's training anchor).  Paths are relative to the vendored URDF
# directory.
# ----------------------------------------------------------------------------
_MESH_FILES: tuple[tuple[str, str], ...] = (
    *[(f"link_{i}_visual", f"new_iiwa14_meshes/visual/link_{i}.stl") for i in range(8)],
    *[(f"link_{i}", f"new_iiwa14_meshes/collision/link_{i}.stl") for i in range(8)],
    ("left_hand_C_MC_visual", "left_sharpa_meshes/left_hand_C_MC_visual.STL"),
    ("left_thumb_CMC_VL", "left_sharpa_meshes/left_thumb_CMC_VL.STL"),
    ("left_thumb_MC_visual", "left_sharpa_meshes/left_thumb_MC_visual.STL"),
    ("left_thumb_MCP_VL_visual", "left_sharpa_meshes/left_thumb_MCP_VL_visual.STL"),
    ("left_thumb_PP_visual", "left_sharpa_meshes/left_thumb_PP_visual.STL"),
    ("left_thumb_DP_visual", "left_sharpa_meshes/left_thumb_DP_visual.STL"),
    ("thumb_elastomer_surface", "left_sharpa_meshes/thumb_elastomer_surface.STL"),
    ("left_MCP_VL_visual", "left_sharpa_meshes/left_MCP_VL_visual.STL"),
    ("left_PP_visual", "left_sharpa_meshes/left_PP_visual.STL"),
    ("left_MP_visual", "left_sharpa_meshes/left_MP_visual.STL"),
    ("left_DP_visual", "left_sharpa_meshes/left_DP_visual.STL"),
    ("elastomer_surface", "left_sharpa_meshes/elastomer_surface.STL"),
    ("left_pinky_MC_visual", "left_sharpa_meshes/left_pinky_MC_visual.STL"),
    # Hand collision meshes (URDF <collision> elements; CMC_VL has none).
    ("left_hand_C_MC_collision", "left_sharpa_meshes/left_hand_C_MC.STL"),
    ("left_thumb_MC_collision", "left_sharpa_meshes/left_thumb_MC.STL"),
    ("left_thumb_PP_collision", "left_sharpa_meshes/left_thumb_PP.STL"),
    ("left_thumb_DP_collision", "left_sharpa_meshes/left_thumb_DP.STL"),
    ("thumb_elastomer_collision", "left_sharpa_meshes/thumb_elastomer.STL"),
    ("left_MCP_VL_collision", "left_sharpa_meshes/MCP_VL.STL"),
    ("left_PP_collision", "left_sharpa_meshes/left_PP.STL"),
    ("left_MP_collision", "left_sharpa_meshes/left_MP.STL"),
    ("left_DP_collision", "left_sharpa_meshes/left_DP.STL"),
    ("elastomer_collision", "left_sharpa_meshes/elastomer.STL"),
    ("left_pinky_MC_collision", "left_sharpa_meshes/left_pinky_MC.STL"),
)

# Contact excludes transcribed from the old tree's kuka_sharpa.xml <contact>
# section (originally derived from the source repo's adjacent-link masks,
# adjacent_links.py:LEFT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS).  Endpoints that
# the weld merges are re-pointed to their kept ancestor and duplicates drop.
_EXCLUDE_PAIRS: tuple[tuple[str, str], ...] = (
    ("iiwa14_link_0", "iiwa14_link_1"),
    ("iiwa14_link_1", "iiwa14_link_2"),
    ("iiwa14_link_2", "iiwa14_link_3"),
    ("iiwa14_link_3", "iiwa14_link_4"),
    ("iiwa14_link_4", "iiwa14_link_5"),
    ("iiwa14_link_5", "iiwa14_link_6"),
    ("iiwa14_link_6", "iiwa14_link_7"),
    ("iiwa14_link_6", "left_hand_C_MC"),
    ("iiwa14_link_7", "left_thumb_CMC_VL"),
    ("left_hand_C_MC", "left_thumb_CMC_VL"),
    ("iiwa14_link_7", "left_thumb_MC"),
    ("left_hand_C_MC", "left_thumb_MC"),
    ("iiwa14_link_7", "left_index_MCP_VL"),
    ("left_hand_C_MC", "left_index_MCP_VL"),
    ("iiwa14_link_7", "left_index_PP"),
    ("left_hand_C_MC", "left_index_PP"),
    ("iiwa14_link_7", "left_middle_MCP_VL"),
    ("left_hand_C_MC", "left_middle_MCP_VL"),
    ("iiwa14_link_7", "left_middle_PP"),
    ("left_hand_C_MC", "left_middle_PP"),
    ("iiwa14_link_7", "left_ring_MCP_VL"),
    ("left_hand_C_MC", "left_ring_MCP_VL"),
    ("iiwa14_link_7", "left_ring_PP"),
    ("left_hand_C_MC", "left_ring_PP"),
    ("iiwa14_link_7", "left_pinky_MC"),
    ("left_hand_C_MC", "left_pinky_MC"),
    ("left_index_MCP_VL", "left_index_PP"),
    ("left_index_MP", "left_index_PP"),
    ("left_index_DP", "left_index_MP"),
    ("left_index_MP", "left_index_elastomer"),
    ("left_middle_MCP_VL", "left_middle_PP"),
    ("left_middle_MP", "left_middle_PP"),
    ("left_middle_DP", "left_middle_MP"),
    ("left_middle_MP", "left_middle_elastomer"),
    ("left_pinky_MC", "left_pinky_MCP_VL"),
    ("left_pinky_MC", "left_pinky_PP"),
    ("left_pinky_MCP_VL", "left_pinky_PP"),
    ("left_pinky_MP", "left_pinky_PP"),
    ("left_pinky_DP", "left_pinky_MP"),
    ("left_pinky_MP", "left_pinky_elastomer"),
    ("left_ring_MCP_VL", "left_ring_PP"),
    ("left_ring_MP", "left_ring_PP"),
    ("left_ring_DP", "left_ring_MP"),
    ("left_ring_MP", "left_ring_elastomer"),
    ("left_thumb_CMC_VL", "left_thumb_MC"),
    ("left_thumb_MC", "left_thumb_MCP_VL"),
    ("left_thumb_MC", "left_thumb_PP"),
    ("left_thumb_MCP_VL", "left_thumb_PP"),
    ("left_thumb_DP", "left_thumb_PP"),
    ("left_thumb_PP", "left_thumb_elastomer"),
)


# ----------------------------------------------------------------------------
# Minimal rotation/frames math (numpy-only; no physics engine)
# ----------------------------------------------------------------------------


def _rpy_to_mat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis rpy -> rotation matrix (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _quat_to_mat(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _mat_to_quat(r: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> MJCF quaternion (wxyz)."""
    t = float(np.trace(r))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w, x, y, z = (
            0.25 * s,
            (r[2, 1] - r[1, 2]) / s,
            (r[0, 2] - r[2, 0]) / s,
            (r[1, 0] - r[0, 1]) / s,
        )
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w, x, y, z = (
            (r[2, 1] - r[1, 2]) / s,
            0.25 * s,
            (r[0, 1] + r[1, 0]) / s,
            (r[0, 2] + r[2, 0]) / s,
        )
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w, x, y, z = (
            (r[0, 2] - r[2, 0]) / s,
            (r[0, 1] + r[1, 0]) / s,
            0.25 * s,
            (r[1, 2] + r[2, 1]) / s,
        )
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w, x, y, z = (
            (r[1, 0] - r[0, 1]) / s,
            (r[0, 2] + r[2, 0]) / s,
            (r[1, 2] + r[2, 1]) / s,
            0.25 * s,
        )
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


def _fmt(value: float) -> str:
    return f"{float(value):.17g}"


def _fmt_vec(values) -> str:
    return " ".join(_fmt(v) for v in values)


# ----------------------------------------------------------------------------
# URDF parse
# ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Inertial:
    mass: float
    com: np.ndarray  # in the link frame
    inertia: np.ndarray  # 3x3, in the link frame at com (all rpy are 0 here)


@dataclass(frozen=True, slots=True)
class _Geometry:
    mesh_file: str | None  # relative mesh path for mesh geometry
    rgba: str | None
    origin_xyz: np.ndarray
    origin_rpy: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class _Link:
    name: str
    inertial: _Inertial | None
    visuals: tuple[_Geometry, ...]
    collisions: tuple[_Geometry, ...]


@dataclass(frozen=True, slots=True)
class _Joint:
    name: str
    type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float] | None
    lower: float | None
    upper: float | None
    effort: float | None


def _parse_geometry(element: ET.Element, materials: dict[str, str]) -> _Geometry:
    origin = element.find("origin")
    xyz = (
        np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
        if origin is not None
        else np.zeros(3)
    )
    rpy = (
        tuple(float(v) for v in origin.get("rpy", "0 0 0").split())
        if origin is not None
        else (0.0, 0.0, 0.0)
    )
    mesh = element.find("geometry/mesh")
    mesh_file = mesh.get("filename") if mesh is not None else None
    material = element.find("material")
    rgba: str | None = None
    if material is not None:
        color = material.find("color")
        if color is not None and color.get("rgba"):
            rgba = color.get("rgba")
        elif material.get("name"):
            rgba = materials.get(material.get("name"))
    return _Geometry(mesh_file=mesh_file, rgba=rgba, origin_xyz=xyz, origin_rpy=rpy)  # type: ignore[arg-type]


def _parse_urdf(path: Path) -> tuple[dict[str, _Link], dict[str, _Joint], str]:
    root = ET.parse(path).getroot()
    materials: dict[str, str] = {}
    for material in root.findall("material"):
        color = material.find("color")
        if material.get("name") and color is not None:
            materials[material.get("name")] = color.get("rgba")

    links: dict[str, _Link] = {}
    for element in root.findall("link"):
        inertial_el = element.find("inertial")
        inertial: _Inertial | None = None
        if inertial_el is not None:
            origin = inertial_el.find("origin")
            rpy = (
                tuple(float(v) for v in origin.get("rpy", "0 0 0").split())
                if origin is not None
                else (0.0, 0.0, 0.0)
            )
            if any(abs(v) > 0.0 for v in rpy):
                raise ValueError(
                    f"{path}: inertial rpy {rpy} on link {element.get('name')!r} is not "
                    "handled by the transcription writer (all vendored inertials are rpy=0)"
                )
            com = (
                np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
                if origin is not None
                else np.zeros(3)
            )
            inertia_el = inertial_el.find("inertia")
            ixx, iyy, izz = (float(inertia_el.get(k)) for k in ("ixx", "iyy", "izz"))
            ixy, ixz, iyz = (float(inertia_el.get(k)) for k in ("ixy", "ixz", "iyz"))
            inertial = _Inertial(
                mass=float(inertial_el.find("mass").get("value")),
                com=com,
                inertia=np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]),
            )
        visuals = tuple(_parse_geometry(v, materials) for v in element.findall("visual"))
        collisions = tuple(_parse_geometry(c, materials) for c in element.findall("collision"))
        links[element.get("name")] = _Link(element.get("name"), inertial, visuals, collisions)

    joints: dict[str, _Joint] = {}
    for element in root.findall("joint"):
        origin = element.find("origin")
        xyz = (
            np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
            if origin is not None
            else np.zeros(3)
        )
        rpy = (
            tuple(float(v) for v in origin.get("rpy", "0 0 0").split())
            if origin is not None
            else (0.0, 0.0, 0.0)
        )
        axis_el = element.find("axis")
        limit = element.find("limit")
        joint = _Joint(
            name=element.get("name"),
            type=element.get("type"),
            parent=element.find("parent").get("link"),
            child=element.find("child").get("link"),
            xyz=xyz,
            rpy=rpy,  # type: ignore[arg-type]
            axis=tuple(float(v) for v in axis_el.get("xyz").split())
            if axis_el is not None
            else None,
            lower=float(limit.get("lower")) if limit is not None and limit.get("lower") else None,
            upper=float(limit.get("upper")) if limit is not None and limit.get("upper") else None,
            effort=float(limit.get("effort"))
            if limit is not None and limit.get("effort")
            else None,
        )
        joints[joint.child] = joint

    child_links = set(joints)
    roots = [name for name in links if name not in child_links]
    if roots != [ROBOT_ROOT_LINK]:
        raise ValueError(f"{path}: expected single root link {ROBOT_ROOT_LINK!r}, got {roots}")
    return links, joints, ROBOT_ROOT_LINK


# ----------------------------------------------------------------------------
# Weld analysis: fixed subtrees merge into the nearest articulating ancestor
# ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _WeldPlan:
    kept: tuple[str, ...]  # kept bodies in DFS (document) order; root first
    kept_parent: dict[str, str | None]  # kept body -> kept parent body
    frame_in_kept: dict[
        str, tuple[np.ndarray, np.ndarray]
    ]  # any link -> (R, p) in kept-ancestor frame
    weld_members: dict[str, tuple[str, ...]]  # kept body -> merged link names (incl. itself)
    kept_joint: dict[str, _Joint]  # kept non-root body -> its revolute joint
    kept_frame_in_parent: dict[
        str, tuple[np.ndarray, np.ndarray]
    ]  # kept body -> (R, p) in kept-parent frame


def _plan_welds(links: dict[str, _Link], joints: dict[str, _Joint], root: str) -> _WeldPlan:
    children: dict[str, list[_Joint]] = {name: [] for name in links}
    for joint in joints.values():
        children[joint.parent].append(joint)

    kept: list[str] = []
    kept_parent: dict[str, str | None] = {}
    frame_in_kept: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    weld_members: dict[str, list[str]] = {}
    kept_joint: dict[str, _Joint] = {}
    kept_frame_in_parent: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def visit(
        link: str,
        ancestor: str,
        r_acc: np.ndarray,
        p_acc: np.ndarray,
        parent_kept: str | None,
        joint: _Joint | None,
    ) -> None:
        is_kept = joint is None or joint.type != "fixed"
        if is_kept:
            kept.append(link)
            kept_parent[link] = parent_kept
            weld_members[link] = [link]
            if joint is not None:
                kept_joint[link] = joint
                kept_frame_in_parent[link] = (r_acc, p_acc)
            ancestor = link
            r_acc, p_acc = np.eye(3), np.zeros(3)
        else:
            weld_members[ancestor].append(link)
        frame_in_kept[link] = (r_acc, p_acc)
        for child_joint in children[link]:
            r_joint = _rpy_to_mat(*child_joint.rpy)
            r_child = r_acc @ r_joint
            p_child = p_acc + r_acc @ child_joint.xyz
            visit(child_joint.child, ancestor, r_child, p_child, ancestor, child_joint)

    visit(root, root, np.eye(3), np.zeros(3), None, None)
    return _WeldPlan(
        kept=tuple(kept),
        kept_parent=kept_parent,
        frame_in_kept=frame_in_kept,
        weld_members={k: tuple(v) for k, v in weld_members.items()},
        kept_joint=kept_joint,
        kept_frame_in_parent=kept_frame_in_parent,
    )


def _merge_inertial(
    members: tuple[str, ...], links: dict[str, _Link], plan: _WeldPlan
) -> tuple[np.ndarray, float, np.ndarray]:
    """Parallel-axis composition of a welded group's inertia (in the kept frame).

    Same math as the composite branch of ``tool_catalog._compute_mass_and_inertia``
    (generate_objects.py:206-218), generalized to rotated frames and full
    (non-diagonal) tensors.
    """
    total_mass = 0.0
    com_weighted = np.zeros(3)
    parts: list[tuple[float, np.ndarray, np.ndarray]] = []
    for name in members:
        inertial = links[name].inertial
        if inertial is None:
            continue
        r_link, p_link = plan.frame_in_kept[name]
        com = p_link + r_link @ inertial.com
        inertia = r_link @ inertial.inertia @ r_link.T
        parts.append((inertial.mass, com, inertia))
        total_mass += inertial.mass
        com_weighted += inertial.mass * com
    if total_mass <= 0.0:
        raise ValueError(f"welded group rooted at {members[0]!r} has no mass")
    com_total = com_weighted / total_mass
    inertia_total = np.zeros((3, 3))
    for mass, com, inertia in parts:
        d = com - com_total
        inertia_total += inertia + mass * ((d @ d) * np.eye(3) - np.outer(d, d))
    return com_total, total_mass, inertia_total


# ----------------------------------------------------------------------------
# MJCF emission
# ----------------------------------------------------------------------------


def _emit_inertial(body: ET.Element, com: np.ndarray, mass: float, inertia: np.ndarray) -> None:
    fullinertia = (
        inertia[0, 0],
        inertia[1, 1],
        inertia[2, 2],
        inertia[0, 1],
        inertia[0, 2],
        inertia[1, 2],
    )
    ET.SubElement(
        body,
        "inertial",
        {"pos": _fmt_vec(com), "mass": _fmt(mass), "fullinertia": _fmt_vec(fullinertia)},
    )


def _mesh_asset_name(mesh_file: str) -> str:
    for name, rel in _MESH_FILES:
        if rel == mesh_file or rel.endswith("/" + mesh_file):
            return name
    raise KeyError(f"mesh file {mesh_file!r} is not in the transcribed asset table")


def _compose_frame(
    base: tuple[np.ndarray, np.ndarray], xyz: np.ndarray, rpy: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    r_base, p_base = base
    r_local = _rpy_to_mat(*rpy)
    return r_base @ r_local, p_base + r_base @ xyz


def _emit_frame_attrs(element: ET.Element, frame: tuple[np.ndarray, np.ndarray]) -> None:
    r, p = frame
    element.set("pos", _fmt_vec(p))
    quat = _mat_to_quat(r)
    if not np.allclose(quat, (1.0, 0.0, 0.0, 0.0), atol=1e-15):
        element.set("quat", _fmt_vec(quat))


def _friction_attr(slide: float) -> str:
    return f"{_fmt(slide)} {_fmt(TORSIONAL_FRICTION_DEFAULT)} {_fmt(ROLLING_FRICTION_DEFAULT)}"


def _emit_visual_geom(
    body: ET.Element, name: str, geometry: _Geometry, frame: tuple[np.ndarray, np.ndarray]
) -> None:
    geom = ET.SubElement(
        body,
        "geom",
        {
            # Every geom is named: the mjbatch variant pack requires unique,
            # non-empty geom names (M-J2).  The old tree left these unnamed.
            "name": name,
            "type": "mesh",
            "mesh": _mesh_asset_name(geometry.mesh_file),
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
            "density": "0",
        },
    )
    if geometry.rgba:
        geom.set("rgba", geometry.rgba)
    _emit_frame_attrs(geom, frame)


def _emit_arm_collision_geom(
    body: ET.Element,
    name: str,
    geometry: _Geometry,
    frame: tuple[np.ndarray, np.ndarray],
    slide: float,
) -> None:
    geom = ET.SubElement(
        body,
        "geom",
        {
            "name": name,
            "type": "mesh",
            "mesh": _mesh_asset_name(geometry.mesh_file),
            "friction": _friction_attr(slide),
        },
    )
    if geometry.rgba:
        geom.set("rgba", geometry.rgba)
    _emit_frame_attrs(geom, frame)


def _emit_hand_collision_geom(
    body: ET.Element, name: str, frame: tuple[np.ndarray, np.ndarray], slide: float, **attrs: str
) -> None:
    geom = ET.SubElement(
        body,
        "geom",
        {
            "name": name,
            "friction": _friction_attr(slide),
            "density": "0",
            "contype": "1",
            "conaffinity": "1",
            "group": "3",
            "rgba": _HAND_COLLISION_RGB,
            "condim": "3",
            "solimp": _HAND_SOLIMP,
            "solref": _HAND_SOLREF,
            "margin": "0",
            "gap": "0",
            **attrs,
        },
    )
    _emit_frame_attrs(geom, frame)


def _identity() -> tuple[np.ndarray, np.ndarray]:
    return np.eye(3), np.zeros(3)


def build_robot_mjcf() -> ET.Element:
    """Build the robot MJCF tree (no scene ownership: no option/keyframe).

    Returns the ``<mujoco>`` root element.  ``write_robot_mjcf`` serializes it.
    """
    physics = load_owner_physics()
    links, joints, root_link = _parse_urdf(ROBOT_URDF)
    plan = _plan_welds(links, joints, root_link)

    mujoco_el = ET.Element("mujoco", {"model": "kuka_sharpa"})
    # MJCF defaults to degrees; the transcription is all-radians (URDF).
    ET.SubElement(mujoco_el, "compiler", {"angle": "radian"})
    asset_el = ET.SubElement(mujoco_el, "asset")
    for name, rel in _MESH_FILES:
        path = _URDF_MESH_ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"vendored mesh missing: {path}")
        attrs = {"name": name, "file": str(path)}
        if path.suffix.lower() == ".stl":
            attrs["content_type"] = "model/stl"
        ET.SubElement(asset_el, "mesh", attrs)

    worldbody = ET.SubElement(mujoco_el, "worldbody")
    fingertip_slide_bodies = set(physics.fingertip_body_names)

    def emit_body(link: str, parent_el: ET.Element) -> ET.Element:
        body = ET.SubElement(parent_el, "body", {"name": link, "gravcomp": "1"})
        if link in plan.kept_frame_in_parent:
            _emit_frame_attrs(body, plan.kept_frame_in_parent[link])

        joint = plan.kept_joint.get(link)
        if joint is not None:
            attrs = {
                "name": joint.name,
                "pos": "0 0 0",
                "axis": _fmt_vec(joint.axis or (0.0, 0.0, 1.0)),
                "damping": "0",
                # Armature from the owner YAML (hand joints only; arm = 0).
                "armature": _fmt(physics.actuator_gains[joint.name].armature),
                # The old tree's frictionloss came from HAND_JOINT_FRICTION,
                # retired as dead code in this repository's config.py.
                "frictionloss": "0",
            }
            if joint.lower is not None and joint.upper is not None:
                attrs["range"] = f"{_fmt(joint.lower)} {_fmt(joint.upper)}"
            if joint.effort is not None:
                attrs["actuatorfrcrange"] = f"{_fmt(-joint.effort)} {_fmt(joint.effort)}"
            ET.SubElement(body, "joint", attrs)

        members = plan.weld_members[link]
        com, mass, inertia = _merge_inertial(members, links, plan)
        _emit_inertial(body, com, mass, inertia)

        for member in members:
            member_frame = plan.frame_in_kept[member]
            is_hand = member.startswith("left_")
            for visual in links[member].visuals:
                frame = _compose_frame(member_frame, visual.origin_xyz, visual.origin_rpy)
                _emit_visual_geom(body, f"{member}_visual_geom", visual, frame)
            for collision in links[member].collisions:
                frame = _compose_frame(member_frame, collision.origin_xyz, collision.origin_rpy)
                if is_hand:
                    # Hand collision = the URDF's own collision meshes (the
                    # Isaac line's training anchor).  Geoms landing on the five
                    # DP bodies (DP mesh + welded elastomer pad) carry the
                    # owner YAML's fingertip slide; everything else carries
                    # the robot slide.
                    slide = (
                        physics.fingertip_slide_friction
                        if link in fingertip_slide_bodies
                        else physics.robot_slide_friction
                    )
                    _emit_hand_collision_geom(
                        body,
                        f"{member}_collision_geom",
                        frame,
                        slide,
                        type="mesh",
                        mesh=_mesh_asset_name(collision.mesh_file),
                    )
                else:
                    _emit_arm_collision_geom(
                        body,
                        f"{member}_collision_geom",
                        collision,
                        frame,
                        physics.robot_slide_friction,
                    )

        # Fingertip frames survive the weld as sites on their DP ancestors.
        for fingertip in FINGERTIP_SITE_LINKS:
            if fingertip in members:
                site = ET.SubElement(body, "site", {"name": fingertip})
                _emit_frame_attrs(site, plan.frame_in_kept[fingertip])

        for child in plan.kept:
            if plan.kept_parent[child] == link:
                emit_body(child, body)
        return body

    emit_body(root_link, worldbody)

    actuator_el = ET.SubElement(mujoco_el, "actuator")
    for name in JOINT_NAMES_CANONICAL:
        gains = physics.actuator_gains[name]
        joint = next(j for j in joints.values() if j.name == name)
        attrs = {
            "name": f"{name}_ctrl",
            "joint": name,
            "kp": _fmt(gains.stiffness),
            "kv": _fmt(gains.damping),
            "inheritrange": "1",
        }
        if joint.effort is not None:
            attrs["forcerange"] = f"{_fmt(-joint.effort)} {_fmt(joint.effort)}"
        ET.SubElement(actuator_el, "position", attrs)

    contact_el = ET.SubElement(mujoco_el, "contact")
    emitted: set[tuple[str, str]] = set()
    for body1, body2 in _EXCLUDE_PAIRS:
        mapped = tuple(
            sorted(
                (
                    next(k for k, v in plan.weld_members.items() if body1 in v),
                    next(k for k, v in plan.weld_members.items() if body2 in v),
                )
            )
        )
        if mapped[0] == mapped[1] or mapped in emitted:
            continue
        emitted.add(mapped)
        ET.SubElement(
            contact_el,
            "exclude",
            {"name": f"exclude_{mapped[0]}__{mapped[1]}", "body1": mapped[0], "body2": mapped[1]},
        )

    ET.indent(mujoco_el, space="  ")
    return mujoco_el


def write_robot_mjcf(path: Path) -> Path:
    """Serialize the robot MJCF (no scene ownership: no option/keyframe)."""
    path.write_text(ET.tostring(build_robot_mjcf(), encoding="unicode") + "\n")
    return path


# ----------------------------------------------------------------------------
# Scene physics-solver option block (old-tree scene.xml), shared by the
# B-form entity sources (M-J5; the A-form scene builders are retired).
# ----------------------------------------------------------------------------


def _scene_option_element() -> ET.Element:
    """The scene's physics-solver option block (old-tree scene.xml).

    Every B-form entity source carries an identical copy because unisim's
    entity composition requires agreeing global options across every source
    (mjcf_compiler ``_options``; timestep excluded — the composer forces
    ``sim_dt``).
    """
    physics = load_owner_physics()
    option = ET.Element(
        "option",
        {
            "timestep": _fmt(physics.sim_dt),
            "gravity": "0 0 -9.81",
            "integrator": "implicitfast",
            "solver": "Newton",
            "cone": "elliptic",
            "impratio": "10",
            "iterations": "100",
            "ls_iterations": "50",
        },
    )
    ET.SubElement(option, "flag", {"contact": "enable", "multiccd": "enable"})
    return option


__all__ = [
    "FINGERTIP_SITE_LINKS",
    "ROBOT_SCENE_POS",
    "ROBOT_URDF",
    "TABLE_URDF",
    "build_robot_mjcf",
    "write_robot_mjcf",
]
