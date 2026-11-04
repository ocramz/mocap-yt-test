"""Skeleton definitions and forward kinematics.

Conventions (shared with MocapAnything's datasets):
  * Y-up, character faces +Z, character's left is +X.
  * Units are meters.
  * Rest pose is a T-pose (arms along +/-X), the convention Mixamo rigs use;
    MocapAnything's obj1k training split is mostly Mixamo-style humanoids.
  * Joints are stored in BVH (depth-first) order, so ``parents[i] < i``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class Skeleton:
    names: list[str]
    parents: np.ndarray  # (J,) int, -1 for the root
    offsets: np.ndarray  # (J, 3) rest offset from parent, meters; root offset = rest root position
    # Per-joint body-part radius (meters) used to build a simple mannequin skin.
    radii: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.parents = np.asarray(self.parents, dtype=np.int64)
        self.offsets = np.asarray(self.offsets, dtype=np.float64)
        assert len(self.names) == len(self.parents) == len(self.offsets)
        for i, p in enumerate(self.parents):
            assert p < i, f"joint {self.names[i]} has parent {p} >= {i}: not in BVH order"

    @property
    def num_joints(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def children(self, j: int) -> list[int]:
        return [i for i, p in enumerate(self.parents) if p == j]

    def leaves(self) -> list[int]:
        return [j for j in range(self.num_joints) if not self.children(j)]

    def rest_positions(self) -> np.ndarray:
        """Global joint positions in the rest pose, (J, 3)."""
        pos = np.zeros((self.num_joints, 3))
        for j, p in enumerate(self.parents):
            pos[j] = self.offsets[j] if p < 0 else pos[p] + self.offsets[j]
        return pos

    def height(self) -> float:
        p = self.rest_positions()
        return float(p[:, 1].max() - p[:, 1].min())


def forward_kinematics(
    skel: Skeleton, local_rot: np.ndarray, root_pos: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Global joint transforms.

    local_rot: (T, J, 3, 3) local rotation matrices (BVH semantics: joint j's frame is
        parent_frame * Translate(offset_j) * local_rot_j).
    root_pos:  (T, 3) root translation; defaults to the rest root offset.
    Returns (global_rot (T,J,3,3), global_pos (T,J,3)).
    """
    T, J = local_rot.shape[:2]
    g_rot = np.zeros_like(local_rot)
    g_pos = np.zeros((T, J, 3))
    for j, p in enumerate(skel.parents):
        if p < 0:
            g_rot[:, j] = local_rot[:, j]
            g_pos[:, j] = skel.offsets[j] if root_pos is None else root_pos
        else:
            g_rot[:, j] = g_rot[:, p] @ local_rot[:, j]
            g_pos[:, j] = g_pos[:, p] + np.einsum("tij,j->ti", g_rot[:, p], skel.offsets[j])
    return g_rot, g_pos


def euler_zyx_deg(local_rot: np.ndarray) -> np.ndarray:
    """(…,3,3) rotation matrices -> intrinsic ZYX euler angles in degrees, as (…,3) [z,y,x].

    Matches BVH channel order ``Zrotation Yrotation Xrotation``.
    """
    shp = local_rot.shape[:-2]
    e = R.from_matrix(local_rot.reshape(-1, 3, 3)).as_euler("ZYX", degrees=True)
    return e.reshape(*shp, 3)


# --------------------------------------------------------------------------------------
# Mixamo-convention humanoid
# --------------------------------------------------------------------------------------

#: Joint list of the Mixamo "no fingers" skeleton LOD (25 joints incl. end joints).
MIXAMO_BODY = [
    # name,            parent
    ("Hips", None),
    ("Spine", "Hips"),
    ("Spine1", "Spine"),
    ("Spine2", "Spine1"),
    ("Neck", "Spine2"),
    ("Head", "Neck"),
    ("HeadTop_End", "Head"),
    ("LeftShoulder", "Spine2"),
    ("LeftArm", "LeftShoulder"),
    ("LeftForeArm", "LeftArm"),
    ("LeftHand", "LeftForeArm"),
    ("RightShoulder", "Spine2"),
    ("RightArm", "RightShoulder"),
    ("RightForeArm", "RightArm"),
    ("RightHand", "RightForeArm"),
    ("LeftUpLeg", "Hips"),
    ("LeftLeg", "LeftUpLeg"),
    ("LeftFoot", "LeftLeg"),
    ("LeftToeBase", "LeftFoot"),
    ("LeftToe_End", "LeftToeBase"),
    ("RightUpLeg", "Hips"),
    ("RightLeg", "RightUpLeg"),
    ("RightFoot", "RightLeg"),
    ("RightToeBase", "RightFoot"),
    ("RightToe_End", "RightToeBase"),
]

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def mixamo_humanoid(height: float = 1.75, fingers: bool = False, prefix: str = "") -> Skeleton:
    """Procedurally generate a Mixamo-convention humanoid skeleton in T-pose.

    Proportions follow standard anthropometric ratios (fractions of stature, after
    Drillis & Contini / Winter, "Biomechanics and Motor Control of Human Movement").
    ``prefix`` lets you emit e.g. ``mixamorig:`` names; MocapAnything strips it when it
    builds joint-name embeddings, so the default is unprefixed.
    """
    H = height
    # Vertical landmarks as fractions of stature (standing, feet at y=0).
    hip_y = 0.530 * H
    knee_y = 0.285 * H
    ankle_y = 0.039 * H
    shoulder_y = 0.818 * H
    neck_base_y = 0.845 * H
    head_base_y = 0.880 * H  # skull base / head pivot
    hip_half_width = 0.095 * H / 1.75 * 1.0
    shoulder_half_width = 0.129 * H  # shoulder joint (glenohumeral) from midline
    clavicle_x = 0.035 * H
    upper_arm = 0.186 * H
    forearm = 0.146 * H
    hand = 0.108 * H
    foot_len_heel_to_ball = 0.110 * H
    toe_len = 0.042 * H

    spine_top_y = neck_base_y - 0.01 * H
    spine_seg = (spine_top_y - hip_y) / 4.0  # Hips->Spine->Spine1->Spine2->Neck

    off: dict[str, tuple[float, float, float]] = {
        "Hips": (0.0, hip_y, 0.0),
        "Spine": (0.0, spine_seg, -0.010 * H / 1.75),
        "Spine1": (0.0, spine_seg, -0.005 * H / 1.75),
        "Spine2": (0.0, spine_seg, 0.0),
        "Neck": (0.0, spine_seg + 0.01 * H, 0.012 * H / 1.75),
        "Head": (0.0, head_base_y - neck_base_y, 0.015 * H / 1.75),
        "HeadTop_End": (0.0, H - head_base_y, 0.0),
        "LeftShoulder": (clavicle_x, shoulder_y - (hip_y + 3 * spine_seg) - 0.0, 0.0),
        "LeftArm": (shoulder_half_width - clavicle_x, 0.0, -0.005 * H / 1.75),
        "LeftForeArm": (upper_arm, 0.0, 0.0),
        "LeftHand": (forearm, 0.0, 0.0),
        "LeftUpLeg": (hip_half_width, -0.045 * H, 0.0),
        "LeftLeg": (0.0, knee_y - (hip_y - 0.045 * H), 0.0),
        "LeftFoot": (0.0, ankle_y - knee_y, -0.005 * H / 1.75),
        "LeftToeBase": (0.0, -ankle_y + 0.012 * H, foot_len_heel_to_ball * 0.85),
        "LeftToe_End": (0.0, -0.006 * H, toe_len),
    }
    # Mirror to the right side.
    for k in list(off):
        if k.startswith("Left"):
            x, y, z = off[k]
            off["Right" + k[4:]] = (-x, y, z)

    joints = list(MIXAMO_BODY)
    if fingers:
        joints = _insert_fingers(joints)
        for side, sx in (("Left", 1.0), ("Right", -1.0)):
            spread = {"Thumb": 0.030, "Index": 0.022, "Middle": 0.0, "Ring": -0.018, "Pinky": -0.034}
            for f in FINGERS:
                base = (
                    (0.030 * sx * H / 1.75, -0.015 * H / 1.75, spread[f] * H / 1.75 + 0.015 * H / 1.75)
                    if f == "Thumb"
                    else (0.090 * sx * H / 1.75, 0.0, spread[f] * H / 1.75)
                )
                seg = 0.030 * H / 1.75 if f != "Pinky" else 0.022 * H / 1.75
                off[f"{side}Hand{f}1"] = base
                for k in (2, 3, 4):
                    off[f"{side}Hand{f}{k}"] = (seg * sx, 0.0, 0.0)
        # Hand bone itself becomes shorter (palm) – hand length now encoded in fingers.

    names = [n for n, _ in joints]
    parents = [(-1 if p is None else names.index(p)) for _, p in joints]
    offsets = np.array([off[n] for n in names], dtype=np.float64)

    # Body radii for the mannequin skin (distance from bone axis).
    rad_table = {
        "Hips": 0.090, "Spine": 0.095, "Spine1": 0.110, "Spine2": 0.120, "Neck": 0.048,
        "Head": 0.090, "Shoulder": 0.045, "Arm": 0.045, "ForeArm": 0.036, "Hand": 0.028,
        "UpLeg": 0.070, "Leg": 0.052, "Foot": 0.040, "ToeBase": 0.032,
    }
    radii = np.zeros(len(names))
    for i, n in enumerate(names):
        key = n.replace("Left", "").replace("Right", "")
        if "Hand" in key and key != "Hand":
            radii[i] = 0.009
        else:
            radii[i] = rad_table.get(key, 0.0)
    radii *= H / 1.75

    names = [prefix + n for n in names]
    return Skeleton(names, np.array(parents), offsets, radii=radii, meta={"height": H, "fingers": fingers})


def _insert_fingers(joints):
    out = []
    for name, parent in joints:
        out.append((name, parent))
        if name in ("LeftHand", "RightHand"):
            for f in FINGERS:
                prev = name
                for k in (1, 2, 3, 4):
                    cur = f"{name}{f}{k}"
                    out.append((cur, prev))
                    prev = cur
    return out
