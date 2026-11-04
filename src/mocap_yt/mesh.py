"""Procedural mannequin skin for a :class:`Skeleton` + linear blend skinning.

The mannequin is a set of (elliptical) capsules, one per bone, rigidly bound to the
bone's parent joint — like an artist's wooden mannequin. It is only used to

  * render the *reference image* MocapAnything conditions on (the model was trained
    on rendered rigged characters, many of them untextured Mixamo-style mannequins),
  * give the exported glTF something visible to skin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .skeleton import Skeleton, forward_kinematics


@dataclass
class SkinnedMesh:
    vertices: np.ndarray  # (V, 3) rest-pose positions
    faces: np.ndarray  # (F, 3) int
    joints: np.ndarray  # (V, 4) int joint indices
    weights: np.ndarray  # (V, 4) float weights (rows sum to 1)
    face_part: np.ndarray  # (F,) int joint index that owns the face (for colouring)

    def dense_weights(self, num_joints: int) -> np.ndarray:
        W = np.zeros((len(self.vertices), num_joints))
        for k in range(4):
            np.add.at(W, (np.arange(len(W)), self.joints[:, k]), self.weights[:, k])
        return W


def _capsule(p0, p1, r0, r1, depth_scale=1.0, n_around=16, n_rings=6):
    """Tapered capsule from p0 (radius r0) to p1 (radius r1).

    ``depth_scale`` squashes the cross-section along the world-Z-most perpendicular
    (front/back) axis, giving torsos an elliptical cross-section.
    Returns (verts (V,3), faces (F,3)).
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    axis = p1 - p0
    L = np.linalg.norm(axis)
    if L < 1e-9:
        axis = np.array([0.0, 1.0, 0.0])
        L = 0.0
    else:
        axis = axis / L
    # perpendicular basis: 'd' is the one closest to world Z (front/back)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(axis @ ref) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    d = ref - (ref @ axis) * axis
    d /= np.linalg.norm(d)
    w = np.cross(axis, d)

    rings = []  # (offset along axis, radius) sample list
    # bottom hemisphere (p0), cylinder, top hemisphere (p1)
    for i in range(n_rings, 0, -1):
        a = (np.pi / 2) * i / n_rings
        rings.append((-r0 * np.sin(a), r0 * np.cos(a), 0.0))
    for i in range(0, n_rings + 1):
        a = (np.pi / 2) * i / n_rings
        rings.append((L + r1 * np.sin(a), r1 * np.cos(a), 1.0))
    # poles
    verts = [p0 - axis * r0]
    theta = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    circ = np.cos(theta)[:, None] * w[None] + depth_scale * np.sin(theta)[:, None] * d[None]
    for t, rad, _ in rings:
        verts.extend(p0 + axis * t + rad * circ)
    verts.append(p1 + axis * r1)
    verts = np.array(verts)
    faces = []
    n = n_around
    # bottom cap fan
    for k in range(n):
        faces.append([0, 1 + (k + 1) % n, 1 + k])
    for ri in range(len(rings) - 1):
        a0 = 1 + ri * n
        b0 = 1 + (ri + 1) * n
        for k in range(n):
            k1 = (k + 1) % n
            faces.append([a0 + k, a0 + k1, b0 + k])
            faces.append([a0 + k1, b0 + k1, b0 + k])
    top = len(verts) - 1
    last = 1 + (len(rings) - 1) * n
    for k in range(n):
        faces.append([last + k, last + (k + 1) % n, top])
    return verts, np.array(faces)


def build_mannequin(skel: Skeleton) -> SkinnedMesh:
    """Capsule mannequin over the skeleton's rest pose (rigid skinning)."""
    assert skel.radii is not None, "skeleton needs per-joint radii to build a skin"
    pos = skel.rest_positions()
    V, F, J, P = [], [], [], []
    nv = 0

    def add(p0, p1, r0, r1, owner, depth=1.0, **kw):
        nonlocal nv
        v, f = _capsule(p0, p1, r0, r1, depth, **kw)
        V.append(v)
        F.append(f + nv)
        J.append(np.full(len(v), owner))
        P.append(np.full(len(f), owner))
        nv += len(v)

    names = [n.split(":")[-1] for n in skel.names]
    for j in range(skel.num_joints):
        kids = skel.children(j)
        rj = skel.radii[j]
        if rj <= 0:
            continue
        base = names[j].replace("Left", "").replace("Right", "")
        if base == "Hand":
            # palm: extends past the wrist along the forearm direction
            d = pos[j] - pos[skel.parents[j]]
            end = pos[j] + d / np.linalg.norm(d) * 0.085 * skel.height() / 1.75
            add(pos[j], end, rj, rj * 0.85, j, depth=0.55)
            continue
        if not kids:
            continue
        torso = base in ("Hips", "Spine", "Spine1", "Spine2")
        depth = 0.72 if torso else 1.0
        if base == "Head":
            # head: ellipsoid-ish capsule between skull base and crown
            c = kids[0]
            top = pos[c] - (pos[c] - pos[j]) * 0.45
            add(pos[j] + (pos[c] - pos[j]) * 0.42, top, rj, rj * 0.95, j, depth=1.1)
            continue
        if base == "Foot":
            c = kids[0]
            add(pos[j], pos[c] + np.array([0, 0.004, 0]), rj, skel.radii[c], j, depth=0.75)
            continue
        if len(kids) > 1:
            # branching joint: bulk towards the mean child + thinner limb roots
            mean_child = pos[kids].mean(0)
            add(pos[j], mean_child, rj, rj * 0.95, j, depth=depth)
            for c in kids:
                rc = skel.radii[c]
                cname = names[c].replace("Left", "").replace("Right", "")
                if cname in ("Shoulder",):
                    add(pos[j] + (pos[c] - pos[j]) * 0.6, pos[c], rc * 1.1, rc, j)
                elif cname == "UpLeg":
                    add(mean_child, pos[c], rj * 0.9, rc, j, depth=0.85)
            continue
        c = kids[0]
        rc = skel.radii[c] if skel.radii[c] > 0 else rj * 0.8
        add(pos[j], pos[c], rj, rc if not torso else rj, j, depth=depth)

    verts = np.concatenate(V)
    faces = np.concatenate(F)
    owner = np.concatenate(J)
    joints = np.zeros((len(verts), 4), dtype=np.int64)
    joints[:, 0] = owner
    weights = np.zeros((len(verts), 4))
    weights[:, 0] = 1.0
    return SkinnedMesh(verts, faces, joints, weights, np.concatenate(P))


def skin(mesh: SkinnedMesh, skel: Skeleton, local_rot: np.ndarray, root_pos: np.ndarray | None = None) -> np.ndarray:
    """Linear blend skinning. local_rot (T,J,3,3) -> deformed vertices (T,V,3)."""
    g_rot, g_pos = forward_kinematics(skel, local_rot, root_pos)
    rest = skel.rest_positions()
    T = local_rot.shape[0]
    out = np.zeros((T, len(mesh.vertices), 3))
    for k in range(4):
        jk = mesh.joints[:, k]
        wk = mesh.weights[:, k]
        if not np.any(wk):
            continue
        rel = mesh.vertices - rest[jk]  # (V,3)
        out += wk[None, :, None] * (np.einsum("tvij,vj->tvi", g_rot[:, jk], rel) + g_pos[:, jk])
    return out
