"""Frames, projection and pose variants shared by the refinement stages.

**Camera frame.** MocapAnything predicts in the frame of its training camera: X image-right,
Y up, Z toward the camera, gravity-aligned, with the camera pitched ``PITCH_DEG`` down. The
camera frame used here undoes that pitch: ``p_c = Rx(PITCH_DEG) p``, ``R0_c = Rx(PITCH_DEG) R0``
for the root rotation (non-root local rotations are frame-free). In the camera frame a point
projects orthographically (weak perspective) as ``u = tx + s X_c``, ``v = ty - s Y_c`` (v down).

**Pose variants** (``rot`` local rotations (..., J, 3, 3) of the rig in the camera frame):

* :func:`twin` -- the depth-mirrored pose ``Z_c -> -Z_c``. It projects exactly where the pose
  does, with left and right exchanged: ``L'_j = Mx L_sigma(j) Mx``, ``R0' = Mz R0 Mx``
  (exact because the rig is mirror-symmetric).
* :func:`spin` -- turned 180 degrees about its own spine: ``R0 Ry(180)``.
* :func:`tilt` -- tipped over in depth: ``Mz R0 Mz``.

**rot90 views.** ``np.rot90(image, k)`` turns the picture 90*k degrees counter-clockwise, which
rotates the scene by ``Rz(90 k)`` about +Z_c; a prediction made on it is brought back with
:func:`unrotate` (``Rz(-90 k)``).
"""

from __future__ import annotations

import numpy as np

PITCH_DEG = 5.3  # MocapAnything's training camera looks this far down
MIRROR_PAIRS = ((7, 11), (8, 12), (9, 13), (10, 14), (15, 20), (16, 21), (17, 22), (18, 23), (19, 24))
MX = np.diag([-1.0, 1.0, 1.0])
MZ = np.diag([1.0, 1.0, -1.0])


def rx(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def ry(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def rz(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def mirror_perm(J: int) -> np.ndarray:
    """Joint permutation exchanging the left/right pairs of the humanoid rig (no fingers)."""
    perm = np.arange(J)
    for a, b in MIRROR_PAIRS:
        if b < J:
            perm[a], perm[b] = b, a
    return perm


def _with_root(rot: np.ndarray, root: np.ndarray) -> np.ndarray:
    out = rot.copy()
    out[..., 0, :, :] = root
    return out


def to_camera(rot: np.ndarray, pos: np.ndarray | None = None, pitch: float = PITCH_DEG):
    """MocapAnything frame -> camera frame: root rotation (and positions (..., 3)) by Rx(pitch)."""
    r = rx(pitch)
    out = _with_root(rot, r @ rot[..., 0, :, :])
    return out if pos is None else (out, pos @ r.T)


def from_camera(rot: np.ndarray, pos: np.ndarray | None = None, pitch: float = PITCH_DEG):
    """Inverse of :func:`to_camera`."""
    return to_camera(rot, pos, -pitch)


def unrotate(rot: np.ndarray, pos: np.ndarray | None = None, k: int = 0):
    """Bring a camera-frame prediction made on ``np.rot90(image, k)`` back to the original
    view: root rotation (and positions) by ``Rz(-90 k)``."""
    r = rz(-90.0 * k)
    out = _with_root(rot, r @ rot[..., 0, :, :])
    return out if pos is None else (out, pos @ r.T)


def twin(rot: np.ndarray) -> np.ndarray:
    """Depth-mirrored pose (camera frame): same projection, left/right exchanged."""
    perm = mirror_perm(rot.shape[-3])
    out = MX @ rot[..., perm, :, :] @ MX
    out[..., 0, :, :] = MZ @ rot[..., 0, :, :] @ MX
    return out


def spin(rot: np.ndarray) -> np.ndarray:
    """The pose turned 180 degrees about its own spine axis (root ``R0 Ry(180)``)."""
    return _with_root(rot, rot[..., 0, :, :] @ ry(180.0))


def tilt(rot: np.ndarray) -> np.ndarray:
    """The pose tipped over in depth (root ``Mz R0 Mz``)."""
    return _with_root(rot, MZ @ rot[..., 0, :, :] @ MZ)


VARIANTS = {"as_is": lambda r: r, "twin": twin, "spin": spin, "tilt": tilt}


def project(points: np.ndarray, s, tx, ty) -> np.ndarray:
    """Weak perspective of camera-frame points (..., 3) -> pixels (..., 2), v down."""
    return np.stack([tx + s * points[..., 0], ty - s * points[..., 1]], -1)


def procrustes_2d(src: np.ndarray, dst: np.ndarray, w: np.ndarray | None = None):
    """Similarity (scale s > 0, in-plane rotation, translation) best mapping 2D ``src`` (N, 2)
    onto ``dst`` (N, 2) with weights ``w``. Returns (s, angle_deg, t (2,), weighted RMS residual).
    The angle is counter-clockwise in a y-up frame."""
    w = np.ones(len(src)) if w is None else np.asarray(w, float)
    ws = w.sum()
    if ws <= 0 or len(src) < 2:
        return 1.0, 0.0, np.zeros(2), np.inf
    ms, md = (w[:, None] * src).sum(0) / ws, (w[:, None] * dst).sum(0) / ws
    a, b = src - ms, dst - md
    cov = (w[:, None, None] * (b[:, :, None] * a[:, None, :])).sum(0)  # sum w b a^T
    u, sv, vt = np.linalg.svd(cov)
    d = np.diag([1.0, np.sign(np.linalg.det(u @ vt)) or 1.0])
    rot = u @ d @ vt
    var = (w * (a**2).sum(1)).sum()
    s = float((sv * np.diag(d)).sum() / max(var, 1e-12))
    t = md - s * ms @ rot.T
    res = dst - (s * src @ rot.T + t)
    return s, float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))), t, float(np.sqrt((w * (res**2).sum(1)).sum() / ws))
