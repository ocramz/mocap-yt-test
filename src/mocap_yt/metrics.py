"""Capture quality without ground truth: flips, spin rate, jitter, floor contact, joint limits,
and the stability of a 2D keypoint track.

Motions are rig local rotations (Y-up, metres; :class:`~mocap_yt.bvh.Motion`). Rates are per
second and smoothing windows are in seconds, so captures at different frame rates compare.
"""

from __future__ import annotations

import numpy as np

from .bvh import Motion
from .skeleton import Skeleton, forward_kinematics

# flexion axis of the hinge joints, in the parent's frame (rest = T-pose facing +Z): knees bend
# the shin backward (+X), elbows bring the forearm forward (-Y on the left arm, +Y on the right)
HINGES = {"LeftLeg": (1.0, 0.0, 0.0), "RightLeg": (1.0, 0.0, 0.0),
          "LeftForeArm": (0.0, -1.0, 0.0), "RightForeArm": (0.0, 1.0, 0.0)}


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees of the rotation taking ``a`` to ``b`` (rotation matrices (..., 3, 3))."""
    r = np.einsum("...ji,...jk->...ik", a, b)
    return np.degrees(np.arccos(np.clip((np.trace(r, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)))


def rotvec(r: np.ndarray) -> np.ndarray:
    """Axis-angle vectors (radians) of rotation matrices (..., 3, 3)."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(r.reshape(-1, 3, 3)).as_rotvec().reshape(r.shape[:-2] + (3,))


def _pct(x, qs=(50, 95, 100)) -> list[float]:
    x = np.asarray(x, float)
    return [round(float(np.percentile(x, q)), 1) for q in qs] if x.size else [0.0] * len(qs)


def root_motion(root: np.ndarray, fps: float) -> dict:
    """Root orientation (T, 3, 3): angular speed (deg/s) and acceleration (deg/s^2)
    percentiles [median, 95th, max]; ``flips``: turns of more than 90 degrees within 1/15 s
    (no body turns that fast); ``blips``: single frames that jump away and back."""
    step = geodesic_deg(root[:-1], root[1:])
    rel = np.einsum("tji,tjk->tik", root[:-1], root[1:])  # body-frame frame-to-frame rotation
    acc = geodesic_deg(rel[:-1], rel[1:]) * fps**2
    k = max(1, round(fps / 15))
    far = geodesic_deg(root[:-k], root[k:]) > 90
    flips = int(np.sum(far[1:] & ~far[:-1]) + (far[0] if len(far) else 0))  # count runs, not frames
    out, back, skip = step[:-1], step[1:], geodesic_deg(root[:-2], root[2:])
    blips = int(np.sum((out > 60) & (back > 60) & (skip < np.minimum(out, back) / 2)))  # away and back
    return {"root_speed_deg_s": _pct(step * fps), "root_acc_deg_s2": _pct(acc), "flips": flips, "blips": blips}


def body_frame(skel: Skeleton, motion: Motion) -> np.ndarray:
    """Joint positions relative to the hips, in the hips' frame (T, J, 3): the pose without
    the global orientation, so a steady spin is not mistaken for motion."""
    g, p = forward_kinematics(skel, motion.local_rot, motion.root_pos)
    return np.einsum("tji,tkj->tki", g[:, 0], p - p[:, :1])


def jitter_mm(q: np.ndarray, fps: float, window_s: float = 0.2) -> list[float]:
    """High-frequency residual of body-frame joints (T, J, 3): distance to a Savitzky-Golay
    fit over ``window_s`` (quadratic), percentiles [median, 95th, max] in mm."""
    from scipy.signal import savgol_filter

    n = len(q)
    w = max(5, int(round(window_s * fps)) | 1)
    if n < w:
        return [0.0, 0.0, 0.0]
    smooth = savgol_filter(q, w, 2, axis=0, mode="interp")
    return _pct(np.linalg.norm(q - smooth, axis=-1).ravel() * 1000)


def floor_contact(skel: Skeleton, motion: Motion) -> dict:
    """Penetration of the body surface (joint height minus limb radius) below y = 0: the
    deepest, in cm, and the share of frames more than 2 cm under."""
    radii = skel.radii if skel.radii is not None else np.zeros(skel.num_joints)
    _, p = forward_kinematics(skel, motion.local_rot, motion.root_pos)
    low = (p[..., 1] - radii).min(1)
    return {"floor_penetration_cm": round(float(max(0.0, -low.min()) * 100), 1),
            "frames_below_2cm": round(float(np.mean(low < -0.02)), 3)}


def hinge_violations(skel: Skeleton, motion: Motion, hyper_deg: float = 10.0) -> dict:
    """Knees and elbows: share of frames bent backward by more than ``hyper_deg``, and the
    median rotation off the hinge axis (deg)."""
    out = {}
    for name, axis in HINGES.items():
        if name not in skel.names:
            continue
        rv = np.degrees(rotvec(motion.local_rot[:, skel.index(name)]))
        flex = rv @ np.asarray(axis)
        off = np.linalg.norm(rv - flex[:, None] * np.asarray(axis), axis=1)
        out[name] = {"hyperextended": round(float(np.mean(flex < -hyper_deg)), 3), "off_hinge_deg": round(float(np.median(off)), 1)}
    return out


def motion_metrics(motion: Motion, skel: Skeleton | None = None, fps: float | None = None) -> dict:
    """All motion measures; ``skel`` (with radii) defaults to the motion's own skeleton."""
    skel = skel or motion.skeleton
    fps = float(fps or motion.fps)
    q = body_frame(skel, motion)
    return {"frames": int(motion.num_frames), "fps": round(fps, 3), **root_motion(motion.local_rot[:, 0], fps),
            "jitter_mm": jitter_mm(q, fps), **floor_contact(skel, motion), "hinges": hinge_violations(skel, motion)}


# ------------------------------------------------------------------------------ 2D tracks
def torso_2d(kps: np.ndarray, min_score: float = 0.4):
    """COCO-17 keypoints (T, 17, 3) -> torso direction (T, 2) hip->shoulder midpoints in image
    coordinates, shoulder vector (T, 2) right->left, and a validity mask (all four confident)."""
    ok = (kps[:, [5, 6, 11, 12], 2] >= min_score).all(1)
    sh = (kps[:, 5, :2] + kps[:, 6, :2]) / 2
    hip = (kps[:, 11, :2] + kps[:, 12, :2]) / 2
    return sh - hip, kps[:, 5, :2] - kps[:, 6, :2], ok


def track_metrics(kps: np.ndarray, min_score: float = 0.4) -> dict:
    """Stability of a 2D keypoint track: between consecutive frames with a confident torso,
    how often the torso direction turns by more than 90 degrees (head<->feet flips) and how
    often the left/right labels swap (the shoulder line reverses while the shoulders are well
    apart in both frames)."""
    d, lr, ok = torso_2d(kps, min_score)
    pair = ok[:-1] & ok[1:]
    cos = np.einsum("ti,ti->t", d[:-1], d[1:]) / (np.linalg.norm(d[:-1], axis=1) * np.linalg.norm(d[1:], axis=1) + 1e-9)
    width = np.linalg.norm(lr, axis=1) / (np.linalg.norm(d, axis=1) + 1e-9)
    wide = pair & (width[:-1] > 0.3) & (width[1:] > 0.3)
    swap = np.einsum("ti,ti->t", lr[:-1], lr[1:]) < 0
    return {"pairs": int(pair.sum()), "torso_flips": int(np.sum(pair & (cos < 0))),
            "lr_swaps": int(np.sum(wide & swap)), "lr_pairs": int(wide.sum())}
