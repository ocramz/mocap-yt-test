"""Temporal clean-up of a 2D keypoint track: per frame, which rotation-TTA view to trust and
whether its left/right labels are swapped, so the track stops flipping.

ViTPose with rotation test-time augmentation (:mod:`mocap_yt.person_sam`) keeps, per frame, the
view with the best mean score. On a windmill the views score almost the same, so the kept view
switches from frame to frame and the track flips head<->feet; on back views left and right
swap, for the whole body or for the arms or legs alone. :func:`clean_track` picks one
(view, swap) state per frame by dynamic programming (:func:`viterbi`): a frame pays for a
low-scoring view and for relabelling, and consecutive frames pay for how far the keypoints
jump, in torso lengths.
"""

from __future__ import annotations

import numpy as np

FACE_PAIRS = [(1, 2), (3, 4)]
ARM_PAIRS = [(5, 6), (7, 8), (9, 10)]
LEG_PAIRS = [(11, 12), (13, 14), (15, 16)]
SWAPS = {"none": [], "arms": ARM_PAIRS, "legs": LEG_PAIRS, "all": FACE_PAIRS + ARM_PAIRS + LEG_PAIRS}


def swap_labels(kps: np.ndarray, pairs) -> np.ndarray:
    """Keypoints (..., 17, C) with the left/right ``pairs`` exchanged."""
    out = kps.copy()
    for a, b in pairs:
        out[..., [a, b], :] = kps[..., [b, a], :]
    return out


def viterbi(unary: np.ndarray, pair_cost) -> tuple[np.ndarray, float]:
    """Cheapest path through states: ``unary`` (T, S) costs, ``pair_cost(t)`` -> (S, S) cost of
    going from each state at ``t`` to each state at ``t + 1``. Returns (path (T,), total cost)."""
    T, S = unary.shape
    acc = unary[0].copy()
    back = np.zeros((T, S), int)
    for t in range(1, T):
        tot = acc[:, None] + pair_cost(t - 1)
        back[t] = tot.argmin(0)
        acc = tot[back[t], np.arange(S)] + unary[t]
    path = np.zeros(T, int)
    path[-1] = int(acc.argmin())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path, float(acc.min())


def torso_length(kps: np.ndarray, min_score: float = 0.4) -> float:
    """Median hip-midpoint to shoulder-midpoint distance over frames where all four are
    confident; keypoints (..., 17, 3)."""
    k = kps.reshape(-1, 17, 3)
    ok = (k[:, [5, 6, 11, 12], 2] >= min_score).all(1)
    if not ok.any():
        return float(np.ptp(k[..., :2].reshape(-1, 2), axis=0).max() / 3 or 1.0)
    d = (k[ok, 5, :2] + k[ok, 6, :2]) / 2 - (k[ok, 11, :2] + k[ok, 12, :2]) / 2
    return float(np.median(np.linalg.norm(d, axis=-1)))


def clean_track(views: np.ndarray, swap_cost: float = 0.5, cap: float = 2.0,
                score_floor: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Choose a (view, swap) per frame from ``views`` (T, R, 17, 3): every rotation-TTA view's
    keypoints, mapped back to the frame. Returns (keypoints (T, 17, 3), choice (T, 2): view
    index and index into ``SWAPS``).

    A state's frame cost is the sum over keypoints of ``-log`` score plus ``swap_cost`` per
    swapped group (arms, legs; 2 for all). Between frames the cost is the score-weighted sum
    over keypoints of ``min(d / L, cap)^2``, d their displacement and L the clip's torso
    length: a head<->feet flip moves every keypoint by about two torso lengths, a left/right
    swap moves a limb by its distance to its twin.
    """
    T, R = views.shape[:2]
    names = list(SWAPS)
    states = [(r, s) for r in range(R) for s in range(len(names))]
    kps = np.stack([swap_labels(views[:, r], SWAPS[names[s]]) for r, s in states], 1)  # (T, S, 17, 3)
    nll = -np.log(np.maximum(np.clip(views[..., 2], 0, 1), score_floor)).sum(-1)  # (T, R)
    groups = {"none": 0, "arms": 1, "legs": 1, "all": 2}
    unary = np.stack([nll[:, r] + swap_cost * groups[names[s]] for r, s in states], 1)
    unary[views[..., 2].max((1, 2)) < score_floor] = 0.0  # no pose this frame (person too small): nothing to say
    L = torso_length(views)

    def pair_cost(t):
        a, b = kps[t], kps[t + 1]  # (S, 17, 3)
        d = np.linalg.norm(a[:, None, :, :2] - b[None, :, :, :2], axis=-1) / L  # (S, S, 17)
        w = np.sqrt(np.clip(a[:, None, :, 2], 0, 1) * np.clip(b[None, :, :, 2], 0, 1))
        return (w * np.minimum(d, cap) ** 2).sum(-1)

    path, _ = viterbi(unary, pair_cost)
    choice = np.array([states[p] for p in path])
    return kps[np.arange(T), path], choice
