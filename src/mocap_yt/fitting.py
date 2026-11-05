"""Fit the captured rig motion to evidence from the video.

**Stage B** (:func:`select_hypotheses`) removes the flips. Each frame gets candidate poses:

* MocapAnything's rotation head on the crop as seen and turned by 90/180/270 degrees
  (:func:`mocap_yt.inference.predict_views`);
* its position head, turned into rotations by :func:`ik_from_joints`;
* MotionBERT's lift of the cleaned 2D track, likewise.

Each candidate also comes as its depth twin, spun about its spine and tipped over
(:data:`mocap_yt.geometry.VARIANTS`).

* **Frame cost:** how well the pose projects onto the cleaned 2D keypoints after a similarity
  fit, its in-plane turn charged; how often it contradicts the monocular depth order of joint
  pairs; and a small price for not being MocapAnything's own answer.
* **Transition cost:** how far the root turns and the body-frame pose moves between frames.

A Viterbi pass picks the cheapest sequence.

All poses are local rotations of the humanoid rig in the camera frame of :mod:`mocap_yt.geometry`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .geometry import VARIANTS, procrustes_2d, rz
from .skeleton import Skeleton, forward_kinematics
from .track2d import SWAPS, swap_labels, viterbi

# COCO-17 keypoint -> rig joint (humanoid, no fingers) for the body keypoints
COCO_TO_RIG = {5: 8, 6: 12, 7: 9, 8: 13, 9: 10, 10: 14, 11: 15, 12: 20, 13: 16, 14: 21, 15: 17, 16: 22}
BODY_KPS = tuple(COCO_TO_RIG)
# joint pairs whose depth order the monocular depth can referee (COCO indices): limb bones
# and the left/right shoulders and hips
DEPTH_PAIRS = ((11, 13), (13, 15), (12, 14), (14, 16), (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12))


@dataclass(frozen=True)
class SelectConfig:
    roll_deg: float = 45.0  # in-plane turn costing 1
    lr_swap: float = 1.0  # cost of matching a pose to the 2D keypoints with left/right swapped
    depth: float = 0.5  # cost per contradicted depth vote
    alt_source: float = 0.3  # price of a candidate that is not MocapAnything's rotation head as seen
    variant: float = 0.5  # price of a twin / spin / tilt
    turn_deg: float = 45.0  # root turn between frames (at 15 fps) costing 1
    turn_cap: float = 12.0
    pose_m: float = 0.15  # body-frame joint displacement between frames costing 1
    pose_cap: float = 6.0

    def tag(self) -> str:
        import json

        return json.dumps(asdict(self), sort_keys=True)


# --------------------------------------------------------------------------------- IK
def _min_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotations (..., 3, 3) taking unit vectors ``a`` to ``b`` (..., 3)."""
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
    v = np.cross(a, b)
    c = np.einsum("...i,...i->...", a, b)
    k = np.zeros(v.shape[:-1] + (3, 3))
    k[..., 0, 1], k[..., 0, 2], k[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    k = k - np.swapaxes(k, -1, -2)
    r = np.eye(3) + k + k @ k / np.maximum(1 + c, 1e-9)[..., None, None]
    # antiparallel: rotate 180 degrees about any axis orthogonal to a
    anti = c < -1 + 1e-9
    if np.any(anti):
        ortho = np.cross(a[anti], [1.0, 0.0, 0.0])
        ortho = np.where(np.linalg.norm(ortho, axis=-1, keepdims=True) < 1e-6, np.cross(a[anti], [0.0, 1.0, 0.0]), ortho)
        ortho /= np.linalg.norm(ortho, axis=-1, keepdims=True)
        r[anti] = 2 * ortho[..., :, None] * ortho[..., None, :] - np.eye(3)
    return r


def _frame(x: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Rotation whose columns are the orthonormalised (x, up, x cross up) axes (..., 3)."""
    y = up / np.maximum(np.linalg.norm(up, axis=-1, keepdims=True), 1e-9)
    x = x - np.einsum("...i,...i->...", x, y)[..., None] * y
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
    return np.stack([x, y, np.cross(x, y)], -1)


def ik_from_joints(skel: Skeleton, j: dict[str, np.ndarray]) -> np.ndarray:
    """Local rotations (T, J, 3, 3) of the rig whose bones point along those of 3D joints ``j``
    (name -> (T, 3); names as in :data:`mocap_yt.lifting.H36M`, camera frame).

    Every frame and bone is matched against the same construction on the rig's own rest pose,
    so the rig's joints are reproduced exactly. The hips and chest frames come from the hip and
    shoulder lines and the pelvis->thorax direction; the bend between them is shared equally by
    Spine, Spine1 and Spine2 (the pelvis pitch is otherwise not observable). Neck aims the head
    top from the thorax, the limbs aim their bones with the smallest rotation (no twist). The
    clavicles, Head, hands, feet and toes keep the rest pose relative to their parents."""
    from scipy.spatial.transform import Rotation, Slerp

    T, J = len(j["pelvis"]), skel.num_joints
    idx = {n.split(":")[-1]: i for i, n in enumerate(skel.names)}
    _, rest_p = forward_kinematics(skel, np.tile(np.eye(3), (1, J, 1, 1)))
    r = {k: v[0] for k, v in rig_joints_dict(skel, rest_p).items()}

    def frame(d, a, b, c, e):
        return _frame(d[a] - d[b], d[c] - d[e])

    hips = frame(j, "l_hip", "r_hip", "thorax", "pelvis") @ frame(r, "l_hip", "r_hip", "thorax", "pelvis").T
    chest = frame(j, "l_shoulder", "r_shoulder", "thorax", "pelvis") @ frame(r, "l_shoulder", "r_shoulder", "thorax", "pelvis").T
    loc = np.tile(np.eye(3), (T, J, 1, 1))
    loc[:, 0] = hips
    rel = np.einsum("tji,tjk->tik", hips, chest)  # hips -> chest
    third = np.stack([Slerp([0, 1], Rotation.from_matrix([np.eye(3), m]))([1 / 3]).as_matrix()[0] for m in rel])
    for name in ("Spine", "Spine1", "Spine2"):
        loc[:, idx[name]] = third
    aims = {"Neck": ("thorax", "head"), "LeftArm": ("l_shoulder", "l_elbow"), "LeftForeArm": ("l_elbow", "l_wrist"),
            "RightArm": ("r_shoulder", "r_elbow"), "RightForeArm": ("r_elbow", "r_wrist"),
            "LeftUpLeg": ("l_hip", "l_knee"), "LeftLeg": ("l_knee", "l_ankle"),
            "RightUpLeg": ("r_hip", "r_knee"), "RightLeg": ("r_knee", "r_ankle")}
    glob = np.zeros((T, J, 3, 3))
    glob[:, 0] = hips
    for jj in range(1, J):
        p = skel.parents[jj]
        g = glob[:, p] @ loc[:, jj]
        name = skel.names[jj].split(":")[-1]
        if name in aims:
            a, b = aims[name]
            g = _min_rotation(g @ (r[b] - r[a]), j[b] - j[a]) @ g  # rest-pose vector a->b, as carried by g
            loc[:, jj] = np.einsum("tji,tjk->tik", glob[:, p], g)
        glob[:, jj] = g
    return loc


def h36m_dict(x: np.ndarray) -> dict[str, np.ndarray]:
    from .lifting import H36M

    return {n: x[:, i] for i, n in enumerate(H36M)}


def rig_joints_dict(skel: Skeleton, pos: np.ndarray) -> dict[str, np.ndarray]:
    """Rig joint positions (T, J, 3) as the H36M-named dict :func:`ik_from_joints` takes."""
    n = skel.index
    mid = lambda a, b: (pos[:, n(a)] + pos[:, n(b)]) / 2
    return {"pelvis": pos[:, n("Hips")], "thorax": mid("LeftArm", "RightArm"), "head": pos[:, n("HeadTop_End")],
            "l_hip": pos[:, n("LeftUpLeg")], "l_knee": pos[:, n("LeftLeg")], "l_ankle": pos[:, n("LeftFoot")],
            "r_hip": pos[:, n("RightUpLeg")], "r_knee": pos[:, n("RightLeg")], "r_ankle": pos[:, n("RightFoot")],
            "l_shoulder": pos[:, n("LeftArm")], "l_elbow": pos[:, n("LeftForeArm")], "l_wrist": pos[:, n("LeftHand")],
            "r_shoulder": pos[:, n("RightArm")], "r_elbow": pos[:, n("RightForeArm")], "r_wrist": pos[:, n("RightHand")]}


# ---------------------------------------------------------------------------- stage B
@dataclass
class Candidates:
    rot: np.ndarray  # (T, S, J, 3, 3) local rotations, camera frame
    pos: np.ndarray  # (T, S, J, 3) root-relative joint positions (FK), camera frame, metres
    names: list[str]  # (S,) "source/variant"
    prior: np.ndarray  # (S,) price of each candidate before any evidence


def build_candidates(skel: Skeleton, sources: dict[str, np.ndarray], cfg: SelectConfig, main: str = "ma_k0") -> Candidates:
    """Every source (name -> local rotations (T, J, 3, 3), camera frame) in every variant."""
    rots, names, prior = [], [], []
    for src, rot in sources.items():
        for var, fn in VARIANTS.items():
            rots.append(fn(rot))
            names.append(f"{src}/{var}")
            prior.append((0.0 if src == main else cfg.alt_source) + (0.0 if var == "as_is" else cfg.variant))
    rot = np.stack(rots, 1)
    T, S = rot.shape[:2]
    _, pos = forward_kinematics(skel, rot.reshape(T * S, *rot.shape[2:]), np.zeros((T * S, 3)))
    return Candidates(rot, pos.reshape(T, S, -1, 3), names, np.array(prior))


COCO_MIRROR = {a: b for x, y in SWAPS["all"] for a, b in ((x, y), (y, x))}


def frame_costs(c: Candidates, kps: np.ndarray, cfg: SelectConfig, torso_px: float,
                depth_votes: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evidence cost of every candidate in every frame.

    ``kps`` (T, 17, 3): the cleaned COCO keypoints; ``torso_px``: the clip's torso length in
    pixels; ``depth_votes`` (T, len(DEPTH_PAIRS)): +1 where the first keypoint of a pair is
    clearly nearer than the second, -1 farther, 0 unclear (:func:`mocap_yt.depth.order_votes`).

    The 2D part is the weighted RMS residual (in torso lengths, squared) of a similarity fit of
    the candidate's projected body joints onto the keypoints, plus its in-plane turn
    ``(angle / roll_deg)^2``; the keypoints' labels may be read swapped at ``lr_swap``. The
    depth part costs ``depth`` per vote the candidate's joints contradict.

    Returns (cost (T, S), fit (T, S, 4): scale, angle (deg), tx, ty of the similarity in a y-up
    image frame, swapped (T, S): whether the swapped labels fitted better)."""
    T, S = c.rot.shape[:2]
    rig_idx = [COCO_TO_RIG[k] for k in BODY_KPS]
    cost = np.tile(c.prior, (T, 1)).astype(float)
    fits = np.zeros((T, S, 4))
    fits[..., 0] = 1.0
    swapped = np.zeros((T, S), bool)
    labelings = (kps, swap_labels(kps, SWAPS["all"]))
    for t in range(T):
        best = np.full(S, np.inf)
        for li, lab in enumerate(labelings):
            w = np.clip(lab[t, list(BODY_KPS), 2], 0, 1)
            w = np.where(w >= 0.3, w, 0.0)
            if w.sum() < 1.5:
                continue
            dst = lab[t, list(BODY_KPS), :2] * [1.0, -1.0]  # y up
            for s in range(S):
                sc, ang, tr, res = procrustes_2d(c.pos[t, s, rig_idx, :2], dst, w)
                e = (res / torso_px) ** 2 + (ang / cfg.roll_deg) ** 2 + (cfg.lr_swap if li else 0.0)
                if e < best[s]:
                    best[s], fits[t, s], swapped[t, s] = e, (sc, ang, tr[0], tr[1]), bool(li)
        cost[t] += np.where(np.isfinite(best), best, 0.0)
    if depth_votes is not None:
        z = c.pos[..., 2]  # (T, S, J): toward the camera
        for k, (a, b) in enumerate(DEPTH_PAIRS):
            v = depth_votes[:, k][:, None]
            for li in (0, 1):
                aa, bb = (a, b) if li == 0 else (COCO_MIRROR[a], COCO_MIRROR[b])
                dz = z[:, :, COCO_TO_RIG[aa]] - z[:, :, COCO_TO_RIG[bb]]
                wrong = np.where(np.abs(dz) < 0.01, 0.5, (np.sign(dz) != v).astype(float)) * (v != 0)
                cost += cfg.depth * wrong * (swapped == bool(li))
    return cost, fits, swapped


def transition_costs(c: Candidates, cfg: SelectConfig, fps: float):
    """``pair_cost(t)`` -> (S, S) cost of going from each candidate at t to each at t + 1: the
    root turn ``min((angle / turn)^2, turn_cap)`` with ``turn`` = turn_deg at 15 fps scaled to
    the frame rate, plus ``min((d / pose_m)^2, pose_cap)`` for the mean body-frame joint
    displacement d."""
    turn = cfg.turn_deg * 15.0 / fps
    root = c.rot[:, :, 0]  # (T, S, 3, 3)
    body = np.einsum("tsji,tskj->tski", root, c.pos)  # root-frame joint positions (T, S, J, 3)

    def pair_cost(t):
        rel = np.einsum("aji,bjk->abik", root[t], root[t + 1])
        ang = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
        d = np.linalg.norm(body[t][:, None] - body[t + 1][None], axis=-1).mean(-1)
        return np.minimum((ang / turn) ** 2, cfg.turn_cap) + np.minimum((d / cfg.pose_m) ** 2, cfg.pose_cap)

    return pair_cost


@dataclass
class Selection:
    rot: np.ndarray  # (T, J, 3, 3) chosen local rotations, camera frame, turned in-plane onto the keypoints
    choice: np.ndarray  # (T,) index into names
    names: list[str]
    fit: np.ndarray  # (T, 4) similarity of the chosen candidate onto the keypoints
    swapped: np.ndarray  # (T,) whether the keypoints' labels were read swapped
    cost: float


def select_hypotheses(skel: Skeleton, sources: dict[str, np.ndarray], kps: np.ndarray, fps: float,
                      cfg: SelectConfig = SelectConfig(), depth_votes: np.ndarray | None = None,
                      torso_px: float | None = None) -> Selection:
    """Stage B: the cheapest sequence of candidates (module doc)."""
    from .track2d import torso_length

    c = build_candidates(skel, sources, cfg)
    cost, fits, swapped = frame_costs(c, kps, cfg, torso_px or torso_length(kps), depth_votes)
    path, total = viterbi(cost, transition_costs(c, cfg, fps))
    T = len(path)
    rot = c.rot[np.arange(T), path].copy()
    fit = fits[np.arange(T), path]
    support = (kps[:, list(BODY_KPS), 2] >= 0.3).sum(1)
    for t in range(T):  # turn each pose by the in-plane angle its 2D fit needed (about +Z_c), where the fit is trustworthy
        if support[t] >= 5:
            rot[t, 0] = rz(float(np.clip(fit[t, 1], -60, 60))) @ rot[t, 0]
    return Selection(rot, path, c.names, fit, swapped[np.arange(T), path], total)


# ---------------------------------------------------------------------------- stage C
@dataclass(frozen=True)
class FitConfig:
    iters: tuple[int, int, int] = (60, 120, 300)  # camera only, + root, + everything
    lr: float = 0.03
    robust: tuple[float, float, float] = (0.5, 0.5, 0.5)  # Geman-McClure scale per stage, in torso lengths
    prior_deg: float = 90.0  # joint rotation away from the stage-B pose costing 1
    root_prior_deg: float = 60.0
    smooth: float = 10.0  # angular acceleration of every joint (at 15 fps; scaled to the frame rate)
    cam_smooth: float = 10.0
    hinge: float = 10.0  # knee/elbow bent the wrong way or off their hinge
    spine: float = 1.0  # unequal bends of Spine, Spine1, Spine2
    depth: float = 0.0  # monocular depth-order hinge (off by default)
    smooth_anchor_s: float = 0.1  # the stage-B pose is smoothed over this many seconds first

    def tag(self) -> str:
        import json

        return json.dumps(asdict(self), sort_keys=True)


def smooth_rotations(rot: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Gaussian smoothing of rotations over time (T, ..., 3, 3) via sign-aligned quaternions."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation

    if sigma_frames <= 0:
        return rot.copy()
    shape = rot.shape
    q = Rotation.from_matrix(rot.reshape(-1, 3, 3)).as_quat().reshape(shape[:-2] + (4,))
    for t in range(1, len(q)):  # keep each quaternion in the hemisphere of the previous one
        flip = np.einsum("...i,...i->...", q[t], q[t - 1]) < 0
        q[t][flip] *= -1
    q = gaussian_filter1d(q, sigma_frames, axis=0, mode="nearest")
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(shape)


def _axis_angle_to_matrix(w):
    import torch

    theta = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-8)
    k = w / theta
    K = torch.zeros(w.shape[:-1] + (3, 3), dtype=w.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - K.transpose(-1, -2)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return torch.eye(3, dtype=w.dtype) + s * K + (1 - c) * K @ K


def fk_torch(parents, offsets, rot):
    """Differentiable FK: local rotations (T, J, 3, 3) -> (global rotations, positions (T, J, 3))
    with the root at the origin."""
    import torch

    g, p = [rot[:, 0]], [torch.zeros(rot.shape[0], 3, dtype=rot.dtype)]
    for j in range(1, rot.shape[1]):
        pj = int(parents[j])
        p.append(p[pj] + torch.einsum("tij,j->ti", g[pj], offsets[j]))
        g.append(g[pj] @ rot[:, j])
    return torch.stack(g, 1), torch.stack(p, 1)


@dataclass
class Fit:
    rot: np.ndarray  # (T, J, 3, 3) local rotations, camera frame
    camera: np.ndarray  # (T, 3): s, tx, ty -- u = tx + s X_c, v = ty - s Y_c
    report: dict


def fit_motion(skel: Skeleton, init_rot: np.ndarray, kps: np.ndarray, fps: float, cfg: FitConfig = FitConfig(),
               depth_votes: np.ndarray | None = None, torso_px: float | None = None) -> Fit:
    """Stage C: refine local rotations (camera frame) and a per-frame weak-perspective camera
    against the 2D keypoints ``kps`` (T, 17, 3), labels as stage B read them."""
    import torch

    from .metrics import HINGES
    from .track2d import torso_length

    torch.manual_seed(0)
    T, J = init_rot.shape[:2]
    L = float(torso_px or torso_length(kps))
    anchor = smooth_rotations(init_rot, cfg.smooth_anchor_s * fps)
    dt = torch.float64
    A = torch.tensor(anchor, dtype=dt)
    parents, offsets = skel.parents, torch.tensor(skel.offsets, dtype=dt)
    kp_idx, rig_idx = list(BODY_KPS), [COCO_TO_RIG[k] for k in BODY_KPS]
    xy = torch.tensor(kps[:, kp_idx, :2], dtype=dt)
    w = torch.tensor(np.where(kps[:, kp_idx, 2] >= 0.3, np.clip(kps[:, kp_idx, 2], 0, 1), 0.0), dtype=dt)
    centre = xy[w > 0].mean(0) if (w > 0).any() else torch.zeros(2, dtype=dt)
    xy_n = (xy - centre) / L  # image coordinates in torso lengths, about the clip's centre

    # camera init: scale/translation of the anchor's projection onto the keypoints (no rotation)
    _, p0 = fk_torch(parents, offsets, A)
    src = p0[:, rig_idx, :2] * torch.tensor([1.0, -1.0], dtype=dt)  # image orientation (v down)
    ws = w.sum(1, keepdim=True).clamp_min(1e-9)
    ms, md = (w[..., None] * src).sum(1) / ws, (w[..., None] * xy_n).sum(1) / ws
    var = (w * ((src - ms[:, None]) ** 2).sum(-1)).sum(1).clamp_min(1e-9)
    cov = (w * ((src - ms[:, None]) * (xy_n - md[:, None])).sum(-1)).sum(1)
    s0 = (cov / var).clamp_min(1e-3)
    s0 = torch.where(w.sum(1) > 1.5, s0, s0[w.sum(1) > 1.5].median() if (w.sum(1) > 1.5).any() else torch.ones(()))
    t0 = md - s0[:, None] * ms

    log_s = torch.log(s0).clone().requires_grad_(True)
    tr = t0.clone().requires_grad_(True)
    w_root = torch.zeros(T, 3, dtype=dt, requires_grad=True)
    w_j = torch.zeros(T, J, 3, dtype=dt, requires_grad=True)
    hinge = [(skel.index(n), torch.tensor(a, dtype=dt), torch.tensor(skel.offsets[[c for c in range(J) if skel.parents[c] == skel.index(n)][0]], dtype=dt))
             for n, a in HINGES.items() if n in skel.names]
    spine = [skel.index(n) for n in ("Spine", "Spine1", "Spine2") if n in skel.names]
    rate = (fps / 15.0) ** 4  # mean squared second differences shrink with dt^4

    def pose():
        res = _axis_angle_to_matrix(w_j)
        res = torch.cat([_axis_angle_to_matrix(w_root)[:, None], res[:, 1:]], 1)
        return A @ res

    def losses(rot, robust):
        _, p = fk_torch(parents, offsets, rot)
        proj = p[:, rig_idx, :2] * torch.tensor([1.0, -1.0], dtype=dt) * torch.exp(log_s)[:, None, None] + tr[:, None]
        r2 = ((proj - xy_n) ** 2).sum(-1)
        out = {"2d": (w * r2 / (r2 + robust**2)).sum() / max(T, 1)}
        out["prior"] = ((w_j[:, 1:] ** 2).sum() / np.radians(cfg.prior_deg) ** 2 + (w_root**2).sum() / np.radians(cfg.root_prior_deg) ** 2) / T
        if T >= 3:
            # angular acceleration: change of the frame-to-frame relative rotation (zero for a
            # steady spin, unlike second differences of the matrices themselves)
            rel = rot[:-1].transpose(-1, -2) @ rot[1:]
            out["smooth"] = cfg.smooth * rate * ((rel[1:] - rel[:-1]) ** 2).sum() / T
            cam = torch.cat([log_s[:, None], tr], 1)
            out["cam"] = cfg.cam_smooth * rate * ((cam[2:] - 2 * cam[1:-1] + cam[:-2]) ** 2).sum() / T
        hv = 0.0
        for j, axis, child in hinge:
            d = torch.einsum("tij,j->ti", rot[:, j], child / child.norm())  # child bone in the parent's frame
            rest = child / child.norm()
            fwd = torch.cross(axis, rest, dim=0)  # direction a positive flexion moves the bone
            hv = hv + torch.relu(-(d @ fwd) - 0.05).pow(2).sum() + ((d @ axis) ** 2).sum()
        out["hinge"] = cfg.hinge * hv / T
        if len(spine) == 3:
            out["spine"] = cfg.spine * (((rot[:, spine[0]] - rot[:, spine[1]]) ** 2).sum() + ((rot[:, spine[1]] - rot[:, spine[2]]) ** 2).sum()) / T
        if cfg.depth > 0 and depth_votes is not None:
            v = torch.tensor(depth_votes, dtype=dt)
            dz = torch.stack([p[:, COCO_TO_RIG[a], 2] - p[:, COCO_TO_RIG[b], 2] for a, b in DEPTH_PAIRS], 1)
            out["depth"] = cfg.depth * (torch.relu(0.02 - v * dz) * (v != 0)).pow(2).sum() / (0.02**2) / T
        return out

    report = {}
    stages = ([log_s, tr], [log_s, tr, w_root], [log_s, tr, w_root, w_j])
    for stage, (n, params, robust) in enumerate(zip(cfg.iters, stages, cfg.robust)):
        opt = torch.optim.Adam(params, lr=cfg.lr)
        for _ in range(n):
            opt.zero_grad()
            ls = losses(pose(), robust)
            total = sum(ls.values())
            total.backward()
            opt.step()
        report[f"stage{stage + 1}"] = {k: round(float(v.detach()), 4) for k, v in ls.items()}
    with torch.no_grad():
        rot = pose().numpy()
        s = torch.exp(log_s).numpy() * L  # pixels per metre
        cam = np.stack([s, (tr[:, 0] * L + centre[0]).numpy(), (tr[:, 1] * L + centre[1]).numpy()], 1)
    return Fit(rot, cam, report)
