"""The ``refine`` stage of a sample (``refine: true``): fit the MocapAnything capture to evidence
from the video, after inference (:mod:`mocap_yt.fitting`).

1. The person's 2D track: ViTPose's rotation-TTA views, cleaned of head<->feet flips and
   left/right swaps (:func:`mocap_yt.track2d.clean_track`).
2. Candidate poses per frame: MocapAnything on the crops as seen and turned by 90/180/270
   degrees (:func:`mocap_yt.inference.predict_views`, cached in ``views.npz``), both of its
   heads, optionally MotionBERT's lift (``lift``) and Depth Anything's votes (``depth``).
3. Stage B picks a flip-free sequence; stage C fits rotations and a per-frame camera.
4. Back to MocapAnything's (gravity-up) frame, grounded on the body surface.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .bvh import Motion
from .fitting import FitConfig, SelectConfig, fit_motion, h36m_dict, ik_from_joints, rig_joints_dict, select_hypotheses
from .geometry import from_camera, to_camera, unrotate
from .skeleton import Skeleton, forward_kinematics
from .track2d import SWAPS, clean_track, swap_labels

VERSION = 1


@dataclass(frozen=True)
class RefineConfig:
    """Settings of the refine stage (top-level ``refine:`` block of the config)."""

    views: tuple[int, ...] = (0, 1, 2, 3)  # MocapAnything on the crops turned by 90 * k degrees
    position_head: bool = True  # also candidates from MocapAnything's position head
    motionbert: bool = False  # MotionBERT candidates (no gain on the tutorial clips; off)
    depth: bool = False  # Depth Anything depth-order votes in stage B (off: see README)
    select: dict = field(default_factory=dict)  # SelectConfig overrides
    fit: dict = field(default_factory=dict)  # FitConfig overrides

    def __post_init__(self):
        object.__setattr__(self, "views", tuple(int(k) % 4 for k in self.views))
        if 0 not in self.views:
            raise ValueError("refine.views must include 0 (the crops as seen)")
        SelectConfig(**self.select), FitConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in self.fit.items()})

    def select_cfg(self) -> SelectConfig:
        return SelectConfig(**self.select)

    def fit_cfg(self) -> FitConfig:
        return FitConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in self.fit.items()})


@dataclass
class Refined:
    motion: Motion  # MocapAnything frame, grounded
    camera: np.ndarray  # (T, 3) s, tx, ty of the fit (camera frame): u = tx + s X_c, v = ty - s Y_c
    camera_rot: np.ndarray  # (T, J, 3, 3) the fit's local rotations in the camera frame (for overlays)
    report: dict


def ground_surface(skel: Skeleton, rot: np.ndarray, fps: float, max_under: float = 0.01) -> np.ndarray:
    """Root positions (T, 3) that keep the lowest point of the body surface (joint height minus
    limb radius) on y = 0: the needed height, smoothed over 0.1 s, but never letting the body
    sink more than ``max_under`` metres. No horizontal trajectory."""
    from scipy.ndimage import gaussian_filter1d

    radii = skel.radii if skel.radii is not None else np.zeros(skel.num_joints)
    _, p = forward_kinematics(skel, rot, np.zeros((len(rot), 3)))
    need = -(p[..., 1] - radii).min(1)
    h = np.maximum(gaussian_filter1d(need, max(0.1 * fps, 1e-3), mode="nearest"), need - max_under)
    return np.stack([np.zeros_like(h), h, np.zeros_like(h)], 1)


def load_views(path: Path, tag: str, ks) -> dict | None:
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    if "tag" not in z.files or str(z["tag"]) != tag:
        return None
    return {k: (z[f"rot6d_k{k}"], z[f"position_k{k}"]) for k in ks if f"rot6d_k{k}" in z.files}


def refine_sample(skel: Skeleton, d: Path, frames: list[Path], ok: list[int], pred, fps: float, rig_root: Path, rig: str,
                  rcfg: RefineConfig, poses, model=None, extractor=None, progress: bool = True) -> Refined:
    """Refine one sample's capture. ``pred``: its :class:`~mocap_yt.inference.Prediction`;
    ``poses``: its :class:`~mocap_yt.person_sam.PersonPoses` (all rotation views)."""
    from .inference import position_to_metres, predict_views, rot6d_to_matrix
    from .metrics import motion_metrics, track_metrics

    ok = np.asarray(ok)
    kps_all, choice = clean_track(poses.views) if poses.views is not None else (poses.keypoints, None)
    kps = kps_all[ok]
    report = {"track2d": {"raw": track_metrics(poses.keypoints), "clean": track_metrics(kps_all)}}

    # MocapAnything as seen (the run's own prediction) and on the turned crops
    feats_tag = json.dumps({"frames": ok.tolist(), "features_mtime": (d / "features.npz").stat().st_mtime_ns})
    views = {0: (pred.rot6d, pred.position)}
    turned = [k for k in rcfg.views if k]
    if turned:
        cached = load_views(d / "views.npz", feats_tag, turned) or {}
        missing = [k for k in turned if k not in cached]
        if missing:
            if model is None or extractor is None:
                raise RuntimeError("refine needs the model and feature extractor to predict the turned views")
            crops = [d / "crops" / frames[i].name for i in ok]
            cached.update(predict_views(model, extractor, crops, rig_root, rig, missing))
            arrays = {"tag": np.array(feats_tag)}
            for k, (r, p) in cached.items():
                arrays[f"rot6d_k{k}"], arrays[f"position_k{k}"] = np.asarray(r, np.float32), np.asarray(p, np.float32)
            np.savez_compressed(d / "views.npz", **arrays)
        views.update(cached)
    sources = {}
    for k in rcfg.views:
        r6, p = views[k]
        rot_c, pos_c = unrotate(*to_camera(rot6d_to_matrix(r6), position_to_metres(p, rig_root, rig)), k=k)
        sources[f"ma_k{k}"] = rot_c
        if rcfg.position_head:
            sources[f"ma_pos_k{k}"] = ik_from_joints(skel, rig_joints_dict(skel, pos_c))
    if rcfg.motionbert:
        from .lifting import MotionBertLifter

        lift = MotionBertLifter()(kps_all)
        for name, x in lift.items():
            sources[f"mb_{name}"] = ik_from_joints(skel, h36m_dict(x[ok]))
    votes = None
    if rcfg.depth:
        votes = depth_votes(d, frames, poses, kps_all)[ok]

    sel = select_hypotheses(skel, sources, kps, fps, rcfg.select_cfg(), depth_votes=votes)
    k_fit = np.where(sel.swapped[:, None, None], swap_labels(kps, SWAPS["all"]), kps)
    fit = fit_motion(skel, sel.rot, k_fit, fps, rcfg.fit_cfg(), depth_votes=votes)

    rot = from_camera(fit.rot)
    motion = Motion(skel, rot, ground_surface(skel, rot, fps), fps)
    raw = sources["ma_k0"]
    raw_motion = Motion(skel, from_camera(raw), ground_surface(skel, from_camera(raw), fps), fps)
    picks, counts = np.unique(np.array(sel.names)[sel.choice], return_counts=True)
    report.update({
        "sources": sorted(sources), "picks": dict(zip(picks.tolist(), counts.tolist())),
        "keypoints_read_swapped": int(sel.swapped.sum()), "fit": fit.report,
        "metrics_raw": motion_metrics(raw_motion, skel, fps), "metrics_refined": motion_metrics(motion, skel, fps),
        "settings": {"refine": asdict(rcfg), "version": VERSION},
    })
    if progress:
        a, b = report["metrics_raw"], report["metrics_refined"]
        print(f"[refine] {d.name}: flips {a['flips']} -> {b['flips']}, root speed max {a['root_speed_deg_s'][2]} -> "
              f"{b['root_speed_deg_s'][2]} deg/s, jitter p95 {a['jitter_mm'][1]} -> {b['jitter_mm'][1]} mm, "
              f"floor {a['floor_penetration_cm']} -> {b['floor_penetration_cm']} cm")
    return Refined(motion, fit.camera, fit.rot, report)


def depth_votes(d: Path, frames: list[Path], poses, kps_all: np.ndarray) -> np.ndarray:
    """Depth Anything nearer/farther votes for :data:`mocap_yt.fitting.DEPTH_PAIRS`, every frame
    (depth maps cached in ``depth.npz``)."""
    import cv2

    from .depth import GRID, DepthEstimator, DepthTrack, order_votes, sample
    from .fitting import DEPTH_PAIRS
    from .person_sam import prompt_region

    tag = "v1"
    dt = DepthTrack.load(d / "depth.npz", tag, len(frames))
    if dt is None:
        de = DepthEstimator()
        regions, maps = np.zeros((len(frames), 4)), np.full((len(frames), GRID, GRID), np.nan, np.float32)
        for i, f in enumerate(frames):
            rgb = cv2.cvtColor(cv2.imread(str(f), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            regions[i] = prompt_region(poses.boxes[i], rgb.shape[:2], 1.4, 160)
            maps[i] = de(rgb, regions[i])
        dt = DepthTrack(regions, maps)
        dt.save(d / "depth.npz", tag)
    return order_votes(sample(dt.maps, dt.regions, kps_all[:, :, :2]), DEPTH_PAIRS, kps_all[:, :, 2])


def overlay_points(skel: Skeleton, camera_rot: np.ndarray, camera: np.ndarray) -> np.ndarray:
    """The fitted rig projected onto the frames: (T, J, 2) pixels."""
    from .geometry import project

    _, p = forward_kinematics(skel, camera_rot, np.zeros((len(camera_rot), 3)))
    return project(p, camera[:, :1], camera[:, 1:2], camera[:, 2:3])
