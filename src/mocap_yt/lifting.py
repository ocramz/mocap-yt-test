"""MotionBERT 2D->3D lifting of a COCO-17 keypoint track (``third_party/MotionBERT``, Apache-2.0).

MotionBERT (DSTformer, fine-tuned on Human3.6M) reads a sequence of Human3.6M-layout 2D joints
with confidences, normalised to [-1, 1] over the clip, and returns root-relative 3D joints in
the same normalised units: x right, y down, z away from the camera. :class:`MotionBertLifter`
converts our COCO-17 track to that layout (:func:`coco_to_h36m`), runs clips of up to 243
frames, and maps the result to the camera frame of :mod:`mocap_yt.geometry`
(X right, Y up, Z toward the camera: ``(x, -y, -z)``, a rotation, not a mirror).

MotionBERT's own inference averages a pass on the mirrored input with the plain one. On an
ambiguous pose those two can disagree in depth and average into a flattened one, so both are
returned separately, as two hypotheses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .upstream import PROJECT_ROOT

MB_DIR = PROJECT_ROOT / "third_party" / "MotionBERT"
MB_REPO = "walterzhu/MotionBERT"
MB_REVISION = "370a9196aa3c89198b134c82476143b01c0fb32c"
MB_CKPT = "checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin"
MAXLEN = 243

# Human3.6M 17-joint layout used by MotionBERT
H36M = ("pelvis", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "spine", "thorax",
        "nose", "head", "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist")
H36M_LEFT, H36M_RIGHT = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]


def coco_to_h36m(k: np.ndarray, head_ext: float = 0.5) -> np.ndarray:
    """COCO-17 (..., 17, 3) -> Human3.6M-17 (..., 17, 3) (x, y, confidence). Pelvis and thorax
    are the hip and shoulder midpoints, the spine their midpoint; the head top is extrapolated
    from the thorax through the face (eye midpoint, else the nose)."""
    k = np.asarray(k, float)
    h = np.zeros(k.shape[:-2] + (17, 3))

    def mid(a, b):
        return np.concatenate([(k[..., a, :2] + k[..., b, :2]) / 2, np.minimum(k[..., a, 2:], k[..., b, 2:])], -1)

    pelvis, thorax = mid(11, 12), mid(5, 6)
    h[..., 0, :] = pelvis
    for dst, src in ((1, 12), (2, 14), (3, 16), (4, 11), (5, 13), (6, 15),
                     (9, 0), (11, 5), (12, 7), (13, 9), (14, 6), (15, 8), (16, 10)):
        h[..., dst, :] = k[..., src, :]
    h[..., 7, :2] = (pelvis[..., :2] + thorax[..., :2]) / 2
    h[..., 7, 2] = np.minimum(pelvis[..., 2], thorax[..., 2])
    h[..., 8, :] = thorax
    eyes = mid(1, 2)
    face = np.where(eyes[..., 2:] >= k[..., 0, 2:], eyes, k[..., 0, :])
    h[..., 10, :2] = face[..., :2] + head_ext * (face[..., :2] - thorax[..., :2])
    h[..., 10, 2] = np.minimum(face[..., 2], thorax[..., 2])
    return h


def normalise(h: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """MotionBERT's ``crop_scale``: x, y of a (T, 17, 3) track mapped to [-1, 1] by the square
    around the confident joints of the whole clip; confidences clipped to [0, 1].
    Returns (normalised track, scale in px per unit, centre in px)."""
    h = np.asarray(h, float).copy()
    h[..., 2] = np.clip(h[..., 2], 0, 1)
    xy = h[h[..., 2] > 0][:, :2]
    if len(xy) < 4:
        return np.zeros_like(h), 1.0, np.zeros(2)
    lo, hi = xy.min(0), xy.max(0)
    scale = float(max(hi - lo)) or 1.0
    centre = (lo + hi) / 2
    h[..., :2] = np.clip((h[..., :2] - centre) / scale * 2, -1, 1)
    return h, scale / 2, centre


def flip_h36m(x: np.ndarray) -> np.ndarray:
    """Mirror a Human3.6M track (x -> -x, left <-> right), MotionBERT's ``flip_data``."""
    out = x.copy()
    out[..., 0] *= -1
    out[..., H36M_LEFT + H36M_RIGHT, :] = out[..., H36M_RIGHT + H36M_LEFT, :]
    return out


def to_camera(x: np.ndarray) -> np.ndarray:
    """MotionBERT axes (x right, y down, z away) -> camera frame (X right, Y up, Z toward)."""
    return x * np.array([1.0, -1.0, -1.0])


class MotionBertLifter:
    def __init__(self, device: str = "cpu"):
        import torch
        import torch.nn as nn
        from functools import partial

        from huggingface_hub import hf_hub_download

        if not (MB_DIR / "lib" / "model" / "DSTformer.py").exists():
            raise RuntimeError(f"MotionBERT not found at {MB_DIR}. Run `git submodule update --init`.")
        if str(MB_DIR) not in sys.path:
            sys.path.insert(0, str(MB_DIR))
        from lib.model.DSTformer import DSTformer

        self.model = DSTformer(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512, depth=5, num_heads=8, mlp_ratio=2,
                               norm_layer=partial(nn.LayerNorm, eps=1e-6), maxlen=MAXLEN, num_joints=17)
        ckpt = torch.load(hf_hub_download(MB_REPO, MB_CKPT, revision=MB_REVISION), map_location="cpu", weights_only=False)
        sd = {k.removeprefix("module."): v for k, v in ckpt["model_pos"].items()}
        self.model.load_state_dict(sd, strict=True)
        self.model.to(device).eval()
        self.device = device

    def _run(self, x: np.ndarray) -> np.ndarray:
        import torch

        out = []
        with torch.no_grad():
            for s in range(0, len(x), MAXLEN):
                chunk = torch.from_numpy(x[s : s + MAXLEN]).float()[None].to(self.device)
                out.append(self.model(chunk)[0].cpu().numpy())
        y = np.concatenate(out)
        return y - y[:, :1]  # root-relative

    def __call__(self, coco: np.ndarray) -> dict[str, np.ndarray]:
        """COCO-17 track (T, 17, 3) -> {"plain", "flip"}: (T, 17, 3) Human3.6M joints in the
        camera frame, root-relative, in units of half the clip's 2D extent."""
        x, _, _ = normalise(coco_to_h36m(coco))
        return {"plain": to_camera(self._run(x)), "flip": to_camera(flip_h36m(self._run(flip_h36m(x))))}
