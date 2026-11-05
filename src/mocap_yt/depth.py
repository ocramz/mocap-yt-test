"""Monocular depth of the person (Depth Anything V2 Small, Apache-2.0, via transformers).

The model gives relative inverse depth (larger = nearer) up to an unknown scale and offset, so
only orderings within one frame mean anything: which of two joints is nearer the camera. That
is the sign the pose's depth twin gets wrong (:func:`mocap_yt.geometry.twin`), so it can
referee between hypotheses. :class:`DepthEstimator` runs on a square region around the person
(enlarged, like the SAM region) and keeps a small depth map per frame; :func:`sample` reads it
at image points, :func:`order_votes` turns it into nearer/farther votes for joint pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEPTH_REPO = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
GRID = 128  # stored depth maps are GRID x GRID over the region


@dataclass
class DepthTrack:
    regions: np.ndarray  # (N, 4) x0, y0, x1, y1 of each frame's depth region
    maps: np.ndarray  # (N, GRID, GRID) relative inverse depth (larger = nearer), NaN where none

    def save(self, path: Path, tag: str):
        np.savez_compressed(path, regions=self.regions.astype(np.float32), maps=self.maps.astype(np.float16), tag=np.array(tag))

    @classmethod
    def load(cls, path: Path, tag: str, n: int) -> "DepthTrack | None":
        if not Path(path).exists():
            return None
        z = np.load(path, allow_pickle=False)
        if str(z["tag"]) != tag or len(z["regions"]) != n:
            return None
        return cls(z["regions"], z["maps"].astype(np.float32))


def sample(maps: np.ndarray, regions: np.ndarray, points: np.ndarray, radius: int = 1) -> np.ndarray:
    """Depth at image ``points`` (N, P, 2) of each frame: the median over a small window of the
    stored map (NaN outside the region)."""
    n, p = points.shape[:2]
    out = np.full((n, p), np.nan)
    for i in range(n):
        x0, y0, x1, y1 = regions[i]
        if not np.isfinite(maps[i]).any() or x1 <= x0:
            continue
        g = (points[i] - [x0, y0]) / [x1 - x0, y1 - y0] * GRID
        for j, (gx, gy) in enumerate(g):
            if 0 <= gx < GRID and 0 <= gy < GRID:
                cx, cy = int(gx), int(gy)
                win = maps[i, max(0, cy - radius) : cy + radius + 1, max(0, cx - radius) : cx + radius + 1]
                out[i, j] = np.nanmedian(win) if np.isfinite(win).any() else np.nan
    return out


def order_votes(d: np.ndarray, pairs, conf: np.ndarray | None = None, min_gap: float = 0.15) -> np.ndarray:
    """Nearer/farther votes for joint pairs (a, b) from sampled depths ``d`` (N, P): +1 when a
    is clearly nearer than b, -1 when clearly farther, 0 when unclear. "Clearly" means a gap
    of at least ``min_gap`` times the frame's robust spread of depths (inter-quartile range of
    the sampled points). Pairs with an unsure keypoint (``conf`` < 0.4) don't vote."""
    d = np.asarray(d, float)
    q75, q25 = np.nanpercentile(d, 75, axis=1), np.nanpercentile(d, 25, axis=1)
    spread = np.maximum(q75 - q25, 1e-6)
    v = np.zeros((len(d), len(pairs)))
    for k, (a, b) in enumerate(pairs):
        gap = (d[:, a] - d[:, b]) / spread
        ok = np.isfinite(gap) & (np.abs(gap) >= min_gap)
        if conf is not None:
            ok &= (conf[:, a] >= 0.4) & (conf[:, b] >= 0.4)
        v[ok, k] = np.sign(gap[ok])
    return v


class DepthEstimator:
    def __init__(self, device: str = "cpu"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(DEPTH_REPO, revision=DEPTH_REVISION)
        self.model = AutoModelForDepthEstimation.from_pretrained(DEPTH_REPO, revision=DEPTH_REVISION).to(device).eval()
        self.device = device

    def __call__(self, rgb: np.ndarray, region) -> np.ndarray:
        """Relative inverse depth (GRID, GRID) of ``region`` (x0, y0, x1, y1) of ``rgb``."""
        import torch
        import torch.nn.functional as F

        x0, y0, x1, y1 = (int(v) for v in region)
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        inputs = self.processor(images=crop, return_tensors="pt")
        with torch.no_grad():
            pred = self.model(pixel_values=inputs["pixel_values"].to(self.device)).predicted_depth
        return F.interpolate(pred[:, None], size=(GRID, GRID), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
