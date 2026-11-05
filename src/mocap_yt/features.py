"""Image -> DINOv2 token features, exactly as MocapAnything preprocesses them, on CPU.

Upstream's ``preprocess.image_process.load_image`` hard-codes ``.cuda()``; this is a
device-agnostic port of the RGB (no alpha) branch:

  1. RMBG-1.4 foreground matte at 1024x1024 -> Otsu threshold -> drop small blobs,
  2. composite on white, crop to the largest foreground contour, pad to a square
     with 10 % margin,
  3. DINOv2-large ``last_hidden_state`` -> (257, 1024) tokens per frame.

That assumes the subject is the most salient thing in the frame, as in upstream's human
demo clips (plain backgrounds). In-the-wild footage breaks it: on a wide-angle ice-rink
shot with a distant skater, RMBG mattes the ceiling rafters; a dancer's mask swallows the
wall bar their foot touches. Passing per-frame person boxes (``track_people`` ->
``encode_images(boxes=...)``, sample option ``subject: person``) changes only *what* is
matted, so the model still sees a person on white:

  a. a COCO person detector (torchvision Faster R-CNN, MobileNetV3-FPN) on every frame,
  b. one person followed through the clip (:func:`track_subject`): anchored on the most
     confident large detection, linked frame to frame by box overlap, gaps interpolated,
  c. RMBG on a square region around that person's box (``person_context`` x its size),
     intersected with a semantic person mask (torchvision DeepLabV3-MobileNetV3, COCO/VOC
     classes; dilated, then closed to bridge gaps inside unusual poses) so walls, bars or
     beams touching the person drop out -- unless that mask misses most of the person, as
     it does for some upside-down poses; thin leftovers are opened away, the mask is clipped
     to the box plus a margin and reduced to the components that belong to the person
     (:func:`select_components`); then steps 2-3.
  d. ``subject: person_sam`` (``estimate_poses`` -> ``encode_images(boxes=..., poses=...)``)
     replaces c: the tracked person's 2D pose (ViTPose) prompts SAM 2.1, see
     :mod:`mocap_yt.person_sam`. Frames SAM can't segment plausibly fall back to c.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import person_sam as psam
from . import upstream

PERSON = 1  # COCO category id in torchvision's detection models


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of box ``a`` (4,) with each row of ``b`` (K, 4); boxes are ``x0, y0, x1, y1``."""
    iw = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def track_subject(dets: list[np.ndarray], min_iou: float = 0.1, anchor_score: float = 0.7) -> np.ndarray | None:
    """Follow one person through a clip.

    ``dets[t]``: (K, 5) person boxes ``x0, y0, x1, y1, score`` detected in frame ``t``.
    The track starts from the detection with the largest ``score * sqrt(area)`` among those
    scoring at least ``anchor_score`` (any, if none does) and grows forwards and backwards:
    each frame takes the box overlapping the last tracked one most, if IoU >= ``min_iou``.
    Frames without a match are interpolated (held constant at the ends).
    Returns (T, 4) boxes, or ``None`` when no frame has a detection.
    """
    T = len(dets)
    cands = [(t, k) for t in range(T) for k in range(len(dets[t]))]
    if not cands:
        return None

    def rank(tk):
        x0, y0, x1, y1, score = dets[tk[0]][tk[1]]
        return score >= anchor_score, score * np.sqrt(max((x1 - x0) * (y1 - y0), 0.0))

    t0, k0 = max(cands, key=rank)
    boxes = np.full((T, 4), np.nan)
    boxes[t0] = dets[t0][k0, :4]
    for step in (1, -1):
        last, t = boxes[t0], t0 + step
        while 0 <= t < T:
            if len(dets[t]):
                ious = box_iou(last, dets[t][:, :4])
                k = int(ious.argmax())
                if ious[k] >= min_iou:
                    boxes[t] = last = dets[t][k, :4]
            t += step
    ok = ~np.isnan(boxes[:, 0])
    idx = np.arange(T)
    for c in range(4):
        boxes[:, c] = np.interp(idx, idx[ok], boxes[ok, c])
    return boxes


def select_components(labels: np.ndarray, box, min_inside: float = 0.5) -> np.ndarray:
    """Foreground components of a labelled image (0 = background) that belong to the person
    in ``box``: every component with at least ``min_inside`` of its area inside the box, plus
    the one overlapping the box most (limbs reaching out of a torso-only box stay attached;
    a ceiling or a wall bar crossing the box does not). Returns a boolean mask."""
    h, w = labels.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    total = np.bincount(labels.ravel())
    inside = np.zeros_like(total)
    if x1 > x0 and y1 > y0:
        inside = np.bincount(labels[y0:y1, x0:x1].ravel(), minlength=len(total))
    total[0] = inside[0] = 0
    if not inside.any():
        return np.zeros(labels.shape, bool)
    keep = inside >= min_inside * np.maximum(total, 1)
    keep[inside.argmax()] = True
    keep[0] = False
    return keep[labels]


@dataclass
class FeatureExtractor:
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    batch_size: int = 16
    person_context: float = 2.0  # side of the region RMBG sees around a person box, in box sizes
    person_margin: float = 0.15  # the person's mask may extend this far (x box size) out of the box
    person_min_cover: float = 0.3  # gate RMBG by the person segmenter only if it covers this much of it
    person_sam: psam.PersonSamConfig = field(default_factory=psam.PersonSamConfig)  # subject: person_sam

    def __post_init__(self):
        upstream.ensure_on_path()
        upstream.torch_threads()
        from preprocess.briarmbg import BriaRMBG
        from transformers import AutoImageProcessor, Dinov2Model

        self.rmbg = BriaRMBG.from_pretrained(upstream.RMBG_REPO).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(upstream.DINO_REPO)
        self.dino_model = Dinov2Model.from_pretrained(upstream.DINO_REPO).to(self.device, self.dtype).eval()
        self._detector = self._segmenter = None
        self.ungated = 0  # person-mode frames where the segmenter gate was skipped
        self._pose = self._sam = None
        self.last_matte: psam.MatteResult | None = None  # person_sam result of the last matte_and_crop
        self.last_failure: psam.NoPersonMask | None = None  # ... or why SAM was not used
        self.mask_sources: np.ndarray | None = None  # per frame of the last encode_images: psam.SOURCES index, -1 skipped
        self.timings = Counter()  # seconds per stage of the last encode_images

    # ------------------------------------------------------------------ preprocessing
    @torch.no_grad()
    def _foreground(self, img: torch.Tensor, min_size: int = 200) -> np.ndarray:
        """RMBG-1.4 matte of a (3, H, W) image in [0, 1] -> boolean foreground mask (H, W)."""
        import cv2
        import torchvision.transforms.functional as TF
        from skimage.measure import label
        from skimage.morphology import remove_small_objects

        h, w = img.shape[1:]
        img1k = TF.resize(img, [1024, 1024], antialias=True)
        if float(img1k.max()) < 1e-3:
            raise ValueError("pure black image")
        alpha = self.rmbg(TF.normalize(img1k, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).unsqueeze(0))[0][0]
        alpha = TF.resize(alpha.squeeze(0), [h, w], antialias=True)
        alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-12)
        a8 = (alpha * 255).to(torch.uint8).squeeze().cpu().numpy()
        _, a8 = cv2.threshold(a8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return remove_small_objects(label(a8), min_size=min_size) > 0

    @staticmethod
    def _square_crop(img: torch.Tensor, mask: np.ndarray, rect, padding_ratio: float) -> np.ndarray:
        """Composite ``img`` on white through ``mask``, crop ``rect`` (x, y, w, h) and pad it
        to a square with a ``padding_ratio`` margin -> (S, S, 3) float image in [0, 1]."""
        import torch.nn.functional as F

        x, y, bw, bh = (int(v) for v in rect)
        m = torch.from_numpy(mask.astype(np.float32)).to(img.device)[None]
        comp = img * m + 1.0 * (1 - m)  # white background
        pad = [0] * 4
        if bw > bh:
            pad[0] = int(bw * padding_ratio)
            pad[2] = int(pad[0] + (bw - bh) / 2)
        else:
            pad[2] = int(bh * padding_ratio)
            pad[0] = int(pad[2] + (bh - bw) / 2)
        pad[1], pad[3] = pad[0], pad[2]
        crop = F.pad(comp[:, y : y + bh, x : x + bw], pad=tuple(pad), mode="constant", value=1.0)
        return crop.permute(1, 2, 0).cpu().numpy()

    @torch.no_grad()
    def matte_and_crop(self, rgb: np.ndarray, padding_ratio: float = 0.1, box=None, pose=None) -> tuple[np.ndarray, np.ndarray]:
        """rgb uint8 (H,W,3) -> (cropped square float image in [0,1] (S,S,3), binary mask (H,W)).

        ``box`` (x0, y0, x1, y1): matte the person in that box (module doc, step c);
        ``None``: upstream's behaviour, the largest salient region.
        ``pose`` (17, 3) keypoints ``x, y, score`` of that person: segment them with
        pose-prompted SAM (step d; :attr:`last_matte` holds the details), falling back to
        step c when SAM gives no plausible mask (then :attr:`last_matte` is ``None``).
        """
        import cv2
        from skimage.measure import label

        h, w = rgb.shape[:2]
        s = 2000 / max(h, w)
        if s < 1:
            rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            h, w = rgb.shape[:2]
            box = None if box is None else np.asarray(box, float) * s
            pose = None if pose is None else np.asarray(pose, float) * [s, s, 1]
        img = torch.from_numpy(rgb).to(self.device).float().permute(2, 0, 1) / 255.0
        if box is None:
            mask = self._foreground(img)
            if not mask.any():
                raise ValueError("no foreground found")
            contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rect = cv2.boundingRect(max(contours, key=cv2.contourArea))
            return self._square_crop(img, mask, rect, padding_ratio), mask

        self.last_matte = self.last_failure = None
        if pose is not None:
            t = time.perf_counter()
            try:
                res = psam.segment_person(rgb, box, pose, self.sam_segmenter(), self.person_sam)
            except psam.NoPersonMask as e:
                res, self.last_failure = None, e  # fall through to the person path below
            self.timings["sam"] += time.perf_counter() - t
            if res is not None:
                self.last_matte = res
                rx0, ry0, rx1, ry1 = res.region
                ys, xs = np.nonzero(res.mask)
                rect = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
                mask = np.zeros((h, w), bool)
                mask[ry0:ry1, rx0:rx1] = res.mask
                return self._square_crop(img[:, ry0:ry1, rx0:rx1], res.mask, rect, padding_ratio), mask

        x0, y0, x1, y1 = box
        half = max(x1 - x0, y1 - y0) * self.person_context / 2
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx0, ry0 = max(0, int(cx - half)), max(0, int(cy - half))
        rx1, ry1 = min(w, int(np.ceil(cx + half))), min(h, int(np.ceil(cy + half)))
        if rx1 - rx0 < 8 or ry1 - ry0 < 8:
            raise ValueError("person box too small")
        sub = img[:, ry0:ry1, rx0:rx1]
        fg = self._foreground(sub, min_size=max(8, min(200, (rx1 - rx0) * (ry1 - ry0) // 200)))
        bx0, by0, bx1, by1 = x0 - rx0, y0 - ry0, x1 - rx0, y1 - ry0  # box in region coordinates
        bw, bh = bx1 - bx0, by1 - by0

        def disk(r):
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

        size = max(bw, bh)
        gate = cv2.dilate((self.person_probability(sub) > 0.5).astype(np.uint8), disk(max(1, int(0.04 * size))))
        # bridge the gaps the segmenter leaves inside an unusual pose (a torso it isn't sure of)
        gate = cv2.morphologyEx(gate, cv2.MORPH_CLOSE, disk(max(1, int(0.12 * size)))) > 0
        inbox = (slice(max(0, int(by0)), int(np.ceil(by1))), slice(max(0, int(bx0)), int(np.ceil(bx1))))
        if (fg & gate)[inbox].sum() >= self.person_min_cover * fg[inbox].sum():
            fg &= gate
        else:  # the segmenter doesn't see this person (some upside-down poses): keep RMBG's mask
            self.ungated += 1
        if (r := int(size / 60)) >= 1:  # bars/beams thinner than ~1/30 of the person
            fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, disk(r)) > 0
        mx, my = self.person_margin * bw, self.person_margin * bh
        clip = np.zeros_like(fg)
        clip[max(0, int(by0 - my)) : int(np.ceil(by1 + my)), max(0, int(bx0 - mx)) : int(np.ceil(bx1 + mx))] = True
        sel = select_components(label(fg & clip), (bx0, by0, bx1, by1))
        if not sel.any():
            raise ValueError("no foreground found on the person")
        ys, xs = np.nonzero(sel)
        rect = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        mask = np.zeros((h, w), bool)
        mask[ry0:ry1, rx0:rx1] = sel
        return self._square_crop(sub, sel, rect, padding_ratio), mask

    # ------------------------------------------------------------------ people
    def detector(self):
        if self._detector is None:
            from torchvision.models.detection import (
                FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                fasterrcnn_mobilenet_v3_large_fpn,
            )

            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
            self._detector = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, box_score_thresh=0.3)
            self._detector.to(self.device).eval()
        return self._detector

    def segmenter(self):
        if self._segmenter is None:
            from torchvision.models.segmentation import (
                DeepLabV3_MobileNet_V3_Large_Weights,
                deeplabv3_mobilenet_v3_large,
            )

            weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
            self._segmenter = deeplabv3_mobilenet_v3_large(weights=weights).to(self.device).eval()
            self._segmenter.person = weights.meta["categories"].index("person")
        return self._segmenter

    @torch.no_grad()
    def person_probability(self, img: torch.Tensor) -> np.ndarray:
        """Per-pixel P(person) of a (3, H, W) image in [0, 1] (DeepLabV3, short side at 520)."""
        import torch.nn.functional as F

        net = self.segmenter()
        h, w = img.shape[1:]
        s = 520 / min(h, w)
        x = F.interpolate(img[None], size=(max(1, round(h * s)), max(1, round(w * s))),
                          mode="bilinear", antialias=True, align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
        p = net((x - mean) / std)["out"].softmax(1)[:, net.person : net.person + 1]
        return F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

    @torch.no_grad()
    def detect_people(self, rgb: np.ndarray) -> np.ndarray:
        """(K, 5) person boxes ``x0, y0, x1, y1, score`` in an RGB uint8 frame."""
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float() / 255.0
        out = self.detector()([x])[0]
        keep = out["labels"] == PERSON
        return torch.cat([out["boxes"][keep], out["scores"][keep, None]], 1).cpu().numpy()

    def track_people(self, paths: list[Path], progress: bool = True) -> np.ndarray | None:
        """Detect people in every frame and follow one through the clip -> (N, 4) boxes."""
        import cv2
        from tqdm import tqdm

        dets = []
        for p in tqdm(paths, desc="people") if progress else paths:
            dets.append(self.detect_people(cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)))
        found = sum(len(d) > 0 for d in dets)
        print(f"[features] person detected in {found}/{len(dets)} frames")
        return track_subject(dets)

    def pose_estimator(self) -> psam.PoseEstimator:
        if self._pose is None:
            self._pose = psam.PoseEstimator(self.person_sam.pose, self.device)
        return self._pose

    def sam_segmenter(self) -> psam.SamSegmenter:
        if self._sam is None:
            self._sam = psam.SamSegmenter(self.person_sam.sam, self.device)
        return self._sam

    def estimate_poses(self, paths: list[Path], boxes: np.ndarray, progress: bool = True) -> psam.PersonPoses:
        """2D pose of the tracked person (``boxes`` (N, 4)) in every frame, with rotation
        test-time augmentation (``person_sam.rotations``). People smaller than
        ``person_sam.min_person`` get no pose (all scores 0)."""
        import cv2
        from tqdm import tqdm

        cfg = self.person_sam
        n, rots = len(paths), cfg.rotations
        kps = np.zeros((n, 17, 3), np.float32)
        rotation, rot_scores = np.zeros(n, np.int16), np.zeros((n, len(rots)), np.float32)
        t = time.perf_counter()
        for i, p in enumerate(tqdm(paths, desc="pose") if progress else paths):
            b = np.asarray(boxes[i], float)
            if max(b[2] - b[0], b[3] - b[1]) < cfg.min_person:
                continue
            rgb = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            allk, means = self.pose_estimator()(rgb, b, rots)
            j = psam.choose_rotation(means, rots, cfg.upright_margin)
            kps[i], rotation[i], rot_scores[i] = allk[j], rots[j], means
        posed = rot_scores.any(1)
        if progress:
            turns = ", ".join(f"{r} deg: {int((rotation[posed] == r).sum())}" for r in rots)
            mean = float(kps[posed, :, 2].mean()) if posed.any() else 0.0
            print(f"[features] pose in {int(posed.sum())}/{n} frames ({(time.perf_counter() - t) / max(n, 1):.2f} s/frame), "
                  f"mean keypoint score {mean:.2f}; kept rotation {turns}")
        return psam.PersonPoses(np.asarray(boxes, np.float32), kps, rotation, rot_scores)

    # ------------------------------------------------------------------ features
    @torch.no_grad()
    def dino(self, crops: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        pils = [Image.fromarray((c * 255).astype(np.uint8)) for c in crops]
        out = []
        for i in range(0, len(pils), self.batch_size):
            px = self.processor(pils[i : i + self.batch_size], return_tensors="pt").pixel_values
            out.append(self.dino_model(px.to(self.device, self.dtype)).last_hidden_state.float().cpu())
        return torch.cat(out).numpy()  # (N, 257, 1024)

    def encode_images(self, paths: list[Path], crops_dir: Path | None = None, progress: bool = True,
                      boxes: np.ndarray | None = None, poses: np.ndarray | None = None):
        """Returns (features (N,257,1024) float32, list of valid indices).

        ``boxes`` (N, 4): per-frame person boxes (:meth:`track_people`); matte only that person.
        ``poses`` (N, 17, 3): that person's keypoints (:meth:`estimate_poses`); segment them with
        pose-prompted SAM. :attr:`mask_sources` then records which matte each frame got.
        """
        import cv2
        from tqdm import tqdm

        crops, ok = [], []
        self.ungated = 0
        self.timings = Counter()
        self.mask_sources = None if poses is None else np.full(len(paths), -1, np.int8)
        it = tqdm(paths, desc="matte+crop") if progress else paths
        for i, p in enumerate(it):
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            t = time.perf_counter()
            try:
                crop, _ = self.matte_and_crop(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), box=None if boxes is None else boxes[i],
                                              pose=None if poses is None else poses[i])
            except ValueError as e:
                print(f"[features] skip {p}: {e}")
                continue
            finally:
                self.timings["matte"] += time.perf_counter() - t
            if poses is not None:
                self.mask_sources[i] = psam.SOURCES.index(self.last_matte.source if self.last_matte else "rmbg")
            crops.append(crop)
            ok.append(i)
            if crops_dir is not None:
                crops_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(crops_dir / Path(p).name), cv2.cvtColor((crop * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        if not crops:
            raise RuntimeError("no frame could be matted")
        if boxes is not None and self.ungated:
            print(f"[features] person segmenter missed the person in {self.ungated}/{len(paths)} frames: RMBG mask kept as is")
        if self.mask_sources is not None:
            counts = {name: int((self.mask_sources == k).sum()) for k, name in enumerate(psam.SOURCES)}
            counts["skipped"] = int((self.mask_sources < 0).sum())
            sam = self.timings["sam"] / len(paths)
            print(f"[features] person_sam masks per source: {counts}; s/frame: SAM {sam:.2f}, "
                  f"RMBG fallback {self.timings['matte'] / len(paths) - sam:.2f}")
        feats = []
        it = range(0, len(crops), self.batch_size)
        for i in tqdm(it, desc="dinov2") if progress else it:
            feats.append(self.dino(crops[i : i + self.batch_size]))
        return np.concatenate(feats), ok
