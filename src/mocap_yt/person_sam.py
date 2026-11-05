"""Pose-prompted SAM matting for ``subject: person_sam``.

``subject: person`` (:mod:`mocap_yt.features`) mattes the tracked person with RMBG intersected
with a semantic person mask. Both lose limbs that are motion-blurred or close in colour to the
background, and the semantic mask misses upside-down bodies. This mode keeps the person
tracker but builds the mask from the person's 2D pose:

  1. 2D pose (:class:`PoseEstimator`): ViTPose-B, COCO-17 keypoints, top-down on the tracked
     box. It runs on the frame turned by 0/90/180/270 degrees; the turn with the best mean
     keypoint score wins (COCO has few upside-down people; a windmill is mostly upside down)
     and its keypoints are mapped back (:func:`choose_rotation`).
  2. Prompt (:func:`build_prompt`): up to ``max_points`` positive points *inside* the body
     (torso centre, head, a point on each forearm/shin next to the wrist/ankle, knees) plus a
     box, the tracked box grown to hold every confident keypoint. Points at the very tips of
     limbs sit on the outline and pull background in; SAM was trained with a handful of clicks.
  3. SAM 2.1 (:class:`SamSegmenter`) on a square region around the person
     (:func:`prompt_region`), encoded once; the decoder tries points + box, then the box alone.
  4. Clean-up (:func:`clean_mask`): fill small holes, open away thin bars except along the
     stick figure, clip to an envelope around the box, keep the parts that hold keypoints.
  5. A plausibility check (:func:`is_plausible`); when nothing passes, :class:`NoPersonMask`
     and the caller falls back to the person mode for that frame.

Everything but the two model wrappers is numpy/OpenCV, testable without weights.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

VERSION = 1  # bump when the algorithm changes: cached person_sam features are recomputed

FACE = (0, 1, 2, 3, 4)  # nose, eyes, ears
SHOULDERS, TORSO = (5, 6), (5, 6, 11, 12)
ARMS = ((5, 7, 9), (6, 8, 10))  # shoulder, elbow, wrist
LEGS = ((11, 13, 15), (12, 14, 16))  # hip, knee, ankle
KNEES = (13, 14)
BONES = ((5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12))
SOURCES = ("sam_pose", "sam_box", "rmbg")  # mask_source codes 0, 1, 2 (-1: frame skipped)


@dataclass(frozen=True)
class PersonSamConfig:
    """Settings of ``subject: person_sam`` (top-level ``person_sam:`` block of the config)."""

    sam: str = "facebook/sam2.1-hiera-small"
    pose: str = "usyd-community/vitpose-base-simple"
    rotations: tuple[int, ...] = (0, 90, 180, 270)  # degrees, counter-clockwise
    upright_margin: float = 0.05  # a turned pose must beat the upright one by this mean score
    min_person: int = 40  # px; smaller people get no pose, only a box prompt
    kp_score: float = 0.4  # keypoints below this score are ignored
    kp_score_distal: float = 0.5  # ... wrists and ankles below this one
    max_points: int = 8
    inset: float = 0.2  # wrist/ankle prompt points move this far toward the elbow/knee
    context: float = 2.0  # side of the region SAM sees, in prompt-box sizes
    min_region: int = 160  # px, so tiny people are enlarged
    box_margin: float = 0.1  # the prompt box grows by this x its size
    envelope: float = 0.25  # the mask is clipped to the prompt box grown by this x person size
    opening: bool = True  # open away thin bars (not along the stick figure)
    min_cover: float = 0.75  # share of the prompt points the mask must hold
    min_iou: float = 0.5  # SAM's own IoU estimate
    fill_range: tuple[float, float] = (0.08, 0.9)  # mask area / prompt box area

    def __post_init__(self):
        object.__setattr__(self, "rotations", tuple(int(r) for r in self.rotations))
        object.__setattr__(self, "fill_range", tuple(float(f) for f in self.fill_range))
        if not self.rotations or any(r % 90 for r in self.rotations):
            raise ValueError(f"person_sam.rotations must be multiples of 90, got {self.rotations}")

    def tag(self) -> str:
        """Everything a person_sam matte depends on (cache key)."""
        return json.dumps({**asdict(self), "version": VERSION}, sort_keys=True)

    def pose_tag(self) -> str:
        """Everything the cached poses depend on."""
        keys = ("pose", "rotations", "upright_margin", "min_person")
        return json.dumps({**{k: getattr(self, k) for k in keys}, "version": VERSION}, sort_keys=True)


def matting_tag(subject: str, cfg: PersonSamConfig | None) -> str:
    """Cache tag of a subject's matting settings: ``""`` for the modes without settings."""
    return (cfg or PersonSamConfig()).tag() if subject == "person_sam" else ""


class NoPersonMask(Exception):
    """SAM gave no plausible person mask for this frame; ``attempts``: stats of what it gave."""

    def __init__(self, msg: str, attempts=()):
        super().__init__(msg)
        self.attempts = list(attempts)


@dataclass
class PersonPoses:
    """Tracked person boxes of a clip and their 2D pose."""

    boxes: np.ndarray  # (N, 4) x0, y0, x1, y1
    keypoints: np.ndarray  # (N, 17, 3) x, y, score of the kept rotation (all 0: not estimated)
    rotation: np.ndarray  # (N,) degrees the frame was turned for the kept pose
    rot_scores: np.ndarray  # (N, R) mean keypoint score for each tried rotation

    def save(self, path: Path, tag: str):
        np.savez_compressed(path, boxes=self.boxes.astype(np.float32), keypoints=self.keypoints.astype(np.float32),
                            rotation=self.rotation.astype(np.int16), rot_scores=self.rot_scores.astype(np.float32),
                            tag=np.array(tag))

    @classmethod
    def load(cls, path: Path, tag: str, n: int) -> "PersonPoses | None":
        """The poses cached at ``path`` if they were made with ``tag`` for ``n`` frames."""
        if not Path(path).exists():
            return None
        z = np.load(path, allow_pickle=False)
        if str(z["tag"]) != tag or len(z["boxes"]) != n:
            return None
        return cls(z["boxes"], z["keypoints"], z["rotation"], z["rot_scores"])


# ------------------------------------------------------------------------------ geometry


def rot90_points(xy, k: int, hw) -> np.ndarray:
    """(x, y) points of an image of size ``hw`` = (H, W) -> their position in
    ``np.rot90(image, k)`` (k counter-clockwise quarter turns; pixel ``i`` spans ``[i, i+1)``)."""
    xy = np.asarray(xy, float)
    x, y = xy[..., 0], xy[..., 1]
    h, w = hw
    k %= 4
    if k == 0:
        nx, ny = x, y
    elif k == 1:
        nx, ny = y, w - x
    elif k == 2:
        nx, ny = w - x, h - y
    else:
        nx, ny = h - y, x
    return np.stack([nx, ny], -1)


def unrot90_points(xy, k: int, hw) -> np.ndarray:
    """Inverse of :func:`rot90_points`: points of ``np.rot90(image, k)`` -> ``image`` (size ``hw``)."""
    return rot90_points(xy, -k, hw if k % 2 == 0 else tuple(hw)[::-1])


def rot90_box(box, k: int, hw) -> np.ndarray:
    """Box ``x0, y0, x1, y1`` of an image of size ``hw`` -> its box in ``np.rot90(image, k)``."""
    x0, y0, x1, y1 = box
    p = rot90_points([[x0, y0], [x1, y1]], k, hw)
    return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])


def choose_rotation(mean_scores, rotations, upright_margin: float = 0.05) -> int:
    """Index of the rotation whose pose scored best; 0 degrees wins unless it is beaten by more
    than ``upright_margin``."""
    s = np.asarray(mean_scores, float)
    best = int(np.argmax(s))
    if 0 in rotations:
        up = list(rotations).index(0)
        if s[best] <= s[up] + upright_margin:
            return up
    return best


def prompt_region(box, hw, context: float = 2.0, min_side: int = 160) -> tuple[int, int, int, int]:
    """Square ``x0, y0, x1, y1`` around ``box``: ``context`` x its longer side, at least
    ``min_side``, shifted to lie inside the image (clipped only where the image is smaller)."""
    h, w = hw
    x0, y0, x1, y1 = box
    side = int(np.ceil(max(context * max(x1 - x0, y1 - y0), min_side)))

    def span(c, n):
        if side >= n:
            return 0, n
        a = min(max(int(round(c - side / 2)), 0), n - side)
        return a, a + side

    rx0, rx1 = span((x0 + x1) / 2, w)
    ry0, ry1 = span((y0 + y1) / 2, h)
    return rx0, ry0, rx1, ry1


# ------------------------------------------------------------------------------- prompts


@dataclass
class Prompt:
    points: np.ndarray  # (P, 2) x, y
    labels: np.ndarray  # (P,) 1 = person, 0 = not the person
    box: np.ndarray | None  # (4,) x0, y0, x1, y1
    kind: str  # "pose": points from the pose; "box": the box (+ whatever points there are)

    def shifted(self, dx: float, dy: float) -> "Prompt":
        return Prompt(self.points - [dx, dy], self.labels, self.box - [dx, dy, dx, dy], self.kind)

    def inside(self, w: int, h: int) -> "Prompt":
        """Drop points outside a ``w`` x ``h`` image and clip the box to it."""
        keep = (self.points[:, 0] >= 0) & (self.points[:, 0] < w) & (self.points[:, 1] >= 0) & (self.points[:, 1] < h)
        box = np.clip(self.box, 0, [w, h, w, h])
        return Prompt(self.points[keep], self.labels[keep], box, self.kind)


def _confident(kps, idx, thr):
    return [i for i in idx if kps[i, 2] >= thr]


def torso_centre(kps: np.ndarray, thr: float) -> np.ndarray | None:
    idx = _confident(kps, TORSO, thr)
    return kps[idx, :2].mean(0) if len(idx) >= 2 else None


def head_point(kps: np.ndarray, thr: float, toward_neck: float = 0.15) -> np.ndarray | None:
    """Mean of the confident face keypoints, moved a little toward the shoulders (into the head)."""
    idx = _confident(kps, FACE, thr)
    if not idx:
        return None
    p = kps[idx, :2].mean(0)
    sh = _confident(kps, SHOULDERS, thr)
    return p + toward_neck * (kps[sh, :2].mean(0) - p) if sh else p


def limb_point(kps: np.ndarray, chain, thr: float, thr_distal: float, inset: float, mid_fallback: bool):
    """A point on the distal segment of a limb: the wrist/ankle moved ``inset`` toward the
    elbow/knee (just the wrist/ankle if the elbow/knee is unsure; the elbow if the wrist is
    unsure and ``mid_fallback``)."""
    _, mid, end = chain
    if kps[end, 2] >= thr_distal:
        if kps[mid, 2] >= thr:
            return kps[end, :2] + inset * (kps[mid, :2] - kps[end, :2])
        return kps[end, :2].copy()
    if mid_fallback and kps[mid, 2] >= thr:
        return kps[mid, :2].copy()
    return None


def build_prompt(kps: np.ndarray | None, box, cfg: PersonSamConfig) -> Prompt:
    """SAM prompt for the person in ``box`` with keypoints ``kps`` (17, 3); see the module doc.

    ``kind="pose"`` needs at least 3 points, else the prompt is the tracked box (plus the points
    there are). Small people (``min_person``) and missing poses always get the box.
    """
    box = np.asarray(box, float)
    size = max(box[2] - box[0], box[3] - box[1])
    none = np.zeros((0, 2)), np.zeros(0, int)
    if kps is None or size < cfg.min_person or not np.isfinite(kps).all():
        return Prompt(*none, box, "box")
    thr, thr_d = cfg.kp_score, cfg.kp_score_distal
    cands = [torso_centre(kps, thr), head_point(kps, thr)]
    cands += [limb_point(kps, leg, thr, thr_d, cfg.inset, mid_fallback=False) for leg in LEGS]
    cands += [limb_point(kps, arm, thr, thr_d, cfg.inset, mid_fallback=True) for arm in ARMS]
    cands += [kps[i, :2] if kps[i, 2] >= thr else None for i in KNEES]
    pts = []
    for p in cands:
        if p is not None and len(pts) < cfg.max_points and all(np.hypot(*(p - q)) >= 0.05 * size for q in pts):
            pts.append(p)
    pts = np.array(pts, float).reshape(-1, 2)
    labels = np.ones(len(pts), int)
    if len(pts) < 3:
        return Prompt(pts, labels, box, "box")
    conf = kps[kps[:, 2] >= thr, :2]
    pb = np.concatenate([np.minimum(box[:2], conf.min(0)), np.maximum(box[2:], conf.max(0))])
    pb += np.array([-1, -1, 1, 1]) * cfg.box_margin * max(pb[2] - pb[0], pb[3] - pb[1])
    return Prompt(pts, labels, pb, "pose")


def box_prompt(kps: np.ndarray | None, box, cfg: PersonSamConfig) -> Prompt:
    """Fallback prompt: the tracked box, plus the torso centre when the pose has one."""
    c = None if kps is None else torso_centre(kps, cfg.kp_score)
    pts = np.zeros((0, 2)) if c is None else c[None]
    return Prompt(pts, np.ones(len(pts), int), np.asarray(box, float), "box")


# --------------------------------------------------------------------------------- masks


def _disk(r: int):
    import cv2

    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def skeleton_mask(kps: np.ndarray | None, hw, thr: float, width: int) -> np.ndarray:
    """Stick figure of the confident bones, ``width`` px thick -> boolean (H, W)."""
    import cv2

    m = np.zeros(hw, np.uint8)
    if kps is None:
        return m > 0
    xy = np.round(kps[:, :2]).astype(int)
    ok = kps[:, 2] >= thr
    for a, b in BONES:
        if ok[a] and ok[b]:
            cv2.line(m, tuple(int(v) for v in xy[a]), tuple(int(v) for v in xy[b]), 1, thickness=max(1, int(width)))
    for i in np.flatnonzero(ok):
        cv2.circle(m, tuple(int(v) for v in xy[i]), max(1, int(width) // 2), 1, -1)
    return m > 0


def fill_small_holes(mask: np.ndarray, max_area: float) -> np.ndarray:
    """Fill the holes (background not connected to the border) of at most ``max_area`` px."""
    import cv2

    m = mask.astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((~m).astype(np.uint8), connectivity=4)
    if n <= 1:
        return m
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    fill = stats[:, cv2.CC_STAT_AREA] <= max_area
    fill[0] = False  # label 0 = the mask itself
    fill[border] = False
    return m | fill[lab]


def clean_mask(mask: np.ndarray, kps: np.ndarray | None, prompt: Prompt, size: float, cfg: PersonSamConfig) -> np.ndarray:
    """Tidy a SAM mask of a person ``size`` px tall (region coordinates; ``kps`` too)."""
    import cv2

    m = fill_small_holes(mask, max(16, 0.01 * mask.sum()))
    r = int(size / 60)  # same rule as the person mode's opening: bars thinner than ~1/30 of the person
    if cfg.opening and r >= 1:
        opened = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, _disk(r)) > 0
        limbs = skeleton_mask(kps, m.shape, cfg.kp_score, max(3, int(0.08 * size)))
        m = opened | (m & (cv2.dilate(limbs.astype(np.uint8), _disk(r)) > 0))
    e = cfg.envelope * size
    x0, y0, x1, y1 = prompt.box
    env = np.zeros_like(m)
    env[max(0, int(y0 - e)) : max(0, int(np.ceil(y1 + e))), max(0, int(x0 - e)) : max(0, int(np.ceil(x1 + e)))] = True
    m &= env
    n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return m
    anchors = [prompt.points[prompt.labels == 1]]
    if kps is not None:
        anchors.append(kps[kps[:, 2] >= cfg.kp_score, :2])
    anchors = np.concatenate(anchors).reshape(-1, 2)
    rad = max(2, int(0.03 * size))
    h, w = m.shape
    keep = np.zeros(n, bool)
    for x, y in np.round(anchors).astype(int):
        if 0 <= x < w and 0 <= y < h:
            keep[np.unique(lab[max(0, y - rad) : y + rad + 1, max(0, x - rad) : x + rad + 1])] = True
    keep[0] = False
    if keep.any():
        return keep[lab]
    from .features import select_components

    return select_components(lab, prompt.box)


def mask_stats(mask: np.ndarray, prompt: Prompt) -> dict:
    """``cover``: share of the positive prompt points in the mask; ``fill``: mask / box area."""
    h, w = mask.shape
    pos = np.round(prompt.points[prompt.labels == 1]).astype(int)
    pos = pos[(pos[:, 0] >= 0) & (pos[:, 0] < w) & (pos[:, 1] >= 0) & (pos[:, 1] < h)]
    cover = float(mask[pos[:, 1], pos[:, 0]].mean()) if len(pos) else 1.0
    x0, y0, x1, y1 = prompt.box
    area = int(mask.sum())
    return {"area": area, "cover": round(cover, 3), "fill": round(float(area / max((x1 - x0) * (y1 - y0), 1.0)), 3)}


def is_plausible(stats: dict, iou: float, obj: float, kind: str, cfg: PersonSamConfig) -> bool:
    lo, hi = cfg.fill_range
    ok = stats["area"] > 0 and obj > 0 and iou >= cfg.min_iou and lo <= stats["fill"] <= hi
    return bool(ok and (kind == "box" or stats["cover"] >= cfg.min_cover))


# -------------------------------------------------------------------------- orchestration


@dataclass
class MatteResult:
    mask: np.ndarray  # (h, w) boolean, region coordinates
    region: tuple[int, int, int, int]  # x0, y0, x1, y1 in the frame
    source: str  # "sam_pose" | "sam_box"
    prompt: Prompt  # region coordinates
    stats: dict  # cover, fill, iou, obj, ... of the kept mask
    attempts: list[dict] = field(default_factory=list)  # stats of every decoder call


def segment_person(rgb: np.ndarray, box, kps: np.ndarray | None, sam, cfg: PersonSamConfig) -> MatteResult:
    """Mask of the person in ``box`` (frame coordinates) with keypoints ``kps`` (17, 3, or
    ``None``). ``sam`` provides ``embed(region_rgb)`` and ``decode(image, prompt) -> (logits,
    iou, obj)`` (:class:`SamSegmenter`). Raises :class:`NoPersonMask`."""
    h, w = rgb.shape[:2]
    box = np.asarray(box, float)
    prompt = build_prompt(kps, box, cfg)
    x0, y0, x1, y1 = prompt_region(prompt.box, (h, w), cfg.context, cfg.min_region)
    rw, rh = x1 - x0, y1 - y0
    if rw < 8 or rh < 8:
        raise NoPersonMask("region too small")
    kps_r = None if kps is None else np.asarray(kps, float) - [x0, y0, 0]
    size = max(box[2] - box[0], box[3] - box[1])
    attempts = [prompt] if prompt.kind == "box" else [prompt, box_prompt(kps, box, cfg)]
    image = sam.embed(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
    tried = []
    for p in attempts:
        p = p.shifted(x0, y0).inside(rw, rh)
        logits, iou, obj = sam.decode(image, p)
        m = clean_mask(logits > 0, kps_r, p, size, cfg)
        st = {**mask_stats(m, p), "iou": round(float(iou), 3), "obj": round(float(obj), 2), "kind": p.kind, "points": len(p.points)}
        tried.append(st)
        if is_plausible(st, iou, obj, p.kind, cfg):
            return MatteResult(m, (x0, y0, x1, y1), "sam_pose" if p.kind == "pose" else "sam_box", p, st, tried)
    raise NoPersonMask(f"no plausible SAM mask: {tried}", tried)


# ---------------------------------------------------------------------------------- models


class PoseEstimator:
    """ViTPose (transformers) on a person box, with rotation test-time augmentation."""

    def __init__(self, repo: str, device: str = "cpu"):
        from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor

        self.processor = VitPoseImageProcessor.from_pretrained(repo)
        self.model = VitPoseForPoseEstimation.from_pretrained(repo).to(device).eval()
        self.device = device

    def __call__(self, rgb: np.ndarray, box, rotations=(0,)) -> tuple[np.ndarray, np.ndarray]:
        """Keypoints (R, 17, 3) ``x, y, score`` in ``rgb``'s coordinates, one set per rotation of
        the frame, and their mean scores (R,)."""
        import torch

        hw = rgb.shape[:2]
        ks = [(r // 90) % 4 for r in rotations]
        images, boxes = [], []
        for k in ks:
            images.append(np.ascontiguousarray(np.rot90(rgb, k)))
            b = rot90_box(box, k, hw)
            boxes.append([[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]])  # COCO xywh
        with torch.no_grad():
            inputs = self.processor(images=images, boxes=boxes, return_tensors="pt")
            out = self.model(pixel_values=inputs["pixel_values"].to(self.device))
        res = self.processor.post_process_pose_estimation(out, boxes=boxes)
        kps = np.zeros((len(ks), 17, 3), np.float32)
        for j, k in enumerate(ks):
            kps[j, :, :2] = unrot90_points(res[j][0]["keypoints"].cpu().numpy(), k, hw)
            kps[j, :, 2] = res[j][0]["scores"].cpu().numpy()
        return kps, kps[:, :, 2].mean(1)


class SamSegmenter:
    """SAM 2.1 (transformers), image mode: one encoder pass per image, many prompts."""

    def __init__(self, repo: str, device: str = "cpu"):
        from transformers import Sam2Model, Sam2Processor
        from transformers.utils import logging

        level = logging.get_verbosity()
        logging.set_verbosity_error()  # the SAM 2.1 checkpoints are `sam2_video`; Sam2Model loads their image part
        try:
            self.processor = Sam2Processor.from_pretrained(repo)
            self.model = Sam2Model.from_pretrained(repo).to(device).eval()
        finally:
            logging.set_verbosity(level)
        self.device = device

    def embed(self, rgb: np.ndarray) -> dict:
        import torch

        inputs = self.processor(images=rgb, return_tensors="pt")
        with torch.no_grad():
            emb = self.model.get_image_embeddings(inputs["pixel_values"].to(self.device))
        return {"embeddings": emb, "original_sizes": inputs["original_sizes"]}

    def decode(self, image: dict, prompt: Prompt) -> tuple[np.ndarray, float, float]:
        """Mask logits (h, w) of the embedded image for ``prompt`` (its pixel coordinates), SAM's
        IoU estimate and object-presence logit."""
        import torch

        kw = {} if prompt.box is None else {"input_boxes": [[[float(v) for v in prompt.box]]]}
        if len(prompt.points):
            kw["input_points"] = [[prompt.points.astype(float).tolist()]]
            kw["input_labels"] = [[prompt.labels.astype(int).tolist()]]
        inputs = self.processor(images=None, original_sizes=image["original_sizes"], return_tensors="pt", **kw)
        prompts = {k: inputs[k].to(self.device) for k in ("input_points", "input_labels", "input_boxes") if k in inputs}
        with torch.no_grad():
            out = self.model(image_embeddings=image["embeddings"], multimask_output=False, **prompts)
        logits = self.processor.post_process_masks(out.pred_masks.cpu(), image["original_sizes"], binarize=False)[0]
        return logits[0, 0].numpy(), float(out.iou_scores.flatten()[0]), float(out.object_score_logits.flatten()[0])
