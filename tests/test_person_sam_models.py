"""person_sam with the real models: skipped unless SAM 2.1 and ViTPose are in the Hugging Face
cache (``mocap-yt setup`` downloads them)."""

import numpy as np
import pytest
from huggingface_hub import try_to_load_from_cache

from mocap_yt.person_sam import (
    PersonSamConfig,
    PoseEstimator,
    Prompt,
    SamSegmenter,
    choose_rotation,
    prompt_region,
    rot90_box,
    rot90_points,
    segment_person,
    torso_centre,
)
from mocap_yt.upstream import PROJECT_ROOT

CFG = PersonSamConfig()
HOCKEY = PROJECT_ROOT / "assets" / "samples" / "Hockey Stop in 3 Minutes - Learn to Stop on Ice _ Tutorial_360p.mp4"


def cached(repo: str) -> bool:
    return isinstance(try_to_load_from_cache(repo, "config.json"), str)


pytestmark = pytest.mark.skipif(not (cached(CFG.sam) and cached(CFG.pose)),
                                reason="SAM 2.1 / ViTPose weights not downloaded (mocap-yt setup)")


@pytest.fixture(scope="module")
def skater():
    """A frame of the hockey tutorial's stance drill (2:44): one skater facing the camera, and a box around them."""
    import cv2

    cap = cv2.VideoCapture(str(HOCKEY))
    cap.set(cv2.CAP_PROP_POS_MSEC, 164_000)
    ok, bgr = cap.read()
    assert ok
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), np.array([215.0, 55, 355, 330])


@pytest.fixture(scope="module")
def pose_model():
    return PoseEstimator(CFG.pose)


@pytest.fixture(scope="module")
def sam_model():
    return SamSegmenter(CFG.sam)


def test_rotation_tta_on_an_upside_down_skater(skater, pose_model):
    rgb, box = skater
    up, _ = pose_model(rgb, box, (0,))
    assert up[0, :, 2].mean() > 0.8
    flipped = np.ascontiguousarray(np.rot90(rgb, 2))
    kps, means = pose_model(flipped, rot90_box(box, 2, rgb.shape[:2]), CFG.rotations)
    j = choose_rotation(means, CFG.rotations, CFG.upright_margin)
    # ViTPose is about as sure of a sideways skater as of an upright one, but not of an upside-down
    # one: TTA keeps an upright or sideways view, whose keypoints match the upright estimate
    assert CFG.rotations[j] != 0 and means[j] > means[0] + 0.15
    np.testing.assert_allclose(kps[j, :, :2], rot90_points(up[0, :, :2], 2, rgb.shape[:2]), atol=8.0)


def test_sam_point_labels(skater, pose_model, sam_model):
    rgb, box = skater
    kps = pose_model(rgb, box, (0,))[0][0]
    x0, y0, x1, y1 = prompt_region(box, rgb.shape[:2])
    image = sam_model.embed(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
    torso, head = torso_centre(kps, CFG.kp_score) - [x0, y0], kps[0, :2] - [x0, y0]
    tx, ty = np.round(torso).astype(int)
    hx, hy = np.round(head).astype(int)
    logits, iou, obj = sam_model.decode(image, Prompt(torso[None], np.array([1]), box - [x0, y0, x0, y0], "pose"))
    assert logits[ty, tx] > 0 and logits[hy, hx] > 0 and obj > 0 and iou > 0.5  # the whole skater
    both = np.stack([torso, head])
    logits, _, _ = sam_model.decode(image, Prompt(both, np.array([1, 1]), None, "pose"))
    assert logits[ty, tx] > 0 and logits[hy, hx] > 0
    logits, _, _ = sam_model.decode(image, Prompt(both, np.array([1, 0]), None, "pose"))
    assert logits[ty, tx] > 0 and logits[hy, hx] < 0  # label 0 excludes the head


def test_segment_person_on_a_real_frame(skater, pose_model, sam_model):
    rgb, box = skater
    kps = pose_model(rgb, box, (0,))[0][0]
    res = segment_person(rgb, box, kps, sam_model, CFG)
    assert res.source == "sam_pose" and res.stats["cover"] == 1.0
    assert 0.2 < res.stats["fill"] < 0.7
    x0, y0, _, _ = res.region
    ys, xs = np.nonzero(res.mask)
    assert ys.min() + y0 < kps[0, 1] and ys.max() + y0 > kps[15:17, 1].max()  # head to skates
