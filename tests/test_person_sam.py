"""Pose-prompted SAM matting helpers (no model weights needed)."""

import numpy as np
import pytest

from mocap_yt.person_sam import (
    NoPersonMask,
    PersonPoses,
    PersonSamConfig,
    Prompt,
    build_prompt,
    choose_rotation,
    clean_mask,
    fill_small_holes,
    is_plausible,
    mask_stats,
    matting_tag,
    prompt_region,
    rot90_box,
    rot90_points,
    segment_person,
    unrot90_points,
)

CFG = PersonSamConfig()


def standing_pose(x=100.0, y=40.0, s=1.0, score=0.9):
    """COCO-17 keypoints of an upright person ~160*s px tall, head at (x, y)."""
    xy = np.array([
        [0, 0], [-4, -3], [4, -3], [-8, 0], [8, 0],  # nose, eyes, ears
        [-20, 25], [20, 25], [-26, 60], [26, 60], [-28, 90], [28, 90],  # shoulders, elbows, wrists
        [-12, 85], [12, 85], [-13, 125], [13, 125], [-14, 160], [14, 160],  # hips, knees, ankles
    ], float) * s + [x, y]
    return np.concatenate([xy, np.full((17, 1), score)], 1)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_rot90_points_follow_np_rot90(k):
    rng = np.random.default_rng(k)
    img = np.zeros((7, 11), int)
    for n, (r, c) in enumerate(zip(rng.integers(0, 7, 5), rng.integers(0, 11, 5)), start=1):
        img[r, c] = n
    rot = np.rot90(img, k)
    for n in np.unique(img[img > 0]):
        r, c = np.argwhere(img == n)[0]
        r2, c2 = np.argwhere(rot == n)[0]
        np.testing.assert_allclose(rot90_points([c + 0.5, r + 0.5], k, img.shape), [c2 + 0.5, r2 + 0.5])
    pts = rng.uniform(0, 7, (10, 2))
    np.testing.assert_allclose(unrot90_points(rot90_points(pts, k, img.shape), k, img.shape), pts)


def test_rot90_box_contains_the_rotated_pixels():
    img = np.zeros((20, 30), bool)
    img[3:8, 10:25] = True
    for k in range(4):
        rot = np.rot90(img, k)
        ys, xs = np.nonzero(rot)
        np.testing.assert_allclose(rot90_box([10, 3, 25, 8], k, img.shape), [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])


def test_choose_rotation_prefers_upright_within_margin():
    rots = (0, 90, 180, 270)
    assert choose_rotation([0.70, 0.73, 0.2, 0.1], rots, 0.05) == 0
    assert choose_rotation([0.50, 0.30, 0.80, 0.1], rots, 0.05) == 2
    assert choose_rotation([0.5, 0.9], (90, 180), 0.05) == 1  # no upright candidate: plain best


def test_prompt_region_is_square_inside_the_frame():
    assert prompt_region((300, 100, 340, 180), (360, 640)) == (240, 60, 400, 220)  # 2 x 80 px, centred
    x0, y0, x1, y1 = prompt_region((0, 0, 20, 30), (360, 640))  # tiny, at the corner: min side, shifted in
    assert (x0, y0, x1 - x0, y1 - y0) == (0, 0, 160, 160)
    x0, y0, x1, y1 = prompt_region((600, 300, 640, 360), (360, 640), min_side=100)
    assert (x1, y1) == (640, 360) and x1 - x0 == y1 - y0 == 120
    assert prompt_region((100, 0, 400, 300), (360, 640)) == (0, 0, 600, 360)  # taller than the frame: clipped


def test_build_prompt_points_inside_the_body():
    kps = standing_pose()
    box = np.array([70, 30, 130, 205.0])
    p = build_prompt(kps, box, CFG)
    assert p.kind == "pose" and 3 <= len(p.points) <= CFG.max_points and (p.labels == 1).all()
    # torso centre, head, shins next to the ankles, forearms next to the wrists, knees
    torso = kps[[5, 6, 11, 12], :2].mean(0)
    assert np.isclose(p.points, torso).all(1).any()
    shin = kps[15, :2] + 0.2 * (kps[13, :2] - kps[15, :2])
    assert np.isclose(p.points, shin).all(1).any()
    assert not np.isclose(p.points, kps[15, :2]).all(1).any()  # not the ankle itself (on the outline)
    # the box holds the tracked box and every keypoint, plus a margin
    assert (p.box[:2] <= np.minimum(box[:2], kps[:, :2].min(0))).all()
    assert (p.box[2:] >= np.maximum(box[2:], kps[:, :2].max(0))).all()


def test_build_prompt_drops_unsure_and_duplicate_points():
    kps = standing_pose()
    kps[[9, 10], 2] = 0.45  # wrists below kp_score_distal -> elbows instead
    kps[15:17, 2] = 0.1  # ankles unsure -> no shin points
    p = build_prompt(kps, [70, 30, 130, 205], CFG)
    assert np.isclose(p.points, kps[7, :2]).all(1).any() and np.isclose(p.points, kps[8, :2]).all(1).any()
    assert not np.isclose(p.points, kps[9, :2]).all(1).any()
    d = np.linalg.norm(p.points[:, None] - p.points[None], axis=-1) + np.eye(len(p.points)) * 1e9
    assert d.min() >= 0.05 * 175
    p = build_prompt(kps, [70, 30, 130, 205], PersonSamConfig(max_points=4))
    assert len(p.points) == 4


def test_build_prompt_falls_back_to_the_box():
    box = [70, 30, 130, 205]
    kps = standing_pose()
    kps[:, 2] = 0.1  # nobody is sure
    p = build_prompt(kps, box, CFG)
    assert p.kind == "box" and len(p.points) == 0
    np.testing.assert_allclose(p.box, box)
    assert build_prompt(None, box, CFG).kind == "box"
    tiny = build_prompt(standing_pose(s=0.15), [95, 38, 105, 65], CFG)  # 27 px tall < min_person
    assert tiny.kind == "box" and len(tiny.points) == 0


def test_fill_small_holes_keeps_the_gap_between_the_legs():
    m = np.zeros((60, 60), bool)
    m[5:55, 10:50] = True
    m[20:23, 20:23] = False  # 9 px hole in the torso
    m[35:55, 25:35] = False  # gap between the legs: open to the border
    m[8:18, 30:45] = False  # a large enclosed hole (150 px)
    out = fill_small_holes(m, 50)
    assert out[20:23, 20:23].all() and not out[35:55, 25:35].any() and not out[8:18, 30:45].any()


def test_clean_mask_keeps_the_person_and_drops_a_bar():
    kps = standing_pose(x=100, y=40)
    p = build_prompt(kps, [70, 30, 130, 205], CFG)
    m = np.zeros((240, 260), bool)
    m[40:200, 85:115] = True  # body
    m[125:130, 0:260] = True  # a wall bar crossing the whole region, 5 px thick
    m[128:136, 120:132] = True  # a blurred hand, detached, on the right wrist keypoint
    m[10:20, 220:240] = True  # something else, far away
    m[60:110, 72:76] = True  # a thin (4 px) blurred forearm along the left arm
    out = clean_mask(m, kps, p, 175, CFG)
    assert out[100, 100] and out[130, 126]  # body and the detached hand
    assert not out[127, 5] and not out[127, 250]  # the bar
    assert not out[15, 230]  # the far blob
    assert out[90, 74]  # the thin forearm, protected by the stick figure


def test_mask_stats_and_plausibility():
    p = Prompt(np.array([[10, 10], [30, 30], [50, 50.0]]), np.ones(3, int), np.array([0, 0, 60, 60.0]), "pose")
    m = np.zeros((60, 60), bool)
    m[5:35, 5:35] = True
    st = mask_stats(m, p)
    assert st["cover"] == pytest.approx(2 / 3, abs=1e-3) and st["fill"] == pytest.approx(900 / 3600, abs=1e-3)
    assert not is_plausible(st, 0.9, 1.0, "pose", CFG)  # a prompt point is outside the mask
    assert is_plausible(st, 0.9, 1.0, "box", CFG)
    m[45:55, 45:55] = True
    st = mask_stats(m, p)
    assert is_plausible(st, 0.9, 1.0, "pose", CFG)
    assert not is_plausible(st, 0.3, 1.0, "pose", CFG)  # SAM is unsure
    assert not is_plausible(st, 0.9, -1.0, "pose", CFG)  # no object
    full = mask_stats(np.ones((60, 60), bool), p)
    assert not is_plausible(full, 0.9, 1.0, "pose", CFG)  # fills the whole box: leaked


class FakeSam:
    """Scripted segmenter: returns the given masks in turn, records the calls."""

    def __init__(self, *masks):
        self.masks, self.embeds, self.prompts = list(masks), 0, []

    def embed(self, rgb):
        self.embeds += 1
        self.shape = rgb.shape[:2]
        return "image"

    def decode(self, image, prompt):
        self.prompts.append(prompt)
        m = self.masks.pop(0)(self.shape, prompt)
        return np.where(m, 5.0, -5.0), 0.9, 2.0


def body(shape, prompt):
    """A filled ellipse around the prompt's points (pose) or inside its box."""
    m = np.zeros(shape, bool)
    pts = prompt.points if prompt.kind == "pose" else prompt.box.reshape(2, 2) + [[3, 3], [-3, -3]]
    x0, y0 = pts.min(0) - 3
    x1, y1 = pts.max(0) + 3
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    m[((xx - (x0 + x1) / 2) / ((x1 - x0) / 2)) ** 2 + ((yy - (y0 + y1) / 2) / ((y1 - y0) / 2)) ** 2 <= 1] = True
    return m


def empty(shape, prompt):
    return np.zeros(shape, bool)


def test_segment_person_pose_then_box_then_give_up():
    rgb = np.zeros((360, 640, 3), np.uint8)
    kps, box = standing_pose(x=300, y=80, s=0.6), np.array([280, 74, 320, 180.0])  # region fits the frame
    sam = FakeSam(body)
    res = segment_person(rgb, box, kps, sam, CFG)
    assert res.source == "sam_pose" and sam.embeds == 1 and res.mask.any()
    x0, y0, x1, y1 = res.region
    assert res.mask.shape == (y1 - y0, x1 - x0) and x1 - x0 == y1 - y0
    sam = FakeSam(empty, body)
    res = segment_person(rgb, box, kps, sam, CFG)
    assert res.source == "sam_box" and sam.embeds == 1 and len(res.attempts) == 2
    assert sam.prompts[1].kind == "box"
    with pytest.raises(NoPersonMask):
        segment_person(rgb, box, kps, FakeSam(empty, empty), CFG)
    sam = FakeSam(empty)  # unsure pose: only the box is tried
    with pytest.raises(NoPersonMask):
        segment_person(rgb, box, None, sam, CFG)
    assert len(sam.prompts) == 1


def test_config_tags_and_rotations():
    assert matting_tag("salient", CFG) == matting_tag("person", CFG) == ""
    assert matting_tag("person_sam", CFG) == CFG.tag() != PersonSamConfig(max_points=4).tag()
    assert CFG.pose_tag() == PersonSamConfig(max_points=4).pose_tag() != PersonSamConfig(rotations=[0]).pose_tag()
    assert PersonSamConfig(rotations=[0, 180]).rotations == (0, 180)
    with pytest.raises(ValueError, match="multiples of 90"):
        PersonSamConfig(rotations=[45])


def test_person_poses_cache(tmp_path):
    n = 3
    poses = PersonPoses(np.ones((n, 4)), np.zeros((n, 17, 3)), np.array([0, 180, 0]), np.zeros((n, 4)))
    path = tmp_path / "poses.npz"
    poses.save(path, "a")
    assert PersonPoses.load(path, "a", n).rotation.tolist() == [0, 180, 0]
    assert PersonPoses.load(path, "b", n) is None  # other settings
    assert PersonPoses.load(path, "a", n + 1) is None  # other clip
    assert PersonPoses.load(tmp_path / "missing.npz", "a", n) is None
