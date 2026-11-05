"""2D keypoint track clean-up (no model weights needed)."""

import numpy as np

from mocap_yt.metrics import track_metrics
from mocap_yt.track2d import SWAPS, clean_track, swap_labels, torso_length, viterbi

FRONT = np.array([  # an upright person facing the camera, ~150 px tall (x, y), y down
    [100, 20], [104, 17], [96, 17], [108, 20], [92, 20],
    [125, 45], [75, 45], [130, 75], [70, 75], [132, 105], [68, 105],
    [115, 100], [85, 100], [116, 135], [84, 135], [117, 168], [83, 168]], float)


def rotate(xy, deg, centre):
    a = np.radians(deg)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (xy - centre) @ r.T + centre


def track(T=30, seed=0):
    """True keypoints drifting right, and 4 'views': the truth plus three rotated misreads."""
    rng = np.random.default_rng(seed)
    truth = np.stack([FRONT + [2.0 * t, 0] for t in range(T)])
    views = np.zeros((T, 4, 17, 3))
    for r, deg in enumerate((0, 90, 180, 270)):
        c = truth.mean(1, keepdims=True)
        views[:, r, :, :2] = np.stack([rotate(truth[t], deg, c[t, 0]) for t in range(T)]) + rng.normal(0, 1, (T, 17, 2))
        views[:, r, :, 2] = 0.80 if r == 0 else 0.78
    return truth, views


def test_viterbi_picks_cheapest_path():
    unary = np.array([[0, 5], [5, 0], [0, 5]], float)
    switch = np.array([[0, 1], [1, 0]], float)
    path, cost = viterbi(unary, lambda t: switch)
    assert path.tolist() == [0, 1, 0] and cost == 2
    path, _ = viterbi(unary, lambda t: switch * 10)  # switching too dear: stay
    assert path.tolist() in ([0, 0, 0], [1, 1, 1])


def test_swap_labels_and_torso_length():
    k = np.concatenate([FRONT, np.ones((17, 1))], 1)
    s = swap_labels(k, SWAPS["all"])
    assert (s[5] == k[6]).all() and (s[0] == k[0]).all() and (swap_labels(s, SWAPS["all"]) == k).all()
    assert torso_length(k[None]) == np.linalg.norm([0, 55])


def test_clean_track_removes_view_switches_and_label_flicker():
    truth, views = track()
    views[[5, 12, 20], 2, :, 2] = 0.83  # the upside-down misread wins these frames on score
    views[8:10, 0] = swap_labels(views[8:10, 0], SWAPS["all"])  # two frames with left/right swapped
    naive = views[np.arange(len(views)), views[..., 2].mean(-1).argmax(1)]
    assert track_metrics(naive)["torso_flips"] == 6  # into and out of each misread frame
    assert track_metrics(naive)["lr_swaps"] == 8  # those, plus into and out of the relabelled pair
    kps, choice = clean_track(views)
    assert (choice[:, 0] == 0).all()
    assert list(SWAPS)[choice[8, 1]] == "all" and list(SWAPS)[choice[0, 1]] == "none"
    np.testing.assert_allclose(kps[..., :2], truth, atol=5)
    t = track_metrics(kps)
    assert t["torso_flips"] == 0 and t["lr_swaps"] == 0


def test_clean_track_keeps_a_genuine_turn():
    """The person really turns upside down mid-clip (all views agree): no reason to fight it."""
    truth, views = track(T=20)
    for t in range(10, 20):
        views[t, :, :, :2] = views[t, [2, 3, 0, 1], :, :2]  # from frame 10 on, view 2 reads the truth
    views[10:, 2, :, 2], views[10:, 0, :, 2] = 0.80, 0.78
    kps, choice = clean_track(views)
    assert choice[:10, 0].tolist() == [0] * 10 and choice[10:, 0].tolist() == [2] * 10
