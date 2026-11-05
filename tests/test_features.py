"""Person-guided matting helpers (no model weights needed)."""

import numpy as np
import pytest

from mocap_yt.features import box_iou, select_components, track_subject


def test_box_iou():
    a = np.array([0, 0, 10, 10.0])
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [5, 0, 15, 10]], float)
    np.testing.assert_allclose(box_iou(a, b), [1, 0, 1 / 3])


def det(x0, y0, x1, y1, score):
    return [x0, y0, x1, y1, score]


def test_track_follows_one_person_through_gaps():
    """A skater grows as they approach; a bystander stands still; two frames have no detection."""
    T = 8
    target = [np.array([100 + 4 * t, 50 - 2 * t, 120 + 8 * t, 90 + 6 * t], float) for t in range(T)]
    dets = []
    for t in range(T):
        rows = [det(10, 60, 18, 80, 0.95)]  # bystander: confident but small and static
        if t not in (3, 4):
            rows.append(det(*target[t], 0.9 if t == 7 else 0.6))
        dets.append(np.array(rows, float))
    boxes = track_subject(dets)
    for t in (0, 1, 2, 5, 6, 7):
        np.testing.assert_allclose(boxes[t], target[t])
    # gap frames are interpolated between frames 2 and 5
    np.testing.assert_allclose(boxes[3], target[2] + (target[5] - target[2]) / 3)
    np.testing.assert_allclose(boxes[4], target[2] + 2 * (target[5] - target[2]) / 3)


def test_track_holds_box_at_the_ends_and_handles_empty():
    dets = [np.zeros((0, 5)), np.array([det(0, 0, 10, 20, 0.9)]), np.zeros((0, 5))]
    np.testing.assert_allclose(track_subject(dets), [[0, 0, 10, 20]] * 3)
    assert track_subject([np.zeros((0, 5))] * 4) is None


def test_select_components_keeps_the_person_only():
    lab = np.zeros((100, 200), int)
    lab[30:80, 90:110] = 1   # body, inside the box
    lab[40:45, 60:90] = 2    # arm sticking out of the box (separate blob): mostly outside -> dropped
    lab[20:24, 0:200] = 3    # wall bar crossing the box: mostly outside -> dropped
    lab[10:30, 150:170] = 4  # someone else, outside the box
    lab[60:70, 95:105] = 5   # blob fully inside the box
    keep = select_components(lab, (85, 15, 115, 85))
    assert set(np.unique(lab[keep])) == {1, 5}


def test_select_components_keeps_best_overlap_even_if_mostly_outside():
    """A torso-only detection box must not cut off a leg that is part of the same blob."""
    lab = np.zeros((50, 100), int)
    lab[20:30, 10:90] = 1  # body + outstretched leg: 80 px wide, box covers 20 px of it
    keep = select_components(lab, (40, 15, 60, 35))
    assert keep.sum() == (lab == 1).sum()
    assert not select_components(lab, (0, 40, 5, 50)).any()  # nothing in the box -> empty


@pytest.mark.parametrize("box", [(-10, -10, 5, 5), (95, 45, 120, 70)])
def test_select_components_clips_boxes_at_the_border(box):
    lab = np.ones((50, 100), int)
    assert select_components(lab, box).all()
