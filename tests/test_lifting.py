"""COCO -> Human3.6M conversion and MotionBERT conventions (the lifting itself needs the weights)."""

import numpy as np
import pytest

from mocap_yt.lifting import H36M, coco_to_h36m, flip_h36m, normalise, to_camera

FRONT = np.array([  # an upright person facing the camera (x, y px), y down: their left is on the right
    [100, 20], [104, 17], [96, 17], [108, 20], [92, 20],
    [125, 45], [75, 45], [130, 75], [70, 75], [132, 105], [68, 105],
    [115, 100], [85, 100], [116, 135], [84, 135], [117, 168], [83, 168]], float)


def coco(conf=0.9):
    return np.concatenate([FRONT, np.full((17, 1), conf)], 1)[None]


def test_coco_to_h36m_layout():
    h = coco_to_h36m(coco())[0]
    j = {n: h[i, :2] for i, n in enumerate(H36M)}
    np.testing.assert_allclose(j["pelvis"], [100, 100])
    np.testing.assert_allclose(j["thorax"], [100, 45])
    np.testing.assert_allclose(j["spine"], [100, 72.5])
    np.testing.assert_allclose(j["l_shoulder"], FRONT[5])
    np.testing.assert_allclose(j["r_ankle"], FRONT[16])
    assert j["head"][1] < j["nose"][1] < j["thorax"][1]  # head top above the face
    assert (h[:, 2] == 0.9).all()


def test_normalise_and_flip():
    x, scale, centre = normalise(coco_to_h36m(np.repeat(coco(1.2), 5, 0)))
    assert np.abs(x[..., :2]).max() == pytest.approx(1.0) and x[..., 2].max() == 1.0  # scores clipped
    np.testing.assert_allclose(x[..., :2] * scale + centre, coco_to_h36m(coco())[0, :, :2][None].repeat(5, 0))
    f = flip_h36m(x)
    np.testing.assert_allclose(flip_h36m(f), x)
    np.testing.assert_allclose(f[:, 11, 0], -x[:, 14, 0])  # left shoulder <- mirrored right shoulder


def test_camera_axes_are_a_rotation():
    m = np.diag([1.0, -1.0, -1.0])
    assert np.linalg.det(m) == 1.0
    p = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(to_camera(p), p @ m.T)
