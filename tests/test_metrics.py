"""Capture quality measures (no model weights needed)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from mocap_yt.bvh import Motion
from mocap_yt.metrics import (
    body_frame,
    floor_contact,
    geodesic_deg,
    hinge_violations,
    jitter_mm,
    motion_metrics,
    root_motion,
    track_metrics,
)
from mocap_yt.skeleton import mixamo_humanoid

SK = mixamo_humanoid()


def spin(T=60, deg_per_frame=20.0):
    """The rest pose spinning about the vertical at a steady rate, grounded."""
    rot = np.tile(np.eye(3), (T, SK.num_joints, 1, 1))
    rot[:, 0] = R.from_euler("y", np.arange(T)[:, None] * deg_per_frame, degrees=True).as_matrix()
    root = np.tile(SK.offsets[0], (T, 1))
    return Motion(SK, rot, root, 15.0)


def test_geodesic():
    a = R.from_euler("x", [[0], [30], [180]], degrees=True).as_matrix()
    np.testing.assert_allclose(geodesic_deg(a[0], a), [0, 30, 180], atol=1e-6)


def test_steady_spin_is_no_flip_and_no_jitter():
    m = spin()
    r = root_motion(m.local_rot[:, 0], 15.0)
    assert r["flips"] == 0 and r["blips"] == 0
    assert r["root_speed_deg_s"][0] == pytest.approx(300.0, abs=0.5) and r["root_acc_deg_s2"][2] < 1
    assert jitter_mm(body_frame(SK, m), 15.0)[2] < 1e-6  # the pose itself is still


def test_flips_and_blips():
    turn = R.from_euler("y", 180, degrees=True).as_matrix()
    root = spin().local_rot[:, 0].copy()
    root[30] = root[30] @ turn  # one frame turned around, then back
    r = root_motion(root, 15.0)
    assert r["flips"] == 1 and r["blips"] == 1
    root = spin().local_rot[:, 0].copy()
    root[30:] = root[30:] @ R.from_euler("x", 150, degrees=True).as_matrix()  # a lasting flip
    r = root_motion(root, 15.0)
    assert r["flips"] == 1 and r["blips"] == 0
    assert root_motion(np.repeat(root, 2, axis=0), 30.0)["flips"] == 1  # same flip at 30 fps: still one


def test_floor_and_hinges():
    m = spin(T=5)
    m.root_pos[:, 1] -= 0.05  # 5 cm too low: the feet go under
    f = floor_contact(SK, m)
    assert f["floor_penetration_cm"] > 4 and f["frames_below_2cm"] == 1.0
    m.local_rot[:, SK.index("LeftLeg")] = R.from_euler("x", -30, degrees=True).as_matrix()  # knee bent forward
    h = hinge_violations(SK, m)
    assert h["LeftLeg"]["hyperextended"] == 1.0 and h["RightLeg"]["hyperextended"] == 0.0
    assert motion_metrics(m, SK)["frames"] == 5


def standing(T=10):
    k = np.zeros((T, 17, 3))
    k[:, :, 2] = 0.9
    k[:, 5, :2], k[:, 6, :2] = (120, 50), (80, 50)  # left shoulder on image right (facing camera)
    k[:, 11, :2], k[:, 12, :2] = (112, 120), (88, 120)
    return k


def test_track_metrics_counts_flips_and_swaps():
    k = standing()
    assert track_metrics(k) == {"pairs": 9, "torso_flips": 0, "lr_swaps": 0, "lr_pairs": 9}
    k[4, :, 1] = 170 - k[4, :, 1]  # frame 4 upside down (head<->feet)
    k[7, [5, 6]] = k[7, [6, 5]]  # frame 7: left/right shoulder labels swapped
    k[7, [11, 12]] = k[7, [12, 11]]
    t = track_metrics(k)
    assert t["torso_flips"] == 2 and t["lr_swaps"] == 2
