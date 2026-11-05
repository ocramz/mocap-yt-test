"""Frames, projection and pose variants (no model weights needed)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from mocap_yt.geometry import (
    MZ,
    from_camera,
    mirror_perm,
    procrustes_2d,
    project,
    rx,
    ry,
    rz,
    spin,
    tilt,
    to_camera,
    twin,
    unrotate,
)
from mocap_yt.person_sam import rot90_points
from mocap_yt.skeleton import forward_kinematics, mixamo_humanoid

SK = mixamo_humanoid()
J = SK.num_joints


def random_pose(n=4, seed=0, spread=0.6):
    rot = R.from_rotvec(np.random.default_rng(seed).normal(0, spread, (n * J, 3))).as_matrix().reshape(n, J, 3, 3)
    return rot


def fk(rot):
    return forward_kinematics(SK, rot, np.zeros((len(rot), 3)))[1]


def test_elementary_rotations():
    for r in (rx(33), ry(-71), rz(140)):
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0)
    np.testing.assert_allclose(rz(90) @ [1, 0, 0], [0, 1, 0], atol=1e-12)  # x -> y: counter-clockwise, y up
    np.testing.assert_allclose(ry(90) @ [0, 0, 1], [1, 0, 0], atol=1e-12)  # +Z (facing camera) -> +X


def test_camera_frame_round_trip_and_pitch():
    rot, pos = random_pose(), np.random.default_rng(1).normal(size=(4, J, 3))
    r2, p2 = from_camera(*to_camera(rot, pos))
    np.testing.assert_allclose(r2, rot, atol=1e-12)
    np.testing.assert_allclose(p2, pos, atol=1e-12)
    # a standing figure seen by a camera looking 5.3 degrees down: its head is nearer than its feet
    rest = np.tile(np.eye(3), (1, J, 1, 1))
    _, p = to_camera(rest, fk(rest))
    assert p[0, SK.index("HeadTop_End"), 2] > p[0, SK.index("LeftFoot"), 2]


def test_twin_projects_identically_with_left_right_swapped():
    rot = random_pose()
    tw = twin(rot)
    perm = mirror_perm(J)
    p, pt = fk(rot), fk(tw)
    np.testing.assert_allclose(pt, (p @ MZ)[:, perm], atol=1e-9)  # z mirrored, labels swapped
    np.testing.assert_allclose(project(pt, 100, 0, 0), project(p, 100, 0, 0)[:, perm], atol=1e-7)
    np.testing.assert_allclose(twin(tw), rot, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(tw), 1.0, atol=1e-9)


@pytest.mark.parametrize("variant", [spin, tilt])
def test_root_variants_are_rotations_and_involutions(variant):
    rot = random_pose()
    v = variant(rot)
    np.testing.assert_allclose(np.linalg.det(v), 1.0, atol=1e-9)
    np.testing.assert_allclose(variant(v), rot, atol=1e-12)
    np.testing.assert_allclose(v[:, 1:], rot[:, 1:])  # only the root changes


@pytest.mark.parametrize("k", [1, 2, 3])
def test_rot90_image_is_rz_of_the_scene(k):
    """Projecting Rz(90 k)-rotated points equals rotating the projected image with np.rot90."""
    W = 200.0
    rot = random_pose(1)
    p = fk(rot)[0] * 60
    uv = project(p, 1.0, W / 2, W / 2)
    rotated_image_uv = rot90_points(uv, k, (W, W))
    np.testing.assert_allclose(project(p @ rz(90 * k).T, 1.0, W / 2, W / 2), rotated_image_uv, atol=1e-9)
    # a prediction made on the rotated image comes back with unrotate
    seen = rot.copy()
    seen[:, 0] = rz(90 * k) @ rot[:, 0]
    r_back, p_back = unrotate(seen, fk(seen), k)
    np.testing.assert_allclose(r_back, rot, atol=1e-9)
    np.testing.assert_allclose(p_back, fk(rot), atol=1e-9)


def test_procrustes_recovers_similarity():
    rng = np.random.default_rng(3)
    src = rng.normal(size=(12, 2))
    a = np.radians(37)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    dst = 2.5 * src @ rot.T + [4.0, -1.0]
    s, ang, t, res = procrustes_2d(src, dst)
    assert s == pytest.approx(2.5) and ang == pytest.approx(37) and res < 1e-9
    np.testing.assert_allclose(t, [4.0, -1.0], atol=1e-9)
    s, ang, t, res = procrustes_2d(src, dst * [-1, 1])  # a mirror image is not a rotation
    assert res > 0.1
