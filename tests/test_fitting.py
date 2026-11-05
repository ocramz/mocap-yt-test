"""Stage B hypothesis selection and IK (no model weights needed)."""

import numpy as np
from scipy.spatial.transform import Rotation as R

from mocap_yt.fitting import (
    COCO_TO_RIG,
    FitConfig,
    SelectConfig,
    fit_motion,
    ik_from_joints,
    rig_joints_dict,
    select_hypotheses,
)
from mocap_yt.geometry import project, spin, twin
from mocap_yt.metrics import root_motion
from mocap_yt.skeleton import forward_kinematics, mixamo_humanoid

SK = mixamo_humanoid()
J = SK.num_joints


def fk(rot):
    return forward_kinematics(SK, rot, np.zeros((len(rot), 3)))[1]


def dancer(T=40, deg=20.0, seed=0):
    """A pose with bent limbs spinning about the vertical at ``deg`` per frame (camera frame),
    with some arm/leg motion."""
    rng = np.random.default_rng(seed)
    rot = np.tile(np.eye(3), (T, J, 1, 1))
    t = np.arange(T)[:, None]
    for name, axis, amp in (("LeftUpLeg", "x", 40), ("RightUpLeg", "x", -30), ("LeftLeg", "x", 50), ("RightLeg", "x", 70),
                            ("LeftArm", "z", -40), ("RightArm", "z", 30), ("LeftForeArm", "y", -60), ("RightForeArm", "y", 50)):
        ang = amp * (1 + 0.3 * np.sin(0.2 * t + rng.uniform(0, 6)))
        rot[:, SK.index(name)] = R.from_euler(axis, ang, degrees=True).as_matrix()
    rot[:, 0] = R.from_euler("y", t * deg, degrees=True).as_matrix() @ R.from_euler("x", 20, degrees=True).as_matrix()
    return rot


def keypoints(rot, s=200.0, c=(320.0, 180.0)):
    """COCO keypoints (T, 17, 3) of the rig's joints, projected (labels as they truly are)."""
    uv = project(fk(rot), s, *c)
    k = np.zeros((len(rot), 17, 3))
    for kp, j in COCO_TO_RIG.items():
        k[:, kp, :2], k[:, kp, 2] = uv[:, j], 0.9
    return k


def test_ik_recovers_joint_positions():
    rot = dancer(T=5)
    p = fk(rot)
    q = fk(ik_from_joints(SK, rig_joints_dict(SK, p)))
    for name in ("LeftLeg", "RightFoot", "LeftForeArm", "RightHand", "HeadTop_End", "LeftArm"):
        np.testing.assert_allclose(q[:, SK.index(name)], p[:, SK.index(name)], atol=1e-6)


def test_ik_aims_bones_when_the_spine_bends():
    """With a bent spine the pelvis pitch is not recoverable from joints (the IK puts the whole
    bend between hips and chest), but every limb still points the right way."""
    rot = dancer(T=5)
    for name in ("Spine", "Spine1", "Spine2"):
        rot[:, SK.index(name)] = R.from_euler("x", 10, degrees=True).as_matrix()
    p = fk(rot)
    q = fk(ik_from_joints(SK, rig_joints_dict(SK, p)))
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    for a, b in (("LeftUpLeg", "LeftLeg"), ("RightLeg", "RightFoot"), ("LeftArm", "LeftForeArm"), ("RightForeArm", "RightHand")):
        ia, ib = SK.index(a), SK.index(b)
        np.testing.assert_allclose(unit(q[:, ib] - q[:, ia]), unit(p[:, ib] - p[:, ia]), atol=1e-6)


def test_selection_removes_spin_and_twin_flips():
    truth = dancer()
    noisy = truth.copy()
    for t in (7, 8, 19, 30):
        noisy[t] = spin(truth[t])  # MocapAnything turned the body around on these frames
    for t in (12, 25):
        noisy[t] = twin(truth[t])
    assert root_motion(noisy[:, 0], 15.0)["flips"] >= 4
    sel = select_hypotheses(SK, {"ma_k0": noisy}, keypoints(truth), 15.0, SelectConfig())
    np.testing.assert_allclose(sel.rot, truth, atol=1e-9)
    assert root_motion(sel.rot[:, 0], 15.0)["flips"] == 0
    assert {sel.names[i] for i in sel.choice[[7, 12, 25]]} == {"ma_k0/spin", "ma_k0/twin"}


def test_selection_keeps_a_steady_fast_spin():
    truth = dancer(deg=40.0)  # 600 deg/s: fast, but steady
    sel = select_hypotheses(SK, {"ma_k0": truth}, keypoints(truth), 15.0, SelectConfig())
    assert all(sel.names[i] == "ma_k0/as_is" for i in sel.choice)


def test_fit_motion_recovers_a_noisy_capture():
    truth = dancer(T=30)
    rng = np.random.default_rng(5)
    noise = R.from_rotvec(rng.normal(0, np.radians(8), (30 * J, 3))).as_matrix().reshape(30, J, 3, 3)
    init = truth @ noise  # every joint jittered by ~8 degrees
    kps = keypoints(truth)
    fit = fit_motion(SK, init, kps, 15.0, FitConfig())
    uv = project(fk(fit.rot), fit.camera[:, :1], fit.camera[:, 1:2], fit.camera[:, 2:3])

    def err2d(rot_uv):
        return np.mean([np.linalg.norm(rot_uv[:, j] - kps[:, k, :2], axis=-1).mean() for k, j in COCO_TO_RIG.items()])

    best = project(fk(init), 200.0, 320.0, 180.0)  # the noisy capture with the true camera
    assert err2d(uv) < 0.4 * err2d(best)
    np.testing.assert_allclose(fit.camera[:, 0], 200.0, rtol=0.1)
    from mocap_yt.metrics import body_frame, jitter_mm
    from mocap_yt.bvh import Motion
    jit = lambda r: jitter_mm(body_frame(SK, Motion(SK, r, np.zeros((30, 3)), 15.0)), 15.0)[0]
    assert jit(fit.rot) < 0.5 * jit(init)


def test_ground_surface_keeps_the_body_above_the_floor():
    from mocap_yt.refine import ground_surface

    rot = dancer(T=30)
    root = ground_surface(SK, rot, 15.0)
    _, p = forward_kinematics(SK, rot, root)
    low = (p[..., 1] - SK.radii).min(1)
    assert low.min() > -0.0101 and np.median(np.abs(low)) < 0.02  # at most 1 cm under, and resting on it
    assert (root[:, [0, 2]] == 0).all()
