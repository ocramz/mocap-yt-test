import numpy as np
import pygltflib
import pytest
from scipy.spatial.transform import Rotation as R

from mocap_yt import upstream
from mocap_yt.bvh import Motion, read_bvh, write_bvh
from mocap_yt.gltf import bone_mesh, export_gltf
from mocap_yt.mesh import build_mannequin, skin
from mocap_yt.rig_assets import RigSpec, calibration_motion, yaw_motion
from mocap_yt.skeleton import forward_kinematics, mixamo_humanoid


@pytest.fixture(scope="module")
def skel():
    return mixamo_humanoid()


def random_motion(skel, T=7, seed=0):
    rng = np.random.default_rng(seed)
    rot = R.from_rotvec(rng.normal(scale=0.6, size=(T * skel.num_joints, 3))).as_matrix().reshape(T, -1, 3, 3)
    root = skel.offsets[0] + rng.normal(scale=0.2, size=(T, 3))
    return Motion(skel, rot, root, 30.0)


def test_skeleton_shape(skel):
    assert skel.num_joints == 25
    pos = skel.rest_positions()
    assert abs(skel.height() - 1.75) < 0.05
    left = [i for i, n in enumerate(skel.names) if n.startswith("Left")]
    assert (pos[left, 0] > 0).all(), "character's left must be +X"
    toe = skel.index("LeftToe_End")
    assert pos[toe, 2] > 0.1, "feet must point to +Z (facing direction)"


def test_bvh_roundtrip_and_upstream_fk(tmp_path, skel):
    m = random_motion(skel)
    p = write_bvh(tmp_path / "m.bvh", m)
    back = read_bvh(p)
    assert back.skeleton.names == skel.names
    np.testing.assert_allclose(back.local_rot, m.local_rot, atol=1e-5)
    np.testing.assert_allclose(back.root_pos, m.root_pos, atol=1e-5)

    # upstream's loader + FK must agree with ours
    upstream.ensure_on_path()
    from utils import animation
    from utils import bvh as BVH

    anim, names, _ = BVH.load(str(p))
    g = animation.transforms_global(anim)
    up_pos = g[:, :, :3, 3] / g[:, :, 3, 3:]
    _, our_pos = forward_kinematics(skel, m.local_rot, m.root_pos)
    np.testing.assert_allclose(up_pos, our_pos, atol=1e-4)


def test_yaw_and_calibration(skel):
    m = calibration_motion(skel, RigSpec())
    assert m.num_frames == 60
    _, p0 = forward_kinematics(skel, m.local_rot, m.root_pos)
    y = yaw_motion(m, 90)
    _, p1 = forward_kinematics(skel, y.local_rot, y.root_pos)
    Ry = R.from_euler("y", 90, degrees=True).as_matrix()
    np.testing.assert_allclose(p1, p0 @ Ry.T, atol=1e-9)
    ends = [i for i, n in enumerate(skel.names) if n.endswith("_End")]
    assert np.allclose(m.local_rot[:, ends], np.eye(3))


def test_skinning_rest_is_identity(skel):
    mesh = build_mannequin(skel)
    v = skin(mesh, skel, np.tile(np.eye(3), (1, skel.num_joints, 1, 1)))[0]
    np.testing.assert_allclose(v, mesh.vertices, atol=1e-9)


@pytest.mark.parametrize("suffix", [".glb", ".gltf"])
def test_gltf_export(tmp_path, skel, suffix):
    mesh = build_mannequin(skel)
    path = export_gltf(tmp_path / f"a{suffix}", skel, {"m": (mesh, (1, 1, 1, 1)), "b": (bone_mesh(skel), (1, 0, 0, 1))}, random_motion(skel))
    g = pygltflib.GLTF2().load(str(path))
    assert len(g.skins[0].joints) == skel.num_joints
    assert len(g.animations[0].channels) == skel.num_joints + 1
    assert [g.nodes[j].name for j in g.skins[0].joints] == skel.names
