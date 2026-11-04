"""glTF 2.0 export of skeletons, skinned mannequins and animations (pure Python).

The joint hierarchy becomes a node tree (translation = rest offset), every mesh is
skinned to it, and a :class:`~mocap_yt.bvh.Motion` becomes one glTF animation with a
rotation channel per joint plus the root translation. Writes ``.glb`` (binary) or
``.gltf`` (JSON with embedded buffer) depending on the file suffix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygltflib as G
from scipy.spatial.transform import Rotation as R

from .bvh import Motion
from .mesh import SkinnedMesh
from .skeleton import Skeleton

_FLOAT, _USHORT, _UINT = 5126, 5123, 5125


class _Builder:
    def __init__(self):
        self.blob = bytearray()
        self.gltf = G.GLTF2(asset=G.Asset(version="2.0", generator="mocap_yt"))

    def accessor(self, arr: np.ndarray, ctype: int, atype: str, target: int | None = None, minmax=False) -> int:
        arr = np.ascontiguousarray(arr)
        while len(self.blob) % 4:
            self.blob.append(0)
        off = len(self.blob)
        self.blob += arr.tobytes()
        bv = G.BufferView(buffer=0, byteOffset=off, byteLength=arr.nbytes, target=target)
        self.gltf.bufferViews.append(bv)
        count = arr.shape[0]
        acc = G.Accessor(bufferView=len(self.gltf.bufferViews) - 1, componentType=ctype, count=count, type=atype)
        if minmax:
            flat = arr.reshape(count, -1)
            acc.min = flat.min(0).astype(float).tolist()
            acc.max = flat.max(0).astype(float).tolist()
        self.gltf.accessors.append(acc)
        return len(self.gltf.accessors) - 1

    def finish(self, path: Path):
        self.gltf.buffers = [G.Buffer(byteLength=len(self.blob))]
        self.gltf.set_binary_blob(bytes(self.blob))
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".glb":
            self.gltf.save_binary(str(path))
        else:
            self.gltf.convert_buffers(G.BufferFormat.DATAURI)
            self.gltf.save_json(str(path))
        return path


def bone_mesh(skel: Skeleton, width_ratio: float = 0.12) -> SkinnedMesh:
    """Octahedral 'Blender-style' bones, each rigidly bound to its parent joint."""
    pos = skel.rest_positions()
    V, F, J = [], [], []
    for c, p in enumerate(skel.parents):
        if p < 0:
            continue
        a, b = pos[p], pos[c]
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        ax = d / L
        ref = np.array([0, 0, 1.0]) if abs(ax[2]) < 0.9 else np.array([1.0, 0, 0])
        u = np.cross(ax, ref)
        u /= np.linalg.norm(u)
        w = np.cross(ax, u)
        mid = a + ax * L * 0.2
        r = L * width_ratio
        ring = [mid + r * u, mid + r * w, mid - r * u, mid - r * w]
        base = len(V)
        V += [a, *ring, b]
        for k in range(4):
            k1 = (k + 1) % 4
            F.append([base, base + 1 + k1, base + 1 + k])
            F.append([base + 5, base + 1 + k, base + 1 + k1])
        J += [p] * 6
    V = np.array(V)
    joints = np.zeros((len(V), 4), np.int64)
    joints[:, 0] = J
    weights = np.zeros((len(V), 4))
    weights[:, 0] = 1
    F = np.array(F)
    return SkinnedMesh(V, F, joints, weights, joints[F[:, 0], 0])


def _vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    n = np.zeros_like(v)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n


def export_gltf(
    path: str | Path,
    skel: Skeleton,
    meshes: dict[str, tuple[SkinnedMesh, tuple[float, float, float, float]]] | None = None,
    motion: Motion | None = None,
    animation_name: str = "mocap",
) -> Path:
    """Write skeleton (+ skinned meshes) (+ animation) as glTF.

    ``meshes`` maps a mesh name to ``(SkinnedMesh, rgba_base_color)``.
    """
    path = Path(path)
    b = _Builder()
    g = b.gltf
    J = skel.num_joints
    rest = skel.rest_positions()

    # --- joint node tree ---
    for j in range(J):
        node = G.Node(name=skel.names[j], translation=[float(x) for x in skel.offsets[j]], rotation=[0.0, 0.0, 0.0, 1.0])
        node.children = skel.children(j)
        g.nodes.append(node)

    # --- skin ---
    ibm = np.tile(np.eye(4, dtype=np.float32), (J, 1, 1))
    ibm[:, :3, 3] = -rest  # rest rotations are identity: inverse bind = translate(-rest)
    ibm_acc = b.accessor(ibm.transpose(0, 2, 1).reshape(J, 16).astype(np.float32), _FLOAT, "MAT4")
    g.skins.append(G.Skin(name=f"{path.stem}_skin", joints=list(range(J)), skeleton=0, inverseBindMatrices=ibm_acc))

    scene_nodes = [0]
    for name, (m, rgba) in (meshes or {}).items():
        mat = G.Material(
            name=f"{name}_mat",
            pbrMetallicRoughness=G.PbrMetallicRoughness(baseColorFactor=list(rgba), metallicFactor=0.0, roughnessFactor=0.7),
            doubleSided=False,
        )
        g.materials.append(mat)
        pos_acc = b.accessor(m.vertices.astype(np.float32), _FLOAT, "VEC3", G.ARRAY_BUFFER, minmax=True)
        nrm_acc = b.accessor(_vertex_normals(m.vertices, m.faces).astype(np.float32), _FLOAT, "VEC3", G.ARRAY_BUFFER)
        jnt_acc = b.accessor(m.joints.astype(np.uint16), _USHORT, "VEC4", G.ARRAY_BUFFER)
        wgt_acc = b.accessor(m.weights.astype(np.float32), _FLOAT, "VEC4", G.ARRAY_BUFFER)
        idx_acc = b.accessor(m.faces.reshape(-1).astype(np.uint32), _UINT, "SCALAR", G.ELEMENT_ARRAY_BUFFER)
        prim = G.Primitive(
            attributes=G.Attributes(POSITION=pos_acc, NORMAL=nrm_acc, JOINTS_0=jnt_acc, WEIGHTS_0=wgt_acc),
            indices=idx_acc,
            material=len(g.materials) - 1,
        )
        g.meshes.append(G.Mesh(name=name, primitives=[prim]))
        g.nodes.append(G.Node(name=name, mesh=len(g.meshes) - 1, skin=0))
        scene_nodes.append(len(g.nodes) - 1)

    g.scenes = [G.Scene(name=path.stem, nodes=scene_nodes)]
    g.scene = 0

    # --- animation ---
    if motion is not None:
        T = motion.num_frames
        times = (np.arange(T) / motion.fps).astype(np.float32)
        t_acc = b.accessor(times, _FLOAT, "SCALAR", minmax=True)
        samplers, channels = [], []
        quats = R.from_matrix(motion.local_rot.reshape(-1, 3, 3)).as_quat().reshape(T, J, 4)  # xyzw
        # keep quaternion sign continuous over time for clean interpolation
        for t in range(1, T):
            flip = (quats[t] * quats[t - 1]).sum(-1) < 0
            quats[t, flip] *= -1
        for j in range(J):
            acc = b.accessor(quats[:, j].astype(np.float32), _FLOAT, "VEC4")
            samplers.append(G.AnimationSampler(input=t_acc, output=acc, interpolation="LINEAR"))
            channels.append(G.AnimationChannel(sampler=len(samplers) - 1, target=G.AnimationChannelTarget(node=j, path="rotation")))
        acc = b.accessor(motion.root_pos.astype(np.float32), _FLOAT, "VEC3")
        samplers.append(G.AnimationSampler(input=t_acc, output=acc, interpolation="LINEAR"))
        channels.append(G.AnimationChannel(sampler=len(samplers) - 1, target=G.AnimationChannelTarget(node=0, path="translation")))
        g.animations.append(G.Animation(name=animation_name, samplers=samplers, channels=channels))

    return b.finish(path)
