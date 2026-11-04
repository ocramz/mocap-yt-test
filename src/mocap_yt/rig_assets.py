"""Build a MocapAnything "species" dataset for a custom rig without Blender.

Given a :class:`Skeleton` (+ procedural mannequin) this writes the same artifacts
upstream's ``preprocess/run_pipeline.sh`` (stages 3-8 + 14b) derives from an FBX:

    <root>/
      characters/<Rig>/{<Rig>_ffs.bvh, rest.bvh, base_mesh.obj, skinning_weights.npy}
      bvh/<Rig>#<clip>/y{0,90,180,270}.bvh       calibration clip, 4 cardinal yaws
      bvh_pose/<Rig>#<clip>/y*.npz              positions / traj / rot6d / scale
      npz_train_image_only/<Rig>#<clip>/y0.npz   DINOv2 tokens of the rendered ref frame
      ref_images/<Rig>.png                       the rendered reference frame
      species_info_dict.npy                      topology + T5 joint-name embeddings
      cache/__mesh2pose1002_species_scale_cache.pkl

The *calibration clip* starts in the reference pose (frame 0 = what the model is
conditioned on) and then wiggles every non-end joint, so upstream's static-joint
detection only flags what real data flags (``*_End`` leaves, root position).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from . import upstream
from .bvh import Motion, write_bvh
from .mesh import SkinnedMesh, build_mannequin, skin
from .render import fit_camera, render_mesh
from .skeleton import Skeleton

YAWS = (0, 90, 180, 270)
SCALE_CACHE = "cache/__mesh2pose1002_species_scale_cache.pkl"


@dataclass
class RigSpec:
    name: str = "Humanoid"
    clip: str = "calib"
    ref_pose: str = "apose"  # "tpose" | "apose"
    frames: int = 60
    fps: float = 30.0
    ref_image_size: int = 512
    mannequin_rgb: tuple[int, int, int] = (110, 190, 185)
    extra: dict = field(default_factory=dict)

    @property
    def seq(self) -> str:
        return f"{self.name}#{self.clip}"


# ----------------------------------------------------------------------------- poses


def _base(name: str) -> str:
    return name.split(":")[-1]


def reference_pose(skel: Skeleton, kind: str = "apose") -> np.ndarray:
    """Local rotations (J,3,3) of the reference pose."""
    rot = np.tile(np.eye(3), (skel.num_joints, 1, 1))
    if kind == "tpose":
        return rot
    if kind != "apose":
        raise ValueError(kind)
    for j, n in enumerate(skel.names):
        b = _base(n)
        if b == "LeftArm":
            rot[j] = R.from_euler("z", -50, degrees=True).as_matrix()
        elif b == "RightArm":
            rot[j] = R.from_euler("z", 50, degrees=True).as_matrix()
        elif b in ("LeftForeArm", "RightForeArm"):
            rot[j] = R.from_euler("y", 10 if b.startswith("Left") else -10, degrees=True).as_matrix()
    return rot


def calibration_motion(skel: Skeleton, spec: RigSpec, seed: int = 0) -> Motion:
    """Ref pose at frame 0, then smooth small rotations on every non-end joint."""
    rng = np.random.default_rng(seed)
    T, J = spec.frames, skel.num_joints
    ref = reference_pose(skel, spec.ref_pose)
    t = np.arange(T) / spec.fps
    rot = np.tile(ref, (T, 1, 1, 1))
    for j, n in enumerate(skel.names):
        if _base(n).endswith("_End") or _base(n).endswith("End"):
            continue  # end joints never rotate in real rigs either
        freq = rng.uniform(0.4, 1.2, size=3)
        phase = rng.uniform(0, 2 * np.pi, size=3)
        amp = np.radians(8.0 if j else 4.0)
        # sin(…)-sin(phase) keeps frame 0 exactly at the reference pose
        ang = amp * (np.sin(2 * np.pi * freq[None] * t[:, None] + phase[None]) - np.sin(phase)[None])
        rot[:, j] = ref[j][None] @ R.from_rotvec(ang).as_matrix()
    root = np.tile(skel.offsets[0], (T, 1))
    root[:, 1] += 0.01 * np.sin(2 * np.pi * 0.5 * t)
    return Motion(skel, rot, root, spec.fps)


def yaw_motion(m: Motion, deg: float) -> Motion:
    """Rotate a motion about world Y (what upstream's rotate_bvh_parallel does)."""
    Ry = R.from_euler("y", deg, degrees=True).as_matrix()
    rot = m.local_rot.copy()
    rot[:, 0] = Ry[None] @ rot[:, 0]
    return Motion(m.skeleton, rot, m.root_pos @ Ry.T, m.fps)


# ---------------------------------------------------------------------------- builder


def _write_obj(path: Path, v: np.ndarray, f: np.ndarray):
    with open(path, "w") as fh:
        fh.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in v)
        fh.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in f)


def _pose_npz(bvh_path: Path, scale: float) -> dict:
    """bvh -> pose npz fields, computed with upstream's own BVH/animation code
    (mirrors preprocess/extract_bvh_pose.py)."""
    upstream.ensure_on_path()
    from utils import animation
    from utils import bvh as BVH

    anim, _, frametime = BVH.load(str(bvh_path))
    gt = animation.transforms_global(anim)
    pos_before = gt[:, :, :3, 3] / gt[:, :, 3, 3:]
    local = animation.transforms_local(anim)[:, :, :3, :3]
    rot6d = local.reshape(local.shape[0], local.shape[1], 9)[:, :, :6].astype(np.float32)
    pos = pos_before * scale
    return dict(
        position_before=pos_before,
        position=pos,
        traj_before=pos_before[:, 0, :],
        traj=pos[:, 0, :],
        rot6d=rot6d,
        frametime=np.float64(frametime),
        scale=np.float64(scale),
    )


def render_reference(mesh: SkinnedMesh, skel: Skeleton, ref_rot: np.ndarray, spec: RigSpec):
    verts = skin(mesh, skel, ref_rot[None])[0]
    cam = fit_camera(verts, azim_deg=0.0, elev_deg=5.0, fov_deg=30.0, margin=1.05)
    s = spec.ref_image_size
    return render_mesh(verts, mesh.faces, cam, (s, s), color=spec.mannequin_rgb)


def build_rig_dataset(
    root: str | Path,
    skel: Skeleton,
    spec: RigSpec = RigSpec(),
    mesh: SkinnedMesh | None = None,
    extractor=None,
    t5_model: str = upstream.T5_REPO,
) -> Path:
    """Write the full dataset for one rig into ``root``. Returns ``root``."""
    root = Path(root)
    mesh = mesh or build_mannequin(skel)
    name, seq = spec.name, spec.seq

    # -- characters/<Rig>: template BVH + mesh + skin weights
    cdir = root / "characters" / name
    cdir.mkdir(parents=True, exist_ok=True)
    write_bvh(cdir / f"{name}_ffs.bvh", Motion.rest(skel))
    write_bvh(cdir / "rest.bvh", Motion.rest(skel))
    _write_obj(cdir / "base_mesh.obj", mesh.vertices, mesh.faces)
    np.save(cdir / "skinning_weights.npy", mesh.dense_weights(skel.num_joints).astype(np.float32))
    (cdir / "front.npy").parent.mkdir(exist_ok=True)
    np.save(cdir / "front.npy", np.array([0.0, 0.0, 1.0]))

    # -- bvh/ + bvh_pose/: calibration clip in 4 yaws
    motion = calibration_motion(skel, spec)
    extent = (mesh.vertices.max(0) - mesh.vertices.min(0)).max()
    scale = 1.0 / float(extent)  # auto_scale='base_mesh' in upstream
    for yaw in YAWS:
        bvh_path = write_bvh(root / "bvh" / seq / f"y{yaw}.bvh", yaw_motion(motion, yaw))
        npz = root / "bvh_pose" / seq / f"y{yaw}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, **_pose_npz(bvh_path, scale))

    # -- scale cache (mirrors preprocess/build_scale_cache.py)
    max_abs = 0.0
    for npz in sorted((root / "bvh_pose").glob(f"{name}#*/*.npz")):
        p = np.load(npz, allow_pickle=False)["position"]
        max_abs = max(max_abs, float(np.abs(p - p[:, :1]).max()))
    cache_path = root / SCALE_CACHE
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if cache_path.exists():
        with open(cache_path, "rb") as fh:  # our own file, written below
            cache = pickle.load(fh)
    cache[name] = {"global_scale": max_abs, "bbox_center": np.zeros(3, np.float32)}
    with open(cache_path, "wb") as fh:
        pickle.dump(cache, fh)

    # -- species_info_dict.npy via upstream's builder (topology, static joints, T5)
    upstream.ensure_on_path()
    from preprocess import build_species_info as bsi

    info = bsi.build(str(root), name_map=None, t5_model=t5_model, only_species={name})
    sid_path = root / "species_info_dict.npy"
    existing = np.load(sid_path, allow_pickle=True).item() if sid_path.exists() else {}  # our own file
    existing.update(info)
    np.save(sid_path, existing)

    # -- reference image + DINOv2 tokens (frame 0 of y0)
    ref_img = render_reference(mesh, skel, motion.local_rot[0], spec)
    (root / "ref_images").mkdir(exist_ok=True)
    ref_png = root / "ref_images" / f"{name}.png"
    ref_img.save(ref_png)
    if extractor is None:
        from .features import FeatureExtractor

        extractor = FeatureExtractor()
    feats, _ = extractor.encode_images([ref_png], crops_dir=root / "ref_images" / "crops", progress=False)
    out = root / "npz_train_image_only" / seq / "y0.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, image_embed=feats[:1].astype(np.float16))

    rigs_path = root / "rigs.json"
    rigs = json.loads(rigs_path.read_text()) if rigs_path.exists() else {}
    rigs[name] = {
        "seq": seq,
        "ref_pose": spec.ref_pose,
        "joints": skel.names,
        "rename_clean": info[name]["rename_clean"],
        "static_joints": info[name]["static_joints"],
        "static_rot_joints": info[name]["static_rot_joints"],
        "global_scale": max_abs,
        "pose_scale": scale,
    }
    rigs_path.write_text(json.dumps(rigs, indent=1))
    return root
