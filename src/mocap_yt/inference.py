"""CPU inference with MocapAnything V2 (``video2pose2rot``) on a custom rig.

Reuses upstream's model classes, config and ``build_inference_batch`` verbatim; adds
CPU feature extraction (:mod:`mocap_yt.features`), feature caching, sliding windows
for clips longer than upstream's 301-frame cap, and BVH/glTF export.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import upstream
from .bvh import Motion, read_bvh, write_bvh

UPSTREAM_MODEL_CONFIG = "examples/custom_rig/inference.yaml"
MAX_WINDOW = 300  # upstream caps a single forward pass at 301 frames


@dataclass
class Prediction:
    rig: str
    position: np.ndarray  # (T, J, 3) root-relative, normalised by the rig's global scale
    rot6d: np.ndarray  # (T, J, 6) local rotations (upstream 6D = first two matrix rows)
    frame_indices: list[int]
    seconds: float


def load_config(rig_root: Path, rig: str, seq: str) -> dict:
    upstream.ensure_on_path()
    from utils.config_utils import load_yaml_config

    cfg = load_yaml_config(str(upstream.UPSTREAM_DIR / UPSTREAM_MODEL_CONFIG))
    cfg["runtime"].update(device="cpu", dtype="float32")
    cfg["data"].update(
        base_dir=str(rig_root),
        character_dir=str(rig_root / "characters"),
        memory_pkl_path="",
        scale_dict_path=str(rig_root / "cache" / "__mesh2pose1002_species_scale_cache.pkl"),
        wild_flag=True,
        wild_mode=True,
    )
    cfg["data"]["retarget"].update(toggle=False, ref_seq=f"{seq}/y0", ref_idx=0)
    return cfg


class MocapModel:
    def __init__(self, ckpt: Path | None = None):
        upstream.ensure_on_path()
        upstream.torch_threads()
        from utils.config_utils import instantiate_from_config

        self.cfg_template = load_config(Path("."), "", "")
        self.model = instantiate_from_config(self.cfg_template["model"]).float().eval()
        ckpt = ckpt or upstream.checkpoint_path()
        sd = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        for k in ("model_state", "model", "model_state_dict", "state_dict"):
            if isinstance(sd, dict) and k in sd:
                sd = sd[k]
                break
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        self.model.load_state_dict(sd)

    @torch.no_grad()
    def predict(
        self,
        feats: np.ndarray,
        rig_root: Path,
        rig: str,
        seq: str,
        overlap: int = 32,
        pad_joints: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """feats (T,257,1024) -> (positions (T,J,3), rot6d (T,J,6)); long clips run in
        overlapping windows blended with a linear cross-fade.

        ``pad_joints``: upstream pads every skeleton to 150 joints and masks the rest.
        No weight depends on that size, so padding only to the rig's joint count gives
        the same result (checked in tests/test_inference.py) at a fraction of the
        cost; ``None`` keeps upstream's 150.
        """
        upstream.ensure_on_path()
        import inference.video2pose2rot as v2p2r
        from inference.video2pose2rot import build_inference_batch

        v2p2r.MAX_JOINTS = pad_joints or 150

        T = len(feats)
        starts = [0] if T <= MAX_WINDOW else list(range(0, T - overlap, MAX_WINDOW - overlap))
        pos_acc = rot_acc = w_acc = None
        J = None
        for s in starts:
            e = min(T, s + MAX_WINDOW)
            cfg = load_config(rig_root, rig, seq)
            batch = build_inference_batch(cfg=cfg, image_embed_np=feats[s:e].astype(np.float32))
            J = int(batch["J_valid"][0])
            out = self.model(batch, attention_kwargs=cfg["model"]["attention_kwargs"])
            pos = out["pred_position"][0, :, :J].cpu().numpy()
            rot = out["pred_rot6d"][0, :, :J].cpu().numpy()
            n = len(pos)
            w = np.ones(n)
            if s > 0:
                ramp = min(overlap, n)
                w[:ramp] = np.linspace(0, 1, ramp + 2)[1:-1]
            if e < T:
                ramp = min(overlap, n)
                w[-ramp:] = np.minimum(w[-ramp:], np.linspace(1, 0, ramp + 2)[1:-1])
            if pos_acc is None:
                pos_acc, rot_acc, w_acc = np.zeros((T, J, 3)), np.zeros((T, J, 6)), np.zeros(T)
            pos_acc[s : s + n] += w[:, None, None] * pos
            rot_acc[s : s + n] += w[:, None, None] * rot
            w_acc[s : s + n] += w
        w_acc = np.maximum(w_acc, 1e-8)
        return pos_acc / w_acc[:, None, None], rot_acc / w_acc[:, None, None]


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Upstream 6D (the first two matrix rows) -> rotation matrices (..., 3, 3)."""
    upstream.ensure_on_path()
    from utils.transforms3d import rotation_6d_to_matrix

    return rotation_6d_to_matrix(torch.from_numpy(np.asarray(rot6d)).float()).numpy().astype(np.float64)


def rot6d_to_motion(rot6d: np.ndarray, template_bvh: Path, fps: float) -> Motion:
    """Upstream 6D (two matrix rows) -> Motion on the rig's template skeleton."""
    tpl = read_bvh(template_bvh)
    mats = rot6d_to_matrix(rot6d)
    root = np.tile(tpl.skeleton.offsets[0], (len(mats), 1))
    return Motion(tpl.skeleton, mats, root, fps)


def position_to_metres(position: np.ndarray, rig_root: Path, rig: str) -> np.ndarray:
    """Normalised predicted joint positions (root-relative) -> metres, as save_prediction does."""
    gs = json.loads((rig_root / "rigs.json").read_text())[rig]["global_scale"]
    scale = float(np.load(next((rig_root / "bvh_pose").glob(f"{rig}#*/y0.npz")), allow_pickle=False)["scale"])
    return np.asarray(position) * gs / scale


def predict_views(model: "MocapModel", extractor, crops: list[Path], rig_root: Path, rig: str,
                  ks=(1, 2, 3)) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """MocapAnything on the matted crops turned by ``np.rot90(crop, k)`` (exact: the crops are
    square and padded alike on all sides, and are what DINOv2 saw). Returns, per k, the raw
    prediction (rot6d (T, J, 6), position (T, J, 3) normalised) in the turned view; bring it
    back with :func:`mocap_yt.geometry.unrotate` in the camera frame."""
    from PIL import Image

    rigs = json.loads((rig_root / "rigs.json").read_text())
    seq, pad = rigs[rig]["seq"], len(rigs[rig]["joints"])
    images = [np.asarray(Image.open(p).convert("RGB")) for p in crops]
    out = {}
    for k in ks:
        feats = extractor.dino_images([Image.fromarray(np.ascontiguousarray(np.rot90(im, k))) for im in images])
        pos, rot = model.predict(feats, rig_root, rig, seq, pad_joints=pad)
        out[k] = (rot, pos)
    return out


def ground_motion(m: Motion, smooth: int = 5) -> Motion:
    """No global trajectory is predicted; keep the lowest joint on the floor (y=0)."""
    from .skeleton import forward_kinematics

    _, gpos = forward_kinematics(m.skeleton, m.local_rot, m.root_pos)
    low = gpos[:, :, 1].min(1)
    if smooth > 1 and len(low) > smooth:
        k = np.ones(smooth) / smooth
        low = np.convolve(np.pad(low, smooth // 2, mode="edge"), k, mode="valid")[: len(low)]
    root = m.root_pos.copy()
    root[:, 1] -= low
    return Motion(m.skeleton, m.local_rot, root, m.fps)


def save_prediction(
    pred: Prediction, out_dir: Path, rig_root: Path, fps: float, ground: bool = True, stem: str | None = None
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{pred.rig}_pose_pred.npy", pred.position.astype(np.float32))
    np.save(out_dir / f"{pred.rig}_rot6d_pred.npy", pred.rot6d.astype(np.float32))
    tpl = rig_root / "characters" / pred.rig / f"{pred.rig}_ffs.bvh"
    motion = rot6d_to_motion(pred.rot6d, tpl, fps)
    if ground:
        motion = ground_motion(motion)
    bvh_path = write_bvh(out_dir / f"{stem or pred.rig}.bvh", motion)

    # consistency: FK(predicted rotations) vs. predicted joint positions (both normalised)
    from .skeleton import forward_kinematics

    gs = json.loads((rig_root / "rigs.json").read_text())[pred.rig]["global_scale"]
    _, fk = forward_kinematics(motion.skeleton, motion.local_rot, motion.root_pos)
    scale = float(np.load(next((rig_root / "bvh_pose").glob(f"{pred.rig}#*/y0.npz")), allow_pickle=False)["scale"])
    fk_n = (fk - fk[:, :1]) * scale / gs
    fk_err = float(np.linalg.norm(fk_n - pred.position, axis=-1).mean())
    meta = {
        "rig": pred.rig,
        "frames": int(len(pred.rot6d)),
        "fps": fps,
        "inference_seconds": round(pred.seconds, 2),
        "fk_vs_pred_position_err_normalised": round(fk_err, 4),
        "fk_vs_pred_position_err_m": round(fk_err * gs / scale, 4),
        "bvh": bvh_path.name,
    }
    (out_dir / "prediction.json").write_text(json.dumps(meta, indent=1))
    return {"motion": motion, **meta}


def run_inference(
    feats: np.ndarray,
    frame_indices: list[int],
    rig_root: Path,
    rig: str,
    model: MocapModel | None = None,
    pad_to_rig: bool = True,
) -> Prediction:
    rigs = json.loads((rig_root / "rigs.json").read_text())
    seq = rigs[rig]["seq"]
    model = model or MocapModel()
    t0 = time.time()
    pad = len(rigs[rig]["joints"]) if pad_to_rig else None
    pos, rot = model.predict(feats, rig_root, rig, seq, pad_joints=pad)
    return Prediction(rig, pos, rot, frame_indices, time.time() - t0)

