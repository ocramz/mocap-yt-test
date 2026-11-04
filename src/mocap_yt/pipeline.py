"""End-to-end orchestration, driven by a YAML file (see ``configs/samples.yaml``).

Every stage writes into ``work/`` and is skipped when its output already exists,
so re-running is cheap; ``publish`` copies the lightweight results into ``assets/``
(those are versioned).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import upstream, video
from .bvh import write_bvh
from .mesh import build_mannequin
from .rig_assets import RigSpec, build_rig_dataset, calibration_motion, reference_pose
from .skeleton import mixamo_humanoid

ASSETS = upstream.PROJECT_ROOT / "assets"


@dataclass
class RigConfig:
    name: str = "Humanoid"
    height: float = 1.75
    fingers: bool = False
    ref_pose: str = "apose"

    def skeleton(self):
        return mixamo_humanoid(self.height, fingers=self.fingers)

    def spec(self) -> RigSpec:
        return RigSpec(name=self.name, ref_pose=self.ref_pose)


@dataclass
class Sample:
    id: str
    source: str
    start: float = 0.0
    duration: float | None = None
    crop: str | None = None
    fps: float = 15.0
    height: int = 720
    cookies: str | None = None
    note: str = ""


@dataclass
class PipelineConfig:
    rig: RigConfig = field(default_factory=RigConfig)
    samples: list[Sample] = field(default_factory=list)
    work: Path = upstream.WORK_DIR

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        d = yaml.safe_load(Path(path).read_text())
        return cls(rig=RigConfig(**d.get("rig", {})), samples=[Sample(**s) for s in d.get("samples", [])])

    @property
    def rig_root(self) -> Path:
        return self.work / "rigs" / self.rig.name


# ------------------------------------------------------------------------------- rig


def build_rig(cfg: PipelineConfig, force: bool = False, extractor=None) -> Path:
    from .preview import export_rig_gltf, rig_previews

    root = cfg.rig_root
    rc = cfg.rig
    skel = rc.skeleton()
    mesh = build_mannequin(skel)
    spec = rc.spec()
    if force or not (root / "rigs.json").exists():
        if root.exists():
            shutil.rmtree(root)
        build_rig_dataset(root, skel, spec, mesh=mesh, extractor=extractor)

    # versioned rig assets
    out = ASSETS / "rigs" / rc.name.lower()
    out.mkdir(parents=True, exist_ok=True)
    calib = calibration_motion(skel, spec)
    write_bvh(out / f"{rc.name.lower()}_tpose.bvh", calib.__class__.rest(skel))
    write_bvh(out / f"{rc.name.lower()}_calibration.bvh", calib)
    export_rig_gltf(skel, mesh, out, motion=calib, stem=rc.name.lower())
    rig_previews(skel, mesh, reference_pose(skel, "tpose"), out, "preview_tpose")
    rig_previews(skel, mesh, reference_pose(skel, spec.ref_pose), out, f"preview_{spec.ref_pose}")
    shutil.copy2(root / "ref_images" / f"{rc.name}.png", out / "reference_render.png")
    crops = sorted((root / "ref_images" / "crops").glob("*.png"))
    if crops:
        shutil.copy2(crops[0], out / "reference_model_input.png")
    info = json.loads((root / "rigs.json").read_text())[rc.name]
    info.update(
        height_m=rc.height,
        parents=skel.parents.tolist(),
        rest_offsets_m=np.round(skel.offsets, 5).tolist(),
        generator="mocap_yt.skeleton.mixamo_humanoid",
    )
    (out / "rig.json").write_text(json.dumps(info, indent=1))
    return root


# ---------------------------------------------------------------------------- sample


def _resolve_source(src: str) -> str:
    """``hf://<repo_type>/<org>/<repo>/<path>`` -> local file via huggingface_hub."""
    if src.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        rtype, org, repo, *rest = src[5:].split("/")
        return hf_hub_download(f"{org}/{repo}", "/".join(rest), repo_type=rtype.rstrip("s"))
    p = Path(src)
    if not p.is_absolute() and (upstream.PROJECT_ROOT / p).exists():
        return str(upstream.PROJECT_ROOT / p)
    return src


def prepare_sample(cfg: PipelineConfig, s: Sample) -> tuple[Path, list[Path]]:
    d = cfg.work / "samples" / s.id
    src = video.fetch(_resolve_source(s.source), d / "download", cookies=s.cookies)
    clip = d / "clip.mp4"
    if not clip.exists():
        video.make_clip(src, clip, s.start, s.duration, s.fps, s.height, s.crop)
    fdir = d / "frames"
    frames = sorted(fdir.glob("*.png")) if fdir.exists() else []
    if not frames:
        frames = video.extract_frames(clip, fdir)
    return d, frames


def sample_features(d: Path, frames: list[Path], extractor=None) -> tuple[np.ndarray, list[int]]:
    cache = d / "features.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z["image_embed"].astype(np.float32), z["frame_indices"].tolist()
    if extractor is None:
        from .features import FeatureExtractor

        extractor = FeatureExtractor()
    feats, ok = extractor.encode_images(frames, crops_dir=d / "crops")
    np.savez_compressed(cache, image_embed=feats.astype(np.float16), frame_indices=np.array(ok))
    return feats, ok


def run_sample(cfg: PipelineConfig, s: Sample, model=None, extractor=None, preview_every: int = 1) -> dict:
    from .gltf import bone_mesh, export_gltf
    from .inference import run_inference, save_prediction
    from .preview import BONE_RGBA, MANNEQUIN_RGBA, contact_strip, motion_video_frames

    d, frames = prepare_sample(cfg, s)
    feats, ok = sample_features(d, frames, extractor)
    pred = run_inference(feats, ok, cfg.rig_root, cfg.rig.name, model=model)
    out = d / "out"
    res = save_prediction(pred, out, cfg.rig_root, fps=s.fps)
    motion = res.pop("motion")

    skel = motion.skeleton
    mesh = build_mannequin(cfg.rig.skeleton())
    glb = export_gltf(out / f"{s.id}_{cfg.rig.name.lower()}.glb", skel,
                      {"mannequin": (mesh, MANNEQUIN_RGBA), "bones": (bone_mesh(skel), BONE_RGBA)}, motion, s.id)
    used = [frames[i] for i in ok]
    pdir = motion_video_frames(motion, mesh, used, d / "preview", every=preview_every)
    mp4 = video.frames_to_video(str(pdir / "*.png"), out / f"{s.id}_preview.mp4", fps=s.fps / preview_every)
    strip = contact_strip(pdir, out / f"{s.id}_preview_strip.png")
    video.contact_sheet(frames, out / f"{s.id}_input_contact_sheet.png")
    res.update(sample=s.id, source=s.source, start=s.start, duration=s.duration, glb=glb.name, preview=mp4.name)
    (out / "prediction.json").write_text(json.dumps(res, indent=1))
    return res


def publish_sample(cfg: PipelineConfig, s: Sample, include_video: bool = True) -> Path:
    src = cfg.work / "samples" / s.id / "out"
    dst = ASSETS / "samples" / s.id
    dst.mkdir(parents=True, exist_ok=True)
    pats = ["*.glb", "*.bvh", "*.png", "prediction.json"] + (["*.mp4"] if include_video else [])
    for pat in pats:
        for f in src.glob(pat):
            shutil.copy2(f, dst / f.name)
    meta = cfg.work / "samples" / s.id / "download" / "meta.json"
    if meta.exists():
        shutil.copy2(meta, dst / "source_meta.json")
    return dst
