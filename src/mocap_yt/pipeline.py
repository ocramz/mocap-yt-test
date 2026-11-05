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
from .person_sam import SOURCES, PersonPoses, PersonSamConfig, matting_tag
from .rig_assets import RigSpec, build_rig_dataset, calibration_motion, reference_pose
from .skeleton import mixamo_humanoid

ASSETS = upstream.PROJECT_ROOT / "assets"
SUBJECTS = ("salient", "person", "person_sam")


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
    """One input clip. ``start``/``end``/``duration`` are seconds or ``[[HH:]MM:]SS[.fff]``
    strings; give ``end`` or ``duration`` (or neither: to the end of the video).

    ``subject``: ``salient`` (upstream's preprocessing: matte the most salient region; fine
    on plain backgrounds), ``person`` (track the main person with a detector and matte
    only them; for in-the-wild footage) or ``person_sam`` (track them likewise, then segment
    them with SAM prompted by their 2D pose; for blur, clothes the colour of the background,
    upside-down poses). See :mod:`mocap_yt.features` and :mod:`mocap_yt.person_sam`.
    """

    id: str
    source: str
    start: float | str = 0.0
    duration: float | str | None = None
    end: float | str | None = None
    crop: str | None = None
    fps: float = 15.0
    height: int = 720
    subject: str = "salient"
    cookies: str | None = None
    note: str = ""

    def __post_init__(self):
        if self.subject not in SUBJECTS:
            raise ValueError(f"sample {self.id!r}: subject must be one of {', '.join(SUBJECTS)}, not {self.subject!r}")
        try:
            self.start = video.parse_time(self.start) or 0.0
            self.duration = video.parse_time(self.duration)
            self.end = video.parse_time(self.end)
        except ValueError as e:
            raise ValueError(f"sample {self.id!r}: {e}") from None
        if self.end is not None:
            if self.duration is not None:
                raise ValueError(f"sample {self.id!r}: give `end` or `duration`, not both")
            if self.end <= self.start:
                raise ValueError(f"sample {self.id!r}: end {self.end} s must be after start {self.start} s")
            self.duration = round(self.end - self.start, 6)
        elif self.duration is not None:
            if self.duration <= 0:
                raise ValueError(f"sample {self.id!r}: duration must be > 0")
            self.end = round(self.start + self.duration, 6)

    def clip_params(self) -> dict:
        """Everything the cached clip/frames/features of this sample depend on."""
        return {"source": self.source, "start": self.start, "end": self.end, "fps": self.fps,
                "height": self.height, "crop": self.crop}


@dataclass
class PipelineConfig:
    rig: RigConfig = field(default_factory=RigConfig)
    samples: list[Sample] = field(default_factory=list)
    work: Path = upstream.WORK_DIR
    person_sam: PersonSamConfig = field(default_factory=PersonSamConfig)  # settings of subject: person_sam

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        d = yaml.safe_load(Path(path).read_text())
        return cls(rig=RigConfig(**d.get("rig", {})), samples=[Sample(**s) for s in d.get("samples", [])],
                   person_sam=PersonSamConfig(**(d.get("person_sam") or {})))

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


def _invalidate(d: Path, params: dict):
    """Drop cached stages of a sample whose clip parameters (range, fps, ...) changed."""
    stamp = d / "clip.json"
    if not stamp.exists():
        return
    old = json.loads(stamp.read_text())
    if old == params:
        return
    print(f"[{d.name}] clip parameters changed, rebuilding clip, frames and features")
    stale = ["clip.mp4", "features.npz", "poses.npz", "frames", "crops", "preview", "out"]
    if old.get("source") != params["source"]:
        stale.append("download")
    for name in stale:
        p = d / name
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()


def prepare_sample(cfg: PipelineConfig, s: Sample) -> tuple[Path, list[Path]]:
    d = cfg.work / "samples" / s.id
    params = s.clip_params()
    _invalidate(d, params)
    src = video.fetch(_resolve_source(s.source), d / "download", cookies=s.cookies)
    clip = d / "clip.mp4"
    if not clip.exists():
        video.check_range(src, s.start, s.end)
        video.make_clip(src, clip, s.start, s.duration, s.fps, s.height, s.crop)
    fdir = d / "frames"
    frames = sorted(fdir.glob("*.png")) if fdir.exists() else []
    if not frames:
        frames = video.extract_frames(clip, fdir)
    if not frames:
        raise RuntimeError(f"{s.id}: no frames in {clip} (start={s.start}, end={s.end})")
    (d / "clip.json").write_text(json.dumps(params, indent=1))
    return d, frames


def sample_features(d: Path, frames: list[Path], extractor=None, subject: str = "salient",
                    person_sam: PersonSamConfig | None = None) -> tuple[np.ndarray, list[int]]:
    """DINOv2 tokens of every frame, cached in ``features.npz`` with the ``subject`` mode
    they were computed with (``salient``: upstream's matting; ``person``: person-guided;
    ``person_sam``: pose-prompted SAM, whose settings ``person_sam`` are part of the cache key)."""
    cache = d / "features.npz"
    tag = matting_tag(subject, person_sam)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        cached = str(z["subject"]) if "subject" in z.files else "salient"
        cached_tag = str(z["matting"]) if "matting" in z.files else ""
        if cached == subject and cached_tag == tag:
            return z["image_embed"].astype(np.float32), z["frame_indices"].tolist()
        if cached != subject:
            print(f"[{d.name}] subject {cached} -> {subject}: recomputing features")
        else:
            print(f"[{d.name}] {subject} settings changed: recomputing features")
    if extractor is None:
        from .features import FeatureExtractor

        extractor = FeatureExtractor(person_sam=person_sam or PersonSamConfig())
    crops = d / "crops"
    if crops.exists():
        shutil.rmtree(crops)
    if subject == "person_sam":
        poses = person_poses(d, frames, extractor, person_sam or PersonSamConfig())
        feats, ok = extractor.encode_images(frames, crops_dir=crops, boxes=poses.boxes, poses=poses.keypoints)
        extra = {"matting": np.array(tag), "person_boxes": poses.boxes.astype(np.float32),
                 "person_keypoints": poses.keypoints.astype(np.float32), "mask_source": extractor.mask_sources}
    else:
        boxes = extractor.track_people(frames) if subject == "person" else None
        feats, ok = extractor.encode_images(frames, crops_dir=crops, boxes=boxes)
        extra = {} if boxes is None else {"person_boxes": boxes.astype(np.float32)}
    np.savez_compressed(cache, image_embed=feats.astype(np.float16), frame_indices=np.array(ok),
                        subject=np.array(subject), **extra)
    return feats, ok


def person_poses(d: Path, frames: list[Path], extractor, cfg: PersonSamConfig) -> PersonPoses:
    """The tracked person's boxes and 2D pose in every frame, cached in ``poses.npz`` (keyed by
    the pose settings, so changing only SAM settings doesn't re-run detector and pose)."""
    path = d / "poses.npz"
    poses = PersonPoses.load(path, cfg.pose_tag(), len(frames))
    if poses is None:
        boxes = extractor.track_people(frames)
        if boxes is None:
            raise RuntimeError(f"{d.name}: no person detected in any frame")
        poses = extractor.estimate_poses(frames, boxes)
        poses.save(path, cfg.pose_tag())
    return poses


def run_sample(cfg: PipelineConfig, s: Sample, model=None, extractor=None, preview_every: int = 1) -> dict:
    from .gltf import bone_mesh, export_gltf
    from .inference import run_inference, save_prediction
    from .preview import BONE_RGBA, MANNEQUIN_RGBA, contact_strip, motion_video_frames

    d, frames = prepare_sample(cfg, s)
    feats, ok = sample_features(d, frames, extractor, subject=s.subject, person_sam=cfg.person_sam)
    pred = run_inference(feats, ok, cfg.rig_root, cfg.rig.name, model=model)
    out = d / "out"
    res = save_prediction(pred, out, cfg.rig_root, fps=s.fps, stem=f"{s.id}_{cfg.rig.name.lower()}")
    motion = res.pop("motion")

    skel = motion.skeleton
    mesh = build_mannequin(cfg.rig.skeleton())
    glb = export_gltf(out / f"{s.id}_{cfg.rig.name.lower()}.glb", skel,
                      {"mannequin": (mesh, MANNEQUIN_RGBA), "bones": (bone_mesh(skel), BONE_RGBA)}, motion, s.id)
    used = [frames[i] for i in ok]
    crops = [d / "crops" / f.name for f in used] if (d / "crops").is_dir() else None
    pdir = motion_video_frames(motion, mesh, used, d / "preview", every=preview_every, crops=crops)
    mp4 = video.frames_to_video(str(pdir / "*.png"), out / f"{s.id}_preview.mp4", fps=s.fps / preview_every)
    strip = contact_strip(pdir, out / f"{s.id}_preview_strip.png")
    video.contact_sheet(frames, out / f"{s.id}_input_contact_sheet.png")
    res.update(sample=s.id, source=s.source, start=s.start, end=s.end, duration=s.duration, subject=s.subject,
               glb=glb.name, preview=mp4.name)
    if s.subject == "person_sam":
        src = np.load(d / "features.npz")["mask_source"]
        counts = {name: int((src == k).sum()) for k, name in enumerate(SOURCES)}
        res["person_sam"] = {"sam": cfg.person_sam.sam, "pose": cfg.person_sam.pose,
                             "mask_source": {**counts, "skipped": int((src < 0).sum())}}
    if s.note:
        res["note"] = s.note
    (out / "prediction.json").write_text(json.dumps(res, indent=1))
    return res


def publish_sample(cfg: PipelineConfig, s: Sample, include_video: bool = True) -> Path:
    src = cfg.work / "samples" / s.id / "out"
    dst = ASSETS / "samples" / s.id
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    pats = ["*.glb", "*.bvh", "*.png", "prediction.json"] + (["*.mp4"] if include_video else [])
    for pat in pats:
        for f in src.glob(pat):
            shutil.copy2(f, dst / f.name)
    meta = cfg.work / "samples" / s.id / "download" / "meta.json"
    if meta.exists():
        shutil.copy2(meta, dst / "source_meta.json")
    return dst
