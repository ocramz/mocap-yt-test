"""Command line entry point: ``uv run mocap-yt <command>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_CONFIG = "configs/samples.yaml"


def _cfg(args):
    from .pipeline import PipelineConfig

    return PipelineConfig.load(args.config)


def _select(cfg, ids):
    if not ids:
        return cfg.samples
    sel = [s for s in cfg.samples if s.id in ids]
    missing = set(ids) - {s.id for s in sel}
    if missing:
        raise SystemExit(f"unknown sample id(s): {sorted(missing)}")
    return sel


def cmd_setup(args):
    """Download every model weight the pipeline needs (idempotent)."""
    from huggingface_hub import snapshot_download

    from . import upstream

    upstream.ensure_on_path()
    print("checkpoint:", upstream.checkpoint_path())
    for repo in (upstream.RMBG_REPO, upstream.DINO_REPO, upstream.T5_REPO):
        print(repo, "->", snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.py", "*.model", "*.txt"]))
    # person detector + segmenter for `subject: person` samples (cached by torch hub)
    from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
    from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights

    for w in (FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1, DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1):
        w.get_state_dict(progress=True)
        print("torchvision ->", w.url)
    # SAM 2.1 + ViTPose for `subject: person_sam` samples (the checkpoints configured under `person_sam:`)
    ps = _cfg(args).person_sam
    for repo in (ps.sam, ps.pose):
        print(repo, "->", snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt"]))
    # refine: Depth Anything V2 Small and MotionBERT (only used when enabled under `refine:`)
    from huggingface_hub import hf_hub_download

    from .depth import DEPTH_REPO, DEPTH_REVISION
    from .lifting import MB_CKPT, MB_REPO, MB_REVISION

    print(DEPTH_REPO, "->", snapshot_download(DEPTH_REPO, revision=DEPTH_REVISION, allow_patterns=["*.json", "*.safetensors"]))
    print(MB_REPO, "->", hf_hub_download(MB_REPO, MB_CKPT, revision=MB_REVISION))


def cmd_rig(args):
    from .pipeline import build_rig

    cfg = _cfg(args)
    print("rig dataset:", build_rig(cfg, force=args.force))


def cmd_fetch(args):
    from .pipeline import prepare_sample

    cfg = _cfg(args)
    for s in _select(cfg, args.ids):
        d, frames = prepare_sample(cfg, s)
        print(f"{s.id}: {len(frames)} frames in {d}")


def time_range(text: str) -> tuple[float, float]:
    """``START-END`` or ``START+DURATION`` (each seconds or ``[[HH:]MM:]SS[.fff]``) -> (start, end)."""
    from .video import parse_time

    try:
        for sep in "+-":
            if sep in text:
                a, b = (parse_time(x) for x in text.split(sep, 1))
                end = a + b if sep == "+" else b
                if end <= a:
                    raise ValueError(f"end must be after start in {text!r}")
                return a, end
        raise ValueError(f"expected START-END or START+DURATION, got {text!r}")
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def cmd_trim(args):
    from .video import format_time, probe, trim

    opts = dict(fps=args.fps, height=args.height, audio=args.audio)
    if args.samples is not None:  # the ranges configured for local samples, one clip per id
        if args.video or args.ranges:
            raise SystemExit("mocap-yt trim: give VIDEO RANGE... or --samples, not both")
        from .pipeline import _resolve_source

        for s in _select(_cfg(args), args.samples):
            if s.source.startswith(("http://", "https://", "hf://")):
                print(f"{s.id}: skipped (not a local file: {s.source})")
                continue
            src = _resolve_source(s.source)
            end = s.end if s.end is not None else probe(src)["duration"]
            out = trim(src, args.out, s.start, end, name=s.id, **opts)
            print(f"{s.id}: {format_time(s.start)}-{format_time(end)} -> {out}")
        return
    if not args.video or not args.ranges:
        raise SystemExit("usage: mocap-yt trim VIDEO RANGE [RANGE ...]  |  mocap-yt trim --samples [ID ...]")
    for start, end in args.ranges:
        out = trim(args.video, args.out, start, end, **opts)
        print(f"{format_time(start)}-{format_time(end)} -> {out}")


def cmd_sheet(args):
    from .video import parse_time, probe, timeline_sheets

    info = probe(args.video)
    dur = f"{info['duration']:.1f} s" if info["duration"] else "unknown duration"
    print(f"{args.video}: {dur}, {info['fps']} fps, {info['width']}x{info['height']}")
    paths = timeline_sheets(args.video, args.out, every=args.every, start=parse_time(args.start) or 0.0,
                            end=parse_time(args.end), cols=args.cols, rows=args.rows, width=args.width)
    for p in paths:
        print(p)


def cmd_gallery(args):
    """Overview image of already-run samples: model input | capture, a few frames each."""
    import numpy as np

    from .bvh import read_bvh
    from .mesh import build_mannequin
    from .preview import gallery

    cfg = _cfg(args)
    rows = []
    for s in _select(cfg, args.ids):
        d = cfg.work / "samples" / s.id
        bvh = d / "out" / f"{s.id}_{cfg.rig.name.lower()}.bvh"
        if not bvh.exists():
            raise SystemExit(f"{s.id}: no capture yet, run `mocap-yt run {s.id}` first")
        frames = sorted((d / "frames").glob("*.png"))
        ok = np.load(d / "features.npz")["frame_indices"]
        rows.append((s.id, read_bvh(bvh), [d / "crops" / frames[i].name for i in ok]))
    print(gallery(rows, build_mannequin(cfg.rig.skeleton()), Path(args.out), n=args.n))


def _frames_arg(text: str) -> tuple[str, list[int] | None]:
    """``ID`` or ``ID:F1,F2,...`` -> (id, frame indices or None)."""
    sid, _, frames = text.partition(":")
    try:
        return sid, [int(f) for f in frames.split(",")] if frames else None
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected ID or ID:F1,F2,..., got {text!r}") from None


def cmd_matte(args):
    """Matte a few frames of samples with several modes, side by side (no DINOv2, no capture)."""
    import cv2
    import numpy as np
    from PIL import Image

    from .features import FeatureExtractor
    from .person_sam import PersonPoses, prompt_region
    from .pipeline import prepare_sample
    from .preview import annotate_person, matte_sheet

    cfg = _cfg(args)
    ps = cfg.person_sam
    picks = dict(args.ids)
    ext = FeatureExtractor(person_sam=ps)
    sections, diag = [], {}
    for s in _select(cfg, list(picks)):
        d, frames = prepare_sample(cfg, s)
        n = len(frames)
        idx = picks[s.id] or (list(range(0, n, args.every)) if args.every else
                              sorted({round(k * (n - 1) / max(1, args.n - 1)) for k in range(args.n)}))
        if bad := [t for t in idx if not 0 <= t < n]:
            raise SystemExit(f"{s.id}: frames {bad} out of range (0-{n - 1})")
        poses = PersonPoses.load(d / "poses.npz", ps.pose_tag(), n)  # left by a person_sam run
        if poses is None:
            boxes = ext.track_people(frames)  # tracking needs the whole clip, pose only the picked frames
            if boxes is None:
                raise SystemExit(f"{s.id}: no person detected")
            poses = ext.estimate_poses([frames[t] for t in idx], boxes[idx], progress=False)
        else:
            poses = PersonPoses(poses.boxes[idx], poses.keypoints[idx], poses.rotation[idx], poses.rot_scores[idx])
        rows, info = [], []
        for j, t in enumerate(idx):
            rgb = cv2.cvtColor(cv2.imread(str(frames[t]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            box, kps = poses.boxes[j], poses.keypoints[j]
            rec = {"frame": t, "rotation": int(poses.rotation[j]), "keypoint_score": round(float(kps[:, 2].mean()), 3)}
            crops, matte = {}, None
            for mode in args.modes:
                try:
                    crop, _ = ext.matte_and_crop(rgb, box=None if mode == "salient" else box,
                                                 pose=kps if mode == "person_sam" else None)
                except ValueError as e:
                    crops[mode] = (None, "skipped")
                    rec[mode] = f"skipped: {e}"
                    continue
                caption = mode
                if mode == "person_sam":
                    matte, fail = ext.last_matte, ext.last_failure
                    caption = matte.source if matte else "rmbg fallback"
                    rec[mode] = {"source": caption, **({"stats": matte.stats, "attempts": matte.attempts} if matte
                                                       else {"attempts": fail.attempts if fail else []})}
                crops[mode] = (Image.fromarray((crop * 255).astype(np.uint8)), caption)
            region = matte.region if matte else prompt_region(box, rgb.shape[:2], ps.context, ps.min_region)
            rows.append((t, annotate_person(rgb, box, kps, region, ps.kp_score, matte), crops))
            info.append(rec)
        sections.append((s.id, rows))
        diag[s.id] = info
    out = Path(args.out) if args.out else cfg.work / "matte" / f"{'+'.join(picks)}.png"
    print(matte_sheet(sections, args.modes, out))
    out.with_suffix(".json").write_text(json.dumps(diag, indent=1))
    print(out.with_suffix(".json"))


def cmd_run(args):
    import copy

    from .features import FeatureExtractor
    from .inference import MocapModel
    from .pipeline import build_rig, publish_sample, run_sample

    cfg = _cfg(args)
    ext = FeatureExtractor(person_sam=cfg.person_sam)
    build_rig(cfg, extractor=ext)
    model = MocapModel()
    for s in _select(cfg, args.ids):
        if getattr(args, "force_refine", False):
            s = copy.copy(s)  # (dataclasses.replace would re-validate the already normalised range)
            s.refine = True
        try:
            res = run_sample(cfg, s, model=model, extractor=ext, preview_every=args.preview_every)
        except Exception as e:  # keep going with the other samples
            print(f"[FAIL] {s.id}: {type(e).__name__}: {e}")
            if args.strict:
                raise
            continue
        print(json.dumps(res, indent=1))
        if args.publish:
            print("published ->", publish_sample(cfg, s, include_video=not args.no_video))


def main(argv=None):
    p = argparse.ArgumentParser(prog="mocap-yt", description=__doc__)
    p.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="download model weights").set_defaults(fn=cmd_setup)
    r = sub.add_parser("rig", help="generate the humanoid rig + its MocapAnything dataset")
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_rig)
    f = sub.add_parser("fetch", help="download/trim/extract frames for samples")
    f.add_argument("ids", nargs="*")
    f.set_defaults(fn=cmd_fetch)
    from .upstream import WORK_DIR

    t = sub.add_parser("trim", help="cut time ranges out of a video (frame-accurate, re-encoded)",
                       description="e.g. mocap-yt trim video.mp4 3:33.5-3:39.3 4:17+15, or "
                                   "mocap-yt trim --samples hockey_stop (the configured range)")
    t.add_argument("video", nargs="?")
    t.add_argument("ranges", nargs="*", type=time_range, metavar="RANGE",
                   help="START-END or START+DURATION; times are seconds or [[HH:]MM:]SS[.fff]")
    t.add_argument("--samples", nargs="*", metavar="ID",
                   help="instead: cut the configured range of these local samples (all if no id) to <id>.mp4")
    t.add_argument("-o", "--out", default=str(WORK_DIR / "clips"), help="output folder (default: work/clips)")
    t.add_argument("--fps", type=float, help="resample (default: keep the source frame rate)")
    t.add_argument("--height", type=int, help="rescale (default: keep the source size)")
    t.add_argument("--audio", action="store_true", help="keep the audio track")
    t.set_defaults(fn=cmd_trim)
    sh = sub.add_parser("sheet", help="timestamped contact sheets of a video, to pick sample ranges")
    sh.add_argument("video")
    sh.add_argument("-o", "--out", default=str(WORK_DIR / "sheets"), help="output folder (default: work/sheets)")
    sh.add_argument("--every", type=float, default=3.0, help="seconds between thumbnails (default: 3)")
    sh.add_argument("--start", default=None, help="from this time (default: 0)")
    sh.add_argument("--end", default=None, help="up to this time (default: end of video)")
    sh.add_argument("--cols", type=int, default=6)
    sh.add_argument("--rows", type=int, default=8)
    sh.add_argument("--width", type=int, default=256, help="thumbnail width in pixels")
    sh.set_defaults(fn=cmd_sheet)
    g = sub.add_parser("gallery", help="overview PNG of already-run samples (model input | capture)")
    g.add_argument("ids", nargs="*")
    g.add_argument("-o", "--out", default=str(WORK_DIR / "gallery.png"))
    g.add_argument("-n", type=int, default=5, help="frames per sample (default: 5)")
    g.set_defaults(fn=cmd_gallery)
    m = sub.add_parser("matte", help="compare matting modes on a few frames (no DINOv2, no capture)",
                       description="e.g. mocap-yt matte windmill:0,71,119 hockey_stop -n 7 --modes person person_sam")
    m.add_argument("ids", nargs="+", type=_frames_arg, metavar="ID[:F1,F2,...]",
                   help="sample id, optionally with the frame indices to show")
    m.add_argument("--modes", nargs="+", default=["person", "person_sam"], choices=["salient", "person", "person_sam"])
    m.add_argument("-n", type=int, default=7, help="evenly spaced frames per sample, when none are given (default: 7)")
    m.add_argument("--every", type=int, help="instead: every N-th frame")
    m.add_argument("-o", "--out", help="PNG to write, plus a .json of per-frame details (default: work/matte/<ids>.png)")
    m.set_defaults(fn=cmd_matte)
    for name, helptext, force in (("run", "end-to-end: rig, features, inference, [refine,] export, previews", False),
                                  ("refine", "run with the refine stage on, whatever the config says (caches reused)", True)):
        x = sub.add_parser(name, help=helptext)
        x.add_argument("ids", nargs="*")
        x.add_argument("--publish", action="store_true", help="copy results into assets/samples/<id>")
        x.add_argument("--no-video", action="store_true", help="don't publish preview mp4s")
        x.add_argument("--preview-every", type=int, default=1)
        x.add_argument("--strict", action="store_true")
        x.set_defaults(fn=cmd_run, force_refine=force)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
