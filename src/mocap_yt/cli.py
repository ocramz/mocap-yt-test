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


def cmd_run(args):
    from .features import FeatureExtractor
    from .inference import MocapModel
    from .pipeline import build_rig, publish_sample, run_sample

    cfg = _cfg(args)
    ext = FeatureExtractor()
    build_rig(cfg, extractor=ext)
    model = MocapModel()
    for s in _select(cfg, args.ids):
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
    x = sub.add_parser("run", help="end-to-end: rig, features, inference, export, previews")
    x.add_argument("ids", nargs="*")
    x.add_argument("--publish", action="store_true", help="copy results into assets/samples/<id>")
    x.add_argument("--no-video", action="store_true", help="don't publish preview mp4s")
    x.add_argument("--preview-every", type=int, default=1)
    x.add_argument("--strict", action="store_true")
    x.set_defaults(fn=cmd_run)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
