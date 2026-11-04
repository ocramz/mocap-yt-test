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
