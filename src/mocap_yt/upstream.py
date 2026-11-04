"""Access to the pinned MocapAnything checkout (git submodule) and its weights.

MocapAnything is not an installable package: its top-level modules (``utils``,
``models``, ``preprocess``, ``inference``) are imported from the repository root.
:func:`ensure_on_path` puts the submodule there; everything else in ``mocap_yt``
imports upstream code lazily through it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = Path(os.environ.get("MOCAPANYTHING_DIR", PROJECT_ROOT / "third_party" / "MocapAnything"))
WORK_DIR = Path(os.environ.get("MOCAP_YT_WORK", PROJECT_ROOT / "work"))

WEIGHTS_REPO = "kehong/MoCapAnythingV2-weights"
CKPT_NAME = "video2pos2rot_epoch60.pt"
RMBG_REPO = "briaai/RMBG-1.4"
DINO_REPO = "facebook/dinov2-large"
T5_REPO = "google-t5/t5-base"  # canonical id of upstream's `t5-base`


def ensure_on_path() -> Path:
    if not (UPSTREAM_DIR / "models" / "v2").is_dir():
        raise RuntimeError(
            f"MocapAnything not found at {UPSTREAM_DIR}. Run `git submodule update --init` "
            "or set MOCAPANYTHING_DIR."
        )
    p = str(UPSTREAM_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    # upstream's matplotlib animations look for an ffmpeg binary; use imageio's bundled one
    try:
        import imageio_ffmpeg
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return UPSTREAM_DIR


def checkpoint_path(ckpt_dir: Path | None = None) -> Path:
    """Path to the released video2pose2rot checkpoint, downloading it if needed."""
    ckpt_dir = Path(ckpt_dir or WORK_DIR / "checkpoints")
    path = ckpt_dir / CKPT_NAME
    if not path.exists():
        from huggingface_hub import hf_hub_download

        hf_hub_download(WEIGHTS_REPO, CKPT_NAME, local_dir=str(ckpt_dir))
    return path


def torch_threads() -> int:
    import torch

    n = int(os.environ.get("MOCAP_YT_THREADS", os.cpu_count() or 1))
    torch.set_num_threads(n)
    return n
