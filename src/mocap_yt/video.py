"""Video acquisition and normalisation.

* :func:`fetch` – YouTube/any yt-dlp URL, or a local file, into ``<work>/videos/<id>/source.*``
  with a ``meta.json`` (title, uploader, license, url) next to it.
* :func:`make_clip` – trim + resample to the model's frame rate (15 fps by default,
  matching MocapAnything's human demo clips) + downscale.
* :func:`extract_frames` – clip -> numbered PNGs.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(cmd: list[str]):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def youtube_id(url: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


def fetch(source: str, out_dir: str | Path, cookies: str | Path | None = None, max_height: int = 720) -> Path:
    """Download ``source`` (URL) or copy it (local path). Returns the local video path.

    ``cookies`` is an optional Netscape-format cookies file passed to yt-dlp; YouTube
    may require one when requests come from datacenter IPs.
    Idempotent: an existing ``source.*`` in ``out_dir`` is reused.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in out_dir.glob("source.*") if p.suffix != ".json")
    if existing:
        return existing[0]
    src = Path(source)
    if src.exists():
        dst = out_dir / f"source{src.suffix}"
        shutil.copy2(src, dst)
        (out_dir / "meta.json").write_text(json.dumps({"source": str(src.resolve())}, indent=1))
        return dst

    import yt_dlp

    opts = {
        "format": f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/b[height<={max_height}][ext=mp4]/b[height<={max_height}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
    }
    if cookies:
        opts["cookiefile"] = str(cookies)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(source, download=True)
    meta = {k: info.get(k) for k in ("id", "title", "uploader", "channel", "license", "duration", "width", "height", "fps", "webpage_url")}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return next(p for p in out_dir.glob("source.*") if p.suffix != ".json")


def make_clip(
    src: str | Path,
    dst: str | Path,
    start: float = 0.0,
    duration: float | None = None,
    fps: float = 15.0,
    height: int = 720,
    crop: str | None = None,
) -> Path:
    """Trim/resample/downscale. ``crop`` is an optional ffmpeg crop expr ``w:h:x:y``."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = []
    if crop:
        vf.append(f"crop={crop}")
    vf += [f"fps={fps}", f"scale=-2:{height}:flags=lanczos"]
    cmd = [ffmpeg_exe(), "-y", "-ss", f"{start}", "-i", str(src)]
    if duration:
        cmd += ["-t", f"{duration}"]
    cmd += ["-vf", ",".join(vf), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(dst)]
    _run(cmd)
    return dst


def extract_frames(clip: str | Path, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    _run([ffmpeg_exe(), "-y", "-i", str(clip), "-start_number", "0", str(out_dir / "%05d.png")])
    return sorted(out_dir.glob("*.png"))


def frames_to_video(frames_glob: str, dst: str | Path, fps: float = 15.0) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run([ffmpeg_exe(), "-y", "-framerate", f"{fps}", "-pattern_type", "glob", "-i", frames_glob,
          "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(dst)])
    return dst


def contact_sheet(frames: list[Path], dst: str | Path, n: int = 8, cols: int = 4, width: int = 320) -> Path:
    from .render import grid

    if not frames:
        raise ValueError("no frames")
    idx = [round(i * (len(frames) - 1) / max(1, n - 1)) for i in range(n)]
    ims = []
    for i in idx:
        im = Image.open(frames[i]).convert("RGB")
        ims.append(im.resize((width, round(im.height * width / im.width))))
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    grid(ims, cols).save(dst)
    return dst
