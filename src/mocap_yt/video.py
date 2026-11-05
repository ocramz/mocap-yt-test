"""Video acquisition and normalisation.

* :func:`fetch` – YouTube/any yt-dlp URL, or a local file, into ``<work>/videos/<id>/source.*``
  with a ``meta.json`` (title, uploader, license, url) next to it.
* :func:`make_clip` – trim + resample to the model's frame rate (15 fps by default,
  matching MocapAnything's human demo clips) + downscale.
* :func:`trim` – frame-accurate cut of a time range, keeping frame rate and size.
* :func:`timeline_sheets` – timestamped thumbnails of a long video, to pick ranges by eye.
* :func:`extract_frames` – clip -> numbered PNGs.

Timestamps (:func:`parse_time`) are seconds or ``[[HH:]MM:]SS[.fff]`` strings.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


def parse_time(t: float | int | str | None) -> float | None:
    """Seconds from a number or a ``[[HH:]MM:]SS[.fff]`` string: ``"3:33.5"`` -> 213.5.

    ``None`` passes through. Raises ``ValueError`` on anything else, negative times included.
    """
    if t is None:
        return None
    bad = ValueError(f"invalid timestamp {t!r}: use seconds or [[HH:]MM:]SS[.fff], e.g. 213.5 or '3:33.5'")
    if isinstance(t, bool):
        raise bad
    if isinstance(t, (int, float)):
        sec = float(t)
    else:
        parts = str(t).strip().split(":")
        if len(parts) > 3 or any(not p.strip() for p in parts):
            raise bad
        *hm, s = parts
        try:
            sec = float(s)
            units = [int(p) for p in hm]  # [MM] or [HH, MM]
        except ValueError:
            raise bad from None
        seconds_ok = not hm or 0 <= sec < 60  # "1:75" is not a time
        minutes_ok = len(hm) < 2 or 0 <= units[1] < 60
        if not (seconds_ok and minutes_ok) or any(u < 0 for u in units):
            raise bad
        for u, mult in zip(reversed(units), (60, 3600)):
            sec += u * mult
    if not math.isfinite(sec) or sec < 0:
        raise bad
    return sec


def format_time(sec: float, compact: bool = False) -> str:
    """213.5 -> ``"3:33.5"`` (or ``"3m33.5s"`` with ``compact``, safe in file names)."""
    ms = round(sec * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s = f"{ms / 1000:06.3f}".rstrip("0").rstrip(".")
    if compact:
        return (f"{h}h{m:02d}m{s}s" if h else f"{m}m{s}s" if m else f"{float(s):g}s")
    return f"{h}:{m:02d}:{s}" if h else f"{m}:{s}"


def probe(path: str | Path) -> dict:
    """``{"duration", "fps", "width", "height"}`` of a video (via imageio-ffmpeg, no ffprobe needed)."""
    import imageio_ffmpeg

    if not Path(path).is_file():
        raise FileNotFoundError(path)
    gen = imageio_ffmpeg.read_frames(str(path))
    try:
        meta = next(gen)
    finally:
        gen.close()
    w, h = meta["size"]
    return {"duration": meta.get("duration"), "fps": meta.get("fps"), "width": w, "height": h}


def check_range(path: str | Path, start: float, end: float | None = None) -> dict:
    """Probe ``path`` and fail early, with a readable message, if ``[start, end]`` isn't inside it."""
    info = probe(path)
    dur = info["duration"]
    if dur:
        if start >= dur:
            raise ValueError(f"start {format_time(start)} is past the end of {Path(path).name} ({format_time(dur)})")
        if end is not None and end > dur + 0.5:
            raise ValueError(f"end {format_time(end)} is past the end of {Path(path).name} ({format_time(dur)})")
    return info


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
        from .upstream import PROJECT_ROOT

        dst = out_dir / f"source{src.suffix}"
        shutil.copy2(src, dst)
        # project files are recorded relative to the repo, so published metadata is portable
        rel = os.path.relpath(src.resolve(), PROJECT_ROOT)
        meta = {"source": str(src.resolve()) if rel.startswith("..") else rel, "file": src.name}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
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
    fps: float | None = 15.0,
    height: int | None = 720,
    crop: str | None = None,
    audio: bool = False,
) -> Path:
    """Trim/resample/rescale (re-encoded, so cuts are frame-accurate).

    ``fps``/``height`` ``None`` keep the source's; ``crop`` is an optional ffmpeg crop
    expr ``w:h:x:y``.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = []
    if crop:
        vf.append(f"crop={crop}")
    if fps:
        vf.append(f"fps={fps}")
    if height:
        vf.append(f"scale=-2:{height}:flags=lanczos")
    cmd = [ffmpeg_exe(), "-y", "-ss", f"{start}", "-i", str(src)]
    if duration:
        cmd += ["-t", f"{duration}"]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-c:a", "aac"] if audio else ["-an"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(dst)]
    _run(cmd)
    return dst


def trim(
    src: str | Path,
    out_dir: str | Path,
    start: float,
    end: float,
    fps: float | None = None,
    height: int | None = None,
    audio: bool = False,
    name: str | None = None,
) -> Path:
    """Cut ``[start, end]`` (seconds) out of ``src`` into ``out_dir/<name>.mp4`` (default name:
    ``<stem>_<start>-<end>``). Keeps the source frame rate and size unless ``fps``/``height``
    are given."""
    if end <= start:
        raise ValueError(f"empty range {format_time(start)}-{format_time(end)}")
    check_range(src, start, end)
    name = name or f"{_slug(src)}_{format_time(start, True)}-{format_time(end, True)}"
    return make_clip(src, Path(out_dir) / f"{name}.mp4", start, end - start, fps=fps, height=height, audio=audio)


def _slug(path: str | Path) -> str:
    """File-name-safe stem: ``"How to Windmill _ Tutorial"`` -> ``"How_to_Windmill_Tutorial"``."""
    return re.sub(r"[^A-Za-z0-9.-]+", "_", Path(path).stem).strip("_")


def _stamp(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 + 6 * len(text), 15], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 230, 0))
    return img


def timeline_sheets(
    src: str | Path,
    out_dir: str | Path,
    every: float = 3.0,
    start: float = 0.0,
    end: float | None = None,
    cols: int = 6,
    rows: int = 8,
    width: int = 256,
) -> list[Path]:
    """One thumbnail every ``every`` seconds, stamped with its source time and tiled into
    ``cols`` x ``rows`` sheets -- to find which ``start``/``end`` to give a sample."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _slug(src)
    if start or end is not None:  # zoomed-in sheets don't overwrite the whole-video ones
        stem += f"_{format_time(start, True)}-{format_time(end, True) if end is not None else 'end'}"
    for old in out_dir.glob(f"{glob.escape(stem)}_sheet*.png"):
        old.unlink()
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [ffmpeg_exe(), "-y", "-ss", f"{start}", "-i", str(src)]
        if end is not None:
            cmd += ["-t", f"{end - start}"]
        _run(cmd + ["-vf", f"fps=1/{every},scale={width}:-2", "-start_number", "0", f"{tmp}/%05d.png"])
        thumbs = [_stamp(Image.open(f).convert("RGB"), format_time(start + i * every))
                  for i, f in enumerate(sorted(Path(tmp).glob("*.png")))]
    if not thumbs:
        raise ValueError(f"no frames in {src} from {format_time(start)}")
    from .render import grid

    per = cols * rows
    paths = []
    for k in range(0, len(thumbs), per):
        p = out_dir / f"{stem}_sheet{k // per:02d}.png"
        grid(thumbs[k : k + per], cols, bg=(0, 0, 0)).save(p)
        paths.append(p)
    return paths


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
