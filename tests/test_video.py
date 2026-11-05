"""Video import / trimming. Only needs ffmpeg (system or imageio-ffmpeg's), no model weights."""

import argparse

import pytest

from mocap_yt import video
from mocap_yt.cli import time_range


def n_frames(path) -> int:
    import imageio_ffmpeg

    return imageio_ffmpeg.count_frames_and_secs(str(path))[0]


@pytest.mark.parametrize(
    "text, sec",
    [(213.5, 213.5), (7, 7.0), ("213.5", 213.5), ("3:33.5", 213.5), (" 4:17 ", 257.0),
     ("1:02:03", 3723.0), ("0:00.25", 0.25), ("75:00", 4500.0), (None, None)],
)
def test_parse_time(text, sec):
    assert video.parse_time(text) == sec


@pytest.mark.parametrize("text", ["1:75", "1:60:00", "-3", -1, "abc", "1::2", "", "1:2:3:4", True, "nan", "1:-5"])
def test_parse_time_rejects(text):
    with pytest.raises(ValueError):
        video.parse_time(text)


def test_format_time_roundtrip():
    assert video.format_time(213.5) == "3:33.5"
    assert video.format_time(3723) == "1:02:03"
    assert video.format_time(213.5, compact=True) == "3m33.5s"
    assert video.format_time(3723, compact=True) == "1h02m03s"
    assert video.format_time(45.25, compact=True) == "45.25s"
    for t in (0.0, 0.5, 59.999, 60.0, 213.5, 3723.25):
        assert video.parse_time(video.format_time(t)) == pytest.approx(t)



def test_time_range_arg():
    assert time_range("3:33.5-3:39.3") == (213.5, 219.3)
    assert time_range("4:17+15") == (257.0, 272.0)
    for bad in ("3:33.5", "5-4", "x-y"):
        with pytest.raises(argparse.ArgumentTypeError):
            time_range(bad)


def test_probe_and_trim(synth, tmp_path):
    info = video.probe(synth)
    assert (info["width"], info["height"], info["fps"]) == (160, 90, 24)
    assert info["duration"] == pytest.approx(10, abs=0.1)

    out = video.trim(synth, tmp_path, 2.0, 4.5)
    assert out.name == "synthetic_clip_2s-4.5s.mp4"
    t = video.probe(out)
    assert (t["width"], t["height"], t["fps"]) == (160, 90, 24)  # source rate and size kept
    assert n_frames(out) == 60  # 2.5 s @ 24 fps, frame-accurate

    small = video.trim(synth, tmp_path / "s", 0, 2, fps=12, height=46)
    assert (video.probe(small)["height"], n_frames(small)) == (46, 24)

    with pytest.raises(ValueError, match="past the end"):
        video.trim(synth, tmp_path, 12, 13)
    with pytest.raises(ValueError, match="past the end"):
        video.trim(synth, tmp_path, 8, 15)


def test_trim_cli(synth, tmp_path):
    from mocap_yt.cli import main

    cfg = tmp_path / "samples.yaml"
    cfg.write_text(f"samples:\n  - id: drill\n    source: '{synth}'\n    start: '0:01'\n    end: '0:02.5'\n"
                   "  - id: web\n    source: https://youtu.be/aaaaaaaaaaa\n")
    main(["-c", str(cfg), "trim", "--samples", "-o", str(tmp_path / "clips")])  # configured ranges
    assert [p.name for p in (tmp_path / "clips").iterdir()] == ["drill.mp4"]  # URL sample skipped
    assert n_frames(tmp_path / "clips" / "drill.mp4") == 36  # 1.5 s @ 24 fps
    main(["trim", str(synth), "0:03-0:04", "8+0.5", "-o", str(tmp_path / "adhoc")])
    assert sorted(p.name for p in (tmp_path / "adhoc").iterdir()) == ["synthetic_clip_3s-4s.mp4", "synthetic_clip_8s-8.5s.mp4"]
    for bad in (["trim", str(synth)], ["-c", str(cfg), "trim", str(synth), "0-1", "--samples"]):
        with pytest.raises(SystemExit):
            main(bad)


def test_timeline_sheets(synth, tmp_path):
    sheets = video.timeline_sheets(synth, tmp_path, every=1.0, cols=4, rows=2, width=80)
    assert [p.name for p in sheets] == ["synthetic_clip_sheet00.png", "synthetic_clip_sheet01.png"]
    from PIL import Image

    assert Image.open(sheets[0]).size == (4 * 80, 2 * 46)  # 8 thumbnails of 80x46 (even height)
    zoom = video.timeline_sheets(synth, tmp_path, every=0.5, start=2, end=4, cols=4, rows=2, width=80)
    assert [p.name for p in zoom] == ["synthetic_clip_2s-4s_sheet00.png"]  # 4 thumbnails, own name
    assert all(p.exists() for p in sheets)  # the whole-video sheets are still there
