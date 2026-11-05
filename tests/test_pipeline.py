"""Sample configuration and the per-sample caches (no model weights needed)."""

import numpy as np
import pytest

from mocap_yt import video
from mocap_yt.pipeline import PipelineConfig, RigConfig, Sample, prepare_sample, sample_features
from mocap_yt.upstream import PROJECT_ROOT


def test_sample_ranges():
    s = Sample("a", "x.mp4", start="3:33.5", end="3:39.3")
    assert (s.start, s.end) == (213.5, 219.3) and s.duration == pytest.approx(5.8)
    s = Sample("b", "x.mp4", start=10, duration="0:05")
    assert (s.end, s.duration) == (15.0, 5.0)
    assert Sample("c", "x.mp4").end is None
    with pytest.raises(ValueError, match="not both"):
        Sample("d", "x.mp4", end=5, duration=5)
    with pytest.raises(ValueError, match="after start"):
        Sample("e", "x.mp4", start="1:00", end="0:59")
    with pytest.raises(ValueError, match="'f'"):
        Sample("f", "x.mp4", start="1:75")
    with pytest.raises(ValueError, match="subject"):
        Sample("g", "x.mp4", subject="cat")


def test_prepare_sample_rebuilds_on_range_change(synth, tmp_path):
    cfg = PipelineConfig(rig=RigConfig(), work=tmp_path)
    d, frames = prepare_sample(cfg, Sample("s", str(synth), start="0:02", end="0:04", fps=15, height=90))
    assert len(frames) == 30
    (d / "features.npz").write_bytes(b"stale")  # stands in for cached DINOv2 features
    d, frames = prepare_sample(cfg, Sample("s", str(synth), start="0:02", end="0:04", fps=15, height=90))
    assert len(frames) == 30 and (d / "features.npz").exists()  # unchanged -> cache kept
    d, frames = prepare_sample(cfg, Sample("s", str(synth), start="0:02", end="0:03", fps=15, height=90))
    assert len(frames) == 15 and not (d / "features.npz").exists()
    with pytest.raises(ValueError, match="past the end"):
        prepare_sample(cfg, Sample("late", str(synth), start=11, end=12))


class StubExtractor:
    """Stands in for FeatureExtractor: records calls, returns zero features."""

    def __init__(self):
        self.calls = []

    def track_people(self, frames):
        self.calls.append("track")
        return np.tile([1.0, 2.0, 3.0, 4.0], (len(frames), 1))

    def encode_images(self, frames, crops_dir=None, boxes=None):
        self.calls.append("encode+boxes" if boxes is not None else "encode")
        return np.zeros((len(frames), 2, 3), np.float32), list(range(len(frames)))


def test_features_are_cached_per_subject(tmp_path):
    frames = [tmp_path / f"{i:05d}.png" for i in range(3)]
    ext = StubExtractor()
    sample_features(tmp_path, frames, ext, subject="salient")
    sample_features(tmp_path, frames, ext, subject="salient")  # cache hit
    assert ext.calls == ["encode"]
    _, ok = sample_features(tmp_path, frames, ext, subject="person")  # other subject -> recompute
    assert ext.calls == ["encode", "track", "encode+boxes"] and ok == [0, 1, 2]
    z = np.load(tmp_path / "features.npz")
    assert str(z["subject"]) == "person" and z["person_boxes"].shape == (3, 4)
    # caches written before `subject` existed count as salient
    np.savez_compressed(tmp_path / "features.npz", image_embed=np.zeros((3, 2, 3), np.float16), frame_indices=np.arange(3))
    sample_features(tmp_path, frames, ext, subject="salient")
    assert ext.calls[-1] == "encode+boxes"


def test_configured_local_samples_are_importable():
    """Every local video in configs/samples.yaml exists and its range lies inside it."""
    cfg = PipelineConfig.load(PROJECT_ROOT / "configs" / "samples.yaml")
    local = [s for s in cfg.samples if not s.source.startswith(("http", "hf://"))]
    assert local, "expected local samples in the config"
    for s in local:
        path = PROJECT_ROOT / s.source
        assert path.is_file(), s.source
        video.check_range(path, s.start, s.end)
