"""Sample configuration and the per-sample caches (no model weights needed)."""

import numpy as np
import pytest

from mocap_yt import video
from mocap_yt.person_sam import PersonPoses, PersonSamConfig
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
    assert Sample("h", "x.mp4", subject="person_sam").subject == "person_sam"


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
        self.mask_sources = None

    def track_people(self, frames):
        self.calls.append("track")
        return np.tile([1.0, 2.0, 3.0, 4.0], (len(frames), 1))

    def estimate_poses(self, frames, boxes):
        self.calls.append("pose")
        n = len(frames)
        return PersonPoses(boxes, np.full((n, 17, 3), 0.5), np.zeros(n, int), np.zeros((n, 4)))

    def encode_images(self, frames, crops_dir=None, boxes=None, poses=None):
        self.calls.append("encode" + ("+boxes" if boxes is not None else "") + ("+poses" if poses is not None else ""))
        if poses is not None:
            self.mask_sources = np.array([0, 2, -1][: len(frames)], np.int8)
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


def test_person_sam_features_and_pose_caches(tmp_path):
    frames = [tmp_path / f"{i:05d}.png" for i in range(3)]
    ext, cfg = StubExtractor(), PersonSamConfig()
    sample_features(tmp_path, frames, ext, subject="person_sam", person_sam=cfg)
    assert ext.calls == ["track", "pose", "encode+boxes+poses"]
    z = np.load(tmp_path / "features.npz")
    assert str(z["subject"]) == "person_sam" and str(z["matting"]) == cfg.tag()
    assert z["person_keypoints"].shape == (3, 17, 3) and z["mask_source"].tolist() == [0, 2, -1]
    assert (tmp_path / "poses.npz").exists()
    sample_features(tmp_path, frames, ext, subject="person_sam", person_sam=cfg)  # cache hit
    assert len(ext.calls) == 3
    sample_features(tmp_path, frames, ext, subject="person_sam", person_sam=PersonSamConfig(max_points=4))
    assert ext.calls[3:] == ["encode+boxes+poses"]  # a SAM setting: re-matte, keep the poses
    sample_features(tmp_path, frames, ext, subject="person_sam", person_sam=PersonSamConfig(max_points=4, rotations=[0]))
    assert ext.calls[4:] == ["track", "pose", "encode+boxes+poses"]  # a pose setting: everything
    sample_features(tmp_path, frames, ext, subject="person")  # other subjects ignore the person_sam settings
    assert ext.calls[7:] == ["track", "encode+boxes"]


def test_person_sam_needs_a_person(tmp_path):
    class Nobody(StubExtractor):
        def track_people(self, frames):
            return None

    with pytest.raises(RuntimeError, match="no person"):
        sample_features(tmp_path, [tmp_path / "00000.png"], Nobody(), subject="person_sam")


def test_person_sam_config_block(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("person_sam:\n  sam: facebook/sam2.1-hiera-tiny\n  rotations: [0, 180]\n"
                 "samples:\n  - {id: a, source: x.mp4, subject: person_sam}\n")
    cfg = PipelineConfig.load(y)
    assert cfg.person_sam.sam == "facebook/sam2.1-hiera-tiny" and cfg.person_sam.rotations == (0, 180)
    assert cfg.samples[0].subject == "person_sam"
    y.write_text("samples: []\n")
    assert PipelineConfig.load(y).person_sam == PersonSamConfig()
    y.write_text("person_sam:\n  sam_checkpoint: x\n")
    with pytest.raises(TypeError):
        PipelineConfig.load(y)


def test_configured_local_samples_are_importable():
    """Every local video in configs/samples.yaml exists and its range lies inside it."""
    cfg = PipelineConfig.load(PROJECT_ROOT / "configs" / "samples.yaml")
    local = [s for s in cfg.samples if not s.source.startswith(("http", "hf://"))]
    assert local, "expected local samples in the config"
    for s in local:
        path = PROJECT_ROOT / s.source
        assert path.is_file(), s.source
        video.check_range(path, s.start, s.end)
