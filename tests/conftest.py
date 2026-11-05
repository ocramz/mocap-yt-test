import subprocess

import pytest

from mocap_yt import video


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    """10 s, 24 fps, 160x90 test-pattern video; the space in the name is deliberate."""
    p = tmp_path_factory.mktemp("vid") / "synthetic clip.mp4"
    cmd = [video.ffmpeg_exe(), "-y", "-f", "lavfi", "-i", "testsrc=duration=10:size=160x90:rate=24",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)]
    subprocess.run(cmd, check=True, capture_output=True)
    return p
