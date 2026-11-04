"""Model-level checks. Need the checkpoint + built rig (``mocap-yt setup && mocap-yt rig``)."""

import json

import numpy as np
import pytest

from mocap_yt import upstream

RIG_ROOT = upstream.WORK_DIR / "rigs" / "Humanoid"
CKPT = upstream.WORK_DIR / "checkpoints" / upstream.CKPT_NAME

pytestmark = pytest.mark.skipif(
    not (CKPT.exists() and (RIG_ROOT / "rigs.json").exists()),
    reason="needs `mocap-yt setup` and `mocap-yt rig`",
)


@pytest.fixture(scope="module")
def model():
    from mocap_yt.inference import MocapModel

    return MocapModel(CKPT)


def test_joint_padding_is_equivalent(model):
    """Padding the skeleton to its own joint count == upstream's padding to 150."""
    rig = json.loads((RIG_ROOT / "rigs.json").read_text())["Humanoid"]
    feats = np.load(RIG_ROOT / "npz_train_image_only" / rig["seq"] / "y0.npz")["image_embed"].astype(np.float32)
    rng = np.random.default_rng(0)
    clip = np.repeat(feats, 12, axis=0) + rng.normal(scale=0.05, size=(12, *feats.shape[1:])).astype(np.float32)
    p150, r150 = model.predict(clip, RIG_ROOT, "Humanoid", rig["seq"], pad_joints=None)
    pJ, rJ = model.predict(clip, RIG_ROOT, "Humanoid", rig["seq"], pad_joints=len(rig["joints"]))
    np.testing.assert_allclose(pJ, p150, atol=1e-4)
    np.testing.assert_allclose(rJ, r150, atol=1e-4)
