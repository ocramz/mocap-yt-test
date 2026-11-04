"""Image -> DINOv2 token features, exactly as MocapAnything preprocesses them, on CPU.

Upstream's ``preprocess.image_process.load_image`` hard-codes ``.cuda()``; this is a
device-agnostic port of the RGB (no alpha) branch:

  1. RMBG-1.4 foreground matte at 1024x1024 -> Otsu threshold -> drop small blobs,
  2. composite on white, crop to the largest foreground contour, pad to a square
     with 10 % margin,
  3. DINOv2-large ``last_hidden_state`` -> (257, 1024) tokens per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import upstream


@dataclass
class FeatureExtractor:
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    batch_size: int = 16

    def __post_init__(self):
        upstream.ensure_on_path()
        upstream.torch_threads()
        from preprocess.briarmbg import BriaRMBG
        from transformers import AutoImageProcessor, Dinov2Model

        self.rmbg = BriaRMBG.from_pretrained(upstream.RMBG_REPO).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(upstream.DINO_REPO)
        self.dino_model = Dinov2Model.from_pretrained(upstream.DINO_REPO).to(self.device, self.dtype).eval()

    # ------------------------------------------------------------------ preprocessing
    @torch.no_grad()
    def matte_and_crop(self, rgb: np.ndarray, padding_ratio: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """rgb uint8 (H,W,3) -> (cropped square float image in [0,1] (S,S,3), binary mask (H,W))."""
        import cv2
        import torch.nn.functional as F
        import torchvision.transforms.functional as TF
        from skimage.measure import label
        from skimage.morphology import remove_small_objects

        h, w = rgb.shape[:2]
        s = 2000 / max(h, w)
        if s < 1:
            rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            h, w = rgb.shape[:2]
        img = torch.from_numpy(rgb).to(self.device).float().permute(2, 0, 1) / 255.0
        img1k = TF.resize(img, [1024, 1024], antialias=True)
        if float(img1k.max()) < 1e-3:
            raise ValueError("pure black image")
        alpha = self.rmbg(TF.normalize(img1k, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).unsqueeze(0))[0][0]
        alpha = TF.resize(alpha.squeeze(0), [h, w], antialias=True)
        alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-12)
        a8 = (alpha * 255).to(torch.uint8).squeeze().cpu().numpy()
        _, a8 = cv2.threshold(a8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = remove_small_objects(label(a8), min_size=200) > 0
        if not mask.any():
            raise ValueError("no foreground found")
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
        m = torch.from_numpy(mask.astype(np.float32)).to(self.device)[None]
        comp = img * m + 1.0 * (1 - m)  # white background
        pad = [0] * 4
        if bw > bh:
            pad[0] = int(bw * padding_ratio)
            pad[2] = int(pad[0] + (bw - bh) / 2)
        else:
            pad[2] = int(bh * padding_ratio)
            pad[0] = int(pad[2] + (bh - bw) / 2)
        pad[1], pad[3] = pad[0], pad[2]
        crop = F.pad(comp[:, y : y + bh, x : x + bw], pad=tuple(pad), mode="constant", value=1.0)
        return crop.permute(1, 2, 0).cpu().numpy(), mask

    # ------------------------------------------------------------------ features
    @torch.no_grad()
    def dino(self, crops: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        pils = [Image.fromarray((c * 255).astype(np.uint8)) for c in crops]
        out = []
        for i in range(0, len(pils), self.batch_size):
            px = self.processor(pils[i : i + self.batch_size], return_tensors="pt").pixel_values
            out.append(self.dino_model(px.to(self.device, self.dtype)).last_hidden_state.float().cpu())
        return torch.cat(out).numpy()  # (N, 257, 1024)

    def encode_images(self, paths: list[Path], crops_dir: Path | None = None, progress: bool = True):
        """Returns (features (N,257,1024) float32, list of valid indices)."""
        import cv2
        from tqdm import tqdm

        crops, ok = [], []
        it = tqdm(paths, desc="matte+crop") if progress else paths
        for i, p in enumerate(it):
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            try:
                crop, _ = self.matte_and_crop(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            except ValueError as e:
                print(f"[features] skip {p}: {e}")
                continue
            crops.append(crop)
            ok.append(i)
            if crops_dir is not None:
                crops_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(crops_dir / Path(p).name), cv2.cvtColor((crop * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        if not crops:
            raise RuntimeError("no frame could be matted")
        feats = []
        it = range(0, len(crops), self.batch_size)
        for i in tqdm(it, desc="dinov2") if progress else it:
            feats.append(self.dino(crops[i : i + self.batch_size]))
        return np.concatenate(feats), ok
