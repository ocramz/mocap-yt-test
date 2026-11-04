"""Tiny CPU renderer (no OpenGL / Blender needed).

* :func:`render_mesh` – flat-shaded painter's-algorithm rasterizer on top of PIL,
  supersampled for anti-aliasing. Good enough for reference images and previews.
* :func:`draw_skeleton` – stick-figure overlay (left = blue, right = red).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    fov_deg: float = 30.0

    def view(self) -> np.ndarray:
        f = self.target - self.eye
        f = f / np.linalg.norm(f)
        r = np.cross(f, self.up)
        r = r / np.linalg.norm(r)
        u = np.cross(r, f)
        M = np.eye(4)
        M[0, :3], M[1, :3], M[2, :3] = r, u, -f
        M[:3, 3] = -M[:3, :3] @ self.eye
        return M

    def project(self, pts: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """World points (N,3) -> pixel coords (N,2) and view depth (N,) (positive = in front)."""
        W, H = size
        V = self.view()
        pc = pts @ V[:3, :3].T + V[:3, 3]
        depth = -pc[:, 2]
        fy = 0.5 * H / np.tan(np.radians(self.fov_deg) / 2)
        z = np.maximum(depth, 1e-6)
        x = W / 2 + fy * pc[:, 0] / z
        y = H / 2 - fy * pc[:, 1] / z
        return np.stack([x, y], 1), depth


def orbit_camera(center, distance, azim_deg=0.0, elev_deg=0.0, fov_deg=30.0) -> Camera:
    """Camera orbiting ``center``; azim 0 = in front of a +Z-facing character."""
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    offs = distance * np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    c = np.asarray(center, float)
    return Camera(eye=c + offs, target=c, fov_deg=fov_deg)


def fit_camera(points: np.ndarray, azim_deg=0.0, elev_deg=5.0, fov_deg=30.0, margin=1.15) -> Camera:
    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) / 2
    radius = np.linalg.norm(hi - lo) / 2
    dist = margin * radius / np.sin(np.radians(fov_deg) / 2)
    return orbit_camera(center, dist, azim_deg, elev_deg, fov_deg)


def render_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    cam: Camera,
    size=(512, 512),
    color=(110, 190, 185),
    face_colors: np.ndarray | None = None,
    bg=(255, 255, 255),
    ssaa: int = 2,
    light_dir=(0.3, 0.8, 0.6),
    ambient: float = 0.35,
    image: Image.Image | None = None,
) -> Image.Image:
    """Z-buffered, smooth-shaded (interpolated vertex normals), fully vectorised."""
    W, H = size
    S = (W * ssaa, H * ssaa)
    uv, depth = cam.project(vertices, S)
    tri = np.asarray(faces)
    v = vertices
    fn = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, tri[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    L = np.asarray(light_dir, float)
    L /= np.linalg.norm(L)
    Lw = cam.view()[:3, :3].T @ L  # light fixed relative to the camera
    vshade = ambient + (1 - ambient) * np.clip(vn @ Lw, 0, 1)

    # cull back faces (screen-space winding) and faces behind the camera
    p = uv[tri]  # (F,3,2)
    area = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    keep = (area < -1e-9) & (depth[tri].min(1) > 1e-3)
    tri, p, area = tri[keep], p[keep], area[keep]
    x0 = np.clip(np.floor(p[..., 0].min(1)), 0, S[0] - 1).astype(np.int64)
    x1 = np.clip(np.ceil(p[..., 0].max(1)), 0, S[0] - 1).astype(np.int64)
    y0 = np.clip(np.floor(p[..., 1].min(1)), 0, S[1] - 1).astype(np.int64)
    y1 = np.clip(np.ceil(p[..., 1].max(1)), 0, S[1] - 1).astype(np.int64)
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    n = bw * bh
    ok = (x1 >= x0) & (y1 >= y0)
    tri, p, area, x0, y0, bw, n = tri[ok], p[ok], area[ok], x0[ok], y0[ok], bw[ok], n[ok]

    img = np.empty((S[1], S[0], 3), np.float64)
    if image is None:
        img[:] = np.asarray(bg, float)
    else:
        img[:] = np.asarray(image.convert("RGB").resize(S), float)
    if len(tri):
        fid = np.repeat(np.arange(len(tri)), n)
        local = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        px = x0[fid] + local % bw[fid] + 0.5
        py = y0[fid] + local // bw[fid] + 0.5
        a, b, c = p[fid, 0], p[fid, 1], p[fid, 2]
        w0 = ((b[:, 0] - px) * (c[:, 1] - py) - (c[:, 0] - px) * (b[:, 1] - py)) / area[fid]
        w1 = ((c[:, 0] - px) * (a[:, 1] - py) - (a[:, 0] - px) * (c[:, 1] - py)) / area[fid]
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        fid, w0, w1, w2, px, py = fid[inside], w0[inside], w1[inside], w2[inside], px[inside], py[inside]
        t = tri[fid]
        z = w0 * depth[t[:, 0]] + w1 * depth[t[:, 1]] + w2 * depth[t[:, 2]]
        pix = py.astype(np.int64) * S[0] + px.astype(np.int64)
        order = np.lexsort((z, pix))
        first = np.unique(pix[order], return_index=True)[1]
        win = order[first]
        sh = w0[win] * vshade[t[win, 0]] + w1[win] * vshade[t[win, 1]] + w2[win] * vshade[t[win, 2]]
        if face_colors is None:
            base = np.asarray(color, float)[None]
        else:
            base = np.asarray(face_colors, float)[keep][ok][fid[win]]
        flat = img.reshape(-1, 3)
        flat[pix[win]] = np.clip(base * sh[:, None], 0, 255)
    out = Image.fromarray(img.astype(np.uint8))
    return out.resize((W, H), Image.LANCZOS) if ssaa > 1 else out


def side_colors(names: list[str]) -> list[tuple[int, int, int]]:
    out = []
    for n in names:
        b = n.split(":")[-1]
        out.append((40, 90, 220) if b.startswith("Left") else (220, 60, 50) if b.startswith("Right") else (60, 60, 60))
    return out


def draw_skeleton(
    img: Image.Image,
    positions: np.ndarray,
    parents: np.ndarray,
    cam: Camera,
    names: list[str] | None = None,
    width: int = 4,
    joint_radius: int = 4,
) -> Image.Image:
    uv, depth = cam.project(positions, img.size)
    draw = ImageDraw.Draw(img)
    cols = side_colors(names) if names else [(60, 60, 60)] * len(parents)
    for j, p in enumerate(parents):
        if p < 0:
            continue
        draw.line([tuple(uv[p]), tuple(uv[j])], fill=cols[j], width=width)
    for j in range(len(parents)):
        x, y = uv[j]
        draw.ellipse([x - joint_radius, y - joint_radius, x + joint_radius, y + joint_radius], fill=cols[j])
    return img


def hstack(images: list[Image.Image], height: int | None = None, bg=(255, 255, 255)) -> Image.Image:
    if height is not None:
        images = [im.resize((max(1, round(im.width * height / im.height)), height)) for im in images]
    H = max(im.height for im in images)
    out = Image.new("RGB", (sum(im.width for im in images), H), bg)
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width
    return out


def grid(images: list[Image.Image], cols: int, bg=(255, 255, 255)) -> Image.Image:
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (w * cols, h * rows), bg)
    for i, im in enumerate(images):
        out.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
    return out
