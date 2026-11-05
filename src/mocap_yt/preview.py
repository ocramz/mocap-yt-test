"""Screenshots and preview videos for rigs and captured motions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from .bvh import Motion
from .gltf import bone_mesh, export_gltf
from .mesh import SkinnedMesh, skin
from .render import draw_skeleton, fit_camera, hstack, render_mesh
from .skeleton import Skeleton, forward_kinematics

MANNEQUIN_RGBA = (0.43, 0.75, 0.72, 1.0)
BONE_RGBA = (0.95, 0.65, 0.15, 1.0)


def label(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 + 7 * len(text), 18], fill=(255, 255, 255))
    d.text((4, 3), text, fill=(20, 20, 20))
    return img


def rig_previews(skel: Skeleton, mesh: SkinnedMesh, local_rot: np.ndarray, out_dir: Path, tag: str, size=480) -> list[Path]:
    """Mannequin + skeleton stills from 3 viewpoints for one pose."""
    out_dir.mkdir(parents=True, exist_ok=True)
    verts = skin(mesh, skel, local_rot[None])[0]
    _, jpos = forward_kinematics(skel, local_rot[None])
    tiles = []
    for az, name in ((0, "front"), (90, "side"), (35, "three-quarter")):
        cam = fit_camera(verts, azim_deg=az, elev_deg=8)
        m = render_mesh(verts, mesh.faces, cam, (size, size), color=(110, 190, 185))
        s = draw_skeleton(Image.new("RGB", (size, size), "white"), jpos[0], skel.parents, cam, skel.names)
        tiles += [label(m, f"mesh {name}"), label(s, f"skeleton {name}")]
    path = out_dir / f"{tag}.png"
    hstack(tiles[:2] + tiles[2:4] + tiles[4:]).save(path)
    return [path]


def export_rig_gltf(skel: Skeleton, mesh: SkinnedMesh, out_dir: Path, motion: Motion | None = None, stem: str = "humanoid") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bones = bone_mesh(skel)
    paths = [
        export_gltf(out_dir / f"{stem}_skeleton.gltf", skel, {"bones": (bones, BONE_RGBA)}),
        export_gltf(out_dir / f"{stem}_rigged.glb", skel, {"mannequin": (mesh, MANNEQUIN_RGBA), "bones": (bones, BONE_RGBA)}),
    ]
    if motion is not None:
        paths.append(export_gltf(out_dir / f"{stem}_calibration.glb", skel, {"mannequin": (mesh, MANNEQUIN_RGBA)}, motion, "calibration"))
    return paths


def motion_video_frames(
    motion: Motion,
    mesh: SkinnedMesh,
    input_frames: list[Path] | None,
    out_dir: Path,
    size: int = 400,
    every: int = 1,
    crops: list[Path] | None = None,
) -> Path:
    """Write per-frame composites: input | model input | mannequin front | mannequin side + skeleton.

    ``crops``: the matted crops DINOv2 saw, one per motion frame (the "model input" tile).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):  # a previous, longer run would leak into the preview mp4
        old.unlink()
    skel = motion.skeleton
    verts = skin(mesh, skel, motion.local_rot, motion.root_pos)
    _, jpos = forward_kinematics(skel, motion.local_rot, motion.root_pos)
    allv = verts.reshape(-1, 3)
    cams = [fit_camera(allv, azim_deg=az, elev_deg=8, margin=1.0) for az in (0, 90)]
    for t in tqdm(range(0, motion.num_frames, every), desc="preview"):
        tiles = []
        if input_frames:
            im = Image.open(input_frames[min(t, len(input_frames) - 1)]).convert("RGB")
            tiles.append(label(im.resize((round(im.width * size / im.height), size)), f"input #{t}"))
        if crops:
            c = crops[min(t, len(crops) - 1)]
            im = Image.open(c).convert("RGB").resize((size, size)) if c.exists() else Image.new("RGB", (size, size), "white")
            tiles.append(label(im, "model input"))
        for cam, name in zip(cams, ("front", "side")):
            img = render_mesh(verts[t], mesh.faces, cam, (size, size), color=(110, 190, 185))
            img = draw_skeleton(img, jpos[t], skel.parents, cam, skel.names, width=2, joint_radius=2)
            tiles.append(label(img, f"capture {name}"))
        hstack(tiles, height=size).save(out_dir / f"{t:05d}.png")
    return out_dir


def gallery(samples: list[tuple[str, Motion, list[Path]]], mesh: SkinnedMesh, dst: Path, n: int = 5, size: int = 180) -> Path:
    """One row per ``(name, motion, crops)``: ``n`` evenly spaced frames, each shown as the
    model input (matted crop) next to the captured pose (front view)."""
    rows = []
    for name, motion, crops in samples:
        skel = motion.skeleton
        verts = skin(mesh, skel, motion.local_rot, motion.root_pos)
        _, jpos = forward_kinematics(skel, motion.local_rot, motion.root_pos)
        cam = fit_camera(verts.reshape(-1, 3), azim_deg=0, elev_deg=8, margin=1.0)
        tiles = []
        for k in range(n):
            t = round(k * (motion.num_frames - 1) / max(1, n - 1))
            c = crops[min(t, len(crops) - 1)] if crops else None
            im = Image.open(c).convert("RGB").resize((size, size)) if c and c.exists() else Image.new("RGB", (size, size), "white")
            cap = render_mesh(verts[t], mesh.faces, cam, (size, size), color=(110, 190, 185))
            cap = draw_skeleton(cap, jpos[t], skel.parents, cam, skel.names, width=2, joint_radius=2)
            tiles += [label(im, f"{name} #{t}" if k == 0 else f"#{t}"), cap]
        rows.append(hstack(tiles))
    out = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)
    return dst


def contact_strip(frames_dir: Path, dst: Path, n: int = 6) -> Path:
    files = sorted(frames_dir.glob("*.png"))
    idx = [round(i * (len(files) - 1) / max(1, n - 1)) for i in range(n)]
    ims = [Image.open(files[i]) for i in idx]
    w, h = ims[0].size
    out = Image.new("RGB", (w, h * n), "white")
    for k, im in enumerate(ims):
        out.paste(im, (0, k * h))
    out.save(dst)
    return dst
