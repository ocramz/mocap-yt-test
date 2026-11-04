"""Minimal BVH writer/reader for :class:`~mocap_yt.skeleton.Skeleton` motions.

Output follows the layout of MocapAnything's datasets (Mixamo-style):
every joint has 3 rotation channels in ``Z Y X`` order, the root additionally has
3 position channels, and leaf joints carry a zero-length ``End Site``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from .skeleton import Skeleton, euler_zyx_deg


@dataclass
class Motion:
    skeleton: Skeleton
    local_rot: np.ndarray  # (T, J, 3, 3)
    root_pos: np.ndarray  # (T, 3)
    fps: float = 30.0

    @property
    def num_frames(self) -> int:
        return self.local_rot.shape[0]

    @classmethod
    def rest(cls, skel: Skeleton, frames: int = 1, fps: float = 30.0) -> "Motion":
        rot = np.tile(np.eye(3), (frames, skel.num_joints, 1, 1))
        root = np.tile(skel.offsets[0], (frames, 1))
        return cls(skel, rot, root, fps)


def write_bvh(path: str | Path, motion: Motion, precision: int = 6) -> Path:
    skel = motion.skeleton
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = f"{{:.{precision}f}}"
    lines: list[str] = ["HIERARCHY"]

    children = {j: skel.children(j) for j in range(skel.num_joints)}

    def emit(j: int, depth: int):
        ind = "\t" * depth
        off = " ".join(f.format(v) for v in skel.offsets[j])
        if skel.parents[j] < 0:
            lines.append(f"{ind}ROOT {skel.names[j]}")
            lines.append(f"{ind}{{")
            lines.append(f"{ind}\tOFFSET {off}")
            lines.append(f"{ind}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
        else:
            lines.append(f"{ind}JOINT {skel.names[j]}")
            lines.append(f"{ind}{{")
            lines.append(f"{ind}\tOFFSET {off}")
            lines.append(f"{ind}\tCHANNELS 3 Zrotation Yrotation Xrotation")
        if children[j]:
            for c in children[j]:
                emit(c, depth + 1)
        else:
            lines.append(f"{ind}\tEnd Site")
            lines.append(f"{ind}\t{{")
            lines.append(f"{ind}\t\tOFFSET {f.format(0)} {f.format(0)} {f.format(0)}")
            lines.append(f"{ind}\t}}")
        lines.append(f"{ind}}}")

    emit(0, 0)
    euler = euler_zyx_deg(motion.local_rot)  # (T, J, 3)
    lines += ["MOTION", f"Frames: {motion.num_frames}", f"Frame Time: {1.0 / motion.fps:.6f}"]
    order = _dfs_order(skel)  # channel order follows the emitted hierarchy (DFS)
    for t in range(motion.num_frames):
        vals = list(motion.root_pos[t])
        for j in order:
            vals += list(euler[t, j])
        lines.append(" ".join(f.format(v) for v in vals))
    path.write_text("\n".join(lines) + "\n")
    return path


def _dfs_order(skel: Skeleton) -> list[int]:
    out: list[int] = []

    def rec(j):
        out.append(j)
        for c in skel.children(j):
            rec(c)

    rec(0)
    return out


def read_bvh(path: str | Path) -> Motion:
    """Read a BVH file with per-joint rotation channels (any order) and root position."""
    text = Path(path).read_text().split("MOTION")
    hier, motion = text[0], text[1]
    names, parents, offsets, channels = [], [], [], []
    stack: list[int] = []
    pending_end = False
    for raw in hier.splitlines():
        tok = raw.strip().split()
        if not tok:
            continue
        if tok[0] in ("ROOT", "JOINT"):
            names.append(" ".join(tok[1:]))
            parents.append(stack[-1] if stack else -1)
            offsets.append(None)
            channels.append([])
        elif tok[0] == "End":
            pending_end = True
        elif tok[0] == "{":
            if pending_end:
                stack.append(-2)  # marker for end site
            else:
                stack.append(len(names) - 1)
        elif tok[0] == "}":
            top = stack.pop()
            if top == -2:
                pending_end = False
        elif tok[0] == "OFFSET" and not pending_end:
            offsets[-1] = [float(v) for v in tok[1:4]]
        elif tok[0] == "CHANNELS":
            channels[-1] = tok[2:]
    mlines = [l for l in motion.strip().splitlines() if l.strip()]
    nframes = int(mlines[0].split(":")[1])
    ftime = float(mlines[1].split(":")[1])
    data = np.array([[float(v) for v in l.split()] for l in mlines[2 : 2 + nframes]])
    skel = Skeleton(names, np.array(parents), np.array(offsets))
    J = len(names)
    rot = np.zeros((nframes, J, 3, 3))
    root = np.tile(skel.offsets[0], (nframes, 1))
    col = 0
    for j in range(J):
        ch = channels[j]
        pos_axes = [c[0].upper() for c in ch if c.lower().endswith("position")]
        rot_axes = "".join(c[0].upper() for c in ch if c.lower().endswith("rotation"))
        block = data[:, col : col + len(ch)]
        col += len(ch)
        pi = [k for k, c in enumerate(ch) if c.lower().endswith("position")]
        ri = [k for k, c in enumerate(ch) if c.lower().endswith("rotation")]
        if pi:
            p = np.zeros((nframes, 3))
            for k, ax in zip(pi, pos_axes):
                p[:, "XYZ".index(ax)] = block[:, k]
            root = p if j == 0 else root
        if ri:
            rot[:, j] = R.from_euler(rot_axes, block[:, ri], degrees=True).as_matrix()
        else:
            rot[:, j] = np.eye(3)
    return Motion(skel, rot, root, fps=round(1.0 / ftime, 4))


_SAFE = re.compile(r"[^A-Za-z0-9_\-]")


def safe_name(s: str) -> str:
    return _SAFE.sub("_", s)
