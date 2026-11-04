# mocap-yt-test

Motion capture from YouTube videos with [MocapAnything V2](https://github.com/phongdaot/MocapAnything)
(`video2pose2rot`), running **CPU-only**, onto a generated Mixamo-convention humanoid rig.
The output is BVH plus animated glTF.

```
YouTube URL / local file / hf:// ──► clip (15 fps, 720p) ──► frames
     ──► RMBG-1.4 matte + crop ──► DINOv2-large tokens (cached)
     ──► MocapAnything video2pose2rot  (conditioned on the Humanoid rig's reference frame)
     ──► joint positions + 6D rotations ──► BVH ──► glTF (skinned mannequin + bones)
                                                └─► preview mp4 / screenshots
```

## Quick start

```bash
git submodule update --init           # third_party/MocapAnything (pinned)
uv sync                               # CPU-only torch from download.pytorch.org/whl/cpu
uv run mocap-yt setup                 # weights: MocapAnything ckpt, RMBG-1.4, DINOv2-large, T5-base
uv run mocap-yt rig                   # generate the humanoid + its MocapAnything dataset
uv run mocap-yt run --publish         # every sample in configs/samples.yaml
uv run mocap-yt run demo_trap --publish   # ...or selected ids
uv run --group dev pytest             # tests
```

To add a video, add an entry to [`configs/samples.yaml`](configs/samples.yaml) with an `id`,
a `source` (a URL, a local path, or `hf://…`), and optionally `start`, `duration`, `crop`
and `cookies`. Every stage writes into `work/` (gitignored) and is skipped if its output
already exists. `--publish` copies the lightweight results into `assets/samples/<id>/`.

### YouTube access

`mocap-yt` downloads with `yt-dlp`. YouTube asks datacenter IPs (cloud VMs, CI) to
"sign in to confirm you're not a bot". If you hit that, either:

* export your browser's YouTube cookies to a Netscape `cookies.txt` and set
  `cookies: path/to/cookies.txt` on the sample, or
* download the video yourself and point `source:` at the local file.

## Layout

| Path | What |
|---|---|
| `src/mocap_yt/skeleton.py` | `Skeleton`, forward kinematics, procedural **Mixamo-convention humanoid** (`mixamo_humanoid`) |
| `src/mocap_yt/mesh.py` | capsule mannequin skin + linear blend skinning |
| `src/mocap_yt/bvh.py` | BVH writer/reader (Mixamo-style channels: `Z Y X`, `End Site` leaves) |
| `src/mocap_yt/gltf.py` | glTF 2.0 export (joint node tree, skin, meshes, animation); `.glb` or `.gltf` |
| `src/mocap_yt/render.py` | CPU z-buffer rasterizer + skeleton overlays (no OpenGL/Blender needed) |
| `src/mocap_yt/rig_assets.py` | builds the MocapAnything **"species" dataset** for a rig without Blender (see below) |
| `src/mocap_yt/features.py` | CPU port of upstream's RMBG + crop + DINOv2 preprocessing (upstream's version hard-codes CUDA) |
| `src/mocap_yt/inference.py` | loads the released checkpoint, runs `video2pose2rot`, sliding windows for clips longer than 300 frames, BVH export |
| `src/mocap_yt/video.py` | yt-dlp download, ffmpeg trim/resample, frame extraction |
| `src/mocap_yt/pipeline.py`, `cli.py` | config-driven orchestration (`mocap-yt …`) |
| `scripts/gltf_screenshots.sh` | renders every published GLB in three.js (headless Chromium via Playwright) |
| `third_party/MocapAnything` | upstream, git submodule pinned at `c99fb22` |
| `assets/rigs/humanoid/` | the generated rig: BVH, glTF/GLB, previews, reference render |
| `assets/samples/<id>/` | per-video results: `.glb`, `.bvh`, preview `.mp4`, screenshots, `prediction.json` |

## The humanoid rig

I didn't find a free, redistributable humanoid that suits this model. Mixamo needs an Adobe
login and its assets can't be redistributed. The CC-BY Khronos samples (`RiggedFigure`,
`CesiumMan`) have 19 joints named like `leg_joint_L_1`, with no head or toe end joints.
MocapAnything embeds every joint **name** with T5. Its `obj1k` training split is mostly
Mixamo-style humanoids. So the rig is **generated** by `mixamo_humanoid()`:

* the 25-joint Mixamo "no fingers" hierarchy (`Hips, Spine, Spine1, Spine2, Neck, Head,
  HeadTop_End, {Left,Right}{Shoulder,Arm,ForeArm,Hand,UpLeg,Leg,Foot,ToeBase,Toe_End}`).
  `fingers: true` adds the 4-segment finger chains;
* anthropometric proportions (fractions of stature, 1.75 m by default), T-pose rest,
  Y-up, facing +Z, meters;
* a capsule "wooden mannequin" skin, rigidly weighted, that is rendered for the
  reference image and used in the glTF.

Files in `assets/rigs/humanoid/`: `humanoid_tpose.bvh`, `humanoid_skeleton.gltf` (bones only),
`humanoid_rigged.glb` (mannequin + bones, skinned), `humanoid_calibration.{bvh,glb}`,
`preview_*.png`, `reference_render.png` / `reference_model_input.png` (what DINOv2 sees),
`rig.json`.

### Rig → MocapAnything dataset (Blender-free)

Upstream builds per-rig data from an FBX with Blender (`preprocess/run_pipeline.sh`).
`rig_assets.build_rig_dataset` writes the same files directly from the procedural rig
into `work/rigs/<Rig>/`:

| file | how |
|---|---|
| `bvh/<Rig>#calib/y{0,90,180,270}.bvh` | a 2 s calibration clip: frame 0 is the reference pose (A-pose by default), then every non-end joint moves a little so static-joint detection matches real data |
| `bvh_pose/…/y*.npz` | upstream's own `utils.bvh` + `utils.animation` (same math as `extract_bvh_pose.py`) |
| `species_info_dict.npy` | upstream's `preprocess/build_species_info.build` (topology, T5 name embeddings, static joints) |
| `cache/__mesh2pose1002_species_scale_cache.pkl` | same definition as `build_scale_cache.py` |
| `npz_train_image_only/…/y0.npz` | DINOv2 tokens of the rendered reference frame |
| `characters/<Rig>/<Rig>_ffs.bvh` | rest-pose template used to write predicted rotations to BVH |

## Results so far

| sample | source | frames | status |
|---|---|---|---|
| `yt_nIfU38nAYYk` | https://youtu.be/nIfU38nAYYk | – | **not downloaded**: YouTube refused this cloud VM (bot check, then HTTP 403 on the media). Needs `cookies:` or a local file, see above |
| `demo_trap` | MocapAnything demo clip `inputs/nbg_human/trap.mp4` | 300 @ 15 fps | ✅ `assets/samples/demo_trap/` |
| `demo_gt` | MocapAnything demo clip `inputs/nbg_human/gt.mp4` | 300 @ 15 fps | ✅ `assets/samples/demo_gt/` |

Each `assets/samples/<id>/` holds:

* `<id>_humanoid.glb`: the skinned mannequin + bones with the captured animation;
* `<id>_humanoid.bvh`;
* `<id>_preview.mp4` and `<id>_preview_strip.png`: input | capture front | capture side;
* `<id>_humanoid_threejs.png`: the GLB played back in three.js, an independent viewer check
  produced by `scripts/gltf_screenshots.sh`;
* `prediction.json`: timings and a consistency metric.

`fk_vs_pred_position_err_m` in `prediction.json` is the mean distance between the
model's predicted joint positions and forward kinematics of its predicted rotations,
around 8–10 cm on the demo clips. The model's two heads agree only approximately.

## Notes and limitations

* **No global trajectory.** The model predicts root-relative pose. The exporter keeps the
  lowest joint on the floor (`ground_motion`), and the root doesn't translate horizontally.
* Upstream caps one forward pass at 301 frames. Longer clips run in 300-frame windows with
  32-frame linear cross-fades.
* Upstream pads every skeleton to 150 joints. No weight depends on that size, so padding only
  to the rig's 25 joints gives the same output (`tests/test_inference.py`, to 1e-4) and cuts the
  forward pass for 300 frames from about 107 s to about 16 s on 4 vCPUs.
* CPU cost is dominated by preprocessing: RMBG-1.4 matting at 1024² plus DINOv2-large takes
  about 10 min per 300 frames on 4 vCPUs. The result is cached in `work/samples/<id>/features.npz`.
* Model licence: MocapAnything code and weights are MIT. RMBG-1.4 (BRIA AI) has its own,
  non-commercial terms.
