#!/usr/bin/env bash
# Render animated GLBs in three.js (headless Chromium) and save <name>_threejs.png next
# to each. Needs node + the preinstalled playwright.
#   scripts/gltf_screenshots.sh                      # every published GLB under assets/
#   scripts/gltf_screenshots.sh assets/samples/windmill/windmill_humanoid.glb ...
set -euo pipefail
cd "$(dirname "$0")/.."
if (( $# )); then
  glbs=("$@")
else
  glbs=(assets/samples/*/*.glb assets/rigs/*/*_calibration.glb)
fi
for glb in "${glbs[@]}"; do
  [[ -f "$glb" ]] || continue
  out="${glb%.glb}_threejs.png"
  times="auto:5"  # 5 evenly spaced times over the clip (0,4,8,12,16 s for a 20 s clip)
  [[ "$glb" == *_calibration.glb ]] && times="0,0.5,1,1.5"
  echo "$glb -> $out"
  node scripts/gltf_screenshot.mjs "$glb" "$out" "$times" | grep -v '^\[page\]'
done
