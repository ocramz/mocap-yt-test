#!/usr/bin/env bash
# Render every published animated GLB under assets/ in three.js (headless Chromium)
# and save <name>_threejs.png next to it. Needs node + the preinstalled playwright.
set -euo pipefail
cd "$(dirname "$0")/.."
for glb in assets/samples/*/*.glb assets/rigs/*/*_calibration.glb; do
  [[ -f "$glb" ]] || continue
  out="${glb%.glb}_threejs.png"
  times="0,4,8,12,16"
  [[ "$glb" == *_calibration.glb ]] && times="0,0.5,1,1.5"
  echo "$glb -> $out"
  node scripts/gltf_screenshot.mjs "$glb" "$out" "$times" | grep -v '^\[page\]'
done
