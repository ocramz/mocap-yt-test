// Render a glTF/GLB animation in three.js (headless Chromium) at several times and
// save a horizontal strip PNG — an independent check that skinning + animation import
// correctly in a standard viewer.
//   node scripts/gltf_screenshot.mjs <file.glb> <out.png> [t0,t1,...]
import { createRequire } from 'node:module';
const { chromium } = createRequire(import.meta.url)('playwright');
import fs from 'node:fs';
import path from 'node:path';

const [glb, out, timesArg] = process.argv.slice(2);
const times = (timesArg || '0,4,8,12,16').split(',').map(Number);
const data = fs.readFileSync(glb).toString('base64');
const html = `<!doctype html><html><body style="margin:0;background:#fff">
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
const W=360,H=420, times=${JSON.stringify(times)};
const r=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true}); r.setSize(W*times.length,H);
r.setClearColor(0xffffff); r.setScissorTest(true); document.body.appendChild(r.domElement);
const scene=new THREE.Scene(); scene.add(new THREE.HemisphereLight(0xffffff,0x666666,2.2));
const d=new THREE.DirectionalLight(0xffffff,1.5); d.position.set(1,2,3); scene.add(d);
scene.add(new THREE.GridHelper(4,8,0xbbbbbb,0xdddddd));
const buf=Uint8Array.from(atob("${data}"),c=>c.charCodeAt(0)).buffer;
new GLTFLoader().parse(buf,'',g=>{
  scene.add(g.scene); const mixer=new THREE.AnimationMixer(g.scene);
  const clip=g.animations[0]; if(clip) mixer.clipAction(clip).play();
  const cam=new THREE.PerspectiveCamera(35,W/H,0.01,100); cam.position.set(1.2,1.3,3.6); cam.lookAt(0,0.9,0);
  times.forEach((t,i)=>{ mixer.setTime(t); g.scene.updateMatrixWorld(true);
    r.setViewport(i*W,0,W,H); r.setScissor(i*W,0,W,H); r.render(scene,cam); });
  window.__info={clip: clip? clip.name:null, duration: clip? clip.duration:0, tracks: clip? clip.tracks.length:0};
  window.__done=true;
}, e=>{window.__err=String(e); window.__done=true;});
</script></body></html>`;
const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 360 * times.length, height: 420 } });
page.on('console', m => console.log('[page]', m.text()));
await page.setContent(html);
await page.waitForFunction(() => window.__done, null, { timeout: 60000 });
const err = await page.evaluate(() => window.__err);
if (err) { console.error(err); process.exit(1); }
console.log(JSON.stringify(await page.evaluate(() => window.__info)));
fs.mkdirSync(path.dirname(out), { recursive: true });
await page.locator('canvas').screenshot({ path: out });
await browser.close();
