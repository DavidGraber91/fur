import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const MP_URL    = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";
const WASM_URL  = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const IRIS_MM = 11.7;

// MediaPipe Face Mesh landmark indices
const L_SIDE = 234, R_SIDE = 454;   // the two sides of the face
const FOREHEAD = 10, CHIN = 152, NOSE = 1;
const IRIS_A = 468, IRIS_B = 473;   // iris centers (the next 4 points are the iris edge)

const $ = id => document.getElementById(id);
const video = $("cam"), canvas = $("out"), ctx = canvas.getContext("2d");
const glCanvas = $("gl"), stage = $("stage"), frame = $("frame");
const statusEl = $("status"), overlay = $("overlay");

const params = new URLSearchParams(location.search);
const DEMO = params.has("demo");   // ?demo : a preview without a camera (also &yaw=30&pitch=10&roll=0)

// ====================== pose math (plain arrays, no THREE) ======================
// <pose>
const v3 = {
  sub: (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]],
  add: (a, b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]],
  mul: (a, s) => [a[0]*s, a[1]*s, a[2]*s],
  dot: (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2],
  len: a => Math.hypot(a[0], a[1], a[2]),
  norm: a => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0]/l, a[1]/l, a[2]/l]; },
  cross: (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]],
};

// Scene coordinates: origin = image center, x right, y UP, z toward the camera, units = pixels.
// MediaPipe: x,y normalized (y down), z "depth" (smaller = closer) on roughly the same scale as x.
function headPose(lm, cw, ch) {
  const S = i => [lm[i].x * cw - cw / 2, ch / 2 - lm[i].y * ch, -lm[i].z * cw];
  const A = S(L_SIDE), B = S(R_SIDE), T = S(FOREHEAD), C = S(CHIN), N = S(NOSE);

  const w3 = v3.len(v3.sub(B, A));                 // true face width (3D, doesn't shrink when turning)
  let up = v3.sub(T, C);
  const faceLen = v3.len(up);
  up = v3.norm(up);

  // "right" axis: cheek-to-cheek, made perpendicular to up
  let right = v3.sub(B, A);
  right = v3.norm(v3.sub(right, v3.mul(up, v3.dot(right, up))));
  // "forward" axis (out of the face) = right x up
  let fwd = v3.cross(right, up);
  // The nose must point forward; if not, the axes are mirrored, so flip them.
  const mid = v3.mul(v3.add(A, B), 0.5);
  if (v3.dot(fwd, v3.sub(N, mid)) < 0) {
    right = v3.mul(right, -1);
    fwd = v3.cross(right, up);
  }
  fwd = v3.norm(fwd);
  return { right, up, fwd, T, w3, faceLen };
}
// </pose>

// ====================== smoothing ======================
const state = {};
function smooth(key, val, alpha = 0.35) {
  state[key] = state[key] === undefined ? val : state[key] + (val - state[key]) * alpha;
  return state[key];
}
function resetSmoothing() { for (const k in state) delete state[k]; }
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

// ====================== 3D scene ======================
const renderer = new THREE.WebGLRenderer({ canvas: glCanvas, alpha: true, antialias: true });
renderer.setPixelRatio(1);
renderer.setClearColor(0x000000, 0);
const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-320, 320, 240, -240, -5000, 5000);
scene.add(new THREE.HemisphereLight(0xffffff, 0x555566, 1.1));
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(-0.4, 0.8, 1);
scene.add(key);

let cw = 640, ch = 480;
function resizeStage(w, h) {
  cw = canvas.width = w; ch = canvas.height = h;
  renderer.setSize(w, h, false);
  camera.left = -w / 2; camera.right = w / 2; camera.top = h / 2; camera.bottom = -h / 2;
  camera.updateProjectionMatrix();
  frame.style.setProperty("--ar", (w / h).toFixed(4));
}
resizeStage(640, 480);

// Materials (the colour is changed from the colour picker)
const mats = {
  main:  new THREE.MeshStandardMaterial({ roughness: 0.65, metalness: 0.05, side: THREE.DoubleSide }),
  dark:  new THREE.MeshStandardMaterial({ roughness: 0.7,  metalness: 0.05, side: THREE.DoubleSide }),
  light: new THREE.MeshStandardMaterial({ roughness: 0.6,  metalness: 0.05, side: THREE.DoubleSide }),
  band:  new THREE.MeshStandardMaterial({ color: 0xb3262e, roughness: 0.6, side: THREE.DoubleSide }),
  cream: new THREE.MeshStandardMaterial({ color: 0xe8e2d0, roughness: 0.6, side: THREE.DoubleSide }),
};
function setHatColor(hex) {
  mats.main.color.set(hex);
  mats.dark.color.set(hex).multiplyScalar(0.75);
  mats.light.color.set(hex).offsetHSL(0, 0, 0.1);
}
setHatColor("#23242a");

// Hats are built with head width = 1 unit. Base of the hat is at y = 0, +y is up.
const mesh = (geo, mat, y = 0) => { const m = new THREE.Mesh(geo, mat); m.position.y = y; return m; };

function buildTopHat() {
  const g = new THREE.Group();
  g.add(mesh(new THREE.CylinderGeometry(0.62, 0.62, 0.035, 64), mats.dark, 0.0175));                 // brim
  g.add(mesh(new THREE.CylinderGeometry(0.43, 0.47, 0.85, 64), mats.main, 0.035 + 0.425));            // crown
  g.add(mesh(new THREE.CylinderGeometry(0.4675, 0.4685, 0.14, 64, 1, true), mats.band, 0.035 + 0.08)); // band
  g.add(mesh(new THREE.CylinderGeometry(0.43, 0.43, 0.01, 64), mats.light, 0.035 + 0.85));            // lighter top
  return g;
}
function buildFedora() {
  const g = new THREE.Group();
  g.add(mesh(new THREE.CylinderGeometry(0.8, 0.8, 0.02, 64), mats.dark, 0.01));                       // wide brim
  const profile = [[0.5,0.02],[0.5,0.2],[0.47,0.42],[0.43,0.56],[0.36,0.6],[0.3,0.6],[0.12,0.54],[0,0.52]]
    .map(([r, y]) => new THREE.Vector2(r, y));
  g.add(mesh(new THREE.LatheGeometry(profile, 64), mats.main));                                       // crown with a dent
  g.add(mesh(new THREE.CylinderGeometry(0.503, 0.503, 0.12, 64, 1, true), mats.cream, 0.09));         // band
  return g;
}
function buildBeanie() {
  const g = new THREE.Group();
  const dome = mesh(new THREE.SphereGeometry(0.5, 48, 32, 0, Math.PI * 2, 0, Math.PI / 2), mats.main, 0.18);
  dome.scale.y = 1.1;
  g.add(dome);
  g.add(mesh(new THREE.CylinderGeometry(0.52, 0.52, 0.2, 64, 1, true), mats.dark, 0.1));              // cuff
  g.add(mesh(new THREE.SphereGeometry(0.1, 24, 16), mats.light, 0.73 + 0.08));                        // pom-pom
  return g;
}

function buildFurHat() {
  const g = new THREE.Group();   // ריק עד שהמודל נטען
  new GLTFLoader().load(
    "fur_hat.glb",
    gltf => g.add(gltf.scene),
    undefined,
    err => { console.error(err); statusEl.textContent = "שגיאה בטעינת fur_hat.glb"; }
  );
  return g;
}

const HATS = {
  fur:    { build: buildFurHat, height: 0.78,  offset: 0, width: 2.00, ownColors: true },
  top:    { build: buildTopHat, height: 0.885, offset: 0.10 },
  fedora: { build: buildFedora, height: 0.62,  offset: 0.06 },
  beanie: { build: buildBeanie, height: 0.91,  offset: -0.04 },
};
for (const k in HATS) { HATS[k].obj = HATS[k].build(); HATS[k].obj.visible = false; }

// headGroup follows the head (position + 3D rotation); inside it:
//   occluder - an invisible ellipsoid in the shape of the head, so the part of the hat
//              that is inside / behind the head is hidden;
//   hatPivot - the hat itself, scaled to the measured head width.
const headGroup = new THREE.Group();
headGroup.visible = false;
scene.add(headGroup);

const occluder = new THREE.Mesh(
  new THREE.SphereGeometry(1, 40, 28),
  new THREE.MeshBasicMaterial({ colorWrite: false })   // writes depth only
);
occluder.renderOrder = -1;
headGroup.add(occluder);

const hatPivot = new THREE.Group();
for (const k in HATS) hatPivot.add(HATS[k].obj);
headGroup.add(hatPivot);

function showHat(type) {
  for (const k in HATS) HATS[k].obj.visible = (k === type);
}

// pose = { T:[x,y,z], quat:THREE.Quaternion, w3, faceLen }
function placeHat(T, quat, w3, faceLen) {
  const type = $("hatType").value;
  const hs = Number($("sH").value), wr = Number($("sW").value), off = Number($("sO").value);
  const Wh = w3 * wr;

  headGroup.position.set(T[0], T[1], T[2]);
  headGroup.quaternion.copy(quat);
  headGroup.visible = true;

  // Head ellipsoid (in head coordinates; origin = forehead point, -z = toward the back of the head)
  //occluder.position.set(0, -0.2 * w3, -0.58 * w3);
  //occluder.scale.set(0.5 * w3, 0.62 * w3, 0.62 * w3);
  
  occluder.position.set(0, -0.2 * w3, -0.68 * w3);
  occluder.scale.set(0.5 * w3, 0.62 * w3, 0.72 * w3);

  // The hat sits above the forehead, centred over the middle of the head
  hatPivot.position.set(0, faceLen * off, -0.58 * w3);
  hatPivot.scale.set(Wh, Wh * hs, Wh * 1.1);
  showHat(type);
  return Wh;
}

// ====================== UI ======================
const sW = $("sW"), sH = $("sH"), sO = $("sO");
function syncLabels() {
  $("vW").textContent = Number(sW.value).toFixed(2) + "×";
  $("vH").textContent = Number(sH.value).toFixed(2) + "×";
  $("vO").textContent = Number(sO.value).toFixed(2);
}
[sW, sH, sO].forEach(e => e.addEventListener("input", syncLabels));
$("hatColor").addEventListener("input", e => setHatColor(e.target.value));

function applyHatDefaults(type) {
  const h = HATS[type];
  sO.value = h.offset;
  if (h.width) sW.value = h.width;
  $("hatColor").disabled = !!h.ownColors;   // לכובע הפרווה יש טקסטורות משלו
  syncLabels();
}
$("hatType").addEventListener("change", e => applyHatDefaults(e.target.value));

applyHatDefaults($("hatType").value);
syncLabels();

// ====================== model + camera ======================
let landmarker = null, stream = null, facing = "user", running = false, lastVideoTime = -1;

async function loadModel() {
  if (landmarker) return;
  statusEl.textContent = "טוען מודל זיהוי פנים…";
  const { FaceLandmarker, FilesetResolver } = await import(MP_URL);
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const make = delegate => FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numFaces: 1,
  });
  try { landmarker = await make("GPU"); }
  catch (e) { console.warn("GPU failed, falling back to CPU", e); landmarker = await make("CPU"); }
}

async function startCamera() {
  fileMode = false;
  video.removeAttribute("src");
  video.loop = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } },
  });
  video.srcObject = stream;
  await video.play();
  await new Promise(r => video.videoWidth ? r() : video.addEventListener("loadedmetadata", r, { once: true }));
  resizeStage(video.videoWidth, video.videoHeight);
  stage.classList.toggle("mirror", facing === "user");
  resetSmoothing();
  lastVideoTime = -1;
}

let fileMode = false;

$("file").addEventListener("change", async e => {
  const f = e.target.files[0];
  if (!f) return;
  try {
    await loadModel();
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    video.srcObject = null;
    video.src = URL.createObjectURL(f);
    video.loop = true;
    video.muted = true;
    await video.play();
    await new Promise(r => video.videoWidth ? r() : video.addEventListener("loadedmetadata", r, { once: true }));
    resizeStage(video.videoWidth, video.videoHeight);
    stage.classList.remove("mirror");      // קובץ לא צריך מראה
    resetSmoothing();
    lastVideoTime = -1;
    fileMode = true;
    overlay.classList.add("hidden");
    if (!running) { running = true; requestAnimationFrame(loop); }
  } catch (err) {
    statusEl.textContent = "שגיאה: " + (err.message || err);
  }
});

$("start").addEventListener("click", async () => {
  try {
    $("start").disabled = true;
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("הדפדפן לא מאפשר מצלמה. צריך להריץ מכתובת https או מ-localhost.");
    }
    await loadModel();
    statusEl.textContent = "מפעיל מצלמה…";
    await startCamera();
    overlay.classList.add("hidden");
    $("flip").disabled = false;
    $("start").textContent = "המצלמה פועלת";
    statusEl.textContent = "";
    if (!running) { running = true; requestAnimationFrame(loop); }
  } catch (err) {
    console.error(err);
    statusEl.textContent = "שגיאה: " + (err.message || err);
    $("start").disabled = false;
  }
});

$("flip").addEventListener("click", async () => {
  facing = facing === "user" ? "environment" : "user";
  try { await startCamera(); } catch (e) { statusEl.textContent = "שגיאה: " + e.message; }
});

// ====================== main loop ======================
const basis = new THREE.Matrix4(), qTarget = new THREE.Quaternion(), qSmooth = new THREE.Quaternion();
let hasQ = false;
const irisDiam = (lm, c) => {
  const p = [1, 2, 3, 4].map(k => ({ x: lm[c + k].x * cw, y: lm[c + k].y * ch }));
  return (Math.hypot(p[0].x - p[2].x, p[0].y - p[2].y) + Math.hypot(p[1].x - p[3].x, p[1].y - p[3].y)) / 2;
};
const deg = r => (r * 180 / Math.PI);

function updateReadout(w3, Wh, hatType, hs, wr, yawDeg, pitchDeg) {
  $("stFacePx").textContent = Math.round(w3) + " px";
  $("stHatPx").textContent = Math.round(Wh) + " px";
  const mm = state.mmPerPx;
  $("stFaceCm").textContent = mm ? (w3 * mm / 10).toFixed(1) + " ס\"מ" : "מכייל…";
  $("stRatio").textContent = (HATS[hatType].height * hs * wr).toFixed(2);
  $("stYaw").textContent = Math.round(yawDeg) + "°";
  $("stPitch").textContent = Math.round(pitchDeg) + "°";
}

function loop(time) {
  requestAnimationFrame(loop);
  if (DEMO) return runDemo(time);
  if (!(stream || fileMode)  || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  ctx.drawImage(video, 0, 0, cw, ch);
  const res = landmarker.detectForVideo(video, performance.now());

  if (!res.faceLandmarks.length) {
    resetSmoothing(); hasQ = false;
    headGroup.visible = false;
    statusEl.textContent = "לא זוהו פנים. התקרב או שפר תאורה.";
    renderer.render(scene, camera);
    return;
  }
  statusEl.textContent = "";

  const lm = res.faceLandmarks[0];
  const pose = headPose(lm, cw, ch);

  // Rotation from the three head axes, then smoothed (slerp)
  basis.makeBasis(new THREE.Vector3(...pose.right), new THREE.Vector3(...pose.up), new THREE.Vector3(...pose.fwd));
  qTarget.setFromRotationMatrix(basis);
  //if (!hasQ) { qSmooth.copy(qTarget); hasQ = true; } else qSmooth.slerp(qTarget, 0.4);
  if (!hasQ) { qSmooth.copy(qTarget); hasQ = true; } else qSmooth.slerp(qTarget, 1);


  // yaw/pitch for the readout
  const yaw = Math.atan2(pose.fwd[0], pose.fwd[2]);
  const pitch = Math.asin(clamp(pose.fwd[1], -1, 1));

  // Real-world scale from the iris, only when the face is roughly frontal
  if (lm.length >= 478 && Math.abs(yaw) < 0.26) {
    const iris = (irisDiam(lm, IRIS_A) + irisDiam(lm, IRIS_B)) / 2;
    if (iris > 2) smooth("mmPerPx", IRIS_MM / iris, 0.08);
  }

  const T = [smooth("tx", pose.T[0], 1), smooth("ty", pose.T[1], 1), smooth("tz", pose.T[2], 1)];
  const w3 = smooth("w3", pose.w3, 0.3), faceLen = smooth("fl", pose.faceLen, 0.3);

  const Wh = placeHat(T, qSmooth, w3, faceLen);
  renderer.render(scene, camera);

  updateReadout(w3, Wh, $("hatType").value, Number(sH.value), Number(sW.value), deg(yaw), deg(pitch));
}

// ====================== demo without a camera (?demo) ======================
let demoInit = false;
function runDemo(time) {
  if (!demoInit) {
    demoInit = true;
    stage.classList.remove("mirror");
    overlay.classList.add("hidden");
    occluder.material.colorWrite = true;           // make the head visible in the preview
    occluder.material.color.set(0xd9a98a);
    if (params.has("hat")) { $("hatType").value = params.get("hat"); sO.value = HATS[params.get("hat")].offset; syncLabels(); }
  }
  const g = ctx.createLinearGradient(0, 0, 0, ch);
  g.addColorStop(0, "#cfd8e3"); g.addColorStop(1, "#8f9bab");
  ctx.fillStyle = g; ctx.fillRect(0, 0, cw, ch);

  const t = time / 1000;
  const yaw   = params.has("yaw")   ? +params.get("yaw")   : 50 * Math.sin(t * 0.8);
  const pitch = params.has("pitch") ? +params.get("pitch") : 20 * Math.sin(t * 0.55);
  const roll  = params.has("roll")  ? +params.get("roll")  : 12 * Math.sin(t * 0.4);
  const e = new THREE.Euler(THREE.MathUtils.degToRad(pitch), THREE.MathUtils.degToRad(yaw), THREE.MathUtils.degToRad(roll), "YXZ");
  qTarget.setFromEuler(e);
  const w3 = 200, faceLen = 260;
  const Wh = placeHat([0, 70, 0], qTarget, w3, faceLen);
  renderer.render(scene, camera);
  updateReadout(w3, Wh, $("hatType").value, Number(sH.value), Number(sW.value), yaw, pitch);
}
if (DEMO) requestAnimationFrame(loop);