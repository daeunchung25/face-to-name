/* =========================================================
   Face Wrapped – script.js

   How it works:
   1. MediaPipe Face Landmarker finds your face in each webcam frame
      and returns 52 "blendshape" scores (0 to 1), e.g. browInnerUp,
      eyeSquintLeft, mouthSmileRight, jawOpen.
   2. For the first few seconds we record your RESTING face (baseline),
      so everything after that is measured relative to how your face
      looks when relaxed.
   3. During the session we count events (eyebrow raises, blinks...),
      track how much each feature moves, and classify every frame
      into an expression (joy, surprise, skepticism, concentration, neutral).
   4. At the end we turn those numbers into swipeable "Wrapped" cards.
   ========================================================= */

import {
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

/* ---------------- Settings you can tune ---------------- */

const CALIBRATION_SECONDS = 3;
const SESSION_SECONDS = 45;

// MediaPipe's "Left"/"Right" can come out mirrored depending on the camera.
// If the site says "left eyebrow" when you moved your right one, set this to true.
const SWAP_SIDES = false;

// Thresholds are measured AFTER subtracting your resting face.
// "On" = the moment an event starts, "Off" = when it's over (prevents double counting).
const T = {
  smile: 0.3,          // average of mouthSmileLeft/Right
  genuineEyes: 0.08,   // cheek + eye squint during a smile = a "real" (Duchenne) smile
  browRaiseOn: 0.25, browRaiseOff: 0.12,
  furrowOn: 0.22,    furrowOff: 0.1,
  blinkOn: 0.5,      blinkOff: 0.3,   // raw values, not baseline-adjusted
  oneBrowOn: 0.2,    oneBrowOff: 0.1,
  smirk: 0.15,
  surpriseJaw: 0.2,
  wideEyes: 0.12,
  frown: 0.12,
  // Expressiveness score: an average intensity of this much = 100/100
  fullScoreIntensity: 0.18,
};

const PROMPTS = [
  "Tell us about the best meal you've ever had.",
  "Think about your most embarrassing moment. Yes, that one.",
  "Someone just said pineapple belongs on pizza. React.",
  "Describe your morning like it's a movie trailer.",
  "Think of someone you miss.",
  "Smile like it's your yearbook photo.",
];

/* ---------------- Labels & copy ---------------- */

const REGIONS = {
  leftBrow:  { label: "Left eyebrow",  short: "Left brow" },
  rightBrow: { label: "Right eyebrow", short: "Right brow" },
  leftEye:   { label: "Left eye",      short: "Left eye" },
  rightEye:  { label: "Right eye",     short: "Right eye" },
  mouth:     { label: "Mouth",         short: "Mouth" },
};

const REGION_QUIPS = {
  leftBrow:  "One eyebrow did most of the talking. It has opinions.",
  rightBrow: "One eyebrow did most of the talking. It has opinions.",
  leftEye:   "Squinting, widening, narrowing. Your eyes reacted before you did.",
  rightEye:  "Squinting, widening, narrowing. Your eyes reacted before you did.",
  mouth:     "Smiles, frowns, smirks. Your mouth did the heavy lifting.",
};

const EMOTIONS = {
  joy:      { label: "Joy",           word: "joy" },
  surprise: { label: "Surprise",      word: "surprise" },
  skeptic:  { label: "Skepticism",    word: "skepticism" },
  focus:    { label: "Concentration", word: "concentration" },
  neutral:  { label: "Neutral",       word: "a perfectly straight face" },
};

const NOUNS = {
  joy:      { name: "Optimist",    desc: "Smiling is your default setting. Even the awkward prompts couldn't keep it down." },
  surprise: { name: "Dramatist",   desc: "Everything is news to you, and your eyebrows make sure everyone knows it." },
  skeptic:  { name: "Skeptic",     desc: "You heard every prompt. Your face would like to see some sources." },
  focus:    { name: "Overthinker", desc: "A furrowed brow, a thinking face. You process everything, visibly." },
  neutral:  { name: "Poker Face",  desc: `${SESSION_SECONDS} seconds of prompts and your face gave away almost nothing.` },
};

const ADJECTIVES = {
  Sunny:      "With a steady streak of smiles underneath it all.",
  Dramatic:   "Plus a flair for the big reaction.",
  Skeptical:  "But one eyebrow was always taking notes.",
  Brooding:   "With a furrowed brow on standby.",
  Unfiltered: "And you show all of it. Your face doesn't do subtle.",
  "Low-Key":  "All delivered at a quiet, steady volume.",
  Polite:     "Plus a lot of polite smiles that never quite reached your eyes.",
  Steady:     "Even-keeled, start to finish.",
  Certified:  "Truly, nothing got through.",
  Mysterious: "Small flickers here and there, but you kept the full story to yourself.",
};
const EMOTION_ADJ = { joy: "Sunny", surprise: "Dramatic", skeptic: "Skeptical", focus: "Brooding" };

// Big line drawings of facial features, shown behind each card
const DECO = {
  face:  `<svg viewBox="0 0 300 300"><path d="M60 95 Q95 68 130 90"/><path d="M170 90 Q205 68 240 95"/><circle cx="95" cy="138" r="10"/><circle cx="205" cy="138" r="10"/><path d="M85 205 Q150 258 215 205"/></svg>`,
  smile: `<svg viewBox="0 0 300 160"><path d="M30 40 Q150 170 270 40"/></svg>`,
  brows: `<svg viewBox="0 0 300 120"><path d="M20 95 Q75 20 135 78"/><path d="M165 78 Q225 20 280 95"/></svg>`,
  eye:   `<svg viewBox="0 0 300 160"><path d="M20 80 Q150 -10 280 80 Q150 170 20 80Z"/><circle cx="150" cy="80" r="26"/></svg>`,
  blink: `<svg viewBox="0 0 300 160"><path d="M20 60 Q150 140 280 60"/><path d="M75 94 L60 132"/><path d="M150 102 L150 146"/><path d="M225 94 L240 132"/></svg>`,
};

/* ---------------- DOM ---------------- */

const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const ctx = overlay.getContext("2d");
const drawer = new DrawingUtils(ctx);
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let faceLandmarker = null;
let stream = null;
let state = null;
let lastVideoTime = -1;
let lastProcessed = 0;

/* ---------------- Screens ---------------- */

function showScreen(name) {
  document.querySelectorAll(".screen").forEach((s) => s.classList.remove("active"));
  $(`screen-${name}`).classList.add("active");
}

function setStatus(text) {
  $("introStatus").textContent = text;
}

/* ---------------- State ---------------- */

function newStat() {
  return { n: 0, sum: 0, sumSq: 0 };
}
function addStat(stat, v) {
  stat.n++;
  stat.sum += v;
  stat.sumSq += v * v;
}
function stdDev(stat) {
  if (stat.n < 2) return 0;
  const mean = stat.sum / stat.n;
  return Math.sqrt(Math.max(0, stat.sumSq / stat.n - mean * mean));
}
function emptyEmotions() {
  return { joy: 0, surprise: 0, skeptic: 0, focus: 0, neutral: 0 };
}

function resetState() {
  state = {
    phase: "idle", // idle | calibrating | session | done
    phaseStart: 0,
    baselineSum: {},
    baselineCount: 0,
    baseline: {},
    faceFrames: 0,
    faceSeconds: 0,
    smileFrames: 0,
    genuineFrames: 0,
    intensitySum: 0,
    events: { browRaise: 0, furrow: 0, blink: 0, oneBrow: 0 },
    active: { browRaise: false, furrow: false, blink: false, oneBrow: false },
    emotions: emptyEmotions(),
    regions: Object.fromEntries(Object.keys(REGIONS).map((k) => [k, newStat()])),
    promptIndex: 0,
    prompts: PROMPTS.map(() => ({ intensity: 0, n: 0, emotions: emptyEmotions() })),
  };
  lastVideoTime = -1;
  lastProcessed = 0;
}

/* ---------------- Model + camera ---------------- */

async function loadModel() {
  if (faceLandmarker) return;
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
  });
  try {
    faceLandmarker = await FaceLandmarker.createFromOptions(fileset, options("GPU"));
  } catch (err) {
    console.warn("GPU not available, using CPU instead.", err);
    faceLandmarker = await FaceLandmarker.createFromOptions(fileset, options("CPU"));
  }
}

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
}

function stopCamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
}

/* ---------------- Start ---------------- */

$("startBtn").addEventListener("click", async () => {
  const btn = $("startBtn");
  btn.disabled = true;
  try {
    setStatus("Loading the face model. This takes a few seconds the first time.");
    await loadModel();
    setStatus("Waiting for camera permission.");
    await startCamera();
    setStatus("");
    resetState();
    showScreen("session");
    beginPhase("calibrating");
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    stopCamera();
    if (err && err.name === "NotAllowedError") {
      setStatus("Camera access is blocked. Allow the camera from your browser's address bar, then press Start again.");
    } else if (err && err.name === "NotFoundError") {
      setStatus("No camera was found. Connect a webcam and press Start again.");
    } else {
      setStatus(`The session couldn't start: ${err && err.message ? err.message : err}`);
    }
  } finally {
    btn.disabled = false;
  }
});

function beginPhase(phase) {
  state.phase = phase;
  state.phaseStart = performance.now();
  if (phase === "calibrating") {
    $("phaseLabel").textContent = "Getting your resting face";
    $("prompt").textContent = "Relax your face and look at the camera.";
  } else if (phase === "session") {
    state.promptIndex = 0;
    $("phaseLabel").textContent = `Prompt 1 of ${PROMPTS.length}`;
    $("prompt").textContent = PROMPTS[0];
  }
}

/* ---------------- Main loop ---------------- */

function loop(now) {
  if (state.phase !== "calibrating" && state.phase !== "session") return;

  // Only run the model when the webcam has a new frame
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const dt = lastProcessed ? Math.min((now - lastProcessed) / 1000, 0.2) : 0;
    lastProcessed = now;
    const result = faceLandmarker.detectForVideo(video, now);
    handleResult(result, dt);
  }

  updateClock(now);
  requestAnimationFrame(loop);
}

function updateClock(now) {
  const elapsed = (now - state.phaseStart) / 1000;

  if (state.phase === "calibrating") {
    $("timerBar").style.width = `${Math.min(100, (elapsed / CALIBRATION_SECONDS) * 100)}%`;
    // Keep calibrating until we've actually seen the face enough times
    if (elapsed >= CALIBRATION_SECONDS && state.baselineCount >= 15) {
      for (const key in state.baselineSum) {
        state.baseline[key] = state.baselineSum[key] / state.baselineCount;
      }
      beginPhase("session");
    }
    return;
  }

  if (state.phase === "session") {
    $("timerBar").style.width = `${Math.min(100, (elapsed / SESSION_SECONDS) * 100)}%`;
    const perPrompt = SESSION_SECONDS / PROMPTS.length;
    const index = Math.min(PROMPTS.length - 1, Math.floor(elapsed / perPrompt));
    if (index !== state.promptIndex) {
      state.promptIndex = index;
      $("phaseLabel").textContent = `Prompt ${index + 1} of ${PROMPTS.length}`;
      $("prompt").textContent = PROMPTS[index];
    }
    if (elapsed >= SESSION_SECONDS) finish();
  }
}

function handleResult(result, dt) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const landmarks = result.faceLandmarks && result.faceLandmarks[0];
  const shapes = result.faceBlendshapes && result.faceBlendshapes[0];

  $("noFace").classList.toggle("hidden", Boolean(landmarks));
  if (!landmarks || !shapes) return;

  drawFeatures(landmarks);
  const m = toMap(shapes.categories);

  if (state.phase === "calibrating") {
    for (const key in m) state.baselineSum[key] = (state.baselineSum[key] || 0) + m[key];
    state.baselineCount++;
    updateMeters(computeSignals(m, false));
  } else if (state.phase === "session") {
    record(m, dt);
  }
}

function drawFeatures(lm) {
  const css = getComputedStyle(document.documentElement);
  const brows = css.getPropertyValue("--bubble").trim();
  const eyes = css.getPropertyValue("--mint").trim();
  const lips = css.getPropertyValue("--butter").trim();
  drawer.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: brows, lineWidth: 4 });
  drawer.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: brows, lineWidth: 4 });
  drawer.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: eyes, lineWidth: 3 });
  drawer.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: eyes, lineWidth: 3 });
  drawer.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: lips, lineWidth: 3 });
}

/* ---------------- Turning blendshapes into signals ---------------- */

function toMap(categories) {
  const m = {};
  for (const c of categories) {
    let name = c.categoryName;
    if (SWAP_SIDES) {
      if (name.endsWith("Left")) name = name.slice(0, -4) + "Right";
      else if (name.endsWith("Right")) name = name.slice(0, -5) + "Left";
    }
    m[name] = c.score;
  }
  return m;
}

const avg = (a, b) => (a + b) / 2;

function computeSignals(m, useBaseline = true) {
  // a(key) = how far this blendshape is above your resting face
  const a = (key) => Math.max(0, (m[key] || 0) - (useBaseline ? state.baseline[key] || 0 : 0));

  const browUpL = Math.max(a("browOuterUpLeft"), a("browInnerUp"));
  const browUpR = Math.max(a("browOuterUpRight"), a("browInnerUp"));
  const browDownL = a("browDownLeft");
  const browDownR = a("browDownRight");
  const smileL = a("mouthSmileLeft");
  const smileR = a("mouthSmileRight");
  const squintL = a("eyeSquintLeft");
  const squintR = a("eyeSquintRight");
  const wideL = a("eyeWideLeft");
  const wideR = a("eyeWideRight");

  const s = {
    browUp: avg(browUpL, browUpR),
    browDown: avg(browDownL, browDownR),
    // one brow up while the other stays down = the "skeptical eyebrow"
    browAsym: Math.abs((a("browOuterUpLeft") - browDownL) - (a("browOuterUpRight") - browDownR)),
    smile: avg(smileL, smileR),
    smirk: Math.abs(smileL - smileR),
    eyeSquint: avg(avg(squintL, squintR), avg(a("cheekSquintLeft"), a("cheekSquintRight"))),
    eyeWide: avg(wideL, wideR),
    blink: avg(m.eyeBlinkLeft || 0, m.eyeBlinkRight || 0), // raw: blinking isn't relative
    jaw: a("jawOpen"),
    frown: avg(a("mouthFrownLeft"), a("mouthFrownRight")),
  };

  // How much
