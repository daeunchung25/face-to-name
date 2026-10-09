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

  // How much each feature is "doing" right now (roughly 0–1)
  s.regions = {
    leftBrow: browUpL + browDownL,
    rightBrow: browUpR + browDownR,
    leftEye: squintL + wideL,
    rightEye: squintR + wideR,
    mouth: Math.max(
      s.smile,
      s.frown,
      a("mouthPucker"),
      a("mouthFunnel"),
      avg(a("mouthStretchLeft"), a("mouthStretchRight"))
    ),
  };

  // Overall expressiveness of this frame (talking/jaw counts for less)
  s.intensity =
    (s.browUp + s.browDown + s.smile + s.frown + s.eyeWide + s.eyeSquint + s.browAsym + s.jaw * 0.5) / 4;

  return s;
}

function classify(s) {
  if (s.smile > T.smile) return "joy";
  if (s.browUp > T.browRaiseOn && (s.eyeWide > T.wideEyes || s.jaw > T.surpriseJaw)) return "surprise";
  if (s.browAsym > T.oneBrowOn || (s.smirk > T.smirk && s.smile > 0.08)) return "skeptic";
  if (s.browDown > T.furrowOn || s.frown > T.frown) return "focus";
  return "neutral";
}

// Counts an event once when the value crosses "on", and resets when it drops below "off"
function edge(name, value, on, off) {
  if (!state.active[name] && value > on) {
    state.active[name] = true;
    state.events[name]++;
  } else if (state.active[name] && value < off) {
    state.active[name] = false;
  }
}

function record(m, dt) {
  const s = computeSignals(m);
  state.faceFrames++;
  state.faceSeconds += dt;
  state.intensitySum += s.intensity;

  if (s.smile > T.smile) {
    state.smileFrames++;
    if (s.eyeSquint > T.genuineEyes) state.genuineFrames++;
  }

  edge("browRaise", s.browUp, T.browRaiseOn, T.browRaiseOff);
  edge("furrow", s.browDown, T.furrowOn, T.furrowOff);
  edge("blink", s.blink, T.blinkOn, T.blinkOff);
  edge("oneBrow", s.browAsym, T.oneBrowOn, T.oneBrowOff);

  const emotion = classify(s);
  state.emotions[emotion]++;

  const blinking = s.blink > 0.4;
  for (const [key, value] of Object.entries(s.regions)) {
    if (blinking && key.endsWith("Eye")) continue; // blinks aren't "expressions"
    addStat(state.regions[key], value);
  }

  const p = state.prompts[state.promptIndex];
  p.intensity += s.intensity;
  p.n++;
  p.emotions[emotion]++;

  updateMeters(s);
}

function updateMeters(s) {
  const pct = (v) => `${Math.min(100, v * 160)}%`;
  $("m-brows").style.width = pct(Math.max(s.browUp, s.browDown));
  $("m-eyes").style.width = pct(Math.max(s.eyeSquint, s.eyeWide));
  $("m-smile").style.width = pct(s.smile);
  $("m-jaw").style.width = pct(s.jaw);
}

/* ---------------- Results ---------------- */

function finish() {
  state.phase = "done";
  stopCamera();

  if (state.faceFrames < 60) {
    showScreen("intro");
    setStatus("Your face was only visible for a moment. Sit facing the camera in good light and press Start again.");
    return;
  }
  renderCards(buildResults());
  showScreen("results");
}

const pctOf = (part, whole) => (whole ? Math.round((100 * part) / whole) : 0);
const topKey = (obj) => Object.keys(obj).reduce((a, b) => (obj[b] > obj[a] ? b : a));

function buildResults() {
  const frames = state.faceFrames;

  const emotionPct = {};
  for (const k in state.emotions) emotionPct[k] = pctOf(state.emotions[k], frames);

  const regionSpread = {};
  for (const k in state.regions) regionSpread[k] = stdDev(state.regions[k]);
  const topRegion = topKey(regionSpread);

  let bestPrompt = 0;
  let bestAvg = -1;
  state.prompts.forEach((p, i) => {
    const a = p.n > 5 ? p.intensity / p.n : -1;
    if (a > bestAvg) { bestAvg = a; bestPrompt = i; }
  });
  const bestPromptEmotion = topKey(state.prompts[bestPrompt].emotions);

  const meanIntensity = state.intensitySum / frames;
  const score = Math.round(100 * Math.min(1, Math.sqrt(meanIntensity / T.fullScoreIntensity)));

  const minutes = Math.max(state.faceSeconds, 1) / 60;

  const r = {
    seconds: Math.round(state.faceSeconds),
    frames,
    smilePct: pctOf(state.smileFrames, frames),
    genuinePct: pctOf(state.genuineFrames, state.smileFrames),
    smileFrames: state.smileFrames,
    events: { ...state.events },
    blinkRate: Math.round(state.events.blink / minutes),
    emotionPct,
    regionSpread,
    topRegion,
    bestPrompt: PROMPTS[bestPrompt],
    bestPromptEmotion,
    score,
  };
  r.persona = pickPersona(r);
  return r;
}

function pickPersona(r) {
  const e = r.emotionPct;
  const ranked = ["joy", "surprise", "skeptic", "focus"].sort((a, b) => e[b] - e[a]);
  const [first, second] = ranked;

  const noun = e.neutral >= 65 || e[first] < 6 ? "neutral" : first;

  let adjective;
  if (noun === "neutral") adjective = r.score < 30 ? "Certified" : "Mysterious";
  else if (r.score >= 75) adjective = "Unfiltered";
  else if (r.events.oneBrow >= 3 && noun !== "skeptic") adjective = "Skeptical";
  else if (e[second] >= 6) adjective = EMOTION_ADJ[second];
  else if (r.score <= 25) adjective = "Low-Key";
  else if (r.smilePct >= 15 && r.genuinePct < 35) adjective = "Polite";
  else adjective = "Steady";

  return {
    name: `The ${adjective} ${NOUNS[noun].name}`,
    desc: `${NOUNS[noun].desc} ${ADJECTIVES[adjective]}`,
  };
}

/* ---------------- Card templates ---------------- */

const count = (n, unit = "") =>
  `<span class="count" data-to="${n}">${REDUCED_MOTION ? n : 0}</span>${unit ? `<span class="unit">${unit}</span>` : ""}`;

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function bars(rows) {
  const max = Math.max(...rows.map((r) => r.value), 0.0001);
  return `<div class="bars">${rows
    .map(
      (r) => `<div class="bar-row"><span>${r.label}</span>
        <div class="bar-track"><div class="bar-fill" style="--w:${Math.round((r.value / max) * 100)}%"></div></div>
        <b>${r.display ?? ""}</b></div>`
    )
    .join("")}</div>`;
}

function featureColor(region) {
  if (region.endsWith("Brow")) return "var(--bubble)";
  if (region.endsWith("Eye")) return "var(--mint)";
  return "var(--butter)";
}

function featureDeco(region) {
  if (region.endsWith("Brow")) return DECO.brows;
  if (region.endsWith("Eye")) return DECO.eye;
  return DECO.smile;
}

function buildCards(r) {
  const ev = r.events;

  // Smile card copy
  let smileBody;
  if (r.smileFrames === 0) smileBody = "Not a single smile registered. Tough crowd.";
  else {
    let quip = "A healthy mix of real smiles and polite ones.";
    if (r.smilePct < 5) quip = "Smiles were rare, so every one of them counted.";
    else if (r.genuinePct >= 60) quip = "Most of those were the real thing, eyes and all.";
    else if (r.genuinePct < 30) quip = "The rest were polite. We noticed.";
    smileBody = `${r.genuinePct}% of those smiles reached your eyes. ${quip}`;
  }

  // Eyebrow card copy
  let browBody = `You furrowed them ${plural(ev.furrow, "time")}, too.`;
  if (ev.oneBrow > 0) browBody += ` And ${plural(ev.oneBrow, "time")} you raised just one. Iconic.`;

  // Score card copy
  let scoreBody = "You keep things close. A raised eyebrow from you means something.";
  if (r.score >= 75) scoreBody = "Your face is an open book. Large print.";
  else if (r.score >= 45) scoreBody = "Expressive when it counts, calm when it doesn't.";

  const regionRows = Object.keys(REGIONS)
    .map((k) => ({ label: REGIONS[k].short, value: r.regionSpread[k] }))
    .sort((a, b) => b.value - a.value);

  const emotionRows = Object.keys(EMOTIONS)
    .map((k) => ({ label: EMOTIONS[k].label, value: r.emotionPct[k], display: `${r.emotionPct[k]}%` }))
    .sort((a, b) => b.value - a.value);
  const topNonNeutral = emotionRows.find((row) => row.label !== "Neutral");

  return [
    {
      bg: "var(--cobalt)", fg: "var(--milk)", deco: DECO.face,
      html: `<p class="kicker">Your face had a lot to say.</p>
        <p class="big">${count(r.seconds, "sec")}</p>
        <p class="body">${r.frames.toLocaleString()} frames of eyebrows, eyes, and smiles. Here's what we found.</p>
        <p class="tap-hint">Tap or swipe to continue</p>`,
    },
    {
      bg: "var(--butter)", fg: "var(--ink)", deco: DECO.smile,
      html: `<p class="kicker">You smiled for</p>
        <p class="big">${count(r.smilePct, "%")}</p>
        <p class="body">of the session. ${smileBody}</p>`,
    },
    {
      bg: "var(--bubble)", fg: "var(--ink)", deco: DECO.brows,
      html: `<p class="kicker">You raised your eyebrows</p>
        <p class="big">${count(ev.browRaise, ev.browRaise === 1 ? "time" : "times")}</p>
        <p class="body">${browBody}</p>`,
    },
    {
      bg: featureColor(r.topRegion), fg: "var(--ink)", deco: featureDeco(r.topRegion),
      html: `<p class="kicker">Your most expressive feature</p>
        <p class="big words">${REGIONS[r.topRegion].label}</p>
        ${bars(regionRows)}
        <p class="body">${REGION_QUIPS[r.topRegion]}</p>`,
    },
    {
      bg: "var(--mint)", fg: "var(--ink)", deco: DECO.blink,
      html: `<p class="kicker">You blinked</p>
        <p class="big">${count(ev.blink, ev.blink === 1 ? "time" : "times")}</p>
        <p class="body">That's about ${r.blinkRate} a minute. A relaxed person usually lands somewhere around 15 to 20.</p>`,
    },
    {
      bg: "var(--cobalt)", fg: "var(--milk)", deco: DECO.face,
      html: `<p class="kicker">Your face came alive when we said</p>
        <p class="big quote">“${r.bestPrompt}”</p>
        <p class="body">Your strongest reaction of the session, and it was mostly ${EMOTIONS[r.bestPromptEmotion].word}.</p>`,
    },
    {
      bg: "var(--milk)", fg: "var(--ink)", deco: DECO.eye,
      html: `<p class="kicker">Your expression mix</p>
        <p class="big words">Mostly ${emotionRows[0].label.toLowerCase()}</p>
        ${bars(emotionRows)}
        <p class="body">${
          emotionRows[0].label === "Neutral" && topNonNeutral
            ? `When your face did move, it leaned toward ${topNonNeutral.label.toLowerCase()}.`
            : "Every frame was sorted into one of these five."
        }</p>`,
    },
    {
      bg: "var(--bubble)", fg: "var(--ink)", deco: DECO.face,
      html: `<p class="kicker">Your expressiveness score</p>
        <p class="big">${count(r.score, "/100")}</p>
        <p class="body">${scoreBody}</p>`,
    },
    {
      bg: "var(--ink)", fg: "var(--milk)", deco: DECO.face,
      html: `<p class="kicker">You are</p>
        <p class="big words">${r.persona.name}</p>
        <p class="body">${r.persona.desc}</p>
        <dl class="summary">
          <div><dt>Smiling</dt><dd>${r.smilePct}%</dd></div>
          <div><dt>Eyebrow raises</dt><dd>${ev.browRaise}</dd></div>
          <div><dt>Top feature</dt><dd>${REGIONS[r.topRegion].short}</dd></div>
          <div><dt>Score</dt><dd>${r.score}/100</dd></div>
        </dl>
        <button class="btn" data-action="restart">Run it again</button>`,
    },
  ];
}

/* ---------------- Story navigation ---------------- */

let cardEls = [];
let cardIndex = 0;
let cardData = [];

function renderCards(results) {
  cardData = buildCards(results);
  const cards = $("cards");
  cards.innerHTML = cardData
    .map(
      (c, i) => `<article class="card" style="--bg:${c.bg};--fg:${c.fg}"
        aria-roledescription="slide" aria-label="${i + 1} of ${cardData.length}">
        <div class="deco" aria-hidden="true">${c.deco}</div>${c.html}</article>`
    )
    .join("");
  $("progress").innerHTML = cardData.map(() => "<i></i>").join("");
  cardEls = [...cards.querySelectorAll(".card")];
  goTo(0);
  cards.focus({ preventScroll: true });
}

function goTo(i) {
  cardIndex = Math.max(0, Math.min(cardEls.length - 1, i));
  cardEls.forEach((el, j) => {
    el.classList.toggle("active", j === cardIndex);
    el.setAttribute("aria-hidden", String(j !== cardIndex));
  });
  [...$("progress").children].forEach((seg, j) => seg.classList.toggle("on", j <= cardIndex));
  $("story").style.setProperty("--fg", cardData[cardIndex].fg);
  animateCounts(cardEls[cardIndex]);
}

function animateCounts(card) {
  card.querySelectorAll(".count").forEach((el) => {
    const to = Number(el.dataset.to);
    if (REDUCED_MOTION || el.dataset.done) {
      el.textContent = to;
      return;
    }
    el.dataset.done = "1";
    const start = performance.now();
    const duration = 900;
    const step = (t) => {
      const p = Math.min(1, (t - start) / duration);
      el.textContent = Math.round(to * (1 - Math.pow(1 - p, 3)));
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
}

// Tap left side = back, right side = forward. Swipe works too.
let pointerStart = null;
$("cards").addEventListener("pointerdown", (e) => {
  pointerStart = { x: e.clientX, y: e.clientY };
});
$("cards").addEventListener("pointerup", (e) => {
  if (!pointerStart) return;
  const dx = e.clientX - pointerStart.x;
  const dy = e.clientY - pointerStart.y;
  pointerStart = null;

  if (e.target.closest("button")) return; // buttons handle their own clicks

  if (Math.abs(dx) > 40 && Math.abs(dx) > Math.abs(dy)) {
    goTo(cardIndex + (dx < 0 ? 1 : -1));
  } else if (Math.abs(dx) < 10 && Math.abs(dy) < 10) {
    const rect = e.currentTarget.getBoundingClientRect();
    goTo(cardIndex + (e.clientX - rect.left < rect.width * 0.3 ? -1 : 1));
  }
});

document.addEventListener("keydown", (e) => {
  if (!$("screen-results").classList.contains("active")) return;
  if (e.key === "ArrowRight") goTo(cardIndex + 1);
  if (e.key === "ArrowLeft") goTo(cardIndex - 1);
});

$("cards").addEventListener("click", (e) => {
  if (e.target.closest("[data-action='restart']")) restart();
});

function restart() {
  setStatus("");
  showScreen("intro");
  $("startBtn").focus();
}
