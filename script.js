/* =========================================================
   Face Wrapped – script.js

   How it works:
   1. MediaPipe Face Landmarker finds your face in each webcam frame
      and returns 52 "blendshape" scores (0 to 1), e.g. browInnerUp,
      eyeSquintLeft, mouthSmileRight, jawOpen.
   2. For the first few seconds we record your RESTING face (baseline),
      so everything after that is measured relative to how your face
      looks when relaxed.
   3. Then we ask 8 questions to answer out loud, taking as long as you
      like and pressing "I'm finished" after each one. Each one secretly
      belongs to a field (medicine, art, tech...). While you talk, we
      measure your "spark": genuine smiles, raised eyebrows, and widened
      eyes count up; furrows and frowns count down. Talking itself
      (jaw and lip movement) is ignored.
   4. The field with the most spark becomes your dream job, and your
      expression style gives it a funny adjective ("The Skeptical Surgeon").
   ========================================================= */

import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

/* ---------------- Settings you can tune ---------------- */

const CALIBRATION_SECONDS = 3;
// There's no time limit per question: the user presses "I'm finished" to move on.

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
  oneBrowOn: 0.2,    oneBrowOff: 0.1,
  smirk: 0.15,
  wideEyes: 0.12,
  frown: 0.12,
  // Expressiveness score: an average intensity of this much = 100/100
  fullScoreIntensity: 0.18,
};

// Each question secretly belongs to a field. The field is only revealed in the results.
// field = the career area, job = the dream job it leads to, prompt = the question shown on screen.
const FUTURES = [
  { field: "Medicine",     job: "Surgeon",            line: "Taking care of people, even when it's hard.",
    prompt: "Tell us about a time you took care of someone who was sick or hurt." },
  { field: "Art",          job: "Gallery Artist",     line: "Making things that didn't exist before.",
    prompt: "Describe something you made that you're proud of." },
  { field: "Tech",         job: "App Founder",        line: "Building the thing everyone uses.",
    prompt: "If you could build any app, what would it do? Explain it." },
  { field: "Science",      job: "Research Scientist", line: "Chasing questions nobody has answered yet.",
    prompt: "What's a question about the world you've always wanted answered?" },
  { field: "Business",     job: "Dealmaker",          line: "Convincing a room and closing the deal.",
    prompt: "Pitch us something you own. Convince us to buy it." },
  { field: "Teaching",     job: "Teacher",            line: "Making something click for someone else.",
    prompt: "Teach us something you know really well." },
  { field: "Performance",  job: "Headliner",          line: "The lights, the crowd, the noise.",
    prompt: "Tell us about the last time you were the center of attention." },
  { field: "The outdoors", job: "Park Ranger",        line: "Fresh air instead of fluorescent lights.",
    prompt: "Describe your favorite place outside. Where is it, and why?" },
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
  leftBrow:  "One eyebrow did most of the deciding. It has opinions about your career.",
  rightBrow: "One eyebrow did most of the deciding. It has opinions about your career.",
  leftEye:   "Squinting, widening, narrowing. Your eyes reacted before you did.",
  rightEye:  "Squinting, widening, narrowing. Your eyes reacted before you did.",
  mouth:     "Smiles, frowns, smirks. Your mouth did the heavy lifting.",
};

const EMOTION_WORDS = {
  joy: "joy",
  surprise: "surprise",
  skeptic: "skepticism",
  focus: "concentration",
  neutral: "quiet interest",
};

// What your face did during the future it liked least
const HARD_NO = {
  joy:      "You smiled, but it never quite reached your eyes.",
  surprise: "Your face looked alarmed. Noted.",
  skeptic:  "One eyebrow went up, and not in a good way.",
  focus:    "Your brow furrowed like it was doing taxes.",
  neutral:  "Your face didn't move. Not even a little.",
};

// Funny adjectives for the dream job, based on HOW you express yourself.
// {job} is replaced with the job title.
const ADJECTIVES = {
  Cheerful:       "You'd be the {job} who's way too happy at 7 a.m.",
  Dramatic:       "You'd be the {job} who reacts to every small win like it's the finale.",
  Skeptical:      "You'd be the {job} who asks for a second opinion on the second opinion.",
  Overthinking:   "You'd be the {job} with a backup plan for the backup plan.",
  "Poker-Faced":  "You'd be the {job} nobody can read. Not even your coworkers.",
  Unfiltered:     "You'd be the {job} whose face says it all before you open your mouth.",
  Polite:         "You'd be the {job} who smiles through every meeting that could've been an email.",
  "Low-Key":      "You'd be the {job} who gets it done without anyone noticing the effort.",
  Steady:         "You'd be the {job} everyone wants around when things go sideways.",
};
const EMOTION_ADJ = { joy: "Cheerful", surprise: "Dramatic", skeptic: "Skeptical", focus: "Overthinking" };

// Big line drawings of facial features, shown behind each card
const DECO = {
  face:  `<svg viewBox="0 0 300 300"><path d="M60 95 Q95 68 130 90"/><path d="M170 90 Q205 68 240 95"/><circle cx="95" cy="138" r="10"/><circle cx="205" cy="138" r="10"/><path d="M85 205 Q150 258 215 205"/></svg>`,
  smile: `<svg viewBox="0 0 300 160"><path d="M30 40 Q150 170 270 40"/></svg>`,
  frown: `<svg viewBox="0 0 300 160"><path d="M30 130 Q150 10 270 130"/></svg>`,
  brows: `<svg viewBox="0 0 300 120"><path d="M20 95 Q75 20 135 78"/><path d="M165 78 Q225 20 280 95"/></svg>`,
  eye:   `<svg viewBox="0 0 300 160"><path d="M20 80 Q150 -10 280 80 Q150 170 20 80Z"/><circle cx="150" cy="80" r="26"/></svg>`,
};

/* ---------------- DOM ---------------- */

const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const ctx = overlay.getContext("2d");
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// The landmark points we draw as dots: eyebrows, eyes, lips, and the face outline
const DOT_INDICES = [
  ...new Set(
    [
      FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
      FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
      FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
      FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
      FaceLandmarker.FACE_LANDMARKS_LIPS,
      FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
    ].flatMap((set) => set.flatMap((c) => [c.start, c.end]))
  ),
];

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
    events: { browRaise: 0, furrow: 0, oneBrow: 0 },
    active: { browRaise: false, furrow: false, oneBrow: false },
    emotions: emptyEmotions(),
    regions: Object.fromEntries(Object.keys(REGIONS).map((k) => [k, newStat()])),
    futureIndex: 0,
    futures: FUTURES.map(() => ({ spark: 0, n: 0, browRaises: 0, emotions: emptyEmotions() })),
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

function showFuture(i) {
  $("phaseLabel").textContent = `Question ${i + 1} of ${FUTURES.length}. Answer out loud, as long as you like.`;
  $("prompt").textContent = FUTURES[i].prompt;
  // The progress bar shows how many questions are done
  $("timerBar").style.width = `${(i / FUTURES.length) * 100}%`;
}

// "I'm finished" moves to the next question, or to the results after the last one
$("nextBtn").addEventListener("click", () => {
  if (state.phase !== "session") return;
  if (state.futureIndex < FUTURES.length - 1) {
    state.futureIndex++;
    showFuture(state.futureIndex);
  } else {
    finish();
  }
});

function beginPhase(phase) {
  state.phase = phase;
  state.phaseStart = performance.now();
  if (phase === "calibrating") {
    $("phaseLabel").textContent = "Getting your resting face";
    $("prompt").textContent = "Relax your face and look at the camera. The questions start in a moment.";
    $("nextBtn").classList.add("hidden");
  } else if (phase === "session") {
    state.futureIndex = 0;
    showFuture(0);
    $("nextBtn").classList.remove("hidden");
    $("nextBtn").focus({ preventScroll: true }); // so Enter or Space also works
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
  }
}

function handleResult(result, dt) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const landmarks = result.faceLandmarks && result.faceLandmarks[0];
  const shapes = result.faceBlendshapes && result.faceBlendshapes[0];

  $("noFace").classList.toggle("hidden", Boolean(landmarks));
  if (!landmarks || !shapes) return;

  drawDots(landmarks);
  const m = toMap(shapes.categories);

  if (state.phase === "calibrating") {
    for (const key in m) state.baselineSum[key] = (state.baselineSum[key] || 0) + m[key];
    state.baselineCount++;
    updateMeters(computeSignals(m, false));
  } else if (state.phase === "session") {
    record(m, dt);
  }
}

// Small, quiet dots on the tracked points (landmark x/y are 0–1, so scale to the canvas)
function drawDots(lm) {
  const r = Math.max(1.2, overlay.width / 420);
  ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
  ctx.beginPath();
  for (const i of DOT_INDICES) {
    const p = lm[i];
    const x = p.x * overlay.width;
    const y = p.y * overlay.height;
    ctx.moveTo(x + r, y);
    ctx.arc(x, y, r, 0, Math.PI * 2);
  }
  ctx.fill();
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
    sneer: avg(a("noseSneerLeft"), a("noseSneerRight")),
  };

  s.genuine = s.smile > T.smile && s.eyeSquint > T.genuineEyes;

  // How much each feature is "doing" right now (roughly 0–1)
  s.regions = {
    leftBrow: browUpL + browDownL,
    rightBrow: browUpR + browDownR,
    leftEye: squintL + wideL,
    rightEye: squintR + wideR,
    // Only smiles and frowns count, so normal talking doesn't make the mouth "win"
    mouth: Math.max(s.smile, s.frown),
  };

  // Overall expressiveness of this frame (jaw is left out because everyone moves it while talking)
  s.intensity =
    (s.browUp + s.browDown + s.smile + s.frown + s.eyeWide + s.eyeSquint + s.browAsym) / 3.5;

  // "Spark" = how drawn-in you look. Real smiles, raised brows and wide eyes add;
  // furrows, frowns and nose wrinkles subtract.
  s.spark =
    s.smile +
    (s.genuine ? s.eyeSquint : 0) +
    0.6 * s.browUp +
    0.6 * s.eyeWide -
    0.5 * s.browDown -
    0.8 * s.frown -
    0.8 * s.sneer;

  return s;
}

function classify(s) {
  if (s.smile > T.smile) return "joy";
  if (s.browUp > T.browRaiseOn && s.eyeWide > T.wideEyes) return "surprise";
  if (s.browAsym > T.oneBrowOn || (s.smirk > T.smirk && s.smile > 0.08)) return "skeptic";
  if (s.browDown > T.furrowOn || s.frown > T.frown) return "focus";
  return "neutral";
}

// Counts an event once when the value crosses "on", and resets when it drops below "off".
// Returns true on the frame the event starts.
function edge(name, value, on, off) {
  if (!state.active[name] && value > on) {
    state.active[name] = true;
    state.events[name]++;
    return true;
  }
  if (state.active[name] && value < off) state.active[name] = false;
  return false;
}

function record(m, dt) {
  const s = computeSignals(m);
  const f = state.futures[state.futureIndex];

  state.faceFrames++;
  state.faceSeconds += dt;
  state.intensitySum += s.intensity;

  if (s.smile > T.smile) {
    state.smileFrames++;
    if (s.genuine) state.genuineFrames++;
  }

  if (edge("browRaise", s.browUp, T.browRaiseOn, T.browRaiseOff)) f.browRaises++;
  edge("furrow", s.browDown, T.furrowOn, T.furrowOff);
  edge("oneBrow", s.browAsym, T.oneBrowOn, T.oneBrowOff);

  const emotion = classify(s);
  state.emotions[emotion]++;

  const blinking = s.blink > 0.4;
  for (const [key, value] of Object.entries(s.regions)) {
    if (blinking && key.endsWith("Eye")) continue; // blinks aren't "expressions"
    addStat(state.regions[key], value);
  }

  f.spark += s.spark;
  f.n++;
  f.emotions[emotion]++;

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
  $("nextBtn").classList.add("hidden");
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

  // Rank the futures by average spark (futures you weren't visible for go last)
  const ranking = state.futures
    .map((f, i) => ({
      ...FUTURES[i],
      spark: f.n > 5 ? f.spark / f.n : -Infinity,
      browRaises: f.browRaises,
      emotion: f.n ? topKey(f.emotions) : "neutral",
    }))
    .sort((a, b) => b.spark - a.spark);

  const best = ranking[0];
  const worst = ranking.filter((f) => f.spark > -Infinity).at(-1);
  const mostBrows = [...ranking].sort((a, b) => b.browRaises - a.browRaises)[0];

  const meanIntensity = state.intensitySum / frames;
  const score = Math.round(100 * Math.min(1, Math.sqrt(meanIntensity / T.fullScoreIntensity)));

  const r = {
    frames,
    smilePct: pctOf(state.smileFrames, frames),
    genuinePct: pctOf(state.genuineFrames, state.smileFrames),
    smileFrames: state.smileFrames,
    events: { ...state.events },
    emotionPct,
    regionSpread,
    topRegion,
    ranking,
    best,
    worst,
    mostBrows,
    score,
  };
  r.persona = pickPersona(r);
  return r;
}

function pickPersona(r) {
  const e = r.emotionPct;
  const topActive = ["joy", "surprise", "skeptic", "focus"].sort((a, b) => e[b] - e[a])[0];

  let adjective;
  if (e.neutral >= 65 || e[topActive] < 6) adjective = "Poker-Faced";
  else if (r.score >= 75) adjective = "Unfiltered";
  else if (r.smilePct >= 15 && r.genuinePct < 35) adjective = "Polite";
  else if (r.events.oneBrow >= 3) adjective = "Skeptical";
  else if (e[topActive] >= 6) adjective = EMOTION_ADJ[topActive];
  else if (r.score <= 25) adjective = "Low-Key";
  else adjective = "Steady";

  return {
    name: `The ${adjective} ${r.best.job}`,
    joke: ADJECTIVES[adjective].replace("{job}", r.best.job.toLowerCase()),
  };
}

/* ---------------- Card templates ---------------- */

const count = (n, unit = "") =>
  `<span class="count" data-to="${n}">${REDUCED_MOTION ? n : 0}</span>${unit ? `<span class="unit">${unit}</span>` : ""}`;

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function bars(rows) {
  const min = Math.min(...rows.map((r) => r.value));
  const shifted = rows.map((r) => ({ ...r, value: r.value - Math.min(0, min) + 0.02 }));
  const max = Math.max(...shifted.map((r) => r.value), 0.0001);
  return `<div class="bars">${shifted
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
  if (r.mostBrows.browRaises > 0) browBody = `Most of them went up during ${r.mostBrows.field.toLowerCase()}. ${browBody}`;
  if (ev.oneBrow > 0) browBody += ` And ${plural(ev.oneBrow, "time")} you raised just one. Iconic.`;

  const regionRows = Object.keys(REGIONS)
    .map((k) => ({ label: REGIONS[k].short, value: r.regionSpread[k] }))
    .sort((a, b) => b.value - a.value);

  const rankingRows = r.ranking
    .filter((f) => f.spark > -Infinity)
    .map((f) => ({ label: f.field, value: f.spark }));

  return [
    {
      bg: "var(--cobalt)", fg: "var(--milk)", deco: DECO.face,
      html: `<p class="kicker">You answered</p>
        <p class="big">${count(FUTURES.length, "questions")}</p>
        <p class="body">Each one was secretly about a different career. Your face reacted while you talked, whether you meant it to or not.</p>
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
      bg: "var(--milk)", fg: "var(--ink)", deco: DECO.eye,
      html: `<p class="kicker">How your face ranked each career</p>
        <p class="big words">${r.best.field} wins</p>
        ${bars(rankingRows)}
        <p class="body">Ranked by spark: real smiles, raised brows, and wide eyes, minus frowns and furrows.</p>`,
    },
    {
      bg: "var(--cobalt)", fg: "var(--milk)", deco: DECO.face,
      html: `<p class="kicker">Your face lit up when we asked</p>
        <p class="big quote">“${r.best.prompt}”</p>
        <p class="body">That one was secretly about ${r.best.field.toLowerCase()}. ${r.best.line} You were mostly showing ${EMOTION_WORDS[r.best.emotion]}.</p>`,
    },
    {
      bg: "var(--mint)", fg: "var(--ink)", deco: DECO.frown,
      html: `<p class="kicker">And it quietly said no to</p>
        <p class="big words">${r.worst.field}</p>
        <p class="body">${HARD_NO[r.worst.emotion]} That happened during “${r.worst.prompt}” Maybe leave that one to someone else.</p>`,
    },
    {
      bg: "var(--ink)", fg: "var(--milk)", deco: DECO.face,
      html: `<p class="kicker">Your dream job</p>
        <p class="big words">${r.persona.name}</p>
        <p class="body">${r.persona.joke}</p>
        <dl class="summary">
          <div><dt>Top career</dt><dd>${r.best.field}</dd></div>
          <div><dt>Hard no</dt><dd>${r.worst.field}</dd></div>
          <div><dt>Smiling</dt><dd>${r.smilePct}%</dd></div>
          <div><dt>Expressiveness</dt><dd>${r.score}/100</dd></div>
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
