/* =========================================================
   Face Wrapped – script.js

   How it works:
   1. MediaPipe Face Landmarker finds your face in each webcam frame
      and returns 52 "blendshape" scores (0 to 1), e.g. browInnerUp,
      eyeSquintLeft, mouthSmileRight, jawOpen.
   2. For the first few seconds we record your RESTING face (baseline),
      so everything after that is measured relative to how your face
      looks when relaxed.
   3. Then we ask 6 ridiculous questions to answer out loud, taking as
      long as you like and pressing "I'm finished" after each one.
      Every smile is tracked as its own "moment" and sorted into a real
      laugh, a polite smile, or a sarcastic smirk (see trackLaugh).
   4. We don't claim to know who you are. The wraps only describe what
      your face DID (smiles, eyebrow raises, straight faces) and end with
      a joke award that admits it's scientifically questionable.
   5. The last card draws a 1080×1920 image for an Instagram story and
      lets you share the site with friends.
   6. The design is satire: every result looks like official paperwork,
      with case numbers, rubber stamps, and fine print.
   7. The twist: while you laugh, the site keeps the single best frame of
      each real laugh. After all the fake "analysis", the last cards drop
      the jokes and simply show you yourself laughing, un-mirrored, the way
      everyone else sees you. The photos never leave your device.
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
  // Laugh audit: how a smile "moment" gets sorted
  laughOff: 0.15,      // a smile moment ends when the smile drops below this
  realEyes: 0.4,       // real laugh: eyes crinkled for at least 40% of the moment
  realJaw: 0.15,       // ...and the mouth opened at least this much, or it lasted long enough
  realSeconds: 1,
  lopsided: 0.15,      // sarcastic: one side of the smile this much higher than the other
  eyeRoll: 0.3,        // sarcastic: eyes rolled up this much during the smile
  // Expressiveness score: an average intensity of this much = 100/100
  fullScoreIntensity: 0.18,
};

// The questions. "short" is used in the results ("during the pineapple pizza question").
const QUESTIONS = [
  { short: "pineapple pizza",     prompt: "Defend pineapple on pizza, even if you’re not a fan." },
  { short: "Google search",       prompt: "Explain your most recent Google search to your grandma." },
  { short: "worst gift",          prompt: "What’s the worst gift you’ve ever received?" },
  { short: "1% battery",          prompt: "Your phone is at 1% battery, you’re somewhere unfamiliar, and you’re alone. Who do you text, and what do you say?" },
  { short: "startup pitch",       prompt: "Pitch a terrible startup idea. Make it sound genius." },
  { short: "embarrassing moment", prompt: "Tell us about the most embarrassing moment of your life." },
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
  leftBrow:  "More range than most actors.",
  rightBrow: "More range than most actors.",
  leftEye:   "Your eyes did the talking.",
  rightEye:  "Your eyes did the talking.",
  mouth:     "Your mouth did all the work.",
};

// The five "moods" every frame gets sorted into
const EMOTIONS = {
  joy:      "Laughing",
  surprise: "Shock",
  skeptic:  "Side-eye",
  focus:    "Deep thought",
  neutral:  "Blank stare",
};

// The joke award at the end. bg = its color on the card and the story image.
const AWARDS = {
  laugh:   { name: "The Human Laugh Track",       bg: "#4a1a6b",
             desc: "You laughed at your own answers. Every time." },
  shock:   { name: "Chief Shock Officer",         bg: "#2233ff",
             desc: "Everything surprised you. Even your own answers." },
  sideeye: { name: "The Professional Side-Eyer",  bg: "#0e5c45",
             desc: "One eyebrow up. Zero trust." },
  thinker: { name: "The Overthinker in Residence", bg: "#1a1033",
             desc: "Deep thoughts. Even about pizza." },
  poker:   { name: "Certified Poker Face",        bg: "#262626",
             desc: "Your face gave away nothing. Are you okay?" },
  gif:     { name: "The Walking Reaction GIF",    bg: "#00546b",
             desc: "Your face doesn’t do subtle." },
  brows:   { name: "Chief Eyebrow Officer",       bg: "#8a1c4a",
             desc: "Your eyebrows did the talking." },
  polite:  { name: "The Customer Service Smile",  bg: "#8a3b12",
             desc: "Lots of smiles. Few real ones." },
  sarcasm: { name: "Head of the Sarcasm Department", bg: "#3b3b98",
             desc: "Lopsided smiles. Rolling eyes." },
};
const EMOTION_AWARD = { joy: "laugh", surprise: "shock", skeptic: "sideeye", focus: "thinker" };

// Card colors: every card gets its own
const PALETTE = {
  cobalt: "#2233ff", butter: "#ffe14d", bubble: "#ff8ac8", lilac: "#b69cff",
  lime: "#c6f432", forest: "#0e5c45", sky: "#8fd3ff", tangerine: "#ff8a3d",
  mint: "#7cf2c2", ink: "#1a1033", milk: "#ffffff",
};

// Big line drawings of facial features, shown behind each card
const DECO = {
  face:  `<svg viewBox="0 0 300 300"><path d="M60 95 Q95 68 130 90"/><path d="M170 90 Q205 68 240 95"/><circle cx="95" cy="138" r="10"/><circle cx="205" cy="138" r="10"/><path d="M85 205 Q150 258 215 205"/></svg>`,
  smile: `<svg viewBox="0 0 300 160"><path d="M30 40 Q150 170 270 40"/></svg>`,
  frown: `<svg viewBox="0 0 300 160"><path d="M30 130 Q150 10 270 130"/></svg>`,
  brows: `<svg viewBox="0 0 300 120"><path d="M20 95 Q75 20 135 78"/><path d="M165 78 Q225 20 280 95"/></svg>`,
  eye:   `<svg viewBox="0 0 300 160"><path d="M20 80 Q150 -10 280 80 Q150 170 20 80Z"/><circle cx="150" cy="80" r="26"/></svg>`,
  flat:  `<svg viewBox="0 0 300 160"><path d="M40 90 L260 90"/></svg>`,
};

/* ---------------- DOM ---------------- */

const $ = (id) => document.getElementById(id);
const video = $("video");
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// The site's own address, used in the share message and on the story image
const SITE_URL = location.origin + location.pathname.replace(/index\.html$/, "");

let faceLandmarker = null;
let stream = null;
let lastResults = null; // results of the most recent session
let storyFile = null;   // the finished story image, as a File (ready to share)
let storyUrl = null;    // the same image as a temporary URL (for preview and download)
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
    laughs: { real: 0, polite: 0, sarcastic: 0 },
    moment: null, // the smile moment happening right now, if any
    photos: [],   // the best frame of each laugh (kept only in this browser tab)
    regions: Object.fromEntries(Object.keys(REGIONS).map((k) => [k, newStat()])),
    questionIndex: 0,
    questions: QUESTIONS.map(() => ({ intensity: 0, n: 0, browRaises: 0, emotions: emptyEmotions() })),
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
    setStatus("Loading…");
    await loadModel();
    setStatus("Allow the camera to start.");
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
      setStatus("Camera blocked. Allow it in your browser, then try again.");
    } else if (err && err.name === "NotFoundError") {
      setStatus("No camera found.");
    } else {
      setStatus(`The session couldn't start: ${err && err.message ? err.message : err}`);
    }
  } finally {
    btn.disabled = false;
  }
});

function showQuestion(i) {
  $("phaseLabel").textContent = `Question ${i + 1} of ${QUESTIONS.length}`;
  $("prompt").textContent = QUESTIONS[i].prompt;
  // The progress bar shows how many questions are done
  $("timerBar").style.width = `${(i / QUESTIONS.length) * 100}%`;
}

// "I'm finished" moves to the next question, or to the results after the last one
$("nextBtn").addEventListener("click", () => {
  if (state.phase !== "session") return;
  if (state.questionIndex < QUESTIONS.length - 1) {
    state.questionIndex++;
    showQuestion(state.questionIndex);
  } else {
    finish();
  }
});

function beginPhase(phase) {
  state.phase = phase;
  state.phaseStart = performance.now();
  if (phase === "calibrating") {
    $("phaseLabel").textContent = "Getting ready";
    $("prompt").textContent = "Relax and look at the camera.";
    $("nextBtn").classList.add("hidden");
  } else if (phase === "session") {
    state.questionIndex = 0;
    showQuestion(0);
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
  const landmarks = result.faceLandmarks && result.faceLandmarks[0];
  const shapes = result.faceBlendshapes && result.faceBlendshapes[0];

  $("noFace").classList.toggle("hidden", Boolean(landmarks));
  if (!landmarks || !shapes) return;

  const m = toMap(shapes.categories);

  if (state.phase === "calibrating") {
    for (const key in m) state.baselineSum[key] = (state.baselineSum[key] || 0) + m[key];
    state.baselineCount++;
    updateMeters(computeSignals(m, false));
  } else if (state.phase === "session") {
    record(m, dt);
  }
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
    smileMax: Math.max(smileL, smileR), // catches one-sided smirks too
    smirk: Math.abs(smileL - smileR),
    eyeRoll: avg(a("eyeLookUpLeft"), a("eyeLookUpRight")),
    eyeSquint: avg(avg(squintL, squintR), avg(a("cheekSquintLeft"), a("cheekSquintRight"))),
    eyeWide: avg(wideL, wideR),
    blink: avg(m.eyeBlinkLeft || 0, m.eyeBlinkRight || 0), // raw: blinking isn't relative
    jaw: a("jawOpen"),
    frown: avg(a("mouthFrownLeft"), a("mouthFrownRight")),
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
  const q = state.questions[state.questionIndex];

  state.faceFrames++;
  state.faceSeconds += dt;
  state.intensitySum += s.intensity;

  if (s.smile > T.smile) {
    state.smileFrames++;
    if (s.genuine) state.genuineFrames++;
  }

  if (edge("browRaise", s.browUp, T.browRaiseOn, T.browRaiseOff)) q.browRaises++;
  edge("furrow", s.browDown, T.furrowOn, T.furrowOff);
  edge("oneBrow", s.browAsym, T.oneBrowOn, T.oneBrowOff);

  trackLaugh(s, dt);

  const emotion = classify(s);
  state.emotions[emotion]++;

  const blinking = s.blink > 0.4;
  for (const [key, value] of Object.entries(s.regions)) {
    if (blinking && key.endsWith("Eye")) continue; // blinks aren't "expressions"
    addStat(state.regions[key], value);
  }

  q.intensity += s.intensity;
  q.n++;
  q.emotions[emotion]++;

  updateMeters(s);
}

/* ---------------- Laugh audit ----------------
   Each smile is tracked from start to finish as one "moment".
   When it ends, we look at what the rest of the face did during it:
   - Real laugh: eyes crinkled (cheeks up) AND the mouth opened or it lasted a while
   - Sarcastic smirk: lopsided smile, or an eye roll, or one eyebrow up
   - Polite smile: everything else (the mouth smiled, the eyes didn't join in)
*/
function trackLaugh(s, dt) {
  if (!state.moment) {
    if (s.smileMax <= T.smile) return; // no smile happening
    state.moment = {
      frames: 0, seconds: 0, eyes: 0, lopsided: 0, eyeRoll: 0, oneBrow: 0, jawMax: 0,
      bestScore: 0, photo: null, question: state.questionIndex,
    };
  }
  const mo = state.moment;

  // Keep a photo of the biggest moment of this laugh
  const laughScore = s.smileMax + s.eyeSquint + 0.5 * s.jaw;
  if (laughScore > mo.bestScore + 0.03) {
    mo.bestScore = laughScore;
    mo.photo = snapshot(mo.photo);
  }
  mo.frames++;
  mo.seconds += dt;
  if (s.eyeSquint > T.genuineEyes) mo.eyes++;
  if (s.smirk > T.lopsided) mo.lopsided++;
  if (s.eyeRoll > T.eyeRoll) mo.eyeRoll++;
  if (s.browAsym > T.oneBrowOn) mo.oneBrow++;
  mo.jawMax = Math.max(mo.jawMax, s.jaw);

  if (s.smileMax < T.laughOff) endLaugh();
}

function endLaugh() {
  const mo = state.moment;
  state.moment = null;
  if (!mo || mo.frames < 4) return; // too short to count

  const share = (n) => n / mo.frames;
  const sarcastic = share(mo.lopsided) > 0.4 || share(mo.eyeRoll) > 0.2 || share(mo.oneBrow) > 0.3;
  const real =
    share(mo.eyes) >= T.realEyes && (mo.jawMax > T.realJaw || mo.seconds >= T.realSeconds);

  if (sarcastic) state.laughs.sarcastic++;
  else if (real) state.laughs.real++;
  else state.laughs.polite++;

  // Only laughs go in the gallery: real ones first, genuine smiles as backup. Never smirks.
  if (!sarcastic && mo.photo) {
    state.photos.push({ canvas: mo.photo, real, score: mo.bestScore, question: mo.question });
    state.photos.sort((a, b) => b.real - a.real || b.score - a.score);
    state.photos.length = Math.min(state.photos.length, 8);
  }
}

// Copies the current webcam frame into a small canvas.
// The raw camera frame is NOT mirrored, so this is how other people see you.
function snapshot(canvas) {
  const w = 360;
  const h = Math.round((w * video.videoHeight) / video.videoWidth) || 270;
  const c = canvas || document.createElement("canvas");
  c.width = w;
  c.height = h;
  c.getContext("2d").drawImage(video, 0, 0, w, h);
  return c;
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
  if (state.moment) endLaugh();
  state.phase = "done";
  $("nextBtn").classList.add("hidden");
  stopCamera();

  if (state.faceFrames < 60) {
    showScreen("intro");
    setStatus("Couldn’t see your face. Try again in better light.");
    return;
  }
  const results = buildResults();
  lastResults = results;
  storyFile = null;
  renderCards(results);
  showScreen("results");

  // Draw the story image in the background so it's ready by the last card
  makeStoryImage(results)
    .then(({ file, url }) => {
      if (storyUrl) URL.revokeObjectURL(storyUrl);
      storyFile = file;
      storyUrl = url;
      const img = $("storyPreview");
      if (img) {
        img.src = url;
        img.alt = `Story image: ${results.award.name}`;
      }
    })
    .catch((err) => console.error("Story image failed:", err));
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

  // Per-question stats (questions where your face was barely visible are skipped)
  const perQuestion = state.questions
    .map((q, i) => ({
      ...QUESTIONS[i],
      n: q.n,
      intensity: q.n ? q.intensity / q.n : 0,
      joyPct: pctOf(q.emotions.joy, q.n),
      browRaises: q.browRaises,
    }))
    .filter((q) => q.n > 5);

  // Funniest = the question you smiled most during (ties go to the more expressive one)
  const funniest = [...perQuestion].sort((a, b) => b.joyPct - a.joyPct || b.intensity - a.intensity)[0];
  // Straightest face = the question your face moved least during
  const calmest =
    [...perQuestion].sort((a, b) => a.intensity - b.intensity).find((q) => q !== funniest) || funniest;
  const mostBrows = [...perQuestion].sort((a, b) => b.browRaises - a.browRaises)[0];

  const meanIntensity = state.intensitySum / frames;
  const score = Math.round(100 * Math.min(1, Math.sqrt(meanIntensity / T.fullScoreIntensity)));

  const r = {
    caseNo: `FW-${Math.floor(1000 + Math.random() * 9000)}`,
    frames,
    smilePct: pctOf(state.smileFrames, frames),
    genuinePct: pctOf(state.genuineFrames, state.smileFrames),
    smileFrames: state.smileFrames,
    events: { ...state.events },
    emotionPct,
    regionSpread,
    topRegion,
    laughs: { ...state.laughs },
    photos: state.photos.slice(0, 4).map((p) => ({
      src: p.canvas.toDataURL("image/jpeg", 0.85),
      short: QUESTIONS[p.question].short,
    })),
    funniest,
    calmest,
    mostBrows,
    score,
  };
  r.award = AWARDS[pickAward(r)];
  return r;
}

// Picks the joke award. It's based on a handful of face numbers, and it says so.
function pickAward(r) {
  const e = r.emotionPct;
  const top = ["joy", "surprise", "skeptic", "focus"].sort((a, b) => e[b] - e[a])[0];

  if (e.neutral >= 65 || e[top] < 6) return "poker";
  if (r.score >= 75) return "gif";
  if (r.events.browRaise >= 15) return "brows";
  const L = r.laughs;
  if (L.sarcastic >= 3 && L.sarcastic > L.real) return "sarcasm";
  if (L.polite >= 3 && L.polite > L.real) return "polite";
  return EMOTION_AWARD[top];
}

/* ---------------- Card templates ---------------- */

const count = (n, unit = "") =>
  `<span class="count" data-to="${n}">${REDUCED_MOTION ? n : 0}</span>${unit ? `<span class="unit">${unit}</span>` : ""}`;

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

function featureDeco(region) {
  if (region.endsWith("Brow")) return DECO.brows;
  if (region.endsWith("Eye")) return DECO.eye;
  return DECO.smile;
}

function buildCards(r) {
  const ev = r.events;
  const P = PALETTE;

  // Smile card copy
  const smileBody = r.smileFrames === 0 ? "Not once. Tough crowd." : `${r.genuinePct}% of them were real.`;

  // Eyebrow card copy
  const browBody =
    r.mostBrows && r.mostBrows.browRaises > 0 ? `Mostly during the ${r.mostBrows.short} question.` : "";

  // Laugh audit copy
  const L = r.laughs;
  const totalLaughs = L.real + L.polite + L.sarcastic;
  const laughRows = [
    { label: "Real laughs", value: L.real, display: String(L.real) },
    { label: "Polite", value: L.polite, display: String(L.polite) },
    { label: "Sarcastic", value: L.sarcastic, display: String(L.sarcastic) },
  ];
  let laughQuip = "No smiles. Tough crowd.";
  if (totalLaughs > 0) {
    const most = laughRows.reduce((a, b) => (b.value > a.value ? b : a));
    if (most.label === "Sarcastic") laughQuip = "Mostly sarcasm. Impressive.";
    else if (most.label === "Polite") laughQuip = "Mostly polite. Very professional.";
    else laughQuip = "Mostly the real thing.";
  }

  const regionRows = Object.keys(REGIONS)
    .map((k) => ({ label: REGIONS[k].short, value: r.regionSpread[k] }))
    .sort((a, b) => b.value - a.value);

  const emotionRows = Object.keys(EMOTIONS)
    .map((k) => ({ label: EMOTIONS[k], value: r.emotionPct[k], display: `${r.emotionPct[k]}%` }))
    .sort((a, b) => b.value - a.value);

  return [
    {
      bg: P.cobalt, fg: P.milk, deco: DECO.face,
      exhibit: "Intake", stamp: "Received",
      html: `<p class="kicker">You answered</p>
        <p class="big">${count(QUESTIONS.length, "questions")}</p>
        <p class="body">Your face had thoughts.</p>
        <p class="tap-hint">Tap to continue</p>`,
    },
    {
      bg: P.butter, fg: P.ink, deco: DECO.smile,
      exhibit: "Exhibit A", stamp: "Verified*",
      html: `<p class="kicker">You smiled for</p>
        <p class="big">${count(r.smilePct, "%")}</p>
        <p class="body">${smileBody}</p>`,
    },
    {
      bg: P.mint, fg: P.ink, deco: DECO.smile,
      exhibit: "Exhibit B", stamp: "Under review",
      html: `<p class="kicker">The laugh audit</p>
        <p class="big">${count(totalLaughs, totalLaughs === 1 ? "smile" : "smiles")}</p>
        ${bars(laughRows)}
        <p class="body">${laughQuip}</p>`,
    },
    {
      bg: P.bubble, fg: P.ink, deco: DECO.brows,
      exhibit: "Exhibit C", stamp: "Suspicious",
      html: `<p class="kicker">You raised your eyebrows</p>
        <p class="big">${count(ev.browRaise, ev.browRaise === 1 ? "time" : "times")}</p>
        ${browBody ? `<p class="body">${browBody}</p>` : ""}`,
    },
    {
      bg: P.lilac, fg: P.ink, deco: featureDeco(r.topRegion),
      exhibit: "Exhibit D", stamp: "Confirmed (probably)",
      html: `<p class="kicker">Your most expressive feature</p>
        <p class="big words">${REGIONS[r.topRegion].label}</p>
        ${bars(regionRows)}
        <p class="body">${REGION_QUIPS[r.topRegion]}</p>`,
    },
    {
      bg: P.lime, fg: P.ink, deco: DECO.smile,
      exhibit: "Exhibit E", stamp: "Hilarious (to you)",
      html: `<p class="kicker">You cracked up during</p>
        <p class="big quote">“${r.funniest.prompt}”</p>`,
    },
    {
      bg: P.forest, fg: P.milk, deco: DECO.flat,
      exhibit: "Exhibit F", stamp: "Redacted",
      html: `<p class="kicker">Straight face during</p>
        <p class="big quote">“${r.calmest.prompt}”</p>
        <p class="body">Either honest, or <span class="redact">hiding</span> something.</p>`,
    },
    {
      bg: P.sky, fg: P.ink, deco: DECO.eye,
      exhibit: "Exhibit G", stamp: "Very scientific",
      html: `<p class="kicker">Your expression mix</p>
        <p class="big words">Mostly ${emotionRows[0].label.toLowerCase()}</p>
        ${bars(emotionRows)}`,
    },
    {
      bg: r.award.bg, fg: P.milk, deco: "", certificate: true,
      exhibit: "Final ruling", stamp: "Official-ish",
      html: `<p class="kicker">Your face’s official title</p>
        <p class="big words">${r.award.name}</p>
        <p class="body">${r.award.desc}</p>
        <dl class="summary">
          <div><dt>Smiling</dt><dd>${r.smilePct}%</dd></div>
          <div><dt>Eyebrow raises</dt><dd>${ev.browRaise}</dd></div>
          <div><dt>Top feature</dt><dd>${REGIONS[r.topRegion].short}</dd></div>
          <div><dt>Expressiveness</dt><dd>${r.score}/100</dd></div>
        </dl>
        <p class="signature"><span class="sig">Dr. A. Lgorithm</span>Not a real doctor</p>`,
    },
    {
      bg: "#0d0d0d", fg: P.milk, deco: "", plain: true,
      html: `<p class="reveal-line">Just in case you haven’t smiled today…</p>
        <p class="tap-hint">Tap</p>`,
    },
    {
      bg: P.milk, fg: P.ink, deco: "", plain: true,
      html: r.photos.length
        ? `<p class="kicker">…you did.</p>
          <div class="gallery${r.photos.length === 1 ? " one" : ""}">
            ${r.photos
              .map((p) => `<figure><img src="${p.src}" alt="You laughing during the ${p.short} question"><figcaption>${p.short}</figcaption></figure>`)
              .join("")}
          </div>
          <p class="gallery-note">Only on your device.</p>`
        : `<p class="reveal-line">We didn’t catch one.</p>
          <p class="body">Run it again and let it out.</p>`,
    },
    {
      bg: P.tangerine, fg: P.ink, deco: "",
      exhibit: "Release form",
      html: `<p class="kicker">Take it with you</p>
        <img id="storyPreview" class="share-preview" alt="">
        <div class="share-actions">
          <button class="btn btn-dark" data-action="save">Save for your story</button>
          <button class="btn btn-line" data-action="share">Share with friends</button>
        </div>
        <p id="shareStatus" class="share-status" role="status" aria-live="polite"></p>
        <button class="btn btn-text" data-action="restart">Run it again</button>`,
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
      (c, i) => `<article class="card${c.certificate ? " certificate" : ""}${c.plain ? " plain" : ""}" style="--bg:${c.bg};--fg:${c.fg}"
        aria-roledescription="slide" aria-label="${i + 1} of ${cardData.length}">
        <div class="deco" aria-hidden="true">${c.deco}</div>
        ${c.plain ? "" : `<div class="card-head" aria-hidden="true">
          <span>File ${results.caseNo} / ${c.exhibit}</span>
        </div>`}
        ${c.stamp ? `<span class="stamp" aria-hidden="true">${c.stamp}</span>` : ""}
        ${c.html}
        ${c.fine ? `<p class="fineprint">${c.fine}</p>` : ""}
      </article>`
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
  const button = e.target.closest("[data-action]");
  if (!button) return;
  if (button.dataset.action === "restart") restart();
  if (button.dataset.action === "save") saveStory();
  if (button.dataset.action === "share") shareWithFriends();
});

function restart() {
  setStatus("");
  showScreen("intro");
  $("startBtn").focus();
}

/* ---------------- Sharing ---------------- */

function setShareStatus(text) {
  const el = $("shareStatus");
  if (el) el.textContent = text;
}

// On a phone, this opens the share sheet so you can pick Instagram (Stories) or "Save Image".
// On a computer, it downloads the PNG.
function saveStory() {
  if (!storyFile) {
    setShareStatus("Almost ready. Try again.");
    return;
  }
  const onPhone = window.matchMedia("(pointer: coarse)").matches;
  if (onPhone && navigator.canShare && navigator.canShare({ files: [storyFile] })) {
    navigator.share({ files: [storyFile] }).catch((err) => {
      if (err.name !== "AbortError") downloadStory();
    });
    return;
  }
  downloadStory();
}

function downloadStory() {
  const a = document.createElement("a");
  a.href = storyUrl;
  a.download = storyFile.name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setShareStatus("Saved!");
}

// Shares a message + the site link. Falls back to copying it if sharing isn't supported.
function shareWithFriends() {
  const text = `My face just earned the title “${lastResults.award.name}.” Get yours:`;
  if (navigator.share) {
    navigator.share({ title: "Face Wrapped", text, url: SITE_URL }).catch((err) => {
      if (err.name !== "AbortError") copyLink(text);
    });
    return;
  }
  copyLink(text);
}

function copyLink(text) {
  navigator.clipboard
    .writeText(`${text} ${SITE_URL}`)
    .then(() => setShareStatus("Link copied!"))
    .catch(() => setShareStatus(`Copy this link: ${SITE_URL}`));
}

/* ---------------- Story image (1080 × 1920, Instagram story size) ---------------- */

const STORY_W = 1080;
const STORY_H = 1920;
const STORY_FONT = '"Bricolage Grotesque", "Arial Narrow", system-ui, sans-serif';

// Splits text into lines that fit within maxWidth
function wrapLines(g, text, maxWidth) {
  const lines = [];
  let line = "";
  for (const word of text.split(" ")) {
    const test = line ? `${line} ${word}` : word;
    if (g.measureText(test).width > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = test;
    }
  }
  if (line) lines.push(line);
  return lines;
}

function roundedRect(g, x, y, w, h, r) {
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
}

async function makeStoryImage(r) {
  // Make sure the fonts are loaded before drawing text on the canvas
  try {
    await Promise.all([
      ...[400, 500, 800].map((w) => document.fonts.load(`${w} 100px "Bricolage Grotesque"`)),
      document.fonts.load(`500 30px "DM Mono"`),
    ]);
  } catch (err) {
    console.warn("Font didn't load, using fallback.", err);
  }

  const canvas = document.createElement("canvas");
  canvas.width = STORY_W;
  canvas.height = STORY_H;
  const g = canvas.getContext("2d");

  const C = PALETTE;
  const left = 90;
  const textWidth = STORY_W - left * 2;

  const MONO = '"DM Mono", ui-monospace, monospace';

  // Background: each award has its own color
  g.fillStyle = r.award.bg;
  g.fillRect(0, 0, STORY_W, STORY_H);

  // Certificate border (two thin lines)
  g.strokeStyle = C.milk;
  g.globalAlpha = 0.5;
  g.lineWidth = 4;
  g.strokeRect(44, 44, STORY_W - 88, STORY_H - 88);
  g.lineWidth = 2;
  g.strokeRect(62, 62, STORY_W - 124, STORY_H - 124);
  g.globalAlpha = 1;

  // Header: wordmark + case number
  g.textBaseline = "alphabetic";
  g.fillStyle = C.milk;
  g.font = `800 48px ${STORY_FONT}`;
  g.fillText("Face Wrapped", left, 170);
  g.font = `500 28px ${MONO}`;
  g.globalAlpha = 0.8;
  g.textAlign = "right";
  g.fillText(`File ${r.caseNo}`, STORY_W - left, 166);
  g.textAlign = "left";
  g.globalAlpha = 1;

  // The floating face: pink brows, mint eyes, yellow smile (same as the site)
  g.save();
  g.translate(315, 190);
  g.scale(1.5, 1.5);
  g.lineCap = "round";
  g.lineWidth = 16;
  g.strokeStyle = C.bubble;
  g.stroke(new Path2D("M60 95 Q95 68 130 90"));
  g.stroke(new Path2D("M170 90 Q205 68 240 95"));
  g.fillStyle = C.mint;
  for (const x of [95, 205]) {
    g.beginPath();
    g.arc(x, 138, 14, 0, Math.PI * 2);
    g.fill();
  }
  g.strokeStyle = C.butter;
  g.stroke(new Path2D("M85 205 Q150 258 215 205"));
  g.restore();

  // Rubber stamp, tilted
  g.save();
  g.translate(830, 540);
  g.rotate(-0.2);
  g.strokeStyle = C.butter;
  g.fillStyle = C.butter;
  g.lineWidth = 6;
  roundedRect(g, -150, -50, 300, 100, 14);
  g.stroke();
  g.font = `500 40px ${MONO}`;
  g.textAlign = "center";
  g.fillText("OFFICIAL-ISH", 0, 14);
  g.restore();
  g.textAlign = "left";

  // Certificate text + the award name, sized to fit
  let y = 680;
  g.fillStyle = C.milk;
  g.globalAlpha = 0.8;
  g.font = `500 30px ${MONO}`;
  g.fillText("CERTIFICATE OF FACIAL ACHIEVEMENT", left, y);
  y += 56;
  g.globalAlpha = 0.9;
  g.font = `500 40px ${STORY_FONT}`;
  g.fillText("My face’s official title:", left, y);
  g.globalAlpha = 1;

  g.font = `400 38px ${STORY_FONT}`;
  const descLines = wrapLines(g, r.award.desc, textWidth);
  const descHeight = descLines.length * 50;

  const textBottom = 1100;
  let size = 140;
  let nameLines;
  for (; size >= 60; size -= 6) {
    g.font = `800 ${size}px ${STORY_FONT}`;
    nameLines = wrapLines(g, r.award.name, textWidth);
    const nameHeight = nameLines.length * size * 0.95;
    if (y + 16 + nameHeight + 30 + descHeight <= textBottom) break;
  }

  y += 16;
  g.font = `800 ${size}px ${STORY_FONT}`;
  for (const line of nameLines) {
    y += size * 0.95;
    g.fillText(line, left, y);
  }

  y += 30;
  g.globalAlpha = 0.9;
  g.font = `400 38px ${STORY_FONT}`;
  for (const line of descLines) {
    y += 50;
    g.fillText(line, left, y);
  }
  g.globalAlpha = 1;

  // Four stat tiles, each a different color
  const tiles = [
    { label: "Smiling", value: `${r.smilePct}%`, bg: C.butter },
    { label: "Eyebrow raises", value: String(r.events.browRaise), bg: C.bubble },
    { label: "Top feature", value: REGIONS[r.topRegion].short, bg: C.lilac },
    { label: "Expressiveness", value: `${r.score}/100`, bg: C.lime },
  ];
  const tileW = 435;
  const tileH = 180;
  tiles.forEach((t, i) => {
    const x = left + (i % 2) * (tileW + 30);
    const ty = 1140 + Math.floor(i / 2) * (tileH + 22);
    roundedRect(g, x, ty, tileW, tileH, 30);
    g.fillStyle = t.bg;
    g.fill();

    g.fillStyle = C.ink;
    g.font = `500 28px ${MONO}`;
    g.fillText(t.label, x + 32, ty + 58);

    let valueSize = 64;
    g.font = `800 ${valueSize}px ${STORY_FONT}`;
    while (g.measureText(t.value).width > tileW - 64 && valueSize > 34) {
      valueSize -= 4;
      g.font = `800 ${valueSize}px ${STORY_FONT}`;
    }
    g.fillText(t.value, x + 32, ty + 142);
  });

  // Signature line
  g.fillStyle = C.milk;
  g.font = `italic 56px "Snell Roundhand", "Segoe Script", "Brush Script MT", cursive`;
  g.fillText("Dr. A. Lgorithm", left, 1610);
  g.fillRect(left, 1626, 480, 3);
  g.globalAlpha = 0.8;
  g.font = `500 26px ${MONO}`;
  g.fillText("Not a real doctor", left, 1666);

  // Fine print + link
  g.fillText("Accuracy: questionable.", left, 1730);
  g.globalAlpha = 1;
  g.font = `800 34px ${STORY_FONT}`;
  g.fillText(`Get yours: ${SITE_URL.replace(/^https?:\/\//, "").replace(/\/$/, "")}`, left, 1786);

  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  const file = new File([blob], "face-wrapped-story.png", { type: "image/png" });
  return { file, url: URL.createObjectURL(blob) };
}
