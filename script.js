/* =========================================================
   Face Wrapped – script.js

   How it works:
   1. MediaPipe Face Landmarker finds your face in each webcam frame
      and returns 52 "blendshape" scores (0 to 1), e.g. browInnerUp,
      eyeSquintLeft, mouthSmileRight, jawOpen.
   2. For the first few seconds we record your RESTING face (baseline),
      so everything after that is measured relative to how your face
      looks when relaxed.
   3. Then we show 8 possible futures (medicine, art, tech...).
      For each one we measure your "spark": genuine smiles, raised
      eyebrows, and widened eyes count up; furrows and frowns count down.
   4. The future with the most spark becomes your dream job, and your
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
const SECONDS_PER_FUTURE = 7.5;

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
  surpriseJaw: 0.2,
  wideEyes: 0.12,
  frown: 0.12,
  // Expressiveness score: an average intensity of this much = 100/100
  fullScoreIntensity: 0.18,
};

// Each future: the field, the dream job it leads to, and the scenario shown on screen.
const FUTURES = [
  { field: "Medicine",     job: "Surgeon",            line: "High stakes, steady hands.",
    prompt: "Imagine you're in an operating room, and someone's life is in your hands." },
  { field: "Art",          job: "Gallery Artist",     line: "Something you made, and people can't look away.",
    prompt: "Imagine your painting in a gallery, and strangers stopping to stare at it." },
  { field: "Tech",         job: "App Founder",        line: "Building the thing everyone uses.",
    prompt: "Imagine building an app that millions of people open every morning." },
  { field: "Science",      job: "Research Scientist", line: "Being the first to know something.",
    prompt: "Imagine discovering something nobody in history has ever known." },
  { field: "Business",     job: "Dealmaker",          line: "The handshake, the win, the room.",
    prompt: "Imagine closing a million-dollar deal while the whole room watches." },
  { field: "Teaching",     job: "Teacher",            line: "Changing someone's life, one class at a time.",
    prompt: "Imagine a former student telling you that you changed their life." },
  { field: "Performance",  job: "Headliner",          line: "The lights, the crowd, the noise.",
    prompt: "Imagine walking onto a stage in front of 10,000 screaming people." },
  { field: "The outdoors", job: "Park Ranger",        line: "Fresh air instead of fluorescent lights.",
    prompt: "Imagine working outside every day, miles from the nearest desk." },
];

const SESSION_SECONDS = FUTURES.length * SECONDS_PER_FUTURE;

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
