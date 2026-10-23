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

// The questions. "short" is
