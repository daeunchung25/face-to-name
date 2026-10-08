
const startBtn = document.getElementById("startBtn");
const home = document.getElementById("home");
const scanner = document.getElementById("scanner");
const camera = document.getElementById("camera");
const statusText = document.getElementById("status");

startBtn.addEventListener("click", async function() {
  home.hidden = true;
  scanner.hidden = false;

  statusText.textContent = "Starting camera...";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });

    camera.srcObject = stream;
    statusText.textContent = "Camera ready!";
  } catch (error) {
    statusText.textContent = "Camera access failed. Please allow camera permission.";
  }
});
