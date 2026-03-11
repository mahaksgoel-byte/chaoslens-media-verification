console.log("🧠 offscreen loaded");

let ws;

async function captureOnce() {
  console.log("📸 Capturing screenshot");

  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: false
  });

  const video = document.createElement("video");
  video.srcObject = stream;
  await video.play();

  await new Promise(r => video.onloadedmetadata = r);

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);

  // stop immediately
  stream.getTracks().forEach(t => t.stop());

  const blob = await new Promise(r =>
    canvas.toBlob(r, "image/webp", 0.9)
  );

  console.log("🖼️ Screenshot ready");

  ws = new WebSocket("ws://localhost:8080");
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    console.log("🔌 WS connected, sending image");
    ws.send(blob);
    ws.close();
  };
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "TAKE_SCREENSHOT") {
    captureOnce();
  }
});