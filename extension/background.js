console.log("🔥 MadMax SW booted");

const PROJECT_ID = "madmax-67ca2";

let offscreenCreated = false;

async function ensureOffscreen() {
  if (!chrome.offscreen) {
    console.error("❌ chrome.offscreen API not available");
    return;
  }

  const exists = await chrome.offscreen.hasDocument();
  if (exists) return;

  await chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["USER_MEDIA"],
    justification: "Single screenshot capture"
  });

  // Give it time to boot so inspector appears
  await new Promise(resolve => setTimeout(resolve, 500));
}



async function fetchOverlayPermission(uid) {
  try {
    const res = await fetch(
      `https://firestore.googleapis.com/v1/projects/${PROJECT_ID}/databases/(default)/documents/settings/${uid}`
    );

    if (!res.ok) return false;

    const data = await res.json();

    return data.fields?.overlay?.booleanValue === true;
  } catch (e) {
    console.error("❌ Firestore REST error", e);
    return false;
  }
}

async function handleUID(uid) {
  if (!uid) return;

  const allowed = await fetchOverlayPermission(uid);

  await chrome.storage.sync.set({ overlayAllowed: allowed });

  chrome.tabs.query({}, tabs => {
    tabs.forEach(tab => {
      if (tab.id) {
        chrome.tabs.sendMessage(tab.id, {
          type: "OVERLAY_PERMISSION",
          allowed
        });
      }
    });
  });
}

/* ================= LISTENERS ================= */

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.get("madmax_uid", ({ madmax_uid }) => {
    if (madmax_uid) handleUID(madmax_uid);
  });
});

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "PING") {
    console.log("👋 SW awake");
  }
});


chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "sync" && changes.madmax_uid) {
    handleUID(changes.madmax_uid.newValue);
  }
});

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "STORE_UID" && msg.uid) {
    // Save UID and immediately fetch overlay permission
    chrome.storage.sync.set({ madmax_uid: msg.uid }, () => {
      handleUID(msg.uid); // fetch overlayAllowed and send message to tabs
    });
  }

  if (msg.type === "REFRESH_PERMISSION") {
    chrome.storage.sync.get("madmax_uid", ({ madmax_uid }) => {
      if (madmax_uid) handleUID(madmax_uid);
    });
  }
});

chrome.runtime.onMessage.addListener(async (msg) => {
  if (msg.type === "START_CAPTURE") {
    // 1️⃣ Take screenshot of visible tab
    const dataUrl = await chrome.tabs.captureVisibleTab(null, {
      format: "png"
    });

    // 2️⃣ Ensure offscreen exists
    await ensureOffscreen();

    // 3️⃣ Send image to offscreen
    chrome.offscreen.sendMessage({
      type: "UPLOAD_SCREENSHOT",
      dataUrl
    });
  }
});

chrome.runtime.onMessage.addListener(async (msg) => {
  if (msg.type === "TAKE_SCREENSHOT") {
    console.log("📸 TAKE_SCREENSHOT received");

    const dataUrl = await chrome.tabs.captureVisibleTab(
      null,
      { format: "png" } // ✅ MUST be png or jpeg
    );

    // Convert base64 → binary
    const res = await fetch(dataUrl);
    const blob = await res.blob();

    // Send to backend
    const ws = new WebSocket("ws://localhost:8080");
    ws.binaryType = "arraybuffer";

    ws.onopen = async () => {
      console.log("🔌 WS connected, sending screenshot");
      ws.send(await blob.arrayBuffer());
      ws.close();
    };
  }
});