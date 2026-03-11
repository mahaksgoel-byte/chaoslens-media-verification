// content.js
console.log("🧩 MadMax content script loaded");

// Wake up service worker early
chrome.runtime.sendMessage({ type: "PING" }, () => {
    if (chrome.runtime.lastError) {
      // SW was asleep – this is OK in MV3
    }
  });
  


window.addEventListener("message", (event) => {
  console.log("🧩 Content script got window message:", event.data);

  if (event.source !== window) return;
  if (event.data?.type !== "MADMAX_UID") return;

  chrome.runtime.sendMessage(
    {
      type: "STORE_UID",
      uid: event.data.uid
    },
    () => {
      // Ignore errors if SW is asleep
      if (chrome.runtime.lastError) {
        console.debug("SW not ready yet, message will retry later");
      }
    }
  );
  
});

chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "sync" && changes.overlayAllowed) {
      applyOverlayState(changes.overlayAllowed.newValue === true);
    }
  });
  

let overlay = null;

/* ===================== OVERLAY UI ===================== */

function createOverlay() {
    if (overlay) return;
  
    overlay = document.createElement("div");
    overlay.id = "madmax-overlay";
  
    overlay.innerHTML = `
      <img src="${chrome.runtime.getURL("icons/icon48.png")}" />
    `;
  
    Object.assign(overlay.style, {
      position: "fixed",
      width: "60px",                 // smaller than 80px
      height: "60px",
      bottom: "24px",
      right: "24px",
      borderRadius: "50%",
      background: "rgba(0, 0, 0, 0.8)", // dark translucent
      border: "4px solid #8B0000",   // thick blood-red border
      boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
      zIndex: "2147483647",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      cursor: "grab",
      userSelect: "none",
      transition: "all 0.2s ease"
    });
  
    // make the icon bigger than the overlay would suggest
    overlay.querySelector("img").style.width = "36px";  
    overlay.querySelector("img").style.height = "36px";
  
    document.documentElement.appendChild(overlay);
    makeDraggable(overlay);
  }
  

function removeOverlay() {
  if (!overlay) return;
  overlay.remove();
  overlay = null;
}

/* ===================== DRAG SUPPORT ===================== */

function makeDraggable(el) {
  let startX = 0, startY = 0, dx = 0, dy = 0;

  el.onmousedown = e => {
    e.preventDefault();
    startX = e.clientX;
    startY = e.clientY;

    document.onmousemove = e => {
      dx += e.clientX - startX;
      dy += e.clientY - startY;
      startX = e.clientX;
      startY = e.clientY;
      el.style.transform = `translate(${dx}px, ${dy}px)`;
    };

    document.onmouseup = () => {
      document.onmousemove = null;
    };
  };
}

/* ===================== STATE HANDLING ===================== */

function applyOverlayState(allowed) {
  if (allowed) createOverlay();
  else removeOverlay();
}

// Initial load
chrome.storage.sync.get("overlayAllowed", ({ overlayAllowed }) => {
  applyOverlayState(overlayAllowed === true);
});

// Listen for live updates
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "OVERLAY_PERMISSION") {
    applyOverlayState(msg.allowed === true);
  }
});