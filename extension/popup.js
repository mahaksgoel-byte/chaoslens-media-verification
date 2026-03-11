document.getElementById("start").onclick = () => {
  console.log("▶️ START clicked");

  chrome.runtime.sendMessage({ type: "TAKE_SCREENSHOT" });
};