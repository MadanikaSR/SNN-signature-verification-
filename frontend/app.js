/* ═══════════════════════════════════════════════════════════════
   SignVerify AI — app.js
   Env-aware API calls, drag-and-drop, preview, cold-start retry
═══════════════════════════════════════════════════════════════ */

"use strict";

// ── Config: env-aware API base URL ─────────────────────────────────────────────
const API_BASE_URL =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1"
    ? "http://localhost:8000"
    : "https://YOUR_RENDER_BACKEND_URL.onrender.com"; // ← replace after deploying to Render

const MAX_FILE_BYTES = 2 * 1024 * 1024; // 2 MB
const COLDSTART_RETRY_DELAY_MS = 5000;
const REQUEST_TIMEOUT_MS = 35000;

// ── DOM refs ───────────────────────────────────────────────────────────────────
const zone1 = document.getElementById("zone-1");
const zone2 = document.getElementById("zone-2");
const file1Input = document.getElementById("file-1");
const file2Input = document.getElementById("file-2");
const preview1 = document.getElementById("preview-1");
const preview2 = document.getElementById("preview-2");
const previewWrap1 = document.getElementById("preview-wrap-1");
const previewWrap2 = document.getElementById("preview-wrap-2");
const placeholder1 = document.getElementById("placeholder-1");
const placeholder2 = document.getElementById("placeholder-2");
const filename1 = document.getElementById("filename-1");
const filename2 = document.getElementById("filename-2");
const removeBtn1 = document.getElementById("remove-1");
const removeBtn2 = document.getElementById("remove-2");
const verifyBtn = document.getElementById("verify-btn");
const btnSpinner = document.getElementById("btn-spinner");
const btnText = document.getElementById("btn-text");
const toast = document.getElementById("toast");
const toastMsg = document.getElementById("toast-msg");
const resultCard = document.getElementById("result-card");
const errorCard = document.getElementById("error-card");
const errorMsg = document.getElementById("error-msg");
const resetBtn = document.getElementById("reset-btn");
const errorResetBtn = document.getElementById("error-reset-btn");

// Result elements
const resultHeader = document.getElementById("result-header");
const resultIcon = document.getElementById("result-icon");
const resultVerdict = document.getElementById("result-verdict");
const resultSub = document.getElementById("result-sub");
const metricScore = document.getElementById("metric-score");
const metricConfidence = document.getElementById("metric-confidence");
const metricThreshold = document.getElementById("metric-threshold");
const confidenceBar = document.getElementById("confidence-bar");
const thresholdMarker = document.getElementById("threshold-marker");

// ── State ──────────────────────────────────────────────────────────────────────
let imageFile1 = null;
let imageFile2 = null;

// ── Utility: human-readable bytes ─────────────────────────────────────────────
function fmtBytes(bytes) {
  return bytes < 1024 * 1024
    ? `${(bytes / 1024).toFixed(1)} KB`
    : `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// ── File validation ────────────────────────────────────────────────────────────
function validateFile(file) {
  const allowed = ["image/png", "image/jpeg", "image/jpg"];
  if (!allowed.includes(file.type)) {
    return `Unsupported file type: ${file.type}. Please use PNG or JPG.`;
  }
  if (file.size > MAX_FILE_BYTES) {
    return `File too large (${fmtBytes(file.size)}). Maximum is 2 MB.`;
  }
  return null;
}

// ── Preview image ──────────────────────────────────────────────────────────────
function showPreview(slot, file) {
  const preview = slot === 1 ? preview1 : preview2;
  const wrap = slot === 1 ? previewWrap1 : previewWrap2;
  const placeholder = slot === 1 ? placeholder1 : placeholder2;
  const fnEl = slot === 1 ? filename1 : filename2;
  const zone = slot === 1 ? zone1 : zone2;

  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    fnEl.textContent = `${file.name} · ${fmtBytes(file.size)}`;
    placeholder.classList.add("hidden");
    wrap.classList.remove("hidden");
    zone.classList.add("has-file");
  };
  reader.readAsDataURL(file);
}

// ── Clear preview ──────────────────────────────────────────────────────────────
function clearPreview(slot) {
  const preview = slot === 1 ? preview1 : preview2;
  const wrap = slot === 1 ? previewWrap1 : previewWrap2;
  const placeholder = slot === 1 ? placeholder1 : placeholder2;
  const zone = slot === 1 ? zone1 : zone2;

  preview.src = "";
  wrap.classList.add("hidden");
  placeholder.classList.remove("hidden");
  zone.classList.remove("has-file");

  if (slot === 1) {
    imageFile1 = null;
    file1Input.value = "";
  } else {
    imageFile2 = null;
    file2Input.value = "";
  }
  updateVerifyBtn();
}

// ── Handle file chosen ─────────────────────────────────────────────────────────
function handleFile(slot, file) {
  if (!file) return;
  const err = validateFile(file);
  if (err) { showError(err); return; }

  if (slot === 1) imageFile1 = file;
  else imageFile2 = file;

  showPreview(slot, file);
  updateVerifyBtn();
}

// ── Update verify button state ─────────────────────────────────────────────────
function updateVerifyBtn() {
  verifyBtn.disabled = !(imageFile1 && imageFile2);
}

// ── Wire upload zones ──────────────────────────────────────────────────────────
function wireZone(zone, input, slot) {
  // Click → open file dialog
  zone.addEventListener("click", (e) => {
    if (e.target.closest(".remove-btn")) return;
    input.click();
  });

  // Keyboard accessibility
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });

  // File input change
  input.addEventListener("change", () => {
    if (input.files[0]) handleFile(slot, input.files[0]);
  });

  // Drag-and-drop
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFile(slot, file);
  });
}

wireZone(zone1, file1Input, 1);
wireZone(zone2, file2Input, 2);

// Remove buttons
removeBtn1.addEventListener("click", (e) => { e.stopPropagation(); clearPreview(1); });
removeBtn2.addEventListener("click", (e) => { e.stopPropagation(); clearPreview(2); });

// ── Loading state ──────────────────────────────────────────────────────────────
function setLoading(on, msg = "Verifying...") {
  verifyBtn.disabled = on;
  verifyBtn.classList.toggle("loading", on);
  btnSpinner.classList.toggle("hidden", !on);
  btnText.textContent = on ? msg : "Verify Signature";
}

// ── Toast ──────────────────────────────────────────────────────────────────────
function showToast(msg) {
  toastMsg.textContent = msg;
  toast.classList.remove("hidden");
}
function hideToast() { toast.classList.add("hidden"); }

// ── Error card ─────────────────────────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = msg;
  errorCard.classList.remove("hidden");
}

// ── API call with AbortController timeout ─────────────────────────────────────
async function callVerifyAPI() {
  const formData = new FormData();
  formData.append("image1", imageFile1);
  formData.append("image2", imageFile2);

  const controller = new AbortController();
  const timerId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(`${API_BASE_URL}/verify`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timerId);

    if (!response.ok) {
      let detail = `Server error (${response.status})`;
      try {
        const body = await response.json();
        detail = body.detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }

    return await response.json();
  } catch (err) {
    clearTimeout(timerId);
    throw err;
  }
}

// ── Display result ─────────────────────────────────────────────────────────────
function displayResult(data) {
  const { similarity_score, is_match, confidence, threshold } = data;

  // Verdict
  resultIcon.textContent = is_match ? "✅" : "❌";
  resultIcon.className = `result-icon ${is_match ? "match" : "no-match"}`;
  resultVerdict.textContent = is_match ? "Signatures Match" : "Signatures Do Not Match";
  resultVerdict.className = `result-verdict ${is_match ? "match" : "no-match"}`;
  resultSub.textContent = is_match
    ? "The neural network determined these signatures are from the same person."
    : "The neural network determined these signatures are from different people.";

  // Metrics
  metricScore.textContent = similarity_score.toFixed(4);
  metricConfidence.textContent = `${confidence.toFixed(1)}%`;
  metricThreshold.textContent = threshold.toFixed(2);

  // Confidence bar — use raw similarity_score as position
  const pct = Math.round(similarity_score * 100);
  confidenceBar.style.width = "0%";
  confidenceBar.className = `confidence-bar-fill ${is_match ? "match" : "no-match"}`;
  thresholdMarker.style.left = `${(threshold * 100).toFixed(1)}%`;

  // Animate bar after short delay
  requestAnimationFrame(() => {
    setTimeout(() => { confidenceBar.style.width = `${pct}%`; }, 80);
  });

  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── Main verify handler ────────────────────────────────────────────────────────
verifyBtn.addEventListener("click", async () => {
  if (!imageFile1 || !imageFile2) return;

  // Hide previous results
  resultCard.classList.add("hidden");
  errorCard.classList.add("hidden");
  hideToast();
  setLoading(true, "Verifying...");

  try {
    const data = await callVerifyAPI();
    setLoading(false);
    displayResult(data);
  } catch (err) {
    // Cold-start retry: on timeout or network error, retry once
    const isTimeout = err.name === "AbortError" || err.message?.includes("network");
    if (isTimeout) {
      showToast(`Waking server... retrying in ${COLDSTART_RETRY_DELAY_MS / 1000}s`);
      setLoading(true, "Waking server...");
      await wait(COLDSTART_RETRY_DELAY_MS);
      hideToast();
      try {
        const data = await callVerifyAPI();
        setLoading(false);
        displayResult(data);
      } catch (retryErr) {
        setLoading(false);
        showError(
          `Request failed after retry. The backend may be unavailable.\n\nDetails: ${retryErr.message}`
        );
      }
    } else {
      setLoading(false);
      showError(err.message || "An unexpected error occurred. Please try again.");
    }
  }
});

// ── Reset ──────────────────────────────────────────────────────────────────────
function resetAll() {
  clearPreview(1);
  clearPreview(2);
  resultCard.classList.add("hidden");
  errorCard.classList.add("hidden");
  hideToast();
  setLoading(false);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

resetBtn.addEventListener("click", resetAll);
errorResetBtn.addEventListener("click", resetAll);

// ── Helpers ────────────────────────────────────────────────────────────────────
function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ── Health check (optional: ping on load to wake Render) ──────────────────────
async function pingBackend() {
  try {
    await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(5000) });
  } catch (_) {
    // Silently ignore — backend may be sleeping, will wake on /verify call
  }
}

// Warmup ping on page load to reduce cold-start delay
pingBackend();
