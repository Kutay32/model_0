const $ = (id) => document.getElementById(id);

const dropzone = $("dropzone");
const fileInput = $("file");
const statusEl = $("status");
const results = $("results");
const predLabel = $("predLabel");
const probBars = $("probBars");
const imgOverlay = $("imgOverlay");
const imgMask = $("imgMask");
const segMeta = $("segMeta");

function setStatus(msg, cls = "") {
  statusEl.textContent = msg;
  statusEl.className = "status " + cls;
}

function showResults(data) {
  results.classList.remove("hidden");
  const c = data.classification;
  predLabel.textContent = `${c.label} · ${(c.confidence * 100).toFixed(1)}%`;

  probBars.innerHTML = "";
  const entries = Object.entries(c.probabilities);
  const maxP = Math.max(...entries.map(([, v]) => v));
  for (const [name, p] of entries) {
    const li = document.createElement("li");
    const pct = (p * 100).toFixed(1);
    const width = maxP > 0 ? (p / maxP) * 100 : 0;
    li.innerHTML = `
      <div class="bar-label"><span>${name}</span><span>${pct}%</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
    `;
    probBars.appendChild(li);
  }

  imgOverlay.src = "data:image/png;base64," + data.images.overlay_png_base64;
  imgMask.src = "data:image/png;base64," + data.images.mask_png_base64;

  const sz = data.image_size;
  segMeta.textContent = `Image ${sz.width}×${sz.height} px · mask coverage ${(
    data.segmentation.mask_coverage * 100
  ).toFixed(2)}% of pixels`;
}

async function runPredict(file) {
  setStatus("Running model…");
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch("/api/predict", { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  const data = await res.json();
  setStatus("Done.", "ok");
  showResults(data);
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  if (f) runPredict(f).catch((e) => setStatus(e.message, "error"));
});

["dragenter", "dragover"].forEach((ev) => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "var(--accent)";
  });
});
["dragleave", "drop"].forEach((ev) => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "";
  });
});
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f && f.type.startsWith("image/")) runPredict(f).catch((err) => setStatus(err.message, "error"));
});

fetch("/api/health")
  .then((r) => (r.ok ? r.json() : r.json()))
  .then((j) => {
    if (j.models_loaded) setStatus("Models ready — upload an image.", "ok");
    else setStatus("Models not loaded: " + (j.error || "unknown"), "error");
  })
  .catch(() => setStatus("Cannot reach API — start the backend (uvicorn).", "error"));
