const API = "";

function headersJson() {
  const h = { "Content-Type": "application/json" };
  const t = document.getElementById("hubToken").value.trim();
  if (t) h.Authorization = `Bearer ${t}`;
  return h;
}

function headersEmpty() {
  const h = {};
  const t = document.getElementById("hubToken").value.trim();
  if (t) h.Authorization = `Bearer ${t}`;
  return h;
}

async function fetchConfig() {
  const r = await fetch(`${API}/api/config`);
  if (!r.ok) throw new Error("config");
  return r.json();
}

function renderStatus(data) {
  const el = document.getElementById("statusBox");
  const lines = [
    `running: ${data.running}`,
    `pid: ${data.pid ?? "—"}`,
    `exit_code: ${data.exit_code ?? "—"}`,
    `started_at: ${data.started_at_unix ? new Date(data.started_at_unix * 1000).toISOString() : "—"}`,
    data.command ? `cmd: ${data.command.join(" ")}` : "",
  ].filter(Boolean);
  el.textContent = lines.join("\n");
  document.getElementById("btnStop").disabled = !data.running;
}

async function pollStatus() {
  try {
    const r = await fetch(`${API}/api/status`);
    const data = await r.json();
    renderStatus(data);
  } catch (e) {
    document.getElementById("statusBox").textContent = "Could not reach hub API.";
  }
}

async function pollLogs() {
  const r = await fetch(`${API}/api/logs?tail=500`);
  const data = await r.json();
  const logView = document.getElementById("logView");
  logView.textContent = (data.lines || []).join("\n");
  logView.scrollTop = logView.scrollHeight;
}

function fillCheckpointSelect(sel, data) {
  sel.innerHTML = "";
  if (!data.items || data.items.length === 0) return;
  data.items.forEach((it) => {
    const o = document.createElement("option");
    o.value = it.name;
    o.textContent = it.name;
    sel.appendChild(o);
  });
}

async function loadArtifacts() {
  const r = await fetch(`${API}/api/artifacts`);
  const data = await r.json();
  const box = document.getElementById("artifactsTable");
  const sel = document.getElementById("inferCkpt");
  const selRand = document.getElementById("inferRandomCkpt");
  sel.innerHTML = "";
  if (selRand) selRand.innerHTML = "";
  if (!data.items || data.items.length === 0) {
    box.textContent = "No .pth files in checkpoints directory yet.";
    return;
  }
  const rows = data.items
    .map((it) => {
      const m = it.metrics || {};
      const snip = m.dsc != null ? `DSC ${Number(m.dsc).toFixed(3)} · IoU ${Number(m.iou).toFixed(3)}` : "—";
      return `<tr>
        <td><code>${escapeHtml(it.name)}</code></td>
        <td>${(it.size_bytes / 1024).toFixed(0)} KB</td>
        <td class="metrics-snippet" title="${escapeHtml(JSON.stringify(m))}">${escapeHtml(snip)}</td>
      </tr>`;
    })
    .join("");
  box.innerHTML = `<table><thead><tr><th>Checkpoint</th><th>Size</th><th>Metrics (from JSON)</th></tr></thead><tbody>${rows}</tbody></table>`;
  fillCheckpointSelect(sel, data);
  if (selRand) fillCheckpointSelect(selRand, data);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

document.getElementById("btnRefresh").addEventListener("click", () => {
  pollStatus();
  pollLogs();
  loadArtifacts();
});

document.getElementById("btnStop").addEventListener("click", async () => {
  const r = await fetch(`${API}/api/train/stop`, { method: "POST", headers: headersEmpty() });
  const j = await r.json();
  if (!r.ok) alert(j.detail || r.statusText);
  pollStatus();
  pollLogs();
});

document.getElementById("startForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = {
    mode: document.getElementById("mode").value,
    preset: document.getElementById("preset").value,
    data_root: document.getElementById("dataRoot").value.trim(),
    batch_size: Number(document.getElementById("batchSize").value) || 4,
  };
  const rounds = document.getElementById("rounds").value.trim();
  const le = document.getElementById("local_epochs").value.trim();
  const ck = document.getElementById("checkpoint").value.trim();
  if (rounds) body.rounds = Number(rounds);
  if (le) body.local_epochs = Number(le);
  if (ck) body.checkpoint = ck;
  const r = await fetch(`${API}/api/train/start`, {
    method: "POST",
    headers: headersJson(),
    body: JSON.stringify(body),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) {
    alert(j.detail || JSON.stringify(j) || r.statusText);
    return;
  }
  pollStatus();
  pollLogs();
});

document.getElementById("inferForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = document.getElementById("inferFile").files[0];
  const ckpt = document.getElementById("inferCkpt").value;
  if (!file || !ckpt) return;
  const fd = new FormData();
  fd.append("file", file);
  fd.append("checkpoint", ckpt);
  const r = await fetch(`${API}/api/infer/preview`, { method: "POST", body: fd });
  const j = await r.json();
  const out = document.getElementById("inferResult");
  if (!r.ok) {
    out.innerHTML = `<p class="hint" style="color:var(--danger)">${escapeHtml(j.detail || "Error")}</p>`;
    return;
  }
  out.innerHTML = `
    <div><p class="hint">mean prob ${j.mean_prob?.toFixed?.(4) ?? j.mean_prob}</p>
    <p class="hint">Mask</p><img alt="mask" src="data:image/png;base64,${j.mask_png_b64}" /></div>
    <div><p class="hint">Overlay</p><img alt="overlay" src="data:image/png;base64,${j.overlay_png_b64}" /></div>
  `;
});

document.getElementById("inferRandomForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const ckpt = document.getElementById("inferRandomCkpt").value;
  const dataRoot = document.getElementById("inferDataRoot").value.trim();
  const n = Number(document.getElementById("inferRandomN").value) || 15;
  const seedRaw = document.getElementById("inferRandomSeed").value.trim();
  if (!ckpt) return;
  const body = { checkpoint: ckpt, n };
  if (dataRoot) body.data_root = dataRoot;
  if (seedRaw !== "") body.seed = Number(seedRaw);

  const out = document.getElementById("inferRandomResult");
  out.innerHTML = '<p class="hint">Running inference…</p>';
  let r = await fetch(`${API}/api/infer/random-npy`, {
    method: "POST",
    headers: headersJson(),
    body: JSON.stringify(body),
  });
  if (r.status === 404) {
    r = await fetch(`${API}/api/infer/random_npy`, {
      method: "POST",
      headers: headersJson(),
      body: JSON.stringify(body),
    });
  }
  const j = await r.json().catch(() => ({}));
  if (!r.ok) {
    out.innerHTML = `<p class="hint" style="color:var(--danger)">${escapeHtml(j.detail || "Error")}</p>`;
    return;
  }
  const items = j.items || [];
  const clipSummary =
    j.clip_class_head === true &&
    j.mean_prob_benign_clip != null &&
    j.mean_prob_malignant_clip != null
      ? ` · CLIP mean ben ${Number(j.mean_prob_benign_clip).toFixed(3)} · mal ${Number(j.mean_prob_malignant_clip).toFixed(3)}`
      : "";
  const summary = `
    <div class="infer-random-summary">
      ${j.n_evaluated ?? items.length} / ${j.n_requested ?? "—"} samples · mean DSC ${Number(j.mean_dsc).toFixed(4)} · mean IoU ${Number(j.mean_iou).toFixed(4)}
      ${clipSummary}
      ${j.pathology_manifest ? " · pathology manifest" : ""}
      ${j.seed != null ? ` · seed ${j.seed}` : ""}
    </div>`;
  const grid = items
    .map((it) => {
      const gt =
        it.pathology_gt != null && it.pathology_gt !== ""
          ? `GT ${escapeHtml(String(it.pathology_gt))}`
          : "GT —";
      const gtScores =
        it.gt_prob_benign != null && it.gt_prob_malignant != null
          ? ` · gt ben ${Number(it.gt_prob_benign).toFixed(2)} · mal ${Number(it.gt_prob_malignant).toFixed(2)}`
          : "";
      const clipLine =
        it.prob_benign != null && it.prob_malignant != null
          ? ` · CLIP ben ${Number(it.prob_benign).toFixed(3)} · mal ${Number(it.prob_malignant).toFixed(3)}`
          : "";
      const cap = `${escapeHtml(it.stem)}<br/>DSC ${Number(it.dsc).toFixed(3)} · IoU ${Number(it.iou).toFixed(3)} · ${escapeHtml(it.client)} · ${gt}${gtScores}${clipLine}`;
      const inp = it.input_png_b64 ? `data:image/png;base64,${it.input_png_b64}` : "";
      const gt = it.gt_mask_png_b64 ? `data:image/png;base64,${it.gt_mask_png_b64}` : "";
      const gto = it.gt_overlay_png_b64 ? `data:image/png;base64,${it.gt_overlay_png_b64}` : "";
      const strip = `
        <div class="infer-random-strip">
          <figure class="infer-random-fig">
            <img alt="" src="${inp}" />
            <figcaption>Girdi</figcaption>
          </figure>
          <figure class="infer-random-fig">
            <img alt="" src="data:image/png;base64,${it.overlay_png_b64}" />
            <figcaption>Tahmin</figcaption>
          </figure>
          <figure class="infer-random-fig">
            <img alt="" src="${gt}" />
            <figcaption>Gerçek maske</figcaption>
          </figure>
          <figure class="infer-random-fig">
            <img alt="" src="${gto}" />
            <figcaption>Gerçek (üstü)</figcaption>
          </figure>
        </div>`;
      return `<div class="infer-random-item">${strip}<div class="meta">${cap}</div></div>`;
    })
    .join("");
  out.innerHTML = `${summary}<div class="infer-random-grid">${grid}</div>`;
});

(async function init() {
  try {
    const cfg = await fetchConfig();
    document.getElementById("dataRoot").value = cfg.default_data_root || "";
    const idr = document.getElementById("inferDataRoot");
    if (idr) idr.value = cfg.default_data_root || "";
    const hint = document.getElementById("dataRootHint");
    if (hint && cfg.full_cbis_data_root && cfg.smoke_data_root) {
      hint.textContent = `Full CBIS (default): ${cfg.full_cbis_data_root} · Smoke subset: ${cfg.smoke_data_root}`;
    }
    if (!document.getElementById("checkpoint").value) {
      document.getElementById("checkpoint").placeholder = "harmonia_checkpoints/cbis_full_centralized.pth";
    }
    const stale = document.getElementById("staleHubBanner");
    if (stale && cfg.infer_random_npy !== true) {
      stale.style.display = "block";
      stale.textContent =
        "Bu Hub süreci güncel değil (random-npy yok). Terminalde Ctrl+C ile durdurup repo kökünden yeniden başlatın: uvicorn harmonia_vision.hub_app:app --host 127.0.0.1 --port 8765";
    }
  } catch (_) {}
  await pollStatus();
  await pollLogs();
  await loadArtifacts();
  setInterval(pollStatus, 2500);
  setInterval(async () => {
    const r = await fetch(`${API}/api/status`);
    const d = await r.json();
    if (d.running) pollLogs();
  }, 2000);
})();
