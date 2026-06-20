// Cross-Modal Satellite Image Retrieval — frontend logic.
//
// Two flows:
//   /index  — upload one or more modalities (+ optional label) -> add to gallery
//   /query  — upload a crop + modality -> top-5 / top-10 ranked results
//
// All backend interaction is plain fetch() against the FastAPI app.

const API = ""; // same origin

// ---- element handles -------------------------------------------------------
const gallerySizeEl = document.getElementById("gallery-size");
const modelStatusEl = document.getElementById("model-status");
const indexBtn = document.getElementById("index-btn");
const indexFeedback = document.getElementById("index-feedback");
const indexLabel = document.getElementById("index-label");
const queryBtn = document.getElementById("query-btn");
const queryFeedback = document.getElementById("query-feedback");
const queryTiming = document.getElementById("query-timing");
const queryModality = document.getElementById("query-modality");
const resultsEmpty = document.getElementById("results-empty");
const top5Gallery = document.querySelector("#results-top5 .gallery");
const top10Gallery = document.querySelector("#results-top10 .gallery");

// ---- per-modality index files ---------------------------------------------
const indexFiles = { sar: null, optical: null, multispectral: null };

function bindIndexDropzones() {
  document.querySelectorAll(".dropzone[data-modality]").forEach((dz) => {
    const modality = dz.dataset.modality;
    const input = dz.querySelector("input[type=file]");

    dz.addEventListener("click", () => input.click());
    input.addEventListener("change", () => {
      if (input.files[0]) setIndexFile(modality, dz, input.files[0]);
    });
    dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
    dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
    dz.addEventListener("drop", (e) => {
      e.preventDefault();
      dz.classList.remove("dragover");
      if (e.dataTransfer.files[0]) setIndexFile(modality, dz, e.dataTransfer.files[0]);
    });
  });
}

function setIndexFile(modality, dz, file) {
  indexFiles[modality] = file;
  dz.classList.add("has-file");
  dz.querySelector(".dz-file")?.remove();
  const tag = document.createElement("div");
  tag.className = "dz-file";
  tag.textContent = file.name + ` (${(file.size / 1024).toFixed(0)} KB)`;
  dz.appendChild(tag);
  refreshIndexButton();
}

function refreshIndexButton() {
  indexBtn.disabled = !Object.values(indexFiles).some((f) => f);
}

// ---- query dropzone --------------------------------------------------------
const queryDz = document.getElementById("query-dropzone");
const queryInput = document.getElementById("query-file");
let queryFile = null;

queryDz.addEventListener("click", () => queryInput.click());
queryInput.addEventListener("change", () => {
  if (queryInput.files[0]) setQueryFile(queryInput.files[0]);
});
queryDz.addEventListener("dragover", (e) => { e.preventDefault(); queryDz.classList.add("dragover"); });
queryDz.addEventListener("dragleave", () => queryDz.classList.remove("dragover"));
queryDz.addEventListener("drop", (e) => {
  e.preventDefault();
  queryDz.classList.remove("dragover");
  if (e.dataTransfer.files[0]) setQueryFile(e.dataTransfer.files[0]);
});

function setQueryFile(file) {
  queryFile = file;
  queryBtn.disabled = false;
  queryDz.classList.add("has-file");
  queryDz.querySelector(".dz-file")?.remove();
  const tag = document.createElement("div");
  tag.className = "dz-file";
  tag.textContent = file.name + ` (${(file.size / 1024).toFixed(0)} KB)`;
  queryDz.appendChild(tag);
}

// ---- API calls -------------------------------------------------------------
async function checkHealth() {
  try {
    const r = await fetch(API + "/health");
    if (!r.ok) throw new Error("health " + r.status);
    const j = await r.json();
    gallerySizeEl.textContent = `gallery: ${j.gallery_size} vectors`;
    modelStatusEl.textContent = "model: ready";
    modelStatusEl.className = "ok";
  } catch (e) {
    // Likely artifacts missing — show the expected setup hint.
    gallerySizeEl.textContent = "gallery: not built yet";
    modelStatusEl.textContent = "model/index not ready — run training + build_index first";
    modelStatusEl.className = "err";
  }
}

indexBtn.addEventListener("click", async () => {
  indexFeedback.textContent = "indexing…";
  indexFeedback.className = "feedback";
  try {
    let added = 0, total = 0;
    for (const modality of Object.keys(indexFiles)) {
      const file = indexFiles[modality];
      if (!file) continue;
      const fd = new FormData();
      fd.append("file", file);
      fd.append("modality", modality);
      if (indexLabel.value !== "") fd.append("label", indexLabel.value);
      const r = await fetch(API + "/index", { method: "POST", body: fd });
      if (!r.ok) {
        const t = await r.text();
        throw new Error(`${modality}: ${r.status} ${t}`);
      }
      const j = await r.json();
      added += j.added;
      total = j.gallery_size;
    }
    indexFeedback.textContent = `Added ${added} image(s). Gallery now has ${total} vectors.`;
    indexFeedback.className = "feedback ok";
    gallerySizeEl.textContent = `gallery: ${total} vectors`;
  } catch (e) {
    indexFeedback.textContent = "Error: " + e.message;
    indexFeedback.className = "feedback err";
  }
});

queryBtn.addEventListener("click", async () => {
  if (!queryFile) return;
  queryFeedback.textContent = "retrieving…";
  queryFeedback.className = "feedback";
  queryTiming.textContent = "";
  top5Gallery.innerHTML = "";
  top10Gallery.innerHTML = "";
  resultsEmpty.style.display = "none";
  try {
    const fd = new FormData();
    fd.append("file", queryFile);
    fd.append("modality", queryModality.value);
    const r = await fetch(API + "/query", { method: "POST", body: fd });
    if (!r.ok) {
      const t = await r.text();
      throw new Error(`${r.status} ${t}`);
    }
    const j = await r.json();
    renderHits(top5Gallery, j.top_5);
    renderHits(top10Gallery, j.top_10);
    queryTiming.textContent =
      `embed ${j.embed_ms.toFixed(1)} ms · search ${j.search_ms.toFixed(1)} ms · total ${j.total_ms.toFixed(1)} ms`;
    queryFeedback.textContent = `Returned ${j.top_10.length} hits (modality: ${j.modality}, gallery: ${j.gallery_size}).`;
    queryFeedback.className = "feedback ok";
  } catch (e) {
    queryFeedback.textContent = "Error: " + e.message;
    queryFeedback.className = "feedback err";
  }
});

function renderHits(container, hits) {
  container.innerHTML = "";
  if (!hits.length) {
    container.innerHTML = '<div class="empty">no hits</div>';
    return;
  }
  for (const h of hits) {
    const scorePct = Math.max(0, Math.min(100, ((h.score + 1) / 2) * 100)); // cosine [-1,1] -> %
    const card = document.createElement("div");
    card.className = "hit";
    const modClass = h.modality === "optical" ? "optical"
                   : h.modality === "multispectral" ? "multispectral" : "sar";
    card.innerHTML = `
      <img src="${h.thumbnail_b64 ? "data:image/png;base64," + h.thumbnail_b64
                                 : ""}" alt="hit" onerror="this.style.display='none'" />
      <div class="meta">
        <span class="rank">#${h.rank}</span>
        <span class="tag ${modClass}">${h.modality}</span>
        score ${h.score.toFixed(3)}
        ${h.class_name ? " · " + h.class_name : ""}
      </div>
      <div class="bar"><span style="width:${scorePct.toFixed(0)}%"></span></div>
    `;
    container.appendChild(card);
  }
}

// ---- init ------------------------------------------------------------------
bindIndexDropzones();
refreshIndexButton();
checkHealth();
