"""FastAPI app for cross-modal satellite image retrieval.

Two user flows (matching the frontend):

* **/index** — upload a SAR / optical / multispectral image (+ optional label)
  → embed → add to the in-memory FAISS gallery. This is how new locations get
  "saved as encoded numbers".
* **/query** — upload a cropped query image + its modality → embed → search the
  gallery → return top-5/top-10 ranked hits with scores and timing.

Run locally::

    uvicorn api.main:app --reload --port 8000

The model + index are loaded lazily so the UI boots even before training has
run; endpoints return a clear 503 until artifacts exist.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from config import load_config
from index.faiss_index import VectorIndex
from retrieval.embedder import build_embedder_from_config
from retrieval.search import Retriever
from api.schemas import HealthResponse, HitItem, IndexResponse, QueryResponse

cfg = load_config()
app = FastAPI(title="Cross-Modal Satellite Image Retrieval", version="1.0.0")

# Static UI files.
_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static)), name="static")

# Lazily-built singletons (built on first use / on startup if artifacts exist).
_state: dict = {"embedder": None, "index": None, "retriever": None}


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
def _onnx_exists() -> bool:
    onnx_dir = Path(cfg["paths"]["onnx_path"]).parent
    return all(
        (onnx_dir / f"retrieval_model_{m}.onnx").exists()
        for m in ("sar", "optical", "multispectral")
    )


def _index_exists() -> bool:
    return Path(cfg["paths"]["gallery_index"]).exists()


def ensure_loaded() -> None:
    """Load embedder + index on first use; raise 503 if missing."""
    if _state["embedder"] is not None and _state["index"] is not None:
        return
    if not _onnx_exists():
        raise HTTPException(
            status_code=503,
            detail="ONNX models not found. Run training/export_onnx.py first.",
        )
    if not _index_exists():
        raise HTTPException(
            status_code=503,
            detail="FAISS gallery index not found. Run index/build_index.py first.",
        )
    _state["embedder"] = build_embedder_from_config(cfg)
    _state["index"] = VectorIndex.load(cfg["paths"]["gallery_index"],
                                       cfg["paths"]["gallery_meta"])
    _state["retriever"] = Retriever(_state["embedder"], _state["index"],
                                    top_k=cfg["retrieval"]["top_k"])


@app.on_event("startup")
def _startup() -> None:
    if _onnx_exists() and _index_exists():
        try:
            ensure_loaded()
        except Exception as exc:  # noqa: BLE001
            print(f"[api] startup load skipped: {exc}")
    else:
        print("[api] artifacts missing — UI will run in demo-pending mode.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _thumbnail_b64(filepath: str, max_edge: int = 96) -> str | None:
    """Return a small base64 PNG of a gallery file (for the UI thumbnails)."""
    try:
        p = Path(filepath)
        if not p.exists():
            return None
        if p.suffix == ".npy":
            arr = np.load(p)  # (C, H, W)
            # Take first 3 channels for display, normalise to 0-255.
            disp = arr[:3]
            lo, hi = float(disp.min()), float(disp.max())
            disp = (np.clip((disp - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype("uint8")
            img = Image.fromarray(np.transpose(disp, (1, 2, 0)))
        else:
            img = Image.open(p)
        img.thumbnail((max_edge, max_edge))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse((_static / "index.html").read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    size = _state["index"].index.ntotal if _state["index"] else 0
    return HealthResponse(gallery_size=size, embed_dim=cfg["model"]["embed_dim"])


@app.post("/index", response_model=IndexResponse)
async def index_image(
    file: UploadFile = File(...),
    modality: str = Form(...),
    label: int | None = Form(None),
):
    """Embed an uploaded image of one modality and add it to the gallery."""
    if modality not in ("sar", "optical", "multispectral"):
        raise HTTPException(400, "modality must be sar|optical|multispectral")
    ensure_loaded()
    embedder = _state["embedder"]
    vidx = _state["index"]

    data = await file.read()
    if len(data) > cfg["api"]["max_upload_mb"] * 1024 * 1024:
        raise HTTPException(413, "file too large")
    vec = embedder.embed_bytes(data, modality)
    meta = [{
        "filepath": f"(upload:{file.filename})",
        "modality": modality,
        "label": int(label) if label is not None else -1,
        "class_name": "",
        "sample_id": -1,
    }]
    vidx.add(vec[None], meta)
    # Persist so the gallery survives restarts.
    vidx.save(cfg["paths"]["gallery_index"], cfg["paths"]["gallery_meta"])
    return IndexResponse(added=1, gallery_size=len(vidx), modality=modality, label=label)


@app.post("/query", response_model=QueryResponse)
async def query_image(
    file: UploadFile = File(...),
    modality: str = Form(...),
):
    """Retrieve top-5/top-10 from the gallery for an uploaded query crop."""
    if modality not in ("sar", "optical", "multispectral"):
        raise HTTPException(400, "modality must be sar|optical|multispectral")
    ensure_loaded()
    retriever: Retriever = _state["retriever"]

    data = await file.read()
    if len(data) > cfg["api"]["max_upload_mb"] * 1024 * 1024:
        raise HTTPException(413, "file too large")
    res = retriever.query_bytes(data, modality, top_k=cfg["retrieval"]["top_k"])

    def to_items(hits) -> list[HitItem]:
        items = []
        for rank, h in enumerate(hits, 1):
            md = h.metadata
            items.append(HitItem(
                rank=rank,
                score=h.score,
                modality=md.get("modality", "?"),
                label=int(md.get("label", -1)),
                class_name=str(md.get("class_name", "")),
                thumbnail_b64=_thumbnail_b64(md.get("filepath", "")),
                filepath=str(md.get("filepath", "")),
            ))
        return items

    return QueryResponse(
        modality=modality,
        top_5=to_items(res.hits[:5]),
        top_10=to_items(res.hits[:10]),
        embed_ms=res.embed_ms,
        search_ms=res.search_ms,
        total_ms=res.total_ms,
        gallery_size=len(_state["index"]),
    )


@app.post("/admin/reload")
def reload_index() -> JSONResponse:
    """Force a re-read of the on-disk index (after a rebuild)."""
    ensure_loaded()
    _state["index"] = VectorIndex.load(cfg["paths"]["gallery_index"],
                                       cfg["paths"]["gallery_meta"])
    _state["retriever"] = Retriever(_state["embedder"], _state["index"],
                                    top_k=cfg["retrieval"]["top_k"])
    return JSONResponse({"gallery_size": len(_state["index"])})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=cfg["api"]["host"], port=cfg["api"]["port"],
                reload=False)
