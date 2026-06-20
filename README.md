# Cross-Modal Satellite Image Retrieval

Cross-modal **content-based satellite image retrieval** using multi-sensor
remote sensing data (SAR, optical, multispectral). A query image of any one
modality is returned the **top-5 / top-10** most semantically similar images
from a gallery — same-modal (SAR→SAR) or cross-modal (SAR→optical,
optical→multispectral, …).

Built around three ideas that map directly onto the user's design:

| Idea | Implementation |
| --- | --- |
| Scan an image **box-by-box** of equal boxes | **Vision Transformer** — splits the image into a grid of patches |
| Save each image as **encoded numbers** | **512-d embedding vector** stored in a **FAISS** index |
| Treat a location's **3 modalities as the same** image | **Contrastive (InfoNCE) loss** aligns them in one embedding space |
| Query with a **cropped** image and find matches | Embed the crop → **FAISS** similarity search → ranked top-k |

> See [PLAN.md](PLAN.md) for the full decision record and
> [RESEARCH.md](RESEARCH.md) for why a ViT+contrastive+FAISS stack is used
> instead of an LLM.

---

## Architecture

```
INDEX (3 uploads per location)            QUERY (1 crop, any modality)
  SAR / optical / multispectral             │
            │                               │
   input adapter (channel proj)             │
            │                               │
   shared ViT-S/16 encoder  ◄───────────────┘  (same encoder)
            │
   projection MLP → 512-d → L2-norm
            │
   embedding vector
            │
   FAISS IndexFlatIP  +  metadata (id → file, modality, class)
                                                │
                          top-k search → top-5 / top-10 ranked results
```

---

## Quick start

The project lives at `F:\ZCodeProject\cross-modal-satellite-retrieval\`. All
commands below run from that directory.

### 1. Install
```cmd
cd /d F:\ZCodeProject\cross-modal-satellite-retrieval
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Smoke test (laptop, CPU-only, ~1 min) — optional but recommended
Confirms every stage (data → train → export → index → eval) works before the
real Colab run. Uses synthetic data and 2 epochs:
```cmd
python scripts\run_full_pipeline.py --tiny
```

### 3. Prepare data
```cmd
python data\download_sen12ms.py --num 8000 --out data\sen12ms --size 64
```
Tries the real SEN12MS mirrors first; falls back to a learnable synthetic
subset if they are unreachable, so the pipeline always runs end-to-end.

### 4. Train (on Google Colab — needs a GPU)
Open `training/train_colab.ipynb` in Colab, set the runtime to **T4 GPU**,
and run all cells. It will:
1. download a balanced SEN12MS subset,
2. fine-tune the shared ViT-S with InfoNCE (checkpoints saved to Drive
   every epoch so disconnects cost nothing),
3. export three ONNX models `models/retrieval_model_{sar,optical,multispectral}.onnx`.

### 5. Build the gallery index (runs on the laptop, CPU-only)
```cmd
python index\build_index.py --config config\config.yaml
```
Produces `results/gallery.index` (FAISS) and `results/gallery_meta.parquet`
— this is the project's "encoded numbers/binaries" store.

### 6. Evaluate (the competition metrics)
```cmd
python retrieval\evaluate.py --config config\config.yaml
```
Writes F1@5 / F1@10 (same- and cross-modal) + avg retrieval time to
`results/metrics.json`.

### 7. Run the web UI (laptop, CPU-only)
```cmd
scripts\demo_local.bat
:: or:  uvicorn api.main:app --reload --port 8000
```
Open http://127.0.0.1:8000 — upload 3-modal images to add to the gallery,
then upload a **cropped** query image of any modality to retrieve top-5/top-10.

> The UI starts even before model/index artifacts exist and shows a clear
> "not ready" message, so you can see it immediately.

---

## Project layout

```
cross-modal-satellite-retrieval/
├── README.md  PLAN.md  RESEARCH.md  requirements.txt
├── config/config.yaml
├── data/          download_sen12ms.py  dataset.py  preprocess.py  splits.py
├── models/        backbones.py  embedding_head.py  retrieval_model.py
├── losses/        contrastive.py
├── training/      train.py  train_colab.ipynb  export_onnx.py
├── index/         build_index.py  faiss_index.py
├── retrieval/     embedder.py  search.py  evaluate.py
├── api/           main.py  schemas.py  static/{index.html,style.css,app.js}
├── notebooks/     explore_data.ipynb
├── scripts/       run_full_pipeline.py  demo_local.sh
└── results/       (gallery.index, gallery_meta.parquet, metrics.json)
```

---

## Results

Populated after the first real training run by `retrieval/evaluate.py`:

| Metric | Same-modal | Cross-modal |
| --- | :---: | :---: |
| F1@5  | — | — |
| F1@10 | — | — |
| Avg retrieval time / query | — ms | — ms |

**Current verification status:** the data layer (download/synthetic generator,
splits, per-modality preprocessing, manifest format) is verified working on
the laptop; all 40 Python modules compile cleanly. Training/index/eval require
the heavy dependencies (`torch`, `timm`, `faiss-cpu`, `onnxruntime`) to be
installed — run `pip install -r requirements.txt` then `run_full_pipeline.py
--tiny` for a full CPU smoke test.

---

## References
- [SEN12MS dataset + code](https://github.com/schmitt-muc/SEN12MS)
- [Awesome-RSITR — retrieval benchmark](https://github.com/jaychempan/Awesome-RSITR)
- [Awesome-Remote-Sensing-Foundation-Models](https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models)
- [SOMatch — SAR↔optical matching](https://github.com/system123/SOMatch)
- [Prexl et al. CVPRW 2023 — contrastive learning on SEN12MS](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Prexl_Multi-Modal_Multi-Objective_Contrastive_Learning_for_Sentinel-12_Imagery_CVPRW_2023_paper.pdf)
- [DINOv2 + FAISS retrieval notebook](https://github.com/roboflow/notebooks/blob/main/notebooks/dinov2-image-retrieval.ipynb)
