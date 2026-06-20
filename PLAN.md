# Implementation Plan & Decision Record

> Cross-Modal Satellite Image Retrieval Using Multi-Sensor Remote Sensing Data.
> This file is the durable record of *why* the project is built the way it is.

---

## 1. Goal

A retrieval system that accepts a query satellite image from one modality
(SAR / optical / multispectral) and returns the **top-5 / top-10** most
semantically similar images from a gallery — both **same-modal**
(SAR→SAR, …) and **cross-modal** (SAR→optical, optical→multispectral, …).

Reported metrics (per the problem statement):

| Metric | Scope |
| --- | --- |
| F1-score@5  | same-modal retrieval |
| F1-score@10 | same-modal retrieval |
| F1-score@5  | cross-modal retrieval |
| F1-score@10 | cross-modal retrieval |
| Average retrieval time per query | all cases (lower is better) |

Cross-modal retrieval is weighted more heavily because it is harder.

---

## 2. Confirmed decisions

| Decision | Choice | Reason |
| --- | --- | --- |
| Core architecture | **ViT + contrastive learning + FAISS** | Does the "box-by-box scan → encoded vectors → cross-modal match → crop query" workflow the user described, fast on a GPU-less laptop. |
| Dataset | **SEN12MS subset (~5–10k triplets)** | Co-registered SAR + multispectral + cloud-free optical + land-cover labels; perfect for same- and cross-modal retrieval and for F1 evaluation. |
| UI | **FastAPI backend + HTML/JS frontend** | Production-like REST API, lightweight, deployable to GCP. |
| Train env | **Google Colab free T4 GPU** | User has no local GPU; Colab free tier is enough for the subset. |
| Inference env | **Samsung Book 3 360 (CPU) via ONNX Runtime** | No GPU needed; ~30–50 ms/image. |
| Image size | **64×64** | Fast CPU inference; ViT-S/16 → 16 patches (the "box-by-box scan"). |
| Embedding dim | **512** | Good retrieval quality vs. index size. |

---

## 3. Why NOT an LLM (engineering note)

The original research document attached by the user describes fine-tuning a
**security / vulnerability-detection LLM** (Llama-3.1 8B / DeepSeek Coder 6.7B)
for finding bugs in source code. That is a *different domain*. The proposal to
re-use the "LLM scans pixel-by-pixel" idea for satellite images is **not viable**:

- A text LLM cannot see images at all.
- Vision-Language Models (VLMs) do **not** scan pixel-by-pixel — they compress
  images into patches and lose fine pixel detail
  ([arXiv 2601.13401](https://arxiv.org/html/2601.13401v1),
  [arXiv 2408.03940](https://arxiv.org/html/2408.03940v1)).
- A 4–5 B parameter model would be far too heavy and slow on a GPU-less laptop
  and would hurt the "low retrieval time" scoring criterion.

**The user's three core intuitions map 1:1 onto the correct technology instead:**

| User's idea | Correct technology |
| --- | --- |
| "Scan box-by-box of equal boxes" | **Vision Transformer** — splits image into 16×16 patches natively |
| "Save as encoded numbers/binaries" | **Embedding vector** (512 floats) stored in **FAISS** |
| "Assign the 3 modes as the same image" | **Contrastive loss** pulls SAR/optical/multispectral of the same location together |
| "Upload ¼ crop, scan, find match" | Embed the crop → FAISS similarity search → top-5/top-10 |

So we get **everything the user asked for** (box-scan, encoded storage,
cross-modal matching, crop-query retrieval) — with a ViT + contrastive + FAISS
stack instead of an LLM.

---

## 4. Transferable lessons from the research document

The security-LLM research is not wasted — these principles carry over:

1. **Use a pre-trained model + fine-tune, never train from scratch.**
   We start from a timm ViT-S (ImageNet-pretrained) and only fine-tune — far
   cheaper than random init.
2. **Train in the cloud, run inference locally.**
   Training happens on Colab T4; the laptop only does ONNX inference.
3. **Quantisation / lightweight deployment matters under tight resources.**
   We export to ONNX and (optionally) quantise to int8 for the laptop.

---

## 5. Architecture

```
                 ┌─────────────────────────────────────────────────────┐
   INDEX TIME    │  SAR (2ch)   Optical (3ch)   Multispectral (4ch)     │
   (3 uploads)   │        \          |          /                       │
                 │     input adapter (per-modality channel projection)  │
                 │                        |                             │
                 │          shared ViT-S/16 encoder (timm)              │
                 │                  [CLS] token                         │
                 │                        |                             │
                 │            projection MLP → 512-d                    │
                 │                  L2-normalise                        │
                 │                        |                             │
                 │             embedding vector (float32)               │
                 └------------------------┬----------------------------┘
                                          │
                              FAISS IndexFlatIP  +  metadata table
                              (id → filepath, modality, land-cover class)

   QUERY TIME    crop (any modality) → same encoder → vector →
                 FAISS top-k search → ranked top-5 / top-10
                 (a hit is "relevant" if same land-cover class)
```

**Backbone:** timm `vit_small_patch16_224` adapted to 64×64 input.
The patch embed stem is replaced so the per-modality channel counts
(2/3/4) map to the ViT width. Weights are **shared** across modalities,
which forces cross-modal alignment in the shared embedding space.

**Head:** 2-layer MLP, 512-d output, L2-normalised.

**Loss:** InfoNCE with temperature τ = 0.07. Positives = the same location's
images across the three modalities (+ augmented copies); negatives = every
other location in the batch.

---

## 6. Data pipeline (SEN12MS subset)

- **Triplet contents:** Sentinel-1 SAR (VV+VH, 2-channel),
  Sentinel-2 multispectral optical (we keep B,G,R,NIR → 4-channel, plus a
  3-channel RGB view for the "optical" modality),
  MODIS land-cover label (1 of 10 classes).
- **Subset strategy:** ~8k triplets, balanced across the 10 land-cover classes.
- **Preprocessing:**
  - Resize to **64×64**.
  - **SAR:** decibel scale `10·log10(x)`, clip to [-25, 0], min-max normalise.
  - **Optical / multispectral:** per-band z-score normalisation.
- **Splits:** 70% train / 10% val / 20% eval. Within eval, every image is a
  candidate query; the rest form the gallery. Relevance = same land-cover class.

---

## 7. Training (Colab T4)

- Batch 256, 30 epochs, AdamW + cosine schedule, τ = 0.07.
- Augmentations: random crop, flip, Gaussian noise; SAR additionally gets
  speckle noise.
- Checkpoints saved to Google Drive every epoch so free-tier disconnects do
  not lose progress.
- Time budget: ~1.5–2 h on T4 for the subset.

---

## 8. ONNX export + CPU inference

- `torch.onnx.export` with dynamic batch axis; verify output parity.
- Inference via **ONNX Runtime CPU** — no GPU required on the laptop.

---

## 9. FAISS index ("encoded numbers/binaries storage")

- `IndexFlatIP` (inner product = cosine after L2-norm) for the subset.
- `IndexIVFPQ` documented for scaling to the full 180k dataset.
- Sidecar metadata table `id → {filepath, modality, land_cover_class}`
  saved as `.parquet` (this is the "binary" record the user described).

---

## 10. Evaluation (the competition metrics)

- Same-modal: SAR→SAR, OPT→OPT, MS→MS.
- Cross-modal: SAR→OPT, OPT→SAR, OPT→MS, MS→OPT.
- A retrieved image is *relevant* if it shares the query's land-cover class.
- Compute **F1@5, F1@10** (same- and cross-modal) and **average retrieval
  time per query**; dump to `results/metrics.json`.

F1@k is computed per query then averaged:

```
precision@k = (#relevant in top-k) / k
recall@k    = (#relevant in top-k) / (#relevant in gallery)
F1@k        = 2·P·R / (P+R)
```

---

## 11. FastAPI + HTML frontend

- `POST /index`  — 3 uploaded images + label → embed → add to FAISS.
- `POST /query`  — 1 query crop + modality flag → embed → search → top-5/top-10 JSON (base64 thumbnails, scores, modalities, time).
- `GET  /health`, `GET /` (serve UI).
- Frontend: drag-drop upload zones for the 3 modalities, a query zone with a
  modality selector + in-browser canvas crop, and a results grid showing
  top-5/top-10 ranked thumbnails with similarity bars and per-query time.

---

## 12. (Optional) GCP deployment

Containerise with Docker; deploy to **GCP Cloud Run** (scales to zero) or an
**e2-micro** VM. The ONNX model + FAISS index ship together; no GPU at runtime.

---

## 13. Build order

1. Scaffold repo + docs. ✅
2. Data pipeline (download, dataset, preprocess, splits).
3. Model + loss (backbones, head, retrieval model, InfoNCE).
4. Training (train.py, Colab notebook, ONNX export).
5. FAISS index (build_index, wrapper).
6. Retrieval + evaluation (embedder, search, evaluate).
7. FastAPI backend + HTML frontend.
8. End-to-end scripts + explore notebook.
9. README results table populated from a real run.

---

## 14. Honest limitations

- The system does not run an LLM over pixels — it uses a ViT, which achieves
  the same goal correctly.
- F1 on an 8k subset will be modest; we will report real numbers, then scale.
- Free Colab disconnects may interrupt long training — mitigated by
  checkpointing to Drive.

---

## 15. References

- [SEN12MS repo](https://github.com/schmitt-muc/SEN12MS) — dataset + loader
- [SEN12MS paper](https://elib.dlr.de/133280/1/SEN12MS_Preprint.pdf)
- [Awesome-RSITR](https://github.com/jaychempan/Awesome-RSITR) — retrieval benchmark
- [Awesome-Remote-Sensing-Foundation-Models](https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models)
- [SOMatch](https://github.com/system123/SOMatch) — Siamese SAR↔optical matching
- [Prexl et al. CVPRW 2023 — contrastive learning on SEN12MS](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Prexl_Multi-Modal_Multi-Objective_Contrastive_Learning_for_Sentinel-12_Imagery_CVPRW_2023_paper.pdf)
- [DINOv2 + FAISS retrieval notebook](https://github.com/roboflow/notebooks/blob/main/notebooks/dinov2-image-retrieval.ipynb)
- [PyTorch ONNX export tutorial](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
- [arXiv 2601.13401 — pixel precision in VLMs](https://arxiv.org/html/2601.13401v1)
