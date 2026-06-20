# Research Notes

> How the prior security-LLM research informed (and was correctly *not* reused
> for) this cross-modal satellite image retrieval project.

---

## A. Origin of the research document

The user provided a detailed research write-up titled
*"A Practical Blueprint for Building a Sub-8B Security LLM on Free Google
Cloud Resources"*. Its subject is **fine-tuning a Large Language Model
(Llama-3.1 8B / DeepSeek Coder 6.7B) for software vulnerability detection** —
i.e. finding security bugs in source code. That is a **text / code** task in
the **cybersecurity** domain.

This project, by contrast, is a **computer-vision** task: cross-modal
satellite image retrieval. The two share *engineering principles* but not
*methods*. This file records what transfers and what does not.

---

## B. What transfers (and is applied here)

### 1. Build-vs-buy → always fine-tune, never train from scratch
The research correctly concludes that training an 8 B-parameter model from
random initialisation is computationally prohibitive, and that **fine-tuning a
pre-trained model** is the only pragmatic path under tight resources.

> **Applied here:** we start from a timm `vit_small_patch16` that is already
> ImageNet-pretrained, and only fine-tune it on the SEN12MS subset. We never
> train a ViT from scratch.

### 2. Train in the cloud, infer locally
The research describes a two-stage workflow: (1) fine-tune offline where GPU
is available, (2) deploy the resulting weights to a cheap CPU instance for
inference.

> **Applied here:** training runs on **Google Colab's free T4**; the laptop
> (Samsung Book 3 360, no GPU) runs only **ONNX Runtime CPU** inference.

### 3. Free-tier reality check
The research tabulates GCP free-tier limits (e2-micro CPU only, no free GPU,
30 GB disk, limited egress) and concludes GPU work must happen elsewhere.

> **Applied here:** same conclusion — we use Colab for the GPU phase and treat
> GCP (Cloud Run / e2-micro) as an *optional* hosting target for the finished
> API, not as a training environment.

### 4. Efficiency techniques under resource constraints
The research advocates quantisation + LoRA (QLoRA) to fit adaptation into
small memory.

> **Applied here:** the analogue is (a) a **small** backbone (ViT-S, ~22 M
> params) and (b) **ONNX export + optional int8 quantisation** so inference
> fits comfortably on a CPU laptop. LoRA itself is not needed because ViT-S
> is small enough to fine-tune fully on a T4.

---

## C. What does NOT transfer (and why)

### 1. "An LLM scans the image pixel-by-pixel"
This is the central misconception to correct. It does not work for images:

- A **text LLM** cannot perceive images at all — it only consumes tokens.
- A **Vision-Language Model (VLM)** does **not** scan pixel-by-pixel. Research
  shows VLMs compress images into patches and **lose** fine pixel detail:
  - *"Vision encoders compress images through patch embeddings, reducing
>     spatial indexing and losing the precise pixel-level tracking required."*
>     — [arXiv 2601.13401](https://arxiv.org/html/2601.13401v1)
  - A Pixel-Value-Prediction benchmark shows current VLMs are poor at
>     perceiving raw pixel values — [arXiv 2408.03940](https://arxiv.org/html/2408.03940v1).
- A 4–5 B parameter model would be far too heavy and slow on a GPU-less
  laptop, and the competition explicitly rewards **low retrieval time**.

### 2. Security datasets (PrimeVul, CVEfixes, Juliet, OWASP Top 10)
These are source-code vulnerability corpora. They have no role in satellite
retrieval and are not used.

### 3. The whole code/vulnerability toolchain (RAG over codebases, DAST
crawling, OWASP mapping)
Not applicable to an image-retrieval pipeline.

---

## D. The pivot: user intuition → correct technology

The user's mental model is actually sound — it just attaches to the wrong
machine-learning family. The mapping is exact:

| User's words | Correct technology (used in this project) |
| --- | --- |
| "Scan box-by-box of equal boxes for high accuracy" | **Vision Transformer** — the patch embed stem literally splits the image into a grid of equal patches (16 patches for a 64×64 input with patch size 16). |
| "Combines all and saves in an encoded format … as numbers or binaries" | **Embedding vector** (512 floats) stored in a **FAISS** index; metadata sidecar saved as `.parquet`. |
| "Assigns it as the same image" for the 3 modalities | **Contrastive (InfoNCE) loss** pulls the SAR, optical, and multispectral views of the *same* location to nearby points in the shared embedding space. |
| "Upload a cropped ¼ image … scan box-by-box … if it matches return that image" | Embed the crop with the same encoder → **FAISS top-k similarity search** → return ranked top-5 / top-10. |

So the project delivers every behaviour the user asked for, via the
ViT + contrastive + FAISS stack.

---

## E. Why this stack is the right one (evidence)

- **ViT + contrastive is the standard** for remote-sensing cross-modal
  retrieval — see the [Awesome-RSITR](https://github.com/jaychempan/Awesome-RSITR)
  benchmark and the
  [CVPRW 2023 contrastive paper on SEN12MS](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Prexl_Multi-Modal_Multi-Objective_Contrastive_Learning_for_Sentinel-12_Imagery_CVPRW_2023_paper.pdf).
- **DINOv2 + FAISS** is the dominant recipe for image-to-image retrieval in
  general vision — see the
  [Roboflow DINOv2 + FAISS notebook](https://github.com/roboflow/notebooks/blob/main/notebooks/dinov2-image-retrieval.ipynb).
  (DINOv2 is frozen / RGB-only, so it is a weaker baseline for SAR — we use a
  fine-tuned ViT as the primary model.)
- **CLIP-family ViT encoders produce dense embeddings ideal for vector
  search**, which is exactly the retrieval primitive we need.

---

## F. Dataset choice rationale

**SEN12MS** ([repo](https://github.com/schmitt-muc/SEN12MS),
[paper](https://elib.dlr.de/133280/1/SEN12MS_Preprint.pdf)) is the canonical
choice because each of its 180,662 samples is a **co-registered triplet** of:

1. Sentinel-1 **SAR** (dual-pol, VV+VH),
2. Sentinel-2 **multispectral** optical (13 bands),
3. MODIS **land-cover label** (one of 10 classes).

That gives us, for free:
- aligned SAR ↔ optical ↔ multispectral pairs (for contrastive positives),
- a semantic label (for F1@k relevance and balanced sampling).

We use a **~8k balanced subset** to keep download (~2–4 GB), training
(~1.5–2 h on a free T4) and laptop storage small, while still producing
meaningful F1 numbers.

---

## G. Summary

- The security-LLM research contributed **engineering principles**
  (fine-tune don't train-from-scratch; train in cloud, infer locally;
  favour small/quantised models under resource limits).
- It did **not** contribute a method, because "LLM over pixels" is not how
  images work.
- The user's *intuition* (box-scan → encode → cross-modal link → crop query)
  is implemented faithfully with **ViT + contrastive + FAISS**.
