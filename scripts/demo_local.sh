#!/usr/bin/env bash
# Launch the retrieval UI locally on the laptop (CPU-only).
#
# Assumes:
#   - a virtualenv is active with `pip install -r requirements.txt` done
#   - ONNX models exist at models/retrieval_model_{sar,optical,multispectral}.onnx
#     (run training/export_onnx.py first, or training/train_colab.ipynb on Colab)
#   - a FAISS gallery exists at results/gallery.index + results/gallery_meta.parquet
#     (run index/build_index.py first)
#
# If the artifacts are missing the UI still starts and shows a clear message.

set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

echo "Starting Cross-Modal Satellite Retrieval UI on http://${HOST}:${PORT}"
echo "Press Ctrl+C to stop."
exec uvicorn api.main:app --host "$HOST" --port "$PORT" --reload
