@echo off
REM Launch the retrieval UI locally on the laptop (CPU-only).
REM
REM Assumes:
REM   - dependencies installed:  pip install -r requirements.txt
REM   - ONNX models present:     models\retrieval_model_{sar,optical,multispectral}.onnx
REM   - FAISS gallery present:   results\gallery.index + results\gallery_meta.parquet
REM
REM The UI still starts if artifacts are missing and shows a clear message.

setlocal
cd /d "%~dp0\.."

if "%PORT%"=="" set PORT=8000
if "%HOST%"=="" set HOST=127.0.0.1

echo Starting Cross-Modal Satellite Retrieval UI on http://%HOST%:%PORT%
echo Press Ctrl+C to stop.
uvicorn api.main:app --host %HOST% --port %PORT% --reload
endlocal
