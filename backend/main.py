"""
main.py
FastAPI backend serving predictions + Grad-CAM heatmaps for uploaded histopathology images.

Run locally:
    uvicorn main:app --reload --port 8000

Make sure model.pth (produced by model/train.py) is in this directory, or set
MODEL_PATH to point at it.
"""

import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model_utils import load_model, predict_with_heatmap

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pth")

app = FastAPI(title="Breast Histopathology Classifier")

# Loosen this to your actual deployed frontend origin before going to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, gradcam, classes = None, None, None


@app.on_event("startup")
def startup_event():
    global model, gradcam, classes
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: {MODEL_PATH} not found. /predict will fail until a trained "
              f"checkpoint is placed here (see model/train.py).")
        return
    model, gradcam, classes = load_model(MODEL_PATH, device)
    print(f"Model loaded. Classes: {classes}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — train and place model.pth first.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()

    try:
        result = predict_with_heatmap(image_bytes, model, gradcam, classes, device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return result
