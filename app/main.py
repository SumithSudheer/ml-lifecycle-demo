from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import traceback

# -------------------------------
# Config
# -------------------------------
MODEL_DIR = "models"
DEFAULT_MODEL = "house_price_pipeline.joblib"  # fallback model name
TARGET_COLUMN = "MEDV"  # for context (not used directly)


# -------------------------------
# Utility: Load latest model
# -------------------------------
def get_latest_model(model_dir: str = MODEL_DIR) -> str:
    """Return path of the latest model file in /models directory."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"❌ Model directory not found: {model_dir}")

    models = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
    if not models:
        raise FileNotFoundError("❌ No model found in /models directory. Train a model first.")

    # Prefer 'house_price_pipeline.joblib' if exists; else use latest timestamped model
    if DEFAULT_MODEL in models:
        return os.path.join(model_dir, DEFAULT_MODEL)

    latest_model = sorted(models)[-1]
    return os.path.join(model_dir, latest_model)


# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(
    title="🏡 House Price Prediction API",
    description="Predict house prices using a trained ML model.",
    version="1.0.0"
)


# -------------------------------
# Input Schema
# -------------------------------
class HouseFeatures(BaseModel):
    RM: float
    LSTAT: float
    PTRATIO: float


# -------------------------------
# Load Model at Startup
# -------------------------------
try:
    model_path = get_latest_model()
    model = joblib.load(model_path)
    print(f"✅ Model loaded successfully from: {model_path}")
except Exception as e:
    model = None
    print(f"❌ Failed to load model: {e}")
    traceback.print_exc()


# -------------------------------
# Root Endpoint
# -------------------------------
@app.get("/")
def root():
    return {
        "message": "🏡 House Price Prediction API",
        "model_loaded": model is not None,
        "model_path": model_path if model else None
    }


# -------------------------------
# Prediction Endpoint
# -------------------------------
@app.post("/predict")
def predict(features: HouseFeatures):
    """Predict house price from RM, LSTAT, and PTRATIO values."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train a model first.")

    try:
        df = pd.DataFrame([features.dict()])
        prediction = model.predict(df)[0]
        return {
            "input": features.dict(),
            "prediction": round(float(prediction), 2),
            "model_path": model_path
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
