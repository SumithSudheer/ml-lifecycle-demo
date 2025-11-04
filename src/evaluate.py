import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------
# Config
# ------------------------------------
MODEL_DIR = "models"
ARTIFACTS_DIR = "artifacts"
TEST_PATH = "data/processed/test.csv"
TARGET_COLUMN = "MEDV"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, f"metrics_{timestamp}.json")


# ------------------------------------
# Utility: Get Latest Model
# ------------------------------------
def get_latest_model(model_dir: str) -> str:
    """Return path of the most recent .joblib model."""
    models = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
    if not models:
        raise FileNotFoundError(f"❌ No model found in '{model_dir}'. Run training first.")
    latest_model = sorted(models)[-1]  # by timestamp in filename
    return os.path.join(model_dir, latest_model)


# ------------------------------------
# Load Artifacts
# ------------------------------------
def load_artifacts():
    """Load the latest trained model and test data."""
    print("🔹 Loading model and test data...")

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"❌ Test dataset not found at {TEST_PATH}")

    model_path = get_latest_model(MODEL_DIR)
    model = joblib.load(model_path)
    test_df = pd.read_csv(TEST_PATH)

    print(f"✅ Loaded model: {model_path}")
    print(f"Test data shape: {test_df.shape}")
    return model, test_df, model_path


# ------------------------------------
# Evaluate Model
# ------------------------------------
def evaluate(model, test_df):
    """Compute RMSE and R² metrics."""
    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(f"❌ Target column '{TARGET_COLUMN}' not found in test data.")

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    print("🔹 Generating predictions...")
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "timestamp": timestamp,
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "test_rows": len(test_df)
    }

    print(f"✅ Evaluation complete | RMSE: {rmse:.4f} | R²: {r2:.4f}")
    return metrics


# ------------------------------------
# Save Metrics
# ------------------------------------
def save_metrics(metrics, model_path):
    """Save evaluation metrics to artifacts directory."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    metrics_data = {
        "model_evaluated": os.path.basename(model_path),
        "timestamp": timestamp,
        "metrics": metrics
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"📁 Metrics saved → {METRICS_PATH}")


# ------------------------------------
# Main
# ------------------------------------
def main():
    model, test_df, model_path = load_artifacts()
    metrics = evaluate(model, test_df)
    save_metrics(metrics, model_path)


if __name__ == "__main__":
    main()
