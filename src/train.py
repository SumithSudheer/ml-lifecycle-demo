import os
import time
import json
import joblib
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------
# Config
# ------------------------------------
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_DIR = "models"
ARTIFACTS_DIR = "artifacts"

TARGET_COLUMN = "MEDV"
MODEL_TYPE = "random_forest"

# Timestamp for versioning
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

MODEL_PATH = os.path.join(MODEL_DIR, f"model_{timestamp}.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, f"metadata_{timestamp}.json")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, f"metrics_{timestamp}.json")


# ------------------------------------
# Data Loading
# ------------------------------------
def load_data():
    print("🔹 Loading preprocessed data...")

    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError("❌ Processed data not found. Run src/preprocess.py first.")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"❌ Target column '{TARGET_COLUMN}' missing in processed data.")

    print(f"✅ Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df


# ------------------------------------
# Model Training
# ------------------------------------
def train_model(train_df, test_df):
    print(f"🔹 Training model: {MODEL_TYPE}")

    # Split features and target
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    # Ensure numeric input (avoids dtype errors)
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

    start = time.time()

    # Initialize model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    elapsed = round(time.time() - start, 2)

    metrics = {
        "model_type": MODEL_TYPE,
        "timestamp": timestamp,
        "mse": mse,
        "r2": r2,
        "training_time_sec": elapsed,
        "train_rows": len(train_df),
        "test_rows": len(test_df)
    }

    print(f"✅ Training complete in {elapsed}s | MSE: {mse:.4f} | R²: {r2:.4f}")
    return model, metrics


# ------------------------------------
# Save Artifacts
# ------------------------------------
def save_artifacts(model, metrics):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Save model
    joblib.dump(model, MODEL_PATH)

    # Save metadata and metrics
    metadata = {
        "model_path": MODEL_PATH,
        "timestamp": timestamp,
        "model_type": MODEL_TYPE,
        "metrics": metrics
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"💾 Model saved → {MODEL_PATH}")
    print(f"🧾 Metadata saved → {METADATA_PATH}")
    print(f"📊 Metrics saved → {METRICS_PATH}")


# ------------------------------------
# Main
# ------------------------------------
def main():
    train_df, test_df = load_data()
    model, metrics = train_model(train_df, test_df)
    save_artifacts(model, metrics)


if __name__ == "__main__":
    main()
