import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# -------------------------------
# Paths and Config
# -------------------------------
RAW_DATA_PATH = "data/raw/housing.csv"
PROCESSED_DIR = "data/processed"
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")

TARGET_COLUMN = "MEDV"  # 🎯 target column


# -------------------------------
# Data Cleaning Function
# -------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe:
    - Encode categorical columns
    - Fill missing values
    - Ensure numeric types
    """
    df = df.copy()

    # Encode categorical columns
    for col in df.select_dtypes(include=["object"]).columns:
        if col != TARGET_COLUMN:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Fill missing numeric values with column means
    df = df.fillna(df.mean(numeric_only=True)).fillna(0)

    # Convert all values to numeric (just in case)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    return df


# -------------------------------
# Main Script
# -------------------------------
def main():
    print("🔹 Loading raw dataset...")
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"❌ Raw dataset not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    print(f"✅ Loaded dataset with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"❌ Target column '{TARGET_COLUMN}' not found in dataset.")

    # Clean data
    print("🔹 Cleaning dataset...")
    df = clean_data(df)

    # Split train/test
    print("🔹 Splitting dataset into train/test (80/20)...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create processed directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Add timestamped copies (optional)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_file = f"train_{timestamp}.csv"
    test_file = f"test_{timestamp}.csv"

    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    # Keep timestamped versions for versioning
    train_df.to_csv(os.path.join(PROCESSED_DIR, train_file), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, test_file), index=False)

    print(f"✅ Preprocessing complete.")
    print(f"Train: {train_df.shape} → saved to {TRAIN_PATH}")
    print(f"Test:  {test_df.shape} → saved to {TEST_PATH}")
    print(f"🕒 Versioned copies saved with timestamp: {timestamp}")


if __name__ == "__main__":
    main()
