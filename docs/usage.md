

# 🧠 Usage Guide: End-to-End ML Workflow

This document explains how to execute the project's entire Machine Learning lifecycle, starting from data preparation all the way through to deploying the prediction API.


---

## 📊 Step 1: Data Preparation

The first step is to load the raw dataset and prepare it for training.

```bash
python src/preprocess.py


  * **Action:** Loads the California Housing dataset from `sklearn.datasets`.
  * **Result:** Splits the data into training and testing sets (e.g., 80/20 split).
  * **Output:** Saves the processed data (e.g., as `.csv` or `.pkl` files) to the `data/processed/` directory.

-----

## 🧮 Step 2: Model Training

Train the machine learning model using the prepared training data.

```bash
python src/train.py
```

  * **Action:** Loads the processed training data.
  * **Model:** Instantiates and trains the selected regressor (e.g., **XGBoost** or **RandomForest**).
  * **Output 1 (Artifact):** The trained model is saved using `joblib` in `models/house_price_model.joblib`.
  * **Output 2 (Metadata):** Key information about the training run is logged in `models/metadata.json`.

**Example `metadata.json` structure:**

```json
{
  "model_name": "house_price_model.joblib",
  "trained_on": "YYYY-MM-DD",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 5
  },
  "initial_rmse": 0.62,
  "initial_r2_score": 0.89
}
```

-----

## 📈 Step 3: Model Evaluation

Assess the performance of the newly trained model against the held-out test data.

```bash
python src/evaluate.py
```

  * **Action:** Loads the saved model artifact (`.joblib`) and the test data from `data/processed/`.
  * **Calculation:** Computes key regression metrics, such as **Root Mean Squared Error (RMSE)** and **R-squared ($R^2$) score**.
  * **Output:** Saves the final, official metrics to `artifacts/metrics.json`.

-----

## 🌐 Step 4: Serve Model with FastAPI

Once the model is trained and evaluated, deploy it as a high-performance web API.

### **Start the API (Development Mode)**

Use `uvicorn` to run the application with automatic reloading:

```bash
uvicorn app.main:app --reload
```

The API documentation (Swagger UI) is available at `http://127.0.0.1:8000/docs`.

### **API Test (cURL)**

Test the `/predict` endpoint by sending a sample request payload containing the **8 California Housing features**.

```bash
curl -X POST [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict) \
-H "Content-Type: application/json" \
-d '{"features":[8.3252,41.0,6.98412341,1.02380952,322.0,2.55555556,37.88,-122.23]}'
```

**Expected JSON response:**

```json
{"predicted_price": 4.526}
```

*(Note: The exact price value may vary based on the specific model trained)*

-----

## 🐳 Step 5: Containerize for Deployment

To ensure maximum reproducibility and ease of deployment to any cloud environment, use Docker.

### **Build the Image**

```bash
docker build -t house-price-api .
```

### **Run the Container**

Map the container's port 8000 to the host machine's port 8000:

```bash
docker run -p 8000:8000 house-price-api
```

Your fully containerized ML model is now running and ready for production traffic\!

-----

## 🔁 Step 6: Reproduce or Update Model (Retraining Loop)

The pipeline is designed to be easily reproducible. When new data is available or if model performance degrades, follow the full lifecycle again:

1.  **Update Data:** Place new or combined raw data into the `data/raw/` folder.
2.  **Re-run Pipeline:** Re-execute the core scripts in sequence:
    ```bash
    python src/preprocess.py
    python src/train.py
    python src/evaluate.py
    ```
3.  **Deploy Updated Artifact:** The newly created model artifact will automatically be picked up by the FastAPI service when the Docker image is rebuilt and deployed.

<!-- end list -->

