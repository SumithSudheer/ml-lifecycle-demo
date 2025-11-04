

```markdown
# 🏠 House Price Prediction — ML Lifecycle Demo

A production-style **end-to-end Machine Learning system** demonstrating the full lifecycle: **data → training → evaluation → deployment → monitoring**. This project is focused on providing a **practical, production-ready blueprint** for deploying an ML model.

---

## 🎯 Project Objective

This project aims to bridge the gap between ML experimentation and production deployment. By completing this demo, you will:

* **Understand the ML Lifecycle:** See how **data → model → deployment** fits together in a structured pipeline.
* **Practice MLOps Fundamentals:** Gain hands-on experience with **model packaging**, **versioning**, and tracking **metadata**.
* **Master Deployment:** Learn to build a robust prediction service using **FastAPI**.
* **Ensure Reproducibility:** Make the entire system **reproducible** and easily portable using **Docker**.

---

## 🧩 Tech Stack

| Component | Tool | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.11 | Primary language for all scripts and the API. |
| **ML Framework** | Scikit-learn / XGBoost | For model training and prediction. |
| **Web API** | FastAPI | High-performance, async framework for model serving. |
| **Packaging** | Joblib | For serializing and loading the trained model object. |
| **Containerization** | Docker | For building and deploying a reproducible API image. |
| **Dataset** | California Housing | A classic dataset from `sklearn.datasets` for regression. |

---

## ⚙️ Quickstart

Follow these steps to set up and run the entire ML pipeline locally.

### 1️⃣ Clone the Repository

```bash
git clone [https://github.com/yourusername/ml-lifecycle-demo.git](https://github.com/yourusername/ml-lifecycle-demo.git)
cd ml-lifecycle-demo
```

### 2️⃣ Setup Environment

Create a virtual environment and install the required dependencies.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3️⃣ Run Full ML Pipeline

Execute the scripts sequentially to process data, train the model, and evaluate performance.

```bash
# 1. Preprocess the raw data
python src/preprocess.py

# 2. Train and save the model
python src/train.py

# 3. Evaluate the model's performance
python src/evaluate.py
```

> **Output Files:** This step generates `data/processed/`, `models/house_price_model.joblib`, and `artifacts/metrics.json`.

-----

## 🚀 Model Serving & Testing

### 4️⃣ Serve the Model (Local Uvicorn)

Start the FastAPI prediction service locally with hot-reloading for development:

```bash
uvicorn app.main:app --reload
```

The API will be accessible at `http://127.0.0.1:8000`. You can view the **auto-generated documentation (Swagger UI)** at `http://127.0.0.1:8000/docs`.

### 5️⃣ Test Prediction (cURL)

Use `curl` to send a POST request to the `/predict` endpoint.

```bash
curl -X POST [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict) \
-H "Content-Type: application/json" \
-d '{"features":[8.3252,41.0,6.98412341,1.02380952,322.0,2.55555556,37.88,-122.23]}'
```

> The `features` list should contain the 8 California Housing features in the correct order. The response will be the model's predicted house price.

-----

## 🐳 Docker Deployment

For production-style deployment, containerize the FastAPI application.

### 6️⃣ Build the Docker Image

```bash
docker build -t house-price-api .
```

### 7️⃣ Run the Container

The application runs inside the container on port 8000. Map this to your host machine's port 8000.

```bash
docker run -p 8000:8000 house-price-api
```

The API is now running inside the Docker container and can be tested via `curl` as shown in step 5.

-----

## 📂 Project Structure

A well-organized structure is key to a maintainable ML project.

```
ml-lifecycle-demo/
│
├── data/
│   ├── raw/                  # Placeholder for raw, original data (e.g., CSV)
│   └── processed/            # Cleaned, split, and feature-engineered data
│
├── src/                      # Core ML logic scripts
│   ├── preprocess.py         # Handles data loading, cleaning, and splitting
│   ├── train.py              # Loads processed data, trains, and saves model
│   ├── evaluate.py           # Loads model and calculates metrics
│   └── predict.py            # Utility function for model inference
│
├── models/                   # Storage for trained models and associated files
│   ├── house_price_model.joblib # The serialized ML model
│   └── metadata.json         # Model version, training date, hyperparameters
│
├── app/                      # FastAPI application for serving
│   └── main.py               # API endpoints for health and prediction
│
├── artifacts/                # Outputs like evaluation reports
│   └── metrics.json          # Key performance metrics (e.g., RMSE, R-squared)
│
├── requirements.txt          # Python dependencies
├── Dockerfile                # Instructions to containerize the application
└── README.md                 # This file
```

-----

## 📈 Workflow Summary

| Stage | Script/Tool | Output | Purpose |
| :--- | :--- | :--- | :--- |
| **Preprocess** | `src/preprocess.py` | `data/processed/` | Prepare data for model training. |
| **Train** | `src/train.py` | `models/house_price_model.joblib` | Train the ML model and persist it. |
| **Evaluate** | `src/evaluate.py` | `artifacts/metrics.json` | Assess model quality and save results. |
| **Serve** | `app/main.py` | FastAPI endpoint | Expose model inference via an API. |
| **Deploy** | `Dockerfile` | Containerized API | Package the service for reliable deployment. |

-----

## 🧠 Key Learnings

By working through this demo, you will have solidified your knowledge on:

  * ✅ Building **reproducible ML pipelines** using isolated scripts.
  * ✅ Properly **packaging models** (`joblib`) and tracking associated **metadata**.
  * ✅ Building a high-performance serving layer with **FastAPI**.
  * ✅ **Dockerizing** a Python application for production readiness.
  * ✅ Establishing a base for future steps like **monitoring** and **retraining** loops.

<!-- end list -->

