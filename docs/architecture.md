

# 🏗️ Architecture Overview: ML Lifecycle

This document explains the system design and core Machine Learning lifecycle architecture for the **House Price Prediction** project, emphasizing modularity and production readiness.

---

## 🎯 Design Goal


The primary objective is to demonstrate a **reproducible, production-grade ML workflow** that establishes a clear, decoupled flow between training and serving. The design integrates:


* **Modular Scripts:** Decoupled Python scripts for data, training, and evaluation.
* **Artifact Management:** Clear separation and versioning of model artifacts.
* **Microservice Deployment:** Deployment as a high-performance **REST API**.
* **Containerization:** Full portability using **Docker**.

````markdown
---

## 🔄 ML Lifecycle Stages

The project follows the standard sequential ML lifecycle, with clear tools assigned to each stage:

| Stage | Description | Key Tool / Framework |
| :--- | :--- | :--- |
| **Data Ingestion** | Load and initial cleaning of the dataset. | Pandas |
| **Preprocessing** | Feature engineering, scaling, and the crucial train/test split. | Scikit-learn |
| **Training** | Model selection and fitting (e.g., hyperparameter tuning). | XGBoost / Scikit-learn |
| **Evaluation** | Compute key performance metrics (RMSE, $R^2$) on the test set. | Scikit-learn Metrics |
| **Packaging** | Serializing the trained model object for later use. | Joblib |
| **Deployment** | Serving the packaged model via a web service. | **FastAPI** |
| **Monitoring** | Logging live predictions and detecting model/data drift. | Custom / EvidentlyAI (Future) |

---

## 🧱 System Data Flow

The architecture is pipeline-centric, where the output of one script becomes the input of the next, culminating in the deployment artifact.

```text
                               ┌─────────────────────┐
                               │ Raw Data (CSV)      │
                               │ data/raw/           │
                               └────────┬────────────┘
                                        │
                         
                                        ▼
               ┌─────────────────────────────────────────────────┐
               │ 1. Preprocessing (src/preprocess.py)            │
               │    → Split Data (Train/Test) + Clean Features   │
               └───────────────────┬───────────────┬─────────────┘
                                   │               │
                                   ▼               ▼
                       ┌─────────────────┐ ┌─────────────────┐
                       │ 2. Training     │ │ 3. Evaluation   │
                       │ (src/train.py)  │ │ (src/evaluate.py)│
                       │ → Saves Model   │ │ → Saves Metrics │
                       └────────┬────────┘ └────────┬────────┘
                                ▼                  │
                       ┌─────────────────┐         │
                       │ Model Artifact  │         │
                       │ models/*.joblib │         │
                       └────────┬────────┘         │
                                ▼                  │
               ┌─────────────────────────────────────────────────┐
               │ 4. Deployment (app/main.py + Dockerfile)        │
               │    → FastAPI REST API for Inference             │
               └─────────────────────────────────────────────────┘
````

-----

## 🧰 Key Artifacts

These files are essential for reproducibility, version control, and model serving. They represent the "hand-offs" between pipeline stages.

| Artifact File | Description | Role in System |
| :--- | :--- | :--- |
| `models/house_price_model.joblib` | The serialized **trained model object**. | Loaded directly by the FastAPI app for predictions. |
| `models/metadata.json` | Records model details (hyperparameters, train date, initial metrics). | **Model Versioning/Audit Trail**. |
| `artifacts/metrics.json` | Final performance metrics (RMSE, $R^2$). | **Evaluation Report**. |
| `data/processed/` | Cleaned and split input files (`train.csv`, `test.csv`). | Input for training and evaluation scripts. |

-----

## 🐳 Deployment Strategy

The final product is packaged as a single, immutable Docker image.

1.  **Preparation:** The ML pipeline is run (`preprocess.py` → `train.py` → `evaluate.py`), generating the necessary model artifact (`.joblib`).
2.  **Container Build:** The `Dockerfile` copies the trained model and the FastAPI application (`app/main.py`) into a lightweight image.
    ```bash
    docker build -t house-price-api .
    ```
3.  **Production Run:** The image is deployed to a server (e.g., AWS EC2, Google Compute Engine, Kubernetes) and run.
    ```bash
    docker run -p 8000:8000 house-price-api
    ```

-----

## 🧩 Future Enhancements (MLOps Roadmap)

This demo establishes a strong foundation. Future work could focus on full MLOps automation and enterprise features:

  * **Experiment Tracking:** Integrate **MLflow** or **DVC** to manage and compare training runs and model versions.
  * **Data Quality/Drift:** Add **EvidentlyAI** or another tool to monitor incoming data and model prediction quality.
  * **Logging:** Implement a structured logger to push predictions, inputs, and latency metrics to a database (e.g., MongoDB, Prometheus).
  * **Automation:** Automate the retraining pipeline using tools like **Celery**, **Apache Airflow**, or cloud-native solutions (e.g., GitHub Actions, AWS Step Functions).

-----

**Result:** A clean, modular, production-grade ML lifecycle template that can be reused and scaled for any predictive system.

-----
