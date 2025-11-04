

````markdown
# ⚙️ Setup Guide: ML Lifecycle Demo

This guide explains how to prepare your environment and install dependencies for the House Price Prediction ML Lifecycle Demo project.

---

## 🧩 1. Prerequisites

Ensure you have the following tools installed on your system before proceeding:

* **Python:** Version **3.10** or higher.
* **Git:** Necessary to clone the repository.
* **Docker:** (Optional but highly recommended) Required for building and running the containerized deployment.
* **IDE:** Recommended integrated development environment (e.g., VSCode, PyCharm).

---
````

## 🏗️ 2. Clone the Repository

Open your terminal or command prompt and execute:

```bash
git clone [https://github.com/yourusername/ml-lifecycle-demo.git](https://github.com/yourusername/ml-lifecycle-demo.git)
cd ml-lifecycle-demo
```

-----

## 💻 3. Create and Activate a Virtual Environment

It is crucial to isolate project dependencies using a virtual environment.

### **Windows**

```bash
python -m venv venv
.\venv\Scripts\activate
```

### **macOS/Linux**

```bash
python -m venv venv
source venv/bin/activate
```

-----

## 📦 4. Install Dependencies

Install all necessary libraries using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### **`requirements.txt` contents:**

```text
fastapi
uvicorn[standard]
scikit-learn
xgboost
pandas
numpy
joblib
```

-----

## 📂 5. Directory Overview

Familiarize yourself with the project structure:

| Folder | Description |
| :--- | :--- |
| `data/` | Stores raw and processed versions of the California Housing dataset. |
| `src/` | Contains the core Python scripts (`preprocess.py`, `train.py`, etc.) for the ML pipeline. |
| `models/` | Storage location for the saved machine learning model (`.joblib`) and associated metadata. |
| `app/` | Houses the **FastAPI** application logic for serving predictions. |
| `artifacts/` | Stores outputs like evaluation reports and performance metrics (`metrics.json`). |
| `docs/` | Internal documentation and guides (like this one). |

-----

## ✅ 6. Test Setup

Verify that the main tools are correctly installed within your active virtual environment:

```bash
# Check FastAPI version
python -m uvicorn --version

# Check Scikit-learn import
python -c "import sklearn; print('Scikit-learn OK')"
```

You are now ready to start the **ML lifecycle workflow** by running the scripts in the `src/` directory\!

