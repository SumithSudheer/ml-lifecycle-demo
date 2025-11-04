# =======================================
#  Stage 1 — Base Environment
# =======================================
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy only requirement files first (for caching)
COPY requirements.txt .

# Install system dependencies (if needed for scikit-learn / xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# =======================================
#  Stage 2 — Copy Project Files
# =======================================
COPY . .

# Create necessary directories (if they don’t exist)
RUN mkdir -p data/raw data/processed models artifacts

# Expose FastAPI port
EXPOSE 8000

# Default command: run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
