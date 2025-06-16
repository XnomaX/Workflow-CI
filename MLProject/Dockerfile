FROM python:3.12.1-slim

WORKDIR /app

# Copy kode dan model
COPY trained_model/ ./trained-model

# Install dependencies
RUN pip install mlflow scikit-learn pandas scipy psutil

# Serve model
CMD ["mlflow", "models", "serve", "-m", "trained-model/", "-p", "5000", "--no-conda"]
