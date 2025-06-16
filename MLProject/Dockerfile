FROM python:3.10-slim

WORKDIR /app

# Copy kode dan model
COPY trained-model/ ./trained-model

# Install dependencies
RUN pip install mlflow scikit-learn pandas

# Serve model
CMD ["mlflow", "models", "serve", "-m", "trained-model/", "-p", "5000", "--no-conda"]
