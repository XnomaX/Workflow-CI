FROM python:3.12.1-slim

WORKDIR /app

# Copy kode dan model
COPY trained_model/ ./trained-model

# Install dependencies
RUN pip install \
    mlflow==2.19.0 \
    scikit-learn==1.4.2 \
    numpy==1.26.4 \
    psutil==7.0.0 \
    scipy==1.15.2 \
    pandas
# Serve model
CMD ["mlflow", "models", "serve", "-m", "trained-model/", "-p", "5000", "--no-conda"]
