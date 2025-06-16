import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import shutil
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import zipfile

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# MLflow setup
mlflow.set_experiment("msml-basic")
mlflow.sklearn.autolog()

# Extract ZIP or read CSV
if args.data_path.endswith(".zip"):
    with zipfile.ZipFile(args.data_path, 'r') as zip_ref:
        zip_ref.extractall()
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    if not csv_files:
        raise ValueError("Tidak ditemukan file CSV di dalam ZIP.")
    data_file = csv_files[0]
else:
    data_file = args.data_path

# Load data
df = pd.read_csv(data_file)
X = df.drop(columns=["survived"])
y = df["survived"]

# Ambil 1000 data untuk training
X_train, _, y_train, _ = train_test_split(
    X, y, train_size=1000, stratify=y, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Model ringan
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
mlflow.log_metric("accuracy", acc)
# Log model ke MLflow (dan simpan manual juga)
mlflow.sklearn.log_model(model, "model", input_example=X_test.iloc[:5])
