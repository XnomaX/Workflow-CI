import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import argparse
import zipfile
import os

# Parsing argumen
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to CSV or ZIP dataset")
args = parser.parse_args()

# Aktifkan tracking
mlflow.set_experiment("msml-basic")
mlflow.sklearn.autolog()
mlflow.set_tracking_uri("file:./mlruns")

# Deteksi apakah input ZIP atau CSV
if args.data_path.endswith(".zip"):
    with zipfile.ZipFile(args.data_path, 'r') as zip_ref:
        zip_ref.extractall()
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    if not csv_files:
        raise ValueError("Tidak ditemukan file CSV di dalam ZIP.")
    data_file = csv_files[0]
else:
    data_file = args.data_path

# Load dataset
df = pd.read_csv(data_file)

# Pisahkan fitur dan target
X = df.drop(columns=['survived'])
y = df['survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Akurasi: {acc}")
mlflow.log_metric("accuracy", acc)
mlflow.sklearn.log_model(model, "model", input_example=X_test.iloc[:5])
