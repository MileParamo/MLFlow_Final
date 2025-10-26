import joblib
import pandas as pd
import json
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import sys
import os

# --- Umbrales de Validación ---
ACCURACY_THRESHOLD = 0.95  # Mínimo 85% de accuracy
F1_THRESHOLD = 0.95  # Mínimo 85% de F1-score

print("=" * 60)
print("VALIDATION STEP - Wine Classification Model")
print("=" * 60)

# --- Cargar el MISMO dataset que en train.py ---
print("\n--- Loading Wine dataset ---")
wine_data = load_wine()
X, y = wine_data.data, wine_data.target

# División de datos (misma semilla que en train.py para consistencia)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"--- Debug: X_test dimensions: {X_test.shape} ---")
print(f"--- Debug: Number of test samples: {len(y_test)} ---")

# --- Cargar modelo previamente entrenado ---
model_filename = "model.pkl"
model_path = os.path.abspath(os.path.join(os.getcwd(), model_filename))
print(f"\n--- Debug: Loading model from: {model_path} ---")

try:
    model = joblib.load(model_path)
    print(f"✅ Model loaded successfully")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Expected features: {model.n_features_in_}")
except FileNotFoundError:
    print(f"❌ ERROR: Model file not found at '{model_path}'")
    print(f"   Make sure the 'train' step saved it correctly in the project root.")
    print(f"\n--- Debug: Files in {os.getcwd()}: ---")
    try:
        files = os.listdir(os.getcwd())
        for f in files:
            print(f"   - {f}")
    except Exception as list_err:
        print(f"   (Could not list directory: {list_err})")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    sys.exit(1)

# --- Cargar métricas guardadas (opcional, para comparación) ---
metrics_filename = "metrics.json"
metrics_path = os.path.join(os.getcwd(), metrics_filename)
saved_metrics = None

if os.path.exists(metrics_path):
    try:
        with open(metrics_path, 'r') as f:
            saved_metrics = json.load(f)
        print(f"\n--- Metrics from training: ---")
        print(f"   Training Accuracy: {saved_metrics.get('accuracy', 'N/A'):.4f}")
        print(f"   Training F1-Score: {saved_metrics.get('f1_score', 'N/A'):.4f}")
    except Exception as e:
        print(f"⚠️  Warning: Could not load metrics.json: {e}")

# --- Predicción y Validación ---
print(f"\n--- Performing predictions on test set ---")
try:
    y_pred = model.predict(X_test)
    print(f"✅ Predictions completed successfully")
except ValueError as pred_err:
    print(f"❌ ERROR during prediction: {pred_err}")
    print(f"   Model expected {model.n_features_in_} features")
    print(f"   X_test has {X_test.shape[1]} features")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error during prediction: {e}")
    sys.exit(1)

# --- Calcular métricas en el conjunto de prueba ---
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n{'=' * 60}")
print(f"VALIDATION RESULTS")
print(f"{'=' * 60}")
print(f"🔍 Model Accuracy:  {accuracy:.4f} (threshold: {ACCURACY_THRESHOLD})")
print(f"🔍 Model F1-Score:  {f1:.4f} (threshold: {F1_THRESHOLD})")
print(f"{'=' * 60}")

# --- Validación contra umbrales ---
passed_accuracy = accuracy >= ACCURACY_THRESHOLD
passed_f1 = f1 >= F1_THRESHOLD

print(f"\n--- Quality Check Results ---")
print(f"   Accuracy Check: {'✅ PASSED' if passed_accuracy else '❌ FAILED'}")
print(f"   F1-Score Check: {'✅ PASSED' if passed_f1 else '❌ FAILED'}")

# --- Decisión final ---
if passed_accuracy and passed_f1:
    print(f"\n{'=' * 60}")
    print(f"✅ SUCCESS: Model meets all quality criteria!")
    print(f"{'=' * 60}")
    sys.exit(0)
else:
    print(f"\n{'=' * 60}")
    print(f"❌ FAILURE: Model does not meet quality thresholds.")
    print(f"   Pipeline stopped.")
    print(f"{'=' * 60}")
    
    # Detalles adicionales del fallo
    if not passed_accuracy:
        print(f"\n   Accuracy {accuracy:.4f} is below threshold {ACCURACY_THRESHOLD}")
    if not passed_f1:
        print(f"   F1-Score {f1:.4f} is below threshold {F1_THRESHOLD}")
    
    sys.exit(1)