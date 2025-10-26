import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
from mlflow.models import infer_signature
import sys
import traceback
import joblib
import json

print(f"--- Debug: Initial CWD: {os.getcwd()} ---")

# --- Define Paths ---
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = "file://" + os.path.abspath(mlruns_dir)
artifact_location = "file://" + os.path.abspath(mlruns_dir)

print(f"--- Debug: Workspace Dir: {workspace_dir} ---")
print(f"--- Debug: MLRuns Dir: {mlruns_dir} ---")
print(f"--- Debug: Tracking URI: {tracking_uri} ---")
print(f"--- Debug: Artifact Location Base: {artifact_location} ---")

# --- Ensure MLRuns directory exists ---
os.makedirs(mlruns_dir, exist_ok=True)

# --- Configure MLflow ---
mlflow.set_tracking_uri(tracking_uri)

# --- Create or Set Experiment ---
experiment_name = "Wine-Classification-Pipeline"
experiment_id = None

try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=artifact_location
    )
    print(f"--- Debug: Created Experiment '{experiment_name}' with ID: {experiment_id} ---")
except mlflow.exceptions.MlflowException as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        print(f"--- Debug: Experiment '{experiment_name}' already exists. Getting ID. ---")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"--- Debug: Existing Experiment ID: {experiment_id} ---")
            print(f"--- Debug: Existing Artifact Location: {experiment.artifact_location} ---")
        else:
            print(f"--- ERROR: Could not get experiment '{experiment_name}' by name. ---")
            sys.exit(1)
    else:
        print(f"--- ERROR creating/getting experiment: {e} ---")
        raise e

if experiment_id is None:
    print(f"--- FATAL ERROR: Could not obtain valid experiment ID for '{experiment_name}'. ---")
    sys.exit(1)

# --- Load Data and Train Model ---
print("--- Loading Wine dataset ---")
wine_data = load_wine()
X, y = wine_data.data, wine_data.target
feature_names = wine_data.feature_names
target_names = wine_data.target_names

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(target_names)}")
print(f"Classes: {target_names}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Hyperparameters ---
n_estimators = 100
max_depth = 10
random_state = 42

print(f"--- Training Random Forest with n_estimators={n_estimators}, max_depth={max_depth} ---")
model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=random_state
)
model.fit(X_train, y_train)

# --- Predictions and Metrics ---
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n--- Model Performance ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# --- Feature Importance ---
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n--- Top 5 Most Important Features ---")
print(feature_importance.head())

# --- Start MLflow Run ---
print(f"\n--- Debug: Starting MLflow run in Experiment ID: {experiment_id} ---")
run = None

try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        actual_artifact_uri = run.info.artifact_uri
        print(f"--- Debug: Run ID: {run_id} ---")
        print(f"--- Debug: Artifact URI: {actual_artifact_uri} ---")

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", 0.2)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log confusion matrix as artifact
        conf_matrix_path = os.path.join(workspace_dir, "confusion_matrix.txt")
        np.savetxt(conf_matrix_path, conf_matrix, fmt='%d')
        mlflow.log_artifact(conf_matrix_path)

        # Log feature importance as artifact
        feature_importance_path = os.path.join(workspace_dir, "feature_importance.csv")
        feature_importance.to_csv(feature_importance_path, index=False)
        mlflow.log_artifact(feature_importance_path)

        # Infer signature
        signature = infer_signature(X_train, y_pred)

        # Log model
        print(f"--- Debug: Logging model with artifact_path='model' ---")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:5]
        )

        print(f"\n✅ Model registered successfully in MLflow!")
        print(f"✅ Run ID: {run_id}")
        print(f"✅ Accuracy: {accuracy:.4f}")
        print(f"✅ F1-Score: {f1:.4f}")

        # Save model locally for validation step
        model_path = os.path.join(workspace_dir, "model.pkl")
        joblib.dump(model, model_path)
        print(f"✅ Model saved locally at: {model_path}")

        # Save metrics for validation
        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        metrics_path = os.path.join(workspace_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"✅ Metrics saved at: {metrics_path}")

except Exception as e:
    print(f"\n--- ERROR during MLflow execution ---")
    traceback.print_exc()
    print(f"--- End of Error Trace ---")
    print(f"Current CWD on error: {os.getcwd()}")
    print(f"Tracking URI used: {mlflow.get_tracking_uri()}")
    print(f"Experiment ID attempted: {experiment_id}")
    if run:
        print(f"Run Artifact URI on error: {run.info.artifact_uri}")
    else:
        print("Run object was not created successfully.")
    sys.exit(1)

print("\n--- Training pipeline completed successfully! ---")