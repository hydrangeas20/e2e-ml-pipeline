import os
import pickle
import yaml
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("e2e-ml-pipeline")

import mlflow
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]


def evaluate(model_path: str, test_split_path: str = "data/splits/test.csv") -> None:
    # Configure MLflow from environment 
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if not os.path.exists(test_split_path):
        raise FileNotFoundError(
            f"Test split not found at {test_split_path}. Run train.py first to create it."
        )

    test_df = pd.read_csv(test_split_path)
    if "Outcome" not in test_df.columns:
        raise ValueError("Test split missing 'Outcome' column.")

    X_test = test_df.drop(columns=["Outcome"])
    y_test = test_df["Outcome"]

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec_1 = precision_score(y_test, y_pred, pos_label=1)
    rec_1 = recall_score(y_test, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1:       {f1:.4f}")
    print(f"Class-1 Precision: {prec_1:.4f}")
    print(f"Class-1 Recall:    {rec_1:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")

    # Log to MLflow
    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision_class_1", prec_1)
        mlflow.log_metric("recall_class_1", rec_1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")
        mlflow.log_artifact(test_split_path)


if __name__ == "__main__":
    evaluate(
        model_path=params["model"],
        test_split_path="data/splits/test.csv",
    )
