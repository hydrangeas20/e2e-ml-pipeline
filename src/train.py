import os
import pickle
import yaml
import pandas as pd

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv


load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("e2e-ml-pipeline")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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
from mlflow.models import infer_signature


def hyperparameter_tuning(X_train, y_train, param_grid, random_state: int):
    rf = RandomForestClassifier(
        random_state=random_state,
        class_weight="balanced",  # improves minority-class handling
        n_jobs=-1,
    )
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1,
        scoring="f1",  # optimize for class balance rather than raw accuracy
    )
    grid_search.fit(X_train, y_train)
    return grid_search


# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]


def train(
    data_path: str,
    model_path: str,
    random_state: int,
    n_estimators=None,
    max_depth=None,
) -> None:
    # Configure MLflow from environment (DON'T hardcode tokens in code! Anyone can see it!)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    data = pd.read_csv(data_path)
    if "Outcome" not in data.columns:
        raise ValueError(
            "Column 'Outcome' not found. Check preprocess step (should keep headers)."
        )

    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    # Reproducible + stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )

    # The test split is saved so evaluate.py can use the same holdout set
    os.makedirs("data/splits", exist_ok=True)
    test_df = X_test.copy()
    test_df["Outcome"] = y_test.values
    test_df.to_csv("data/splits/test.csv", index=False)

    # Hyperparameter grid 
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    with mlflow.start_run():
        # Hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid, random_state)
        best_model = grid_search.best_estimator_

        # Predictions + probabilities
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec_1 = precision_score(y_test, y_pred, pos_label=1)
        rec_1 = recall_score(y_test, y_pred, pos_label=1)

        # AUC metrics
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        print(f"Accuracy: {acc:.4f}")
        print(f"F1:       {f1:.4f}")
        print(f"Class-1 Precision: {prec_1:.4f}")
        print(f"Class-1 Recall:    {rec_1:.4f}")
        print(f"ROC-AUC:  {roc_auc:.4f}")
        print(f"PR-AUC:   {pr_auc:.4f}")

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision_class_1", prec_1)
        mlflow.log_metric("recall_class_1", rec_1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)

        # Log best params
        for k, v in grid_search.best_params_.items():
            mlflow.log_param(f"best_{k}", v)

        # Confusion matrix + report
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")

        # Log test split 
        mlflow.log_artifact("data/splits/test.csv")

        # Model signature (input -> output)
        signature = infer_signature(X_train, best_model.predict(X_train))

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            registered_model_name="BestModel",
        )

        # Model is locally for DVC tracking / FastAPI serving
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train(
        data_path=params["data"],
        model_path=params["model"],
        random_state=params.get("random_state", 42),
        n_estimators=params.get("n_estimators"),
        max_depth=params.get("max_depth"),
    )




