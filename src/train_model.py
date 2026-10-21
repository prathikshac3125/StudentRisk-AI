from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

from src.load_data import load_student_data
from src.preprocessing import prepare_features_and_target

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

def train():
    df = load_student_data()
    X, y, feature_names = prepare_features_and_target(df)

    if y.nunique() < 2:
        raise ValueError("The dataset needs examples from both risk classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])

    pipeline = Pipeline([
        ("preprocessing", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision_at_risk": float(precision_score(y_test, preds, pos_label=1, zero_division=0)),
        "recall_at_risk": float(recall_score(y_test, preds, pos_label=1, zero_division=0)),
        "f1_at_risk": float(f1_score(y_test, preds, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, preds, labels=[0, 1],
            target_names=["Not identified as at risk", "At risk"],
            output_dict=True, zero_division=0
        ),
        "features": feature_names,
        "target_definition": "At risk = final grade G3 below 10/20"
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "features": feature_names,
        "target_definition": metrics["target_definition"]
    }, MODEL_DIR / "best_model.pkl")
    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training completed.")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"At-risk precision: {metrics['precision_at_risk']:.3f}")
    print(f"At-risk recall: {metrics['recall_at_risk']:.3f}")
    print(f"At-risk F1: {metrics['f1_at_risk']:.3f}")
    print("Confusion matrix (rows=true, columns=predicted; classes 0,1):")
    print(metrics["confusion_matrix"])
    print(f"Saved model to: {MODEL_DIR / 'best_model.pkl'}")
    print(f"Saved metrics to: {REPORT_DIR / 'metrics.json'}")
    return metrics

if __name__ == "__main__":
    train()
