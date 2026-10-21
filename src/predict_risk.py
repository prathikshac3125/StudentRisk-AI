from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.pkl"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. First run: python -m src.train_model"
        )
    saved = joblib.load(MODEL_PATH)
    return saved["pipeline"], saved["features"]

def predict_student(student_data):
    """student_data can be a dict or a one-row DataFrame."""
    pipeline, features = load_model()
    row = pd.DataFrame([student_data]) if isinstance(student_data, dict) else student_data.copy()
    for feature in features:
        if feature not in row.columns:
            row[feature] = None
    row = row[features]
    pred = int(pipeline.predict(row)[0])
    probability = None
    if hasattr(pipeline, "predict_proba"):
        probability = float(pipeline.predict_proba(row)[0][1])
    return {
        "prediction": pred,
        "risk_label": "At risk" if pred == 1 else "Not identified as at risk",
        "risk_probability": probability
    }

if __name__ == "__main__":
    print("Use this module from Python or run the Streamlit app after training.")
