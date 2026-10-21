from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.pkl"

def get_feature_importance(top_n=15):
    """Return Random Forest feature importance after preprocessing."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first: python -m src.train_model")
    saved = joblib.load(MODEL_PATH)
    pipeline = saved["pipeline"]
    prep = pipeline.named_steps["preprocessing"]
    clf = pipeline.named_steps["classifier"]
    names = prep.get_feature_names_out()
    values = clf.feature_importances_
    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
