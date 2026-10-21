from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "reports" / "metrics.json"

def load_metrics():
    if not METRICS_PATH.exists():
        raise FileNotFoundError("Metrics not found. Run: python -m src.train_model")
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)
