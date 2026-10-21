from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "student_data.csv"

def download_student_data(save=True):
    """Download the UCI Student Performance dataset using ucimlrepo."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise ImportError(
            "ucimlrepo is missing. Run: pip install -r requirements.txt"
        ) from exc

    dataset = fetch_ucirepo(id=320)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    df = pd.concat([features, targets], axis=1)

    if save:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_FILE, index=False)
    return df


def load_student_data(path=None):
    """Load the student dataset from the CSV file."""
    path = Path(path) if path else RAW_FILE

    if path.exists():
        df = pd.read_csv(path, sep=None, engine="python")
        return df

    return download_student_data(save=True)


