
import pandas as pd

DEFAULT_FEATURES = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc",
    "health", "absences", "G1"
]

TARGET_COLUMN = "G3"
PASSING_THRESHOLD = 10


def prepare_features_and_target(df, features=None):
    """Create an early-risk label from final grade G3."""
    # Normalize column names and remove accidental spaces.
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Dataset must contain '{TARGET_COLUMN}'. "
            f"Found columns: {df.columns.tolist()}"
        )

    chosen = features or [
        col for col in DEFAULT_FEATURES if col in df.columns
    ]

    if not chosen:
        raise ValueError("No expected feature columns were found.")

    missing = [col for col in chosen if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[chosen].copy()
    y = (pd.to_numeric(df[TARGET_COLUMN], errors="coerce") < PASSING_THRESHOLD).astype(int)

    # Fill missing values according to each column's data type.
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            median_value = X[col].median()
            X[col] = X[col].fillna(
                median_value if pd.notna(median_value) else 0
            )
        else:
            X[col] = X[col].fillna("Unknown").astype(str)

    return X, y, chosen
