import pandas as pd

def summarize_grade_progression(df):
    """Summarize G1, G2 and G3 trends when those columns exist.
    G2/G3 are for retrospective analysis only, not model input.
    """
    grade_cols = [c for c in ["G1", "G2", "G3"] if c in df.columns]
    if not grade_cols:
        return pd.DataFrame()
    return df[grade_cols].describe().T
