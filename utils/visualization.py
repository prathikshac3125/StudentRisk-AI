import matplotlib.pyplot as plt

def plot_feature_importance(importance_df):
    fig, ax = plt.subplots(figsize=(9, 5))
    view = importance_df.sort_values("importance", ascending=True)
    ax.barh(view["feature"], view["importance"])
    ax.set_title("Most influential model features")
    ax.set_xlabel("Feature importance")
    fig.tight_layout()
    return fig
