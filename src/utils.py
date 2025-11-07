from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix

# ──────────────────────────────
# Paths (LOCAL VERSION)
# ──────────────────────────────
# Assume your project root looks like:
# ├── src/
# ├── data/
# ├── models/
# ├── results/

ROOT        = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"
RESULTS_DIR = ROOT / "results"
META_PATH   = MODELS_DIR / "metadata.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────
# Data loading
# ──────────────────────────────
def _require_cols(df: pd.DataFrame, cols: list[str], file_hint: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{file_hint} must contain columns {cols}. "
            f"Found: {list(df.columns)}; Missing: {missing}"
        )

def load_datasets() -> pd.DataFrame:
    """
    Read standardized CSVs and attach a 'domain' column.
    Each CSV must have: Review, Label in {'genuine','fake'}.
    """
    files = [
        (DATA_DIR / "apps_fake_genuine_data.csv", "app"),
        (DATA_DIR / "deceptive-opinion.csv",      "hotel"),
        (DATA_DIR / "Labelled Yelp Dataset.csv",  "yelp"),
    ]

    dfs = []
    for path, dom in files:
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")
        df = pd.read_csv(path)

        # Normalize headers
        df.columns = df.columns.str.strip().str.lower()
        _require_cols(df, ["review", "label"], file_hint=path.name)
        df = df.rename(columns={"review": "Review", "label": "Label"})

        # Attach domain
        df["domain"] = dom
        dfs.append(df[["Review", "Label", "domain"]])

    out = pd.concat(dfs, ignore_index=True)
    out = out.dropna(subset=["Review", "Label"]).reset_index(drop=True)
    return out

# ──────────────────────────────
# Leave-One-Domain-Out split
# ──────────────────────────────
def leave_one_domain_out(df: pd.DataFrame, holdout_domain: str):
    train_df = df[df["domain"].astype(str) != holdout_domain].reset_index(drop=True)
    test_df  = df[df["domain"].astype(str) == holdout_domain].reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise ValueError(f"Empty split for domain='{holdout_domain}'. Check your 'domain' values.")
    return train_df, test_df

# ──────────────────────────────
# Evaluation helpers
# ──────────────────────────────
def cross_validate_model(model, X, y, cv_splits: int = 5):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    f1w = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")
    print(
        f" Cross-Validation ({cv_splits}-fold): "
        f"Accuracy {acc.mean():.4f}±{acc.std():.4f} | "
        f"F1_w {f1w.mean():.4f}±{f1w.std():.4f}"
    )
    return {"cv_accuracy": float(acc.mean()), "cv_f1_weighted": float(f1w.mean())}

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=labels or ["Fake", "Genuine"],
        yticklabels=labels or ["Fake", "Genuine"],
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    plt.tight_layout()
    plt.show()

def save_eval_results(df: pd.DataFrame, path: str = "results/cross_domain_eval.csv"):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f" Evaluation results saved to: {out}")

def print_table_summary(df: pd.DataFrame):
    print("\n Leave-One-Domain-Out Summary:")
    print(df.to_string(index=False))
    print("\n Averages (numeric columns):")
    print(df.mean(numeric_only=True).round(3))

# ──────────────────────────────
# Metadata for Streamlit picker
# ──────────────────────────────
def read_metadata() -> dict:
    if META_PATH.exists():
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"label_map": {"genuine": 0, "fake": 1}, "models": [], "thresholds": {}}

def write_metadata(best_model: str | None = None,
                   add_model: str | None = None,
                   thresholds: dict | None = None):
    meta = read_metadata()
    if add_model and add_model not in meta.get("models", []):
        meta.setdefault("models", []).append(add_model)
    if thresholds:
        meta.setdefault("thresholds", {}).update(thresholds)
    if best_model:
        meta["best_by_auc"] = best_model
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
