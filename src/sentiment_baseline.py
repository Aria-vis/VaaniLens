"""Baseline sentiment classifier: TF-IDF + Logistic Regression (owner: Swapnil).

Trains on data/en_train.csv and evaluates on data/en_test.csv.
Saves:
- models/baseline.joblib (< 25 MB)
- results/sentiment_baseline_metrics.json
- results/confusion_baseline.png
"""

import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_baseline():
    """Trains TF-IDF + Logistic Regression baseline on 5,000 reviews and evaluates on 1,000 test reviews."""
    train_path = DATA_DIR / "en_train.csv"
    test_path = DATA_DIR / "en_test.csv"

    if not train_path.exists() or not test_path.exists():
        from src.data_prep import main as prep_data
        prep_data()

    print("Loading datasets for baseline...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train["text"].fillna("")
    y_train = df_train["label"].astype(int)
    X_test = df_test["text"].fillna("")
    y_test = df_test["label"].astype(int)

    print(f"Training TF-IDF + LogisticRegression on {len(X_train)} samples...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    print("Evaluating baseline...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": round(acc, 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm,
    }

    metrics_path = RESULTS_DIR / "sentiment_baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}: Acc={acc:.4f}, F1={f1:.4f}")

    cm_display = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=["Negative", "Positive"])
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_display.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Baseline Sentiment Confusion Matrix")
    plt.tight_layout()
    
    cm_plot_path = RESULTS_DIR / "confusion_baseline.png"
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_plot_path}")

    model_path = MODELS_DIR / "baseline.joblib"
    joblib.dump(pipeline, model_path)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Saved model to {model_path} ({model_size_mb:.2f} MB)")
    assert model_size_mb < 25, "Model exceeds 25 MB limit!"
    
    return pipeline, metrics


if __name__ == "__main__":
    train_baseline()

