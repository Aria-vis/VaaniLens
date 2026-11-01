"""Transformer sentiment classifier: DistilBERT SST-2 (owner: Swapnil).

Uses pretrained distilbert-base-uncased-finetuned-sst-2-english.
Evaluates on data/en_test.csv.
Saves:
- results/sentiment_transformer_metrics.json
- results/confusion_transformer.png
"""

import os
import json
from pathlib import Path

# Ensure offline execution with cached models
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
_pipeline = None


def get_transformer_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            device=-1,  # CPU
            truncation=True,
            max_length=256
        )
    return _pipeline


def evaluate_transformer():
    test_path = DATA_DIR / "en_test.csv"
    if not test_path.exists():
        from src.data_prep import main as prep_data
        prep_data()

    print("Loading test data for DistilBERT evaluation...")
    df_test = pd.read_csv(test_path)
    texts = df_test["text"].fillna("").tolist()
    y_true = df_test["label"].astype(int).tolist()

    clf = get_transformer_pipeline()
    print(f"Evaluating {len(texts)} samples with DistilBERT on CPU (batch_size=32)...")

    predictions = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        preds = clf(batch)
        predictions.extend(preds)

    # SST-2 labels: 'POSITIVE' -> 1, 'NEGATIVE' -> 0
    y_pred = [1 if p["label"].upper() == "POSITIVE" else 0 for p in predictions]

    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "model": "DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)",
        "test_samples": len(texts),
        "accuracy": round(acc, 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm,
    }

    metrics_path = RESULTS_DIR / "sentiment_transformer_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"DistilBERT Metrics saved to {metrics_path}: Acc={acc:.4f}, F1={f1:.4f}")

    # Plot Confusion Matrix
    cm_display = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=["Negative", "Positive"])
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_display.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("DistilBERT SST-2 Confusion Matrix")
    plt.tight_layout()
    cm_plot_path = RESULTS_DIR / "confusion_transformer.png"
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_plot_path}")

    return metrics


if __name__ == "__main__":
    evaluate_transformer()
