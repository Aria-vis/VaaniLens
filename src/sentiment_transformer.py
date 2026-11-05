"""Transformer sentiment classifier: DistilBERT SST-2 (owner: Swapnil).

Uses pretrained distilbert-base-uncased-finetuned-sst-2-english.
Evaluates on data/en_test.csv.
Saves:
- results/sentiment_transformer_metrics.json
- results/confusion_transformer.png
"""

# Import operating system interface for inspecting runtime environment flags
import os
# JSON module for saving structured evaluation benchmark results
import json
# Pathlib for resolving repository directories
from pathlib import Path

# Enforce offline execution mode when VAANI_OFFLINE=1 is present in the environment
if os.environ.get("VAANI_OFFLINE") == "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Matplotlib for generating scientific confusion matrix visualization plots
import matplotlib.pyplot as plt
# NumPy for matrix manipulations
import numpy as np
# Pandas for loading CSV test split
import pandas as pd
# Scikit-learn evaluation metrics for classification benchmarking
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

# Resolve repository paths relative to current script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face identifier for pre-trained DistilBERT model fine-tuned on Stanford Sentiment Treebank
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
# Global singleton instance for Hugging Face pipeline
_pipeline = None


def get_transformer_pipeline():
    """Initializes and caches the Hugging Face sentiment-analysis pipeline on CPU."""
    global _pipeline
    # Initialize pipeline only on first request to preserve memory
    if _pipeline is None:
        from transformers import pipeline
        # Build sentiment-analysis pipeline explicitly bound to CPU (device=-1)
        _pipeline = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            device=-1,           # Explicitly target host CPU without requiring CUDA
            truncation=True,     # Truncate sequences that exceed maximum length
            max_length=256       # Match the project's global token sequence cap
        )
        
    # Return initialized singleton pipeline
    return _pipeline


def evaluate_transformer():
    """Evaluates DistilBERT on 1,000 held-out test reviews and generates traceable metrics and confusion matrix."""
    # Define target path for held-out test split
    test_path = DATA_DIR / "en_test.csv"
    # Ensure test dataset exists, streaming from Hugging Face if missing
    if not test_path.exists():
        from src.data_prep import main as prep_data
        prep_data()

    print("Loading test data for DistilBERT evaluation...")
    # Load test split into DataFrame
    df_test = pd.read_csv(test_path)
    # Extract text strings and true integer sentiment labels (0 or 1)
    texts = df_test["text"].fillna("").tolist()
    y_true = df_test["label"].astype(int).tolist()

    # Retrieve cached DistilBERT pipeline
    clf = get_transformer_pipeline()
    print(f"Evaluating {len(texts)} samples with DistilBERT on CPU (batch_size=32)...")

    predictions = []
    # Process inputs in mini-batches of 32 to maximize CPU vectorized throughput
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Execute transformer forward pass and classification head
        preds = clf(batch)
        predictions.extend(preds)

    # Convert SST-2 string label conventions ('POSITIVE'/'NEGATIVE') to binary integers (1/0)
    y_pred = [1 if p["label"].upper() == "POSITIVE" else 0 for p in predictions]

    # Calculate overall classification accuracy on held-out test split
    acc = float(accuracy_score(y_true, y_pred))
    # Compute macro-averaged Precision, Recall, and F1 score across both sentiment classes
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    # Generate 2x2 confusion matrix array [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Compile structured evaluation dictionary matching project schema
    metrics = {
        "model": "DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)",
        "test_samples": len(texts),
        "accuracy": round(acc, 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm,
    }

    # Save quantitative metrics to JSON file for Streamlit UI dynamic loading
    metrics_path = RESULTS_DIR / "sentiment_transformer_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"DistilBERT Metrics saved to {metrics_path}: Acc={acc:.4f}, F1={f1:.4f}")

    # Generate Confusion Matrix display plot
    cm_display = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=["Negative", "Positive"])
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_display.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("DistilBERT SST-2 Confusion Matrix")
    plt.tight_layout()
    
    # Save confusion matrix plot image for the Streamlit dashboard
    cm_plot_path = RESULTS_DIR / "confusion_transformer.png"
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_plot_path}")

    # Return dictionary of computed evaluation metrics
    return metrics


# Execute evaluation if script is run directly from command line
if __name__ == "__main__":
    evaluate_transformer()
