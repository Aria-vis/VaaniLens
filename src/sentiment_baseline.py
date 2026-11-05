"""Baseline sentiment classifier: TF-IDF + Logistic Regression (owner: Swapnil).

Trains on data/en_train.csv and evaluates on data/en_test.csv.
Saves:
- models/baseline.joblib (< 25 MB)
- results/sentiment_baseline_metrics.json
- results/confusion_baseline.png
"""

# JSON module for serializing performance evaluation metrics to disk
import json
# Pathlib for platform-independent filesystem directory traversal
from pathlib import Path
# Joblib for fast compression and serialization of Scikit-Learn Pipeline objects
import joblib
# Matplotlib for generating scientific confusion matrix visualization plots
import matplotlib.pyplot as plt
# NumPy for multi-dimensional array operations on confusion matrices
import numpy as np
# Pandas for loading and filtering CSV training and test splits
import pandas as pd
# TF-IDF feature extraction module converting text into sparse n-gram frequency matrices
from sklearn.feature_extraction.text import TfidfVectorizer
# Logistic Regression linear classification model with L2 regularization
from sklearn.linear_model import LogisticRegression
# Quantitative evaluation metrics for classification benchmarks
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
# Scikit-Learn Pipeline chaining preprocessing and estimator stages into a single callable
from sklearn.pipeline import Pipeline

# Resolve base directories relative to repository root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure output directories exist before writing models and evaluation charts
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_baseline():
    """Trains TF-IDF + Logistic Regression baseline on 5,000 reviews and evaluates on 1,000 test reviews."""
    # Define paths to training (5,000 samples) and test (1,000 samples) datasets
    train_path = DATA_DIR / "en_train.csv"
    test_path = DATA_DIR / "en_test.csv"

    # Automatically trigger data streaming and curation if datasets are not present locally
    if not train_path.exists() or not test_path.exists():
        from src.data_prep import main as prep_data
        prep_data()

    print("Loading datasets for baseline...")
    # Read training and test CSV splits into pandas DataFrames
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Extract review text columns, imputing missing or null values with empty strings
    X_train = df_train["text"].fillna("")
    y_train = df_train["label"].astype(int)
    X_test = df_test["text"].fillna("")
    y_test = df_test["label"].astype(int)

    print(f"Training TF-IDF + LogisticRegression on {len(X_train)} samples...")
    # Construct linear pipeline chaining n-gram TF-IDF vectorization with Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),    # Capture single words and adjacent word pairs (bigrams)
            min_df=2,              # Ignore idiosyncratic hapax legomena appearing only once
            max_features=30000,    # Cap vocabulary dimension to 30,000 features to maintain tiny memory footprint
            sublinear_tf=True      # Apply sublinear scaling 1 + log(tf) to dampen excessive frequency dominance
        )),
        ("clf", LogisticRegression(
            max_iter=1000,         # Sufficient iterations to ensure full gradient convergence
            random_state=42        # Deterministic random seed for reproducibility
        ))
    ])

    # Fit pipeline on 5,000 training reviews (word frequencies -> L2 regularized weights)
    pipeline.fit(X_train, y_train)

    print("Evaluating baseline...")
    # Predict discrete binary sentiment classes on held-out test split
    y_pred = pipeline.predict(X_test)
    # Predict continuous posterior class probabilities for calibration analysis
    y_proba = pipeline.predict_proba(X_test)

    # Compute overall classification accuracy
    acc = float(accuracy_score(y_test, y_pred))
    # Compute macro-averaged Precision, Recall, and F1 across both classes
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    # Generate 2x2 confusion matrix array [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Compile structured metrics dictionary for reporting
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

    # Persist JSON evaluation report to results directory
    metrics_path = RESULTS_DIR / "sentiment_baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}: Acc={acc:.4f}, F1={f1:.4f}")

    # Generate visual confusion matrix plot
    cm_display = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=["Negative", "Positive"])
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_display.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Baseline Sentiment Confusion Matrix")
    plt.tight_layout()
    
    # Save confusion matrix figure to results directory
    cm_plot_path = RESULTS_DIR / "confusion_baseline.png"
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_plot_path}")

    # Serialize trained pipeline to models directory using joblib compression
    model_path = MODELS_DIR / "baseline.joblib"
    joblib.dump(pipeline, model_path)
    # Verify file size complies with project constraints (< 25 MB)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Saved model to {model_path} ({model_size_mb:.2f} MB)")
    assert model_size_mb < 25, "Model exceeds 25 MB limit!"
    
    # Return trained pipeline and computed evaluation metrics
    return pipeline, metrics


# Run training and evaluation pipeline if script is executed directly
if __name__ == "__main__":
    train_baseline()

