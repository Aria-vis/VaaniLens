"""Fairness evaluation: English vs Translated Hindi (owner: Swapnil).

Translates Hindi test reviews using MarianMT, performs sentiment analysis,
and compares performance metrics (Accuracy, Macro-F1) against the English test set.
Computes the F1 fairness gap with a 1,000-sample bootstrap 95% confidence interval.

Caveat: English and Hindi sets originate from different corpora/distributions,
so the observed performance gap is a composite of domain shift and translation degradation.

Outputs:
- results/fairness_en_vs_hi.json
- results/fairness_en_vs_hi.png
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_bootstrap_ci(y_true_en, y_pred_en, y_true_hi, y_pred_hi, n_bootstraps=1000, seed=42):
    """Computes 95% bootstrap confidence interval for (F1_en - F1_hi)."""
    np.random.seed(seed)
    gaps = []
    n_en = len(y_true_en)
    n_hi = len(y_true_hi)

    for _ in range(n_bootstraps):
        idx_en = np.random.choice(n_en, size=n_en, replace=True)
        idx_hi = np.random.choice(n_hi, size=n_hi, replace=True)

        _, _, f1_en_b, _ = precision_recall_fscore_support(
            y_true_en[idx_en], y_pred_en[idx_en], average="macro", zero_division=0
        )
        _, _, f1_hi_b, _ = precision_recall_fscore_support(
            y_true_hi[idx_hi], y_pred_hi[idx_hi], average="macro", zero_division=0
        )
        gaps.append(f1_en_b - f1_hi_b)

    ci_lower = float(np.percentile(gaps, 2.5))
    ci_upper = float(np.percentile(gaps, 97.5))
    return ci_lower, ci_upper, gaps


def evaluate_fairness(model_name: str = "transformer"):
    from src.translate import translate_hi_to_en
    from src.sentiment import predict_sentiment, predict_proba

    en_test_path = DATA_DIR / "en_test.csv"
    hi_test_path = DATA_DIR / "hi_reviews_test.csv"

    if not en_test_path.exists() or not hi_test_path.exists():
        from src.data_prep import main as prep_data
        prep_data()

    print("Loading test datasets for fairness evaluation...")
    df_en = pd.read_csv(en_test_path)
    df_hi = pd.read_csv(hi_test_path)

    # 1. Evaluate English
    print(f"Evaluating English test set ({len(df_en)} samples)...")
    y_true_en = df_en["label"].astype(int).values
    en_texts = df_en["text"].fillna("").tolist()
    
    # Process in batches of 32 for fast CPU inference
    batch_size = 32
    y_pred_en_list = []
    for i in range(0, len(en_texts), batch_size):
        batch = en_texts[i : i + batch_size]
        probs = predict_proba(batch, model=model_name)
        preds = np.argmax(probs, axis=1)
        y_pred_en_list.extend(preds)
    y_pred_en = np.array(y_pred_en_list)

    acc_en = float(accuracy_score(y_true_en, y_pred_en))
    p_en, r_en, f1_en, _ = precision_recall_fscore_support(y_true_en, y_pred_en, average="macro")

    # 2. Evaluate Hindi (via MarianMT translation)
    print(f"Translating and evaluating Hindi test set ({len(df_hi)} samples)...")
    y_true_hi = df_hi["label"].astype(int).values
    hi_texts = df_hi["text"].fillna("").tolist()
    
    translated_texts = []
    for text in hi_texts:
        translated_texts.append(translate_hi_to_en(text))
    
    y_pred_hi_list = []
    for i in range(0, len(translated_texts), batch_size):
        batch = translated_texts[i : i + batch_size]
        probs = predict_proba(batch, model=model_name)
        preds = np.argmax(probs, axis=1)
        y_pred_hi_list.extend(preds)
    y_pred_hi = np.array(y_pred_hi_list)

    acc_hi = float(accuracy_score(y_true_hi, y_pred_hi))
    p_hi, r_hi, f1_hi, _ = precision_recall_fscore_support(y_true_hi, y_pred_hi, average="macro")

    # 3. Compute Gap and Bootstrap 95% CI
    f1_gap = float(f1_en - f1_hi)
    ci_lower, ci_upper, _ = compute_bootstrap_ci(y_true_en, y_pred_en, y_true_hi, y_pred_hi)

    results = {
        "model": model_name,
        "english_metrics": {
            "samples": len(y_true_en),
            "accuracy": round(acc_en, 4),
            "precision": round(float(p_en), 4),
            "recall": round(float(r_en), 4),
            "macro_f1": round(float(f1_en), 4),
        },
        "hindi_translated_metrics": {
            "samples": len(y_true_hi),
            "accuracy": round(acc_hi, 4),
            "precision": round(float(p_hi), 4),
            "recall": round(float(r_hi), 4),
            "macro_f1": round(float(f1_hi), 4),
        },
        "f1_gap": round(f1_gap, 4),
        "f1_gap_95_bootstrap_ci": [round(ci_lower, 4), round(ci_upper, 4)],
        "caveat": "The Hindi and English datasets come from different sources/distributions, so the observed performance gap reflects both domain divergence and machine translation errors."
    }

    json_path = RESULTS_DIR / "fairness_en_vs_hi.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Fairness evaluation results saved to {json_path}")

    # Plot Comparison & Gap with CI
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Metric comparison bar chart
    metrics_names = ["Accuracy", "Macro-F1"]
    en_vals = [acc_en, f1_en]
    hi_vals = [acc_hi, f1_hi]
    x = np.arange(len(metrics_names))
    width = 0.35

    ax1.bar(x - width/2, en_vals, width, label="English (Direct)", color="#2ecc71")
    ax1.bar(x + width/2, hi_vals, width, label="Hindi (MarianMT -> English)", color="#e67e22")
    ax1.set_ylabel("Score")
    ax1.set_title("Cross-Lingual Sentiment Performance")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names)
    ax1.set_ylim(0, 1.05)
    ax1.legend()

    # Gap and CI error bar
    ax2.errorbar(
        [0], [f1_gap],
        yerr=[[f1_gap - ci_lower], [ci_upper - f1_gap]],
        fmt='o', color='#e74c3c', ecolor='#c0392b', elinewidth=3, capsize=8, markersize=8
    )
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_xticks([0])
    ax2.set_xticklabels([f"F1 Gap: {f1_gap:.3f}"])
    ax2.set_ylabel("Macro-F1 Disparity (En - Hi)")
    ax2.set_title(f"Fairness F1 Gap (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")

    plt.tight_layout()
    chart_path = RESULTS_DIR / "fairness_en_vs_hi.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Fairness plot saved to {chart_path}")

    return results


if __name__ == "__main__":
    evaluate_fairness()
