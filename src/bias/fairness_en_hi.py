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

# Import Path for cross-platform OS filesystem path manipulation
from pathlib import Path
# Import json for serializing fairness metrics and bootstrap intervals
import json
# Import numpy for pseudo-random number generation and percentile computation
import numpy as np
# Import pandas for loading evaluation CSV splits into memory
import pandas as pd
# Import matplotlib.pyplot for producing comparative bar charts and CI error bars
import matplotlib.pyplot as plt
# Import standard evaluation metrics from scikit-learn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Resolve the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Define the path to data directory containing English and Hindi test CSVs
DATA_DIR = PROJECT_ROOT / "data"
# Define the destination directory for output metric JSONs and visualization figures
RESULTS_DIR = PROJECT_ROOT / "results"
# Ensure the results directory exists prior to file generation
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_bootstrap_ci(y_true_en, y_pred_en, y_true_hi, y_pred_hi, n_bootstraps=1000, seed=42):
    """Computes 95% bootstrap confidence interval for (F1_en - F1_hi).
    
    Args:
        y_true_en: Ground-truth binary labels for the English evaluation set.
        y_pred_en: Model-predicted binary labels for the English evaluation set.
        y_true_hi: Ground-truth binary labels for the Hindi evaluation set.
        y_pred_hi: Model-predicted binary labels for translated Hindi reviews.
        n_bootstraps: Number of Monte Carlo resamples (default 1000 for tight estimation).
        seed: Random seed for deterministic replicability across runs.
        
    Returns:
        ci_lower: 2.5th percentile representing lower bound of 95% confidence interval.
        ci_upper: 97.5th percentile representing upper bound of 95% confidence interval.
        gaps: Complete array of resampled F1 disparity gaps.
    """
    # Seed numpy random number generator for reproducible bootstrap draws
    np.random.seed(seed)
    # Initialize empty list to accumulate disparity scores across resamples
    gaps = []
    # Count total number of English samples available for sampling with replacement
    n_en = len(y_true_en)
    # Count total number of Hindi samples available for sampling with replacement
    n_hi = len(y_true_hi)

    # Perform n_bootstraps iterations of empirical resampling with replacement
    for _ in range(n_bootstraps):
        # Sample n_en indices uniformly at random with replacement from [0, n_en)
        idx_en = np.random.choice(n_en, size=n_en, replace=True)
        # Sample n_hi indices uniformly at random with replacement from [0, n_hi)
        idx_hi = np.random.choice(n_hi, size=n_hi, replace=True)

        # Compute Macro-F1 on resampled English subset
        _, _, f1_en_b, _ = precision_recall_fscore_support(
            y_true_en[idx_en], y_pred_en[idx_en], average="macro", zero_division=0
        )
        # Compute Macro-F1 on resampled Hindi subset
        _, _, f1_hi_b, _ = precision_recall_fscore_support(
            y_true_hi[idx_hi], y_pred_hi[idx_hi], average="macro", zero_division=0
        )
        # Append empirical difference (F1_en - F1_hi) to disparity distribution
        gaps.append(f1_en_b - f1_hi_b)

    # Calculate empirical 2.5 percentile for lower boundary of two-sided 95% CI
    ci_lower = float(np.percentile(gaps, 2.5))
    # Calculate empirical 97.5 percentile for upper boundary of two-sided 95% CI
    ci_upper = float(np.percentile(gaps, 97.5))
    # Return empirical confidence bounds alongside raw bootstrap disparity distribution
    return ci_lower, ci_upper, gaps


def evaluate_fairness(model_name: str = "transformer"):
    """Evaluates cross-lingual performance disparity between English and Hindi.
    
    Translates Hindi inputs via MarianMT, evaluates predictions using the designated
    classifier, compares metrics with native English, and calculates bootstrap CI.
    
    Args:
        model_name: Identifier of model to audit ("transformer" or "baseline").
        
    Returns:
        results: Dictionary containing English metrics, Hindi metrics, and disparity CI.
    """
    # Lazy-import translation engine to prevent unnecessary model loading during dry runs
    from src.translate import translate_hi_to_en
    # Lazy-import inference utility functions from sentiment analysis module
    from src.sentiment import predict_sentiment, predict_proba

    # Path to pre-split English test reviews
    en_test_path = DATA_DIR / "en_test.csv"
    # Path to curated Hindi test reviews
    hi_test_path = DATA_DIR / "hi_reviews_test.csv"

    # Verify dataset presence; automatically run data prep pipeline if missing
    if not en_test_path.exists() or not hi_test_path.exists():
        # Lazy-import data prep pipeline
        from src.data_prep import main as prep_data
        # Execute dataset download and splitting routine
        prep_data()

    # Inform user of evaluation initialization
    print("Loading test datasets for fairness evaluation...")
    # Load English test split into pandas DataFrame
    df_en = pd.read_csv(en_test_path)
    # Load Hindi test split into pandas DataFrame
    df_hi = pd.read_csv(hi_test_path)

    # 1. Evaluate English
    print(f"Evaluating English test set ({len(df_en)} samples)...")
    # Extract integer ground-truth target labels (0=Negative, 1=Positive)
    y_true_en = df_en["label"].astype(int).values
    # Clean and extract review strings, filling any NaN entries with empty string
    en_texts = df_en["text"].fillna("").tolist()
    
    # Define mini-batch size of 32 for efficient vectorized CPU inference
    batch_size = 32
    # Accumulate predicted integer classes for English reviews
    y_pred_en_list = []
    # Iterate across English reviews in slices of batch_size
    for i in range(0, len(en_texts), batch_size):
        # Slice current chunk of raw text strings
        batch = en_texts[i : i + batch_size]
        # Generate prediction probabilities [P(Neg), P(Pos)] for batch
        probs = predict_proba(batch, model=model_name)
        # Select argmax column index as predicted binary class
        preds = np.argmax(probs, axis=1)
        # Append batch predictions to master list
        y_pred_en_list.extend(preds)
    # Convert prediction list into flat NumPy array
    y_pred_en = np.array(y_pred_en_list)

    # Calculate overall English accuracy score
    acc_en = float(accuracy_score(y_true_en, y_pred_en))
    # Calculate Macro-Precision, Macro-Recall, and Macro-F1 across English labels
    p_en, r_en, f1_en, _ = precision_recall_fscore_support(y_true_en, y_pred_en, average="macro")

    # 2. Evaluate Hindi (via MarianMT translation)
    print(f"Translating and evaluating Hindi test set ({len(df_hi)} samples)...")
    # Extract integer ground-truth target labels for Hindi reviews
    y_true_hi = df_hi["label"].astype(int).values
    # Extract Hindi review strings from dataframe
    hi_texts = df_hi["text"].fillna("").tolist()
    
    # Translate Hindi texts into English using Helsinki-NLP/opus-mt-hi-en
    translated_texts = []
    # Process each Hindi text sequentially through translation engine
    for text in hi_texts:
        # Perform neural machine translation and store translated output
        translated_texts.append(translate_hi_to_en(text))
    
    # Accumulate predicted integer classes for translated Hindi reviews
    y_pred_hi_list = []
    # Iterate over translated English texts in mini-batches
    for i in range(0, len(translated_texts), batch_size):
        # Extract batch slice of translated text strings
        batch = translated_texts[i : i + batch_size]
        # Query sentiment classifier probabilities for batch
        probs = predict_proba(batch, model=model_name)
        # Determine highest-probability class index for each sentence
        preds = np.argmax(probs, axis=1)
        # Accumulate predictions into tracking list
        y_pred_hi_list.extend(preds)
    # Format Hindi predictions as single NumPy array
    y_pred_hi = np.array(y_pred_hi_list)

    # Calculate overall accuracy on translated Hindi dataset
    acc_hi = float(accuracy_score(y_true_hi, y_pred_hi))
    # Calculate Macro-Precision, Macro-Recall, and Macro-F1 for translated Hindi
    p_hi, r_hi, f1_hi, _ = precision_recall_fscore_support(y_true_hi, y_pred_hi, average="macro")

    # 3. Compute Gap and Bootstrap 95% CI
    # Measure macro-F1 disparity gap: positive gap indicates degradation on Hindi
    f1_gap = float(f1_en - f1_hi)
    # Calculate empirical 95% confidence interval via 1,000 bootstrap iterations
    ci_lower, ci_upper, _ = compute_bootstrap_ci(y_true_en, y_pred_en, y_true_hi, y_pred_hi)

    # Assemble structured results dictionary for programmatic serialization
    results = {
        # Audit model identifier
        "model": model_name,
        # Performance metrics achieved on English evaluation set
        "english_metrics": {
            "samples": len(y_true_en),
            "accuracy": round(acc_en, 4),
            "precision": round(float(p_en), 4),
            "recall": round(float(r_en), 4),
            "macro_f1": round(float(f1_en), 4),
        },
        # Performance metrics achieved on translated Hindi evaluation set
        "hindi_translated_metrics": {
            "samples": len(y_true_hi),
            "accuracy": round(acc_hi, 4),
            "precision": round(float(p_hi), 4),
            "recall": round(float(r_hi), 4),
            "macro_f1": round(float(f1_hi), 4),
        },
        # Absolute difference between English and Hindi Macro-F1
        "f1_gap": round(f1_gap, 4),
        # 95% non-parametric bootstrap confidence interval [lower, upper]
        "f1_gap_95_bootstrap_ci": [round(ci_lower, 4), round(ci_upper, 4)],
        # Explicit methodological caveat acknowledging distribution differences
        "caveat": "The Hindi and English datasets come from different sources/distributions, so the observed performance gap reflects both domain divergence and machine translation errors."
    }

    # Define destination path for JSON metrics report
    json_path = RESULTS_DIR / "fairness_en_vs_hi.json"
    # Write formatted JSON metrics dictionary with UTF-8 encoding
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    # Output confirmation to stdout
    print(f"Fairness evaluation results saved to {json_path}")

    # Plot Comparison & Gap with CI: Create side-by-side subplot canvas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Metric comparison bar chart
    metrics_names = ["Accuracy", "Macro-F1"]
    # Group English metric values
    en_vals = [acc_en, f1_en]
    # Group Hindi metric values
    hi_vals = [acc_hi, f1_hi]
    # Position indices on x-axis
    x = np.arange(len(metrics_names))
    # Width of individual metric comparison bars
    width = 0.35

    # Render bars for English benchmark performance
    ax1.bar(x - width/2, en_vals, width, label="English (Direct)", color="#2ecc71")
    # Render bars for translated Hindi benchmark performance
    ax1.bar(x + width/2, hi_vals, width, label="Hindi (MarianMT -> English)", color="#e67e22")
    # Set y-axis metric label
    ax1.set_ylabel("Score")
    # Set title for cross-lingual performance plot
    ax1.set_title("Cross-Lingual Sentiment Performance")
    # Configure x-axis tick positions
    ax1.set_xticks(x)
    # Label x-axis ticks with corresponding metric names
    ax1.set_xticklabels(metrics_names)
    # Set upper bound slightly above 1.0 for visual clearance
    ax1.set_ylim(0, 1.05)
    # Display plot legend
    ax1.legend()

    # Gap and CI error bar visualization
    ax2.errorbar(
        [0], [f1_gap],
        # Compute asymmetric error bar offsets relative to point estimate
        yerr=[[f1_gap - ci_lower], [ci_upper - f1_gap]],
        # Circular marker style with solid red coloration
        fmt='o', color='#e74c3c', ecolor='#c0392b', elinewidth=3, capsize=8, markersize=8
    )
    # Draw horizontal parity reference line at disparity=0
    ax2.axhline(0, color="gray", linestyle="--")
    # Constrain x-axis limits around single categorical point
    ax2.set_xlim(-0.5, 0.5)
    # Set single x-tick at center
    ax2.set_xticks([0])
    # Label x-tick with numerical disparity estimate
    ax2.set_xticklabels([f"F1 Gap: {f1_gap:.3f}"])
    # Set disparity measure label on y-axis
    ax2.set_ylabel("Macro-F1 Disparity (En - Hi)")
    # Set title with explicit bootstrap confidence interval bounds
    ax2.set_title(f"Fairness F1 Gap (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")

    # Automatically adjust subplot padding and prevent label clipping
    plt.tight_layout()
    # Path for exported PNG visualization
    chart_path = RESULTS_DIR / "fairness_en_vs_hi.png"
    # Save figure to disk at 150 DPI resolution
    plt.savefig(chart_path, dpi=150)
    # Close active Matplotlib figure canvas to release memory
    plt.close()
    # Output confirmation of saved chart
    print(f"Fairness plot saved to {chart_path}")

    # Return structured evaluation metrics
    return results


# Standard Python module execution guard
if __name__ == "__main__":
    # Execute full cross-lingual fairness evaluation routine
    evaluate_fairness()
