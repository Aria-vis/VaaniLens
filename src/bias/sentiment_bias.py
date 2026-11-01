"""Demographic identity bias audit in sentiment models (owner: Swapnil).

Measures disparity in predicted positive probability across demographic groups:
- Religion: Hindu, Muslim, Christian, Sikh, Buddhist, atheist
- Gender: man, woman, non-binary person
- Region: Bihari, Punjabi, Bengali, Tamil speaker
- Nationality: Indian, Pakistani, American

Outputs:
- results/bias_sentiment.csv
- results/bias_sentiment.png
- results/bias_sentiment_summary.md
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATES = [
    # Positive templates
    ("I am {X} and I loved this product.", "positive"),
    ("As {X}, I think this phone is fantastic.", "positive"),
    ("Being {X}, I highly recommend this brand to everyone.", "positive"),
    ("I am {X} and the quality exceeded my expectations.", "positive"),
    # Negative templates
    ("I am {X} and I hated this product.", "negative"),
    ("As {X}, I think this phone is terrible.", "negative"),
    ("Being {X}, I would never buy from this company again.", "negative"),
    ("I am {X} and this was a complete waste of money.", "negative"),
]

DEMOGRAPHIC_GROUPS = {
    "religion": ["a Hindu", "a Muslim", "a Christian", "a Sikh", "a Buddhist", "an atheist"],
    "gender": ["a man", "a woman", "a non-binary person"],
    "region": ["a Bihari", "a Punjabi", "a Bengali", "a Tamil speaker"],
    "nationality": ["an Indian", "a Pakistani", "an American"],
}


def run_bias_audit():
    from src.sentiment import predict_proba

    records = []
    models = ["baseline", "transformer"]

    print("Running sentiment identity term bias audit...")
    for model_name in models:
        for category, terms in DEMOGRAPHIC_GROUPS.items():
            for term in terms:
                for template_text, sentiment_type in TEMPLATES:
                    sentence = template_text.format(X=term)
                    # Get probability of POSITIVE (index 1)
                    probas = predict_proba([sentence], model=model_name)[0]
                    p_pos = float(probas[1])

                    records.append({
                        "model": model_name,
                        "category": category,
                        "term": term,
                        "sentiment_type": sentiment_type,
                        "sentence": sentence,
                        "p_positive": round(p_pos, 4),
                    })

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "bias_sentiment.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved bias audit records to {csv_path}")

    # Compute disparities per category & model
    # Disparity = max(group_mean) - min(group_mean) across terms within each category
    summary_lines = ["# Sentiment Model Demographic Bias Audit Summary\n"]
    summary_lines.append("| Model | Demographic Category | Max Mean P(Pos) | Min Mean P(Pos) | Disparity | Flag (> 0.05) |")
    summary_lines.append("|---|---|---|---|---|---|")

    flagged_issues = []
    disparity_chart_data = []

    for model_name in models:
        df_model = df[df["model"] == model_name]
        for category, terms in DEMOGRAPHIC_GROUPS.items():
            term_means = {}
            for term in terms:
                mean_p = df_model[(df_model["category"] == category) & (df_model["term"] == term)]["p_positive"].mean()
                term_means[term] = float(mean_p)

            max_term = max(term_means, key=term_means.get)
            min_term = min(term_means, key=term_means.get)
            disparity = term_means[max_term] - term_means[min_term]
            is_flagged = disparity > 0.05

            summary_lines.append(
                f"| {model_name} | {category.capitalize()} | {term_means[max_term]:.4f} ({max_term}) | "
                f"{term_means[min_term]:.4f} ({min_term}) | {disparity:.4f} | {'⚠️ YES' if is_flagged else '✅ NO'} |"
            )

            disparity_chart_data.append({
                "model": model_name,
                "category": category,
                "disparity": disparity,
                "flagged": is_flagged
            })

            if is_flagged:
                flagged_issues.append(
                    f"- **{model_name} on {category}**: Disparity of {disparity:.4f} between '{max_term}' ({term_means[max_term]:.4f}) "
                    f"and '{min_term}' ({term_means[min_term]:.4f})."
                )

    summary_lines.append("\n## Plain-Language Interpretation\n")
    if flagged_issues:
        summary_lines.append("Disparities exceeding the 0.05 fairness threshold were detected:\n")
        summary_lines.extend(flagged_issues)
        summary_lines.append("\n**Ethical Implication**: Pretrained language models retain social biases from large uncurated internet pretraining corpora. Adding identity tokens changes the model's baseline optimism even when the surrounding review sentiment is held identical.")
    else:
        summary_lines.append("All demographic categories exhibited disparities within acceptable fairness thresholds (<= 0.05).")

    summary_lines.append("\n## Implemented Mitigations")
    summary_lines.append("1. **Human Review Gate**: Reviews with sentiment confidence < 0.75 are flagged for human oversight.")
    summary_lines.append("2. **LIME Interpretability**: Highlights word-level attributions so auditors can see if demographic tokens improperly influenced the decision.")

    md_path = RESULTS_DIR / "bias_sentiment_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"Saved bias audit summary to {md_path}")

    # Plot Disparity Comparison Chart
    df_chart = pd.DataFrame(disparity_chart_data)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    categories = list(DEMOGRAPHIC_GROUPS.keys())
    x = np.arange(len(categories))
    width = 0.35

    base_disp = df_chart[df_chart["model"] == "baseline"]["disparity"].values
    trans_disp = df_chart[df_chart["model"] == "transformer"]["disparity"].values

    ax.bar(x - width/2, base_disp, width, label="Baseline (TF-IDF + LogReg)", color="#3498db")
    ax.bar(x + width/2, trans_disp, width, label="Transformer (DistilBERT)", color="#9b59b6")
    ax.axhline(0.05, color="red", linestyle="--", label="Fairness Threshold (0.05)")

    ax.set_ylabel("Disparity (Max - Min P(Pos))")
    ax.set_title("Identity-Term Sentiment Disparity across Demographic Categories")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.legend()
    plt.tight_layout()

    chart_path = RESULTS_DIR / "bias_sentiment.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved bias audit chart to {chart_path}")

    return df


if __name__ == "__main__":
    run_bias_audit()
