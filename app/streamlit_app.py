"""VaaniLens: Multilingual Sentiment, Entity, Translation & Explainability Analyzer.

Main Streamlit Application.
Tabs:
1. Analyze (Multilingual review analysis with LIME & NER)
2. Models (Baseline vs Transformer benchmark)
3. MT Showdown (Rule-based vs SMT vs Neural MT)
4. Bias Audit (Demographic sentiment disparity & translation gender bias)
5. About (Ethics, Model Card, and Privacy)
"""

import sys
import os
import json
from pathlib import Path
import streamlit as st
import pandas as pd

# Set up paths and offline environment
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if os.environ.get("VAANI_OFFLINE") == "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.pipeline import analyze
from src.explain import highlight_html
from src.ner import LABEL_DESCRIPTIONS
from src.pixel_trail import render_pixel_trail

st.set_page_config(
    page_title="VaaniLens | Multilingual NLP & Ethics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("🔍 VaaniLens")
st.sidebar.markdown(
    "**Module V Mini-Project: NLP Applications and Ethics**\n\n"
    "*Languages*: Hindi (हिन्दी) & English\n\n"
    "---"
)
st.sidebar.info(
    "🛡️ **Privacy & Ethics Disclaimer**:\n\n"
    "- Automated AI predictions may contain errors.\n"
    "- Inputs are processed strictly in-memory.\n"
    "- No user data or reviews are stored on disk."
)

RESULTS_DIR = PROJECT_ROOT / "results"

# Interactive WebGL2 PixelTrail Hero Banner (React Bits Component)
render_pixel_trail(height=185)

tab_analyze, tab_models, tab_mt, tab_bias, tab_about = st.tabs([
    "🔍 Analyze",
    "📊 Sentiment Models",
    "🌐 MT Showdown",
    "⚖️ Bias & Fairness Audit",
    "ℹ️ About & Ethics"
])

# -------------------------------------------------------------
# TAB 1: ANALYZE
# -------------------------------------------------------------
with tab_analyze:
    st.header("Multilingual Review Analyzer")
    st.markdown("Paste a customer product review in **Hindi** or **English** to analyze sentiment, entities, and explanations.")

    col_model, col_explain = st.columns([2, 1])
    with col_model:
        selected_model = st.selectbox(
            "Select Sentiment Model",
            options=["transformer", "baseline"],
            format_func=lambda x: "DistilBERT SST-2 (Neural Transformer)" if x == "transformer" else "TF-IDF + Logistic Regression (Baseline)"
        )
    with col_explain:
        enable_lime = st.checkbox("Generate LIME Explanations", value=True)

    # Example review buttons from PROJECT.md Appendix
    st.markdown("**Quick Test Examples:**")
    example_cols = st.columns(5)
    examples = [
        ("Hindi Positive", "यह फ़ोन बहुत अच्छा है, बैटरी पूरे दिन चलती है।"),
        ("Hindi Mixed + Brand", "डिलीवरी बहुत देर से आई और सैमसंग का चार्जर खराब निकला।"),
        ("English Sarcasm", "Oh great, another charger that stops working after a week. Just wonderful."),
        ("English Mixed Entities", "The Redmi Note arrived in Patna two days late, but the camera is fantastic."),
        ("Gender Neutral", "वह डॉक्टर है।")
    ]

    selected_text = ""
    for i, (label, ex_text) in enumerate(examples):
        with example_cols[i]:
            if st.button(label, key=f"ex_btn_{i}"):
                st.session_state["review_input"] = ex_text

    review_text = st.text_area(
        "Enter Review Text:",
        value=st.session_state.get("review_input", "यह फ़ोन बहुत अच्छा है, बैटरी पूरे दिन चलती है।"),
        height=100
    )

    if st.button("🚀 Analyze Review", type="primary"):
        with st.spinner("Processing through neural pipeline..."):
            result = analyze(review_text, model=selected_model, explain_text=enable_lime)

        # Warnings / Human Review Flag
        if result["needs_human_review"]:
            st.warning("⚠️ **Human Oversight Required**: " + " | ".join(result["warnings"]))
        elif result["warnings"]:
            st.info("ℹ️ " + " | ".join(result["warnings"]))

        col1, col2, col3 = st.columns(3)
        with col1:
            lang_label = "Hindi (हिन्दी)" if result["language"] == "hi" else "English"
            st.metric("Detected Language", lang_label)
        with col2:
            sentiment_label = result["sentiment"]["label"]
            score = result["sentiment"]["score"]
            st.metric("Sentiment", f"{sentiment_label} ({score:.1%})")
        with col3:
            st.metric("Pipeline Latency", f"{result['latency_ms']} ms")

        # Translation section if Hindi
        if result["language"] == "hi" and result["translated_text"]:
            st.subheader("🌐 Neural Machine Translation (MarianMT)")
            st.markdown(f"> **English Translation**: *{result['translated_text']}*")

        # LIME Explanations
        if result["explanation"]:
            st.subheader("💡 Explainability (LIME Word Attributions)")
            st.markdown(
                "Words highlighted in <span style='background-color:#2ecc71; color:white; padding:2px 6px; border-radius:4px;'>Green</span> pushed prediction toward **POSITIVE**; "
                "<span style='background-color:#e74c3c; color:white; padding:2px 6px; border-radius:4px;'>Red</span> pushed toward **NEGATIVE**.",
                unsafe_allow_html=True
            )
            html_viz = highlight_html(result["text_for_analysis"], result["explanation"])
            st.markdown(f"<div style='font-size:18px; line-height:2.0; padding:12px; background:#f8f9fa; border-radius:8px;'>{html_viz}</div>", unsafe_allow_html=True)

        # Named Entities
        st.subheader("🏷️ Extracted Named Entities (spaCy)")
        if result["entities"]:
            ent_df = pd.DataFrame(result["entities"])
            ent_df["Description"] = ent_df["label"].map(lambda l: LABEL_DESCRIPTIONS.get(l, "General Entity"))
            st.dataframe(ent_df[["text", "label", "Description", "start", "end"]], use_container_width=True)
        else:
            st.write("No named entities detected.")

# -------------------------------------------------------------
# TAB 2: SENTIMENT MODELS BENCHMARK
# -------------------------------------------------------------
with tab_models:
    st.header("📊 Sentiment Classification Benchmark")
    st.markdown("Comparison between the **TF-IDF + Logistic Regression Baseline** and **Pretrained DistilBERT SST-2 Transformer** on Amazon Polarity (1,000 test reviews).")

    base_metrics_file = RESULTS_DIR / "sentiment_baseline_metrics.json"
    trans_metrics_file = RESULTS_DIR / "sentiment_transformer_metrics.json"

    if base_metrics_file.exists() and trans_metrics_file.exists():
        with open(base_metrics_file, "r") as f:
            b_m = json.load(f)
        with open(trans_metrics_file, "r") as f:
            t_m = json.load(f)

        metrics_comparison = pd.DataFrame([
            {
                "Model": "Baseline (TF-IDF + Logistic Regression)",
                "Accuracy": f"{b_m['accuracy']:.2%}",
                "Precision": f"{b_m['precision']:.2%}",
                "Recall": f"{b_m['recall']:.2%}",
                "Macro-F1": f"{b_m['f1']:.2%}",
                "Model Size": "1.34 MB",
                "Speed": "Instantaneous (< 2 ms/sample)"
            },
            {
                "Model": "Transformer (DistilBERT SST-2)",
                "Accuracy": f"{t_m['accuracy']:.2%}",
                "Precision": f"{t_m['precision']:.2%}",
                "Recall": f"{t_m['recall']:.2%}",
                "Macro-F1": f"{t_m['f1']:.2%}",
                "Model Size": "268 MB",
                "Speed": "~ 40 ms/sample (CPU)"
            }
        ])
        st.dataframe(metrics_comparison, use_container_width=True)

        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            st.subheader("Baseline Confusion Matrix")
            base_cm_img = RESULTS_DIR / "confusion_baseline.png"
            if base_cm_img.exists():
                st.image(str(base_cm_img), use_container_width=True)
        with col_cm2:
            st.subheader("DistilBERT Confusion Matrix")
            trans_cm_img = RESULTS_DIR / "confusion_transformer.png"
            if trans_cm_img.exists():
                st.image(str(trans_cm_img), use_container_width=True)

# -------------------------------------------------------------
# TAB 3: MT SHOWDOWN
# -------------------------------------------------------------
with tab_mt:
    st.header("🌐 Machine Translation Paradigm Showdown")
    st.markdown("Quantitative & qualitative comparison across the three eras of MT: **Rule-Based**, **Statistical (IBM Model 1)**, and **Neural MT (MarianMT)**.")

    mt_comp_file = RESULTS_DIR / "mt_comparison.md"
    mt_plot_file = RESULTS_DIR / "mt_bleu.png"

    if mt_plot_file.exists():
        st.image(str(mt_plot_file), caption="BLEU vs chrF Benchmark across Architectures", width=700)

    if mt_comp_file.exists():
        with open(mt_comp_file, "r", encoding="utf-8") as f:
            st.markdown(f.read())

# -------------------------------------------------------------
# TAB 4: BIAS & FAIRNESS AUDIT
# -------------------------------------------------------------
with tab_bias:
    st.header("⚖️ Algorithmic Bias & Cross-Lingual Fairness Audit")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.subheader("1. Identity-Term Sentiment Bias")
        st.markdown("Disparity across demographic terms (Religion, Gender, Region, Nationality) in identical templates.")
        bias_sent_img = RESULTS_DIR / "bias_sentiment.png"
        if bias_sent_img.exists():
            st.image(str(bias_sent_img), use_container_width=True)
        bias_sent_summary = RESULTS_DIR / "bias_sentiment_summary.md"
        if bias_sent_summary.exists():
            with open(bias_sent_summary, "r", encoding="utf-8") as f:
                st.markdown(f.read())

    with col_b2:
        st.subheader("2. NMT Gender Pronoun Bias")
        st.markdown("Inferred English pronouns (he/she/they) from gender-neutral Hindi inputs across 20 professions.")
        bias_trans_img = RESULTS_DIR / "bias_translation.png"
        if bias_trans_img.exists():
            st.image(str(bias_trans_img), use_container_width=True)
        bias_trans_summary = RESULTS_DIR / "bias_translation_summary.md"
        if bias_trans_summary.exists():
            with open(bias_trans_summary, "r", encoding="utf-8") as f:
                st.markdown(f.read())

    st.markdown("---")
    st.subheader("3. Cross-Lingual Fairness: English vs Translated Hindi")
    fairness_img = RESULTS_DIR / "fairness_en_vs_hi.png"
    if fairness_img.exists():
        st.image(str(fairness_img), width=750)
    fairness_json = RESULTS_DIR / "fairness_en_vs_hi.json"
    if fairness_json.exists():
        with open(fairness_json, "r", encoding="utf-8") as f:
            fair_data = json.load(f)
        st.info(f"**Disparity Gap**: {fair_data['f1_gap']:.4f} Macro-F1 gap | **95% Bootstrap CI**: [{fair_data['f1_gap_95_bootstrap_ci'][0]:.4f}, {fair_data['f1_gap_95_bootstrap_ci'][1]:.4f}].\n\n*Caveat*: {fair_data['caveat']}")

# -------------------------------------------------------------
# TAB 5: ABOUT & ETHICS
# -------------------------------------------------------------
with tab_about:
    st.header("ℹ️ About VaaniLens & Ethical Considerations")
    st.markdown("""
    ### Intended Use
    VaaniLens is designed for customer feedback intelligence and product review analysis across Hindi and English markets.

    ### Out-of-Scope Uses
    - Automated content moderation or punitive user sanctioning without human review.
    - Automated hiring or resume screening.
    - Medical, legal, or high-stakes financial decision-making.

    ### Privacy Architecture
    - **Zero Storage**: Raw user text is processed in memory and never logged or serialized to disk.
    - **Zero API Dependency**: All models (DistilBERT, MarianMT, spaCy, scikit-learn) run completely locally on CPU with 0 API cost.

    ### Human-in-the-Loop Oversight Gate
    - Predictions with confidence $< 0.75$ trigger a yellow review banner (`needs_human_review = True`).
    - Hallucinated gender pronouns in machine translation automatically surface an inference warning.
    """)

