"""VaaniLens: Multilingual Sentiment, Entity, Translation & Explainability Analyzer.

Main Streamlit Application.
Tabs:
1. Analyze (Multilingual review analysis with LIME & NER)
2. Models (Baseline vs Transformer benchmark)
3. MT Showdown (Rule-based vs SMT vs Neural MT)
4. Bias Audit (Demographic sentiment disparity & translation gender bias)
5. About (Ethics, Model Card, and Privacy)
"""

# Import sys for manipulating Python module search paths
import sys
# Import os for inspecting and setting system environment variables
import os
# Import json for parsing metric summaries and benchmark records
import json
# Import Path for platform-independent filesystem path resolution
from pathlib import Path
# Import Streamlit framework for interactive web application UI rendering
import streamlit as st
# Import pandas for tabulating evaluation benchmarks and entity records
import pandas as pd

# Resolve the absolute path to the root of the project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Ensure the project root is present at the front of sys.path for absolute imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enforce offline Hugging Face operation when VAANI_OFFLINE flag is asserted
if os.environ.get("VAANI_OFFLINE") == "1":
    # Prevent Hugging Face Hub from polling remote API servers for model weights
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # Force transformers library to use locally cached weights exclusively
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Import the core analysis pipeline orchestrator
from src.pipeline import analyze
# Import HTML visualization helper for LIME feature attribution spans
from src.explain import highlight_html
# Import human-readable spaCy entity label definitions
from src.ner import LABEL_DESCRIPTIONS
# Import WebGL2 hero banner component from custom Streamlit extension
from src.pixel_trail import render_pixel_trail

# Configure Streamlit browser tab title, favicon, and wide page layout
st.set_page_config(
    page_title="VaaniLens | Multilingual NLP & Ethics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar configuration: Display project branding and module context
st.sidebar.title("🔍 VaaniLens")
# Render metadata regarding academic scope and supported language pairs
st.sidebar.markdown(
    "**Module V Mini-Project: NLP Applications and Ethics**\n\n"
    "*Languages*: Hindi (हिन्दी) & English\n\n"
    "---"
)
# Render ethical and privacy notice in the persistent sidebar panel
st.sidebar.info(
    "🛡️ **Privacy & Ethics Disclaimer**:\n\n"
    "- Automated AI predictions may contain errors.\n"
    "- Inputs are processed strictly in-memory.\n"
    "- No user data or reviews are stored on disk."
)

# Define path to directory containing precomputed benchmark artifacts and plots
RESULTS_DIR = PROJECT_ROOT / "results"

# Interactive WebGL2 PixelTrail Hero Banner (React Bits Component)
render_pixel_trail(height=185)

# Initialize application navigation tabs across five primary analytical views
tab_analyze, tab_models, tab_mt, tab_bias, tab_about = st.tabs([
    "🔍 Analyze",
    "📊 Sentiment Models",
    "🌐 MT Showdown",
    "⚖️ Bias & Fairness Audit",
    "ℹ️ About & Ethics"
])

# -------------------------------------------------------------
# TAB 1: ANALYZE (Live Multilingual Review Inference)
# -------------------------------------------------------------
with tab_analyze:
    # Display primary section header for interactive review inference
    st.header("Multilingual Review Analyzer")
    # Instructional text guiding user input in either Hindi or English
    st.markdown("Paste a customer product review in **Hindi** or **English** to analyze sentiment, entities, and explanations.")

    # Split top controls into two responsive columns (model selection vs explanation toggle)
    col_model, col_explain = st.columns([2, 1])
    # Model selection dropdown in left column
    with col_model:
        selected_model = st.selectbox(
            "Select Sentiment Model",
            options=["transformer", "baseline"],
            # Provide descriptive labels indicating underlying architecture
            format_func=lambda x: "DistilBERT SST-2 (Neural Transformer)" if x == "transformer" else "TF-IDF + Logistic Regression (Baseline)"
        )
    # Checkbox to toggle compute-intensive LIME explanations in right column
    with col_explain:
        enable_lime = st.checkbox("Generate LIME Explanations", value=True)

    # Preset benchmark review examples for instant testing
    st.markdown("**Quick Test Examples:**")
    # Allocate five equal-width columns for preset buttons
    example_cols = st.columns(5)
    # Curated test instances covering diverse linguistic and ethical edge cases
    examples = [
        ("Hindi Positive", "यह फ़ोन बहुत अच्छा है, बैटरी पूरे दिन चलती है।"),
        ("Hindi Mixed + Brand", "डिलीवरी बहुत देर से आई और सैमसंग का चार्जर खराब निकला।"),
        ("English Sarcasm", "Oh great, another charger that stops working after a week. Just wonderful."),
        ("English Mixed Entities", "The Redmi Note arrived in Patna two days late, but the camera is fantastic."),
        ("Gender Neutral", "वह डॉक्टर है।")
    ]

    # Initialize session state review input buffer if not already present
    if "review_input" not in st.session_state:
        st.session_state["review_input"] = "यह फ़ोन बहुत अच्छा है, बैटरी पूरे दिन चलती है।"

    # Callback function to populate text area upon button click
    def _set_example(text):
        st.session_state["review_input"] = text

    # Render each test button and bind it to its corresponding text sample
    for i, (label, ex_text) in enumerate(examples):
        with example_cols[i]:
            st.button(label, key=f"ex_btn_{i}", on_click=_set_example, args=(ex_text,))

    # User input text area bound to session state key
    review_text = st.text_area(
        "Enter Review Text:",
        key="review_input",
        height=100
    )

    # Submission button triggering end-to-end pipeline analysis
    if st.button("🚀 Analyze Review", type="primary"):
        # Display animated spinner during neural inference and LIME perturbation
        with st.spinner("Processing through neural pipeline..."):
            result = analyze(review_text, model=selected_model, explain_text=enable_lime)

        # Warnings / Human Review Flag: Surface alert when low confidence or hallucination occurs
        if result["needs_human_review"]:
            st.warning("⚠️ **Human Oversight Required**: " + " | ".join(result["warnings"]))
        elif result["warnings"]:
            st.info("ℹ️ " + " | ".join(result["warnings"]))

        # Render key quantitative metrics across three KPI cards
        col1, col2, col3 = st.columns(3)
        # Column 1: Detected language identification
        with col1:
            lang_label = "Hindi (हिन्दी)" if result["language"] == "hi" else "English"
            st.metric("Detected Language", lang_label)
        # Column 2: Sentiment classification and probability score
        with col2:
            sentiment_label = result["sentiment"]["label"]
            score = result["sentiment"]["score"]
            st.metric("Sentiment", f"{sentiment_label} ({score:.1%})")
        # Column 3: Wall-clock pipeline execution latency in milliseconds
        with col3:
            st.metric("Pipeline Latency", f"{result['latency_ms']} ms")

        # Translation section: Conditionally displayed when input language is Hindi
        if result["language"] == "hi" and result["translated_text"]:
            st.subheader("🌐 Neural Machine Translation (MarianMT)")
            st.markdown(f"> **English Translation**: *{result['translated_text']}*")

        # Explainability section: Display LIME token attributions when enabled
        if result["explanation"]:
            st.subheader("💡 Explainability (LIME Word Attributions)")
            # Clarify that attributions operate on English text representation
            if result["language"] == "hi":
                st.caption("ℹ️ Local token attributions computed on the neural English translation used by the sentiment engine.")
            # Color-coded interpretation legend
            st.markdown(
                "Words highlighted in <span style='background-color:#2ecc71; color:white; padding:2px 6px; border-radius:4px;'>Green</span> pushed prediction toward **POSITIVE**; "
                "<span style='background-color:#e74c3c; color:white; padding:2px 6px; border-radius:4px;'>Red</span> pushed toward **NEGATIVE**.",
                unsafe_allow_html=True
            )
            # Generate XSS-safe highlighted HTML snippet
            html_viz = highlight_html(result["text_for_analysis"], result["explanation"])
            # Render highlighted review text container
            st.markdown(f"<div style='font-size:18px; line-height:2.0; padding:12px; background:#f8f9fa; border-radius:8px;'>{html_viz}</div>", unsafe_allow_html=True)

        # Named Entity Recognition section: Surface extracted entities from spaCy
        st.subheader("🏷️ Extracted Named Entities (spaCy)")
        # Note language pipeline behavior for Hindi reviews
        if result["language"] == "hi":
            st.caption("ℹ️ Entities and character offsets are extracted from the translated English text via spaCy en_core_web_sm.")
        # If entities are identified, format them into an interactive data table
        if result["entities"]:
            ent_df = pd.DataFrame(result["entities"])
            # Map OntoNotes 5.0 uppercase entity tags to friendly descriptions
            ent_df["Description"] = ent_df["label"].map(lambda l: LABEL_DESCRIPTIONS.get(l, "General Entity"))
            # Render formatted entity table spanning container width
            st.dataframe(ent_df[["text", "label", "Description", "start", "end"]], use_container_width=True)
        else:
            # Fallback message when no entities are detected
            st.write("No named entities detected.")

# -------------------------------------------------------------
# TAB 2: SENTIMENT MODELS BENCHMARK
# -------------------------------------------------------------
with tab_models:
    # Section header for offline benchmark comparison
    st.header("📊 Sentiment Classification Benchmark")
    # Subtitle describing evaluation corpora and sample count
    st.markdown("Comparison between the **TF-IDF + Logistic Regression Baseline** and **Pretrained DistilBERT SST-2 Transformer** on Amazon Polarity (1,000 test reviews).")

    # Paths to serialized evaluation metric JSON files
    base_metrics_file = RESULTS_DIR / "sentiment_baseline_metrics.json"
    trans_metrics_file = RESULTS_DIR / "sentiment_transformer_metrics.json"

    # Verify both benchmark metric files exist before rendering
    if base_metrics_file.exists() and trans_metrics_file.exists():
        # Load baseline TF-IDF metrics from JSON
        with open(base_metrics_file, "r") as f:
            b_m = json.load(f)
        # Load DistilBERT transformer metrics from JSON
        with open(trans_metrics_file, "r") as f:
            t_m = json.load(f)

        # Tabulate comparative architectural and performance metrics
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
        # Display comparative summary table
        st.dataframe(metrics_comparison, use_container_width=True)

        # Allocate two columns for side-by-side confusion matrix figures
        col_cm1, col_cm2 = st.columns(2)
        # Render Baseline confusion matrix plot
        with col_cm1:
            st.subheader("Baseline Confusion Matrix")
            base_cm_img = RESULTS_DIR / "confusion_baseline.png"
            if base_cm_img.exists():
                st.image(str(base_cm_img), use_container_width=True)
        # Render DistilBERT confusion matrix plot
        with col_cm2:
            st.subheader("DistilBERT Confusion Matrix")
            trans_cm_img = RESULTS_DIR / "confusion_transformer.png"
            if trans_cm_img.exists():
                st.image(str(trans_cm_img), use_container_width=True)

# -------------------------------------------------------------
# TAB 3: MT SHOWDOWN (Rule-Based vs SMT vs Neural MT)
# -------------------------------------------------------------
with tab_mt:
    # Section header for machine translation paradigm comparison
    st.header("🌐 Machine Translation Paradigm Showdown")
    # Subtitle outlining the historical evolution of MT paradigms
    st.markdown("Quantitative & qualitative comparison across the three eras of MT: **Rule-Based**, **Statistical (IBM Model 1)**, and **Neural MT (MarianMT)**.")

    # Paths to generated comparative report and SacreBLEU evaluation plot
    mt_comp_file = RESULTS_DIR / "mt_comparison.md"
    mt_plot_file = RESULTS_DIR / "mt_bleu.png"

    # Display BLEU and chrF++ comparative bar chart if available
    if mt_plot_file.exists():
        st.image(str(mt_plot_file), caption="BLEU vs chrF Benchmark across Architectures", width=700)

    # Render detailed Markdown comparison report
    if mt_comp_file.exists():
        with open(mt_comp_file, "r", encoding="utf-8") as f:
            st.markdown(f.read())

# -------------------------------------------------------------
# TAB 4: BIAS & FAIRNESS AUDIT
# -------------------------------------------------------------
with tab_bias:
    # Section header for algorithmic fairness and demographic audits
    st.header("⚖️ Algorithmic Bias & Cross-Lingual Fairness Audit")

    # Allocate two columns for Identity Bias and Pronoun Bias audits
    col_b1, col_b2 = st.columns(2)
    # Column 1: Identity-term counterfactual sentiment bias
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

    # Column 2: Neural machine translation gender pronoun inference audit
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

    # Horizontal divider between demographic and cross-lingual audits
    st.markdown("---")
    # Subheader for English vs Translated Hindi performance gap analysis
    st.subheader("3. Cross-Lingual Fairness: English vs Translated Hindi")
    # Display cross-lingual metric bar chart and bootstrap CI error bar
    fairness_img = RESULTS_DIR / "fairness_en_vs_hi.png"
    if fairness_img.exists():
        st.image(str(fairness_img), width=750)
    # Load and display numerical bootstrap CI values and distribution caveat
    fairness_json = RESULTS_DIR / "fairness_en_vs_hi.json"
    if fairness_json.exists():
        with open(fairness_json, "r", encoding="utf-8") as f:
            fair_data = json.load(f)
        st.info(f"**Disparity Gap**: {fair_data['f1_gap']:.4f} Macro-F1 gap | **95% Bootstrap CI**: [{fair_data['f1_gap_95_bootstrap_ci'][0]:.4f}, {fair_data['f1_gap_95_bootstrap_ci'][1]:.4f}].\n\n*Caveat*: {fair_data['caveat']}")

# -------------------------------------------------------------
# TAB 5: ABOUT & ETHICS
# -------------------------------------------------------------
with tab_about:
    # Section header for governance, intended use, and limitations
    st.header("ℹ️ About VaaniLens & Ethical Considerations")
    # Document core ethical principles, data minimization, and oversight
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
