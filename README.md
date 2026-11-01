---
title: VaaniLens - Multilingual Review & Ethics Analyzer
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.40.0
app_file: app/streamlit_app.py
pinned: false
license: mit
---

# 🔍 VaaniLens: Multilingual, Explainable Sentiment, Entity, and Translation Analyzer
**Module V Mini-Project: NLP Applications and Ethics**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange.svg)](https://huggingface.co/)
[![Tests Passing](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen.svg)]()

VaaniLens is a comprehensive, production-grade Natural Language Processing system for multilingual review analysis in **Hindi** and **English**. It combines cross-lingual neural machine translation, dual-paradigm sentiment classification, named entity recognition, local LIME explainability, algorithmic bias and fairness audits, and an interactive dialogue assistant (**VaaniBot**)—engineered to execute **100% locally on CPU at ₹0 / $0 budget** with strict data privacy.

---

## 🌟 Core Highlights

1. **Multilingual Pipeline**: Automatic Devanagari script detection ($\ge 30\%$ threshold) and batch translation via MarianMT (`Helsinki-NLP/opus-mt-hi-en`).
2. **Dual Sentiment Engines**:
   - **Deep Learning**: DistilBERT SST-2 (`distilbert-base-uncased-finetuned-sst-2-english`) — **89.80% Accuracy**, **89.79% Macro-F1**.
   - **Classical Statistical**: TF-IDF + Logistic Regression — **84.80% Accuracy**, **84.80% Macro-F1**, 0.15 ms/sample inference.
3. **Machine Translation Paradigm Benchmark**: Historical progression across **Rule-Based MT** (BLEU 1.88), **IBM Model 1 Mini-SMT** (BLEU 4.68), and **MarianMT Neural MT** (BLEU 29.84).
4. **Local Interpretability (LIME)**: Word-level feature attribution ($N=200$ local perturbations) highlighting positive (green) and negative (red) contributions.
5. **Named Entity Recognition**: spaCy `en_core_web_sm` extracting brands, locations, dates, and products (Precision: 74.24%, Recall: 57.65%, F1: 64.90%).
6. **Active Safety Harness & Ethics**:
   - *Confidence Review Gate*: Predictions with confidence $< 0.75$ automatically set `needs_human_review = True`.
   - *Translation Gender Guard*: Surfaces warnings when neural translation infers binary gender from gender-neutral Hindi pronouns (*वह*, *वो*).
   - *Demographic Sentiment Audit*: Measures disparity across Religion, Gender, Region, and Nationality.
   - *Cross-Lingual Fairness*: English vs Hindi Macro-F1 gap evaluated with 1,000 bootstrap confidence intervals.
   - *Privacy by Design*: Purely in-memory evaluation; zero user reviews or personal texts are stored to disk.
7. **VaaniBot Dialogue System**: Intent-based conversational assistant (90.9% test accuracy across 11 intents) with empathy escalation and transparent disclosure.

---

## 📊 Summary of Results

Every figure in this table is strictly traceable to artifacts in the `results/` directory:

| Component / Evaluation | Metric | Score / Value | Reference Artifact |
|---|---|---|---|
| **DistilBERT Sentiment (English, N=1,000)** | Macro-F1 / Accuracy | **89.79% / 89.80%** | `results/sentiment_transformer_metrics.json` |
| **Baseline Sentiment (English, N=1,000)** | Macro-F1 / Accuracy | 84.80% / 84.80% | `results/sentiment_baseline_metrics.json` |
| **Cross-Lingual Fairness Gap** | Macro-F1 Gap (En - Hi) | **0.0683** (95% CI: `[-0.003, 0.151]`) | `results/fairness_en_vs_hi.json` |
| **Neural MT (MarianMT)** | BLEU / chrF | **29.84 / 52.93** | `results/mt_nmt.json` |
| **Statistical MT (IBM Model 1)** | BLEU / chrF | 4.68 / 32.56 | `results/mt_smt.json` |
| **Rule-Based MT** | BLEU / chrF | 1.88 / 22.65 | `results/mt_rule_based.json` |
| **spaCy Named Entity Recognition** | Strict Macro-F1 | **64.90%** (P: 74.24%, R: 57.65%) | `results/ner_eval.json` |
| **VaaniBot Dialogue System** | Intent Test Accuracy | **90.9%** (40/44 correct) | `results/chatbot_eval.json` |

---

## 📁 Repository Layout

```
VaaniLens/
├── README.md                      # Hugging Face Spaces metadata & documentation
├── MODEL_CARD.md                  # Mitchell et al. Model Card
├── PROJECT.md                     # Master project specification & task tracking
├── AGENTS.md                      # Agent protocol reference
├── requirements.txt               # Top-level dependencies
├── requirements.lock.txt          # Exact locked dependencies
├── pytest.ini                     # Pytest configuration
├── app/
│   └── streamlit_app.py           # Multi-tab interactive Streamlit dashboard
├── src/
│   ├── config.py                  # Thresholds (CONF_THRESHOLD=0.75, LIME_SAMPLES=200)
│   ├── pipeline.py                # analyze() orchestrator with safety gates
│   ├── sentiment.py               # Sentiment contract wrapper (lazy-loading)
│   ├── sentiment_baseline.py      # TF-IDF + Logistic Regression trainer
│   ├── sentiment_transformer.py   # DistilBERT SST-2 evaluator
│   ├── translate.py               # Devanagari detection & MarianMT translator
│   ├── ner.py                     # spaCy entity extractor & evaluator
│   ├── explain.py                 # LIME local explainability module
│   ├── chatbot.py                 # VaaniBot intent dialog system
│   ├── mt_rule_based.py           # Rule-based MT with SOV->SVO reordering
│   ├── mt_smt.py                  # IBM Model 1 statistical MT
│   ├── mt_evaluation.py           # SacreBLEU & chrF benchmark runner
│   ├── data_prep.py               # Data streamer & curator
│   └── bias/
│       ├── sentiment_bias.py      # Demographic identity sentiment audit
│       ├── translation_bias.py    # Gender translation bias audit
│       └── fairness_en_hi.py      # English vs Hindi fairness metric with bootstrap CI
├── data/                          # Diagnostic datasets (all < 5 MB)
├── models/                        # Serialized models (baseline.joblib, smt_ibm1.pkl)
├── results/                       # Traceable JSON metrics, CSV logs, PNG plots
├── reports/
│   ├── REPORT.md                  # Comprehensive academic synthesis report
│   ├── ETHICS_AUDIT.md            # Detailed ethics, bias & fairness audit
│   ├── slides.md                  # 10 Marp presentation slides
│   └── fragments/                 # Component report fragments
├── scripts/                       # Verification scripts (env_check, integration test)
└── tests/                         # Pytest unit & integration test suite
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/Aria-vis/VaaniLens.git
cd VaaniLens

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run All Tests
```bash
pytest -q
```

### 3. Launch Interactive Streamlit App
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 🛡️ Ethics, Privacy & Safety Disclosures

- **Data Privacy**: VaaniLens executes purely in memory. No customer reviews, user messages, or personal texts are ever logged, serialized to disk, or sent across the network.
- **Human Oversight**: Any prediction where model confidence is below $0.75$ is automatically flagged with an amber banner (`needs_human_review = True`), ensuring human-in-the-loop oversight.
- **Representational Harm Mitigation**: MarianMT's tendency to invent male or female pronouns from gender-neutral Hindi is explicitly detected and surfaced to the operator.

---

## 📜 License & Citation

Distributed under the MIT License. Developed for Module V: NLP Applications and Ethics.
