# VaaniLens Model Card (TA-06)

Following the standard Model Cards for Model Reporting framework (Mitchell et al., 2019).

---

## 1. Model Details

- **Model Name**: VaaniLens Multilingual Sentiment & Translation Pipeline
- **Pipeline Architecture**:
  - **Language Detection**: Rule-based Devanagari Unicode ratio thresholding ($\ge 30\%$ alphabetic characters).
  - **Machine Translation**: MarianMT (`Helsinki-NLP/opus-mt-hi-en`), Seq2Seq Transformer, 74M parameters, 304 MB checkpoint.
  - **Named Entity Recognition**: spaCy `en_core_web_sm` (v3.8.0), Transition-based CNN parser.
  - **Sentiment Analysis (Primary)**: DistilBERT SST-2 (`distilbert-base-uncased-finetuned-sst-2-english`), 6-layer Transformer, 66M parameters, 268 MB checkpoint.
  - **Sentiment Analysis (Baseline Benchmark)**: TF-IDF (1-2 grams, 50,000 features) + Logistic Regression ($C=1.0$, L2 penalty).
  - **Explainability**: Local Interpretable Model-agnostic Explanations (LIME), $N=200$ local perturbations.
  - **Dialogue System**: VaaniBot, TF-IDF + Logistic Regression intent classifier across 11 intents with empathy escalation.
- **Developers**: Swapnil (Lead Developer & Engineer)
- **License**: Apache 2.0 / MIT Academic Open Source
- **Release Date**: November 2026

---

## 2. Intended Use

- **Primary Intended Uses**:
  - Automated triage and sentiment analysis of consumer feedback and product reviews in English and Hindi.
  - Interpretability auditing via token-level attribution to understand why automated decisions are made.
  - Cross-paradigm machine translation comparison for academic evaluation (Rule-Based vs Statistical vs Neural MT).
- **Out-of-Scope / Prohibited Uses**:
  - Automated employment screening, credit evaluation, criminal justice risk assessment, or legal document interpretation.
  - High-stakes clinical decision-making or medical triage.
  - Unsupervised punitive action against employees or customers based solely on sentiment scores.

---

## 3. Factors & Domains

- **Languages**: English (`en`), Hindi (`hi`, Devanagari script).
- **Domain**: E-commerce product reviews, consumer electronics, consumer services, and general customer feedback.
- **Demographic Factors Audited**: Religion (6 terms), Gender (3 terms), Region (4 terms), Nationality (3 terms).

---

## 4. Evaluation Metrics

- **Sentiment Classification**: Accuracy, Macro-Precision, Macro-Recall, Macro-F1 Score, Confusion Matrix.
- **Machine Translation**: sacreBLEU (corpus BLEU), chrF (character n-gram F-score).
- **Named Entity Recognition**: Strict exact-span Precision, Recall, Macro-F1.
- **Fairness & Bias**: Demographic disparity ($\max(P) - \min(P)$ on positive sentiment), cross-lingual F1 gap with 1,000-sample bootstrap 95% confidence interval.

---

## 5. Training Data

1. **Baseline Sentiment Model**:
   - 5,000 balanced customer reviews sampled from `fancyzhx/amazon_polarity` (Train split), stratified 50% positive and 50% negative.
2. **DistilBERT SST-2**:
   - Stanford Sentiment Treebank (SST-2), binary sentence classification.
3. **MarianMT (`opus-mt-hi-en`)**:
   - OPUS multilingual parallel corpora including sub-corpora from TED talks, Ubuntu, and Wikipedia.
4. **IBM Model 1 SMT**:
   - 500 parallel sentence pairs trained via Expectation-Maximization (EM, 5 iterations) for lexical alignment.

---

## 6. Evaluation Data

1. **English Sentiment Test Set**: 1,000 balanced held-out reviews from `fancyzhx/amazon_polarity` (Test split).
2. **Hindi Sentiment Test Set**: 100 curated authentic reviews (`data/hi_reviews_test.csv`) covering complex negation, colloquial phrasing, and sarcasm.
3. **Machine Translation Benchmark Set**: 50 parallel Hindi-English test pairs (`data/hi_en_test_manual.csv`) covering diverse sentence lengths and idioms.
4. **NER Gold Test Set**: 40 annotated sentences (`data/ner_gold.json`) spanning ORG, PERSON, GPE, DATE, and PRODUCT.
5. **Chatbot Held-out Test Set**: 44 held-out natural language queries across 11 conversational intents.

---

## 7. Quantitative Analyses

### 7.1 Sentiment Performance Comparison (N=1,000)

| Model | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | False Positives | Inference Latency |
|---|---|---|---|---|---|---|
| **TF-IDF + Logistic Regression** | 0.8480 | 0.8484 | 0.8480 | 0.8480 | 68 | **0.15 ms / sample** |
| **DistilBERT SST-2** | **0.8980** | **0.8994** | **0.8980** | **0.8979** | **36** | 25.0 ms / sample |

### 7.2 Machine Translation Paradigm Showdown (N=50)

| Paradigm | Architecture | BLEU Score | chrF Score | Inference Speed |
|---|---|---|---|---|
| **Rule-Based MT** | Dict lookup (220+ words) + SOV $\to$ SVO rules | 1.88 | 22.65 | **0.02 ms / sent** |
| **Statistical MT** | IBM Model 1 EM Lexical Alignment + Monotone Decoding | 4.68 | 32.56 | 0.05 ms / sent |
| **Neural MT** | MarianMT 6-layer Transformer | **29.84** | **52.93** | 680 ms / sent |

### 7.3 Named Entity Recognition Evaluation (N=40)

| Metric | Precision | Recall | F1-Score |
|---|---|---|---|
| **spaCy `en_core_web_sm`** | 0.7424 (74.24%) | 0.5765 (57.65%) | 0.6490 (64.90%) |

### 7.4 Cross-Lingual Fairness (English vs Translated Hindi)

- **English Macro-F1**: 0.8979
- **Hindi Translated Macro-F1**: 0.8296
- **F1 Fairness Gap**: **0.0683** (95% Bootstrap CI: **[-0.0030, +0.1510]**)

---

## 8. Ethical Considerations & Audit Highlights

1. **Gender Skew in Machine Translation**:
   - Neural MT systematically infers male pronouns (*he/his*) for traditionally male-dominated professions (Doctor, Driver, Lawyer, Professor, Scientist) and female pronouns (*she/her*) for Nurse, Dancer, Cook, Cleaner, despite gender-neutral Hindi source inputs (*वह* / *ne*).
2. **Demographic Sentiment Sensitivity**:
   - DistilBERT exhibits elevated sentiment disparity on Religion ($\Delta = 0.1129$) and Gender ($\Delta = 0.1055$) when identical review templates are evaluated.

---

## 9. Implemented Safeguards & Caveats

1. **Human Oversight Gate**: Reviews yielding sentiment confidence $< 0.75$ or failing stages automatically set `needs_human_review = True`.
2. **Translation Gender Warning**: If gender-neutral Hindi pronouns are translated into gendered English pronouns, a prominent warning is attached to the output.
3. **Data Privacy**: Purely in-memory evaluation; zero logging of user inputs or personal texts to disk.
4. **Local Execution**: All models run strictly offline on CPU with zero cloud API dependencies.
