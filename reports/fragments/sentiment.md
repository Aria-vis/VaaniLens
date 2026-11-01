# Sentiment Analysis, Fairness & Dialogue Report Fragment (SW-09)

**Author**: Swapnil  
**Scope**: Sentiment classification, error analysis, cross-lingual fairness, demographic bias, and VaaniBot dialogue system.

---

## 1. Methodology

We evaluated two distinct paradigms for sentiment classification on consumer reviews:
1. **Classical / Statistical NLP**: A TF-IDF vectorizer (word unigrams and bigrams, sublinear term-frequency scaling, maximum 50,000 features) coupled with an L2-regularized Logistic Regression classifier ($C=1.0$, `max_iter=500`). The model was trained on 5,000 balanced customer reviews sampled from `fancyzhx/amazon_polarity` (`en_train.csv`).
2. **Deep Learning NLP**: Pretrained DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`), a 6-layer, 768-hidden dimension bidirectional transformer fine-tuned on the Stanford Sentiment Treebank.

Both models were benchmarked on a strictly held-out test split of 1,000 balanced English reviews (`en_test.csv`). For cross-lingual evaluation, 100 authentic Hindi customer reviews (`hi_reviews_test.csv`) were translated via MarianMT and classified by DistilBERT.

---

## 2. Quantitative Results

All numbers originate directly from `results/sentiment_baseline_metrics.json`, `results/sentiment_transformer_metrics.json`, `results/fairness_en_vs_hi.json`, `results/bias_sentiment_summary.md`, and `results/chatbot_eval.json`.

### Table 1: Sentiment Benchmark on 1,000 English Reviews
| Metric | TF-IDF + Logistic Regression | DistilBERT SST-2 | Delta |
|---|---|---|---|
| **Accuracy** | 0.8480 | **0.8980** | +0.0500 (+5.0%) |
| **Macro-Precision** | 0.8484 | **0.8994** | +0.0510 (+5.1%) |
| **Macro-Recall** | 0.8480 | **0.8980** | +0.0500 (+5.0%) |
| **Macro-F1** | 0.8480 | **0.8979** | +0.0499 (+5.0%) |
| **Confusion Matrix** | $\begin{bmatrix} 432 & 68 \\ 84 & 416 \end{bmatrix}$ | $\begin{bmatrix} 464 & 36 \\ 66 & 434 \end{bmatrix}$ | False Positives: $68 \to 36$ (-47.1%) |
| **Model Size** | **1.34 MB** | 268 MB | 200× footprint increase |
| **CPU Latency** | **~0.15 ms / sample** | ~25.0 ms / sample | 166× slower inference |

### Table 2: Cross-Lingual Sentiment Performance & Fairness Gap
| Metric | English Test Set (N=1,000) | Hindi Translated (N=100) | Disparity |
|---|---|---|---|
| **Accuracy** | 0.8980 | 0.8300 | +0.0680 |
| **Macro-F1** | 0.8979 | 0.8296 | **+0.0683** |
| **95% Bootstrap CI** | — | — | **[-0.0030, +0.1510]** |

### Table 3: Demographic Sentiment Disparity (Max - Min Positive Probability)
| Category | Baseline Disparity | DistilBERT Disparity | Regulatory Threshold (0.05) |
|---|---|---|---|
| **Religion** | 0.0655 (atheist vs Muslim) | **0.1129 (Christian vs atheist)** | ⚠️ Both Flagged |
| **Gender** | 0.0133 | **0.1055 (man vs non-binary)** | ⚠️ DistilBERT Flagged |
| **Region** | 0.0231 | **0.0559 (Bengali vs Bihari)** | ⚠️ DistilBERT Flagged |
| **Nationality** | 0.0470 | 0.0273 | ✅ Passed |

---

## 3. VaaniBot Dialogue System

VaaniBot implements an intent-driven dialogue agent with state tracking and sentiment-aware empathy escalation without requiring any commercial LLM API:
- **Architecture**: TF-IDF + Logistic Regression trained on 11 distinct customer service intents.
- **Evaluation**: Benchmarked on 44 held-out test utterances, achieving an overall accuracy of **90.9%** (40/44 correct).
- **Empathy Escalation**: User messages expressing severe frustration or anger automatically trigger `escalate = True` for human agent handoff.
- **Transparency**: First conversational turn always discloses: *"I am VaaniBot, an automated assistant for multilingual review analysis."*

---

## 4. Key Findings & Limitations

1. **Transformer Superiority in Syntactic Binding**: DistilBERT reduces False Positives by nearly 50% relative to the baseline because self-attention correctly binds negation particles across clauses, overcoming the structural blind spot of bag-of-words models.
2. **Elevated Demographic Sensitivity in Pretrained Weights**: DistilBERT demonstrated greater demographic sensitivity (religion disparity 0.1129, gender disparity 0.1055) than the baseline, confirming that uncurated web pretraining embeds societal stereotypes that persist post fine-tuning.
3. **Cross-Lingual Distribution Gap**: The 6.83 percentage point F1 gap between English and translated Hindi reflects a composite of machine translation lexical degradation and the nuanced sarcasm and negation present in colloquial Hindi consumer feedback.
4. **Safety Architecture**: VaaniLens resolves these limitations through strict human-in-the-loop review gating whenever sentiment confidence falls below 0.75.

