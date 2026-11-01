---
marp: true
theme: gaia
_class: lead
paginate: true
backgroundColor: #f5f6fa
color: #2f3640
style: |
  section {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    padding: 40px;
  }
  h1 {
    color: #273c75;
  }
  h2 {
    color: #192a56;
  }
  table {
    font-size: 20px;
  }
  footer {
    font-size: 14px;
    color: #718093;
  }
---

<!-- Slide 1 -->
# 🔍 VaaniLens
### Multilingual, Explainable Sentiment, Entity, and Translation Analyzer
**Module V: NLP Applications and Ethics**

**Author & Lead Engineer**: Swapnil  
*GitHub*: [github.com/Aria-vis/VaaniLens](https://github.com/Aria-vis/VaaniLens)  
*All figures verified against `results/` artifacts. Rs 0 budget, 100% CPU.*

---

<!-- Slide 2 -->
## 🎯 Motivation & Core Constraints

- **The Challenge**: Indian e-commerce feedback is multilingual (Hindi & English), colloquial, and high-volume.
- **Why Cloud APIs Fail**:
  - 💸 **Cost**: Linear API fees make bulk triage expensive.
  - 🔒 **Privacy**: Sending raw user complaints to external APIs violates data sovereignty.
  - 🙈 **Black-Box**: Zero local interpretability for human review teams.
  - ⚖️ **Bias**: Pretrained cloud models silently propagate demographic biases.
- **The VaaniLens Mandate**:
  - **₹0 / $0 Budget**: Zero paid APIs.
  - **100% Local CPU**: DistilBERT, MarianMT, spaCy, and scikit-learn.
  - **Active Safety**: Pronoun inference warnings & confidence gating ($<0.75$).

---

<!-- Slide 3 -->
## 🏗️ System Pipeline Architecture

```
Raw Text (Hindi / English)
   │
   ▼
Devanagari Unicode Ratio (>= 30%) ──[Hindi]──► MarianMT NMT
   │                                                 │
   ├───────────[English]─────────────────────────────┤
   ▼                                                 ▼
Sentiment Classification                   Gender Pronoun Audit
(DistilBERT SST-2 vs Baseline)             (Source 'वह' -> Output 'he'/'she')
   │
   ├──────────────────────────────┬──────────────────────────────┐
   ▼                              ▼                              ▼
spaCy NER                     LIME Explainer                Confidence Gate
(ORG, GPE, DATE)             (N=200 Perturbations)       (Conf < 0.75 -> Flag)
   │                              │                              │
   └──────────────────────────────┴──────────────────────────────┘
                                  ▼
             Streamlit Dashboard & VaaniBot Dialog System
```

---

<!-- Slide 4 -->
## 📊 Sentiment Analysis: Baseline vs DistilBERT

Evaluated on 1,000 held-out reviews from `amazon_polarity`:

| Metric | Baseline (TF-IDF + LogReg) | DistilBERT SST-2 (Transformer) | Delta |
|---|---|---|---|
| **Accuracy** | 0.8480 (84.8%) | **0.8980 (89.8%)** | **+5.0%** |
| **Macro Precision** | 0.8484 | **0.8994** | +5.1% |
| **Macro Recall** | 0.8480 | **0.8980** | +5.0% |
| **Macro-F1** | 0.8480 | **0.8979** | **+5.0%** |
| **False Positives** | 68 | **36** | **-47.1% (Halved!)** |
| **Model Size** | **1.34 MB** | 268 MB | 200× increase |
| **CPU Latency** | **~0.15 ms / sample** | ~25.0 ms / sample | 166× slower |

*Key Insight: DistilBERT resolves negation and contrastive discourse markers.*

---

<!-- Slide 5 -->
## 🔍 Error Analysis: 4 Linguistic Failure Modes

Audit of 35 genuine misclassifications (`results/error_analysis.md`):

1. **Mixed Sentiment (37.1%)**: Competing clauses (*"Loved the book, hated the movie"*).
2. **Negation Scope (28.6%)**: Baseline bag-of-words misses long-distance negation (*"not worth the $2.99"*).
3. **Sarcasm & Irony (20.0%)**: Positive surface words masking critical intent (*"phone turns into a heater"*).
4. **Structural Length (14.3%)**: Truncation past 256 tokens or ultra-terse fragments (*"meh."*).

**Mitigation**: Confidence thresholding ($P < 0.75$) safely flags ambiguous reviews for human intervention.

---

<!-- Slide 6 -->
## 🌐 Machine Translation Paradigm Showdown

Benchmarked on 50 parallel Hindi-English sentence pairs (`results/mt_comparison.md`):

| Paradigm | Architecture | BLEU Score | chrF Score | CPU Latency | Size |
|---|---|---|---|---|---|
| **Rule-Based MT** | Dict (220+ words) + SOV $\to$ SVO rules | 1.88 | 22.65 | **0.02 ms** | **15 KB** |
| **Statistical MT** | IBM Model 1 EM Lexical Alignment | 4.68 | 32.56 | 0.05 ms | 45 KB |
| **Neural MT** | MarianMT Seq2Seq Transformer | **29.84** | **52.93** | 680 ms | 304 MB |

- **Rule-Based**: Rigid rules collapse when encountering unseen domain vocabulary.
- **Statistical**: Learns word alignments, but monotone decoding produces word salads.
- **Neural MT**: Deep self-attention achieves fluent, compositional translations.

---

<!-- Slide 7 -->
## 🏷️ Named Entity Recognition (spaCy)

Evaluated on 40 gold-annotated consumer review sentences (`results/ner_eval.json`):

- **Strict Precision**: **0.7424 (74.24%)**
- **Strict Recall**: **0.5765 (57.65%)**
- **Strict Macro-F1**: **0.6490 (64.90%)**
- **Sample Distribution**: 49 True Positives, 17 False Positives, 36 False Negatives.

**Key Findings**:
- ✅ Strong detection of geographic entities (GPE: *Patna*, *Mumbai*) and corporate brands (ORG: *Amazon*, *Dell*).
- ⚠️ Vulnerability on compound product names (*Redmi Note*, *Kindle Paperwhite*), frequently split or misclassified.

---

<!-- Slide 8 -->
## ⚖️ Ethical AI: Fairness & Bias Audits

1. **Machine Translation Gender Bias (60 Sentences, 20 Occupations)**:
   - Neutral Hindi (*वह डॉक्टर है*) $\to$ Male English (*He is a doctor*).
   - Neutral Hindi (*वह नर्स है*) $\to$ Female English (*She is a nurse*).
   - **Mitigation**: Active alert when pronoun gender is inferred, not stated.
2. **Sentiment Demographic Bias (16 Terms across 4 Categories)**:
   - DistilBERT Disparity: Religion = **0.1129**, Gender = **0.1055**, Region = **0.0559**.
   - Proves pretrained models absorb societal stereotypes from uncurated web data.
3. **Cross-Lingual Fairness (English vs Translated Hindi)**:
   - English F1: 0.8979 vs Hindi Translated F1: 0.8296.
   - **F1 Gap = 0.0683** (95% Bootstrap CI: **[-0.0030, +0.1510]**).

---

<!-- Slide 9 -->
## 🤖 VaaniBot & Active Safety Harness

- **VaaniBot Architecture**:
  - TF-IDF + Logistic Regression across 11 customer service intents.
  - **90.9% Accuracy** (40/44 correct on strictly held-out test queries).
  - Empathy & Escalation: Frustrated messages trigger `escalate = True`.
  - Transparency: Always identifies as an automated assistant on turn 1.
- **Operational Safety Gates**:
  - 🛡️ **Confidence Review Gate**: $P < 0.75 \to$ `needs_human_review = True`.
  - 🔒 **Zero-Retention Privacy**: Purely in-memory evaluation; zero logging to disk.
  - 💡 **LIME Interpretability**: Token-level feature attribution highlights green/red weights.

---

<!-- Slide 10 -->
## 🏁 Conclusion & Key Lessons Learned

1. **Pretrained Transformers Win on Semantics, But Need Safety Rails**:
   - DistilBERT halves False Positives (+5.0% F1), but exhibits higher demographic sensitivity than the classical baseline.
2. **Translation is Not Ethically Neutral**:
   - Neural MT actively introduces societal gender stereotypes from gender-neutral languages; automated pronoun detection is mandatory.
3. **Production NLP on a ₹0 Budget is Achievable**:
   - High-accuracy multilingual triage, explainability, and dialog systems can run completely offline on local CPU hardware with 0 API costs.

**Repository**: [github.com/Aria-vis/VaaniLens](https://github.com/Aria-vis/VaaniLens)

