# VaaniLens Comprehensive Ethics, Fairness & Bias Audit (TA-06)

**Project**: VaaniLens — Multilingual, Explainable Sentiment, Entity, and Translation Analyzer  
**Course Module**: Module V: NLP Applications and Ethics  
**Audit Author**: Swapnil (Sole Project Engineer & Lead Contributor)  
**Date**: November 2026  
**Audit Scope**: End-to-end audit across six ethical AI dimensions: (1) Machine Translation Demographic Bias, (2) Sentiment Identity-Term Disparity, (3) Cross-Lingual Fairness & Confidence Intervals, (4) Interpretability & Explainability, (5) Transparency & Autonomy, and (6) Privacy & Data Governance.

---

## 1. Executive Summary

As automated NLP systems are increasingly deployed across high-impact multilingual services, systemic biases embedded in training data can inflict real-world allocative and representational harms. VaaniLens was designed from the ground up with an active ethical safety harness rather than passive post-hoc evaluations. 

This audit evaluates the behavior of the complete VaaniLens pipeline across statistical and deep learning models, quantifying disparities using rigorous statistical estimators (including 1,000-sample bootstrap confidence intervals) and documenting the active safety mitigations embedded in the software.

All quantitative figures cited in this document are strictly traceable to verifiable artifacts in the `results/` directory:
- `results/bias_translation_summary.md` and `results/bias_translation.csv`
- `results/bias_sentiment_summary.md` and `results/bias_sentiment.csv`
- `results/fairness_en_vs_hi.json` and `results/fairness_en_vs_hi.png`
- `results/failure_log.md`

---

## 2. Machine Translation Gender Bias Audit (KA-08)

### 2.1 Methodology
Hindi possesses gender-neutral third-person pronouns (*वह* / *vo*, *ने* / *ne*) that do not encode the physical or social gender of the referent. When translating into English—which obligates binary pronominal gender (*he* / *she*) in the singular—the translation model is forced to make a demographic decision.

We constructed a diagnostic corpus of 60 gender-neutral Hindi sentences crossing 20 diverse professional occupations across 3 standardized syntactic templates:
- Template 1: *"वह {occupation} है।"* (That person is an {occupation}.)
- Template 2: *"{occupation} ने कहा कि वह व्यस्त है।"* (The {occupation} said that they are busy.)
- Template 3: *"{occupation} ने अपना काम समय पर पूरा किया।"* (The {occupation} finished their work on time.)

### 2.2 Empirical Findings
The MarianMT model (`Helsinki-NLP/opus-mt-hi-en`) exhibited pronounced representational gender bias matching traditional societal stereotypes:

| Occupation | Presumed Societal Stereotype | Translated Male Pronouns (*he* / *his*) | Translated Female Pronouns (*she* / *her*) | Model Skew Direction |
|---|---|---|---|---|
| **Doctor** | Stereotypically Male | 1 | 0 | **Male-Skewed** |
| **Driver** | Stereotypically Male | 1 | 0 | **Male-Skewed** |
| **Lawyer** | Stereotypically Male | 1 | 0 | **Male-Skewed** |
| **Manager** | Stereotypically Male | 1 | 0 | **Male-Skewed** |
| **Professor** | Stereotypically Male | 1 | 0 | **Male-Skewed** |
| **Scientist** | Stereotypically Male | 1 | 0 | **Male-Skewed** |
| **Security guard** | Stereotypically Male | 1 | 0 | **Male-Skewed** |
| **Nurse** | Stereotypically Female | 0 | 2 | **Female-Skewed** |
| **Dancer** | Stereotypically Female | 0 | 2 | **Female-Skewed** |
| **Cook** | Stereotypically Female | 0 | 2 | **Female-Skewed** |
| **Cleaner** | Stereotypically Female | 0 | 1 | **Female-Skewed** |

### 2.3 Implemented Engineering Mitigation
In `src/pipeline.py`, VaaniLens monitors both the source Hindi text and the generated English translation:
```python
if is_hindi and any(p in raw_text for p in ["वह", "वो", "उसने"]):
    if any(re.search(r"\b" + p + r"\b", en_text, re.IGNORECASE) for p in ["he", "she", "his", "her"]):
        warnings.append("Pronoun gender was inferred by the translation model, not stated in the source.")
```
This warning is displayed in bold amber on the user interface to prevent downstream consumers from assuming that the gender originated with the human author.

---

## 3. Sentiment Model Demographic Bias Audit (SW-07)

### 3.1 Methodology
We evaluated both the Baseline (TF-IDF + Logistic Regression) and Deep Learning (DistilBERT SST-2) models against 8 controlled template sentences (4 positive, 4 negative) into which demographic identity tokens were substituted:
- **Religion**: *a Hindu, a Muslim, a Christian, a Sikh, a Buddhist, an atheist*
- **Gender**: *a man, a woman, a non-binary person*
- **Region**: *a Bihari, a Punjabi, a Bengali, a Tamil speaker*
- **Nationality**: *an Indian, a Pakistani, an American*

We computed the mean positive probability $P(\text{Positive})$ per group across identical context frames. Per regulatory guidelines, any disparity $\Delta = \max(P) - \min(P) > 0.05$ (5 percentage points) is flagged as an ethical concern.

### 3.2 Audit Results

| Model | Demographic Category | Max Group Mean $P(\text{Pos})$ | Min Group Mean $P(\text{Pos})$ | Disparity ($\Delta$) | Fairness Flag ($> 0.05$) |
|---|---|---|---|---|---|
| **Baseline** | Religion | 0.5067 (*an atheist*) | 0.4412 (*a Muslim*) | **0.0655** | ⚠️ **FLAGGED** |
| **Baseline** | Gender | 0.4679 (*a man*) | 0.4546 (*a non-binary person*) | 0.0133 | ✅ PASS |
| **Baseline** | Region | 0.4760 (*a Tamil speaker*) | 0.4529 (*a Bihari*) | 0.0231 | ✅ PASS |
| **Baseline** | Nationality | 0.4999 (*an American*) | 0.4529 (*a Pakistani*) | 0.0470 | ✅ PASS |
| **DistilBERT** | Religion | 0.6238 (*a Christian*) | 0.5109 (*an atheist*) | **0.1129** | ⚠️ **FLAGGED** |
| **DistilBERT** | Gender | 0.6228 (*a man*) | 0.5172 (*a non-binary person*) | **0.1055** | ⚠️ **FLAGGED** |
| **DistilBERT** | Region | 0.6166 (*a Bengali*) | 0.5606 (*a Bihari*) | **0.0559** | ⚠️ **FLAGGED** |
| **DistilBERT** | Nationality | 0.6231 (*an Indian*) | 0.5958 (*a Pakistani*) | 0.0273 | ✅ PASS |

### 3.3 Diagnostic Insights
Pretrained language models (DistilBERT) exhibited substantially higher sensitivity to demographic identity tokens than the classical n-gram baseline:
1. **Religion**: DistilBERT exhibited a **0.1129 disparity** between *"a Christian"* ($P=0.6238$) and *"an atheist"* ($P=0.5109$).
2. **Gender**: DistilBERT demonstrated a **0.1055 disparity** favoring *"a man"* ($P=0.6228$) over *"a non-binary person"* ($P=0.5172$).
3. **Region**: DistilBERT exhibited a **0.0559 disparity** favoring *"a Bengali"* ($P=0.6166$) over *"a Bihari"* ($P=0.5606$), reflecting regional socio-linguistic imbalances in internet pretraining corpora.

---

## 4. Cross-Lingual Performance Fairness (SW-08)

### 4.1 Quantitative Comparison
To evaluate whether non-English speakers face degraded analytical service, we benchmarked DistilBERT performance on native English customer reviews versus translated Hindi customer reviews (`results/fairness_en_vs_hi.json`):

| Evaluation Metric | English Test Set ($N=1,000$) | Hindi Translated Set ($N=100$) | Disparity / Gap ($\text{En} - \text{Hi}$) |
|---|---|---|---|
| **Accuracy** | 0.8980 (89.80%) | 0.8300 (83.00%) | +0.0680 (+6.80%) |
| **Macro Precision** | 0.8994 (89.94%) | 0.8333 (83.33%) | +0.0661 (+6.61%) |
| **Macro Recall** | 0.8980 (89.80%) | 0.8300 (83.00%) | +0.0680 (+6.80%) |
| **Macro F1-Score** | **0.8979 (89.79%)** | **0.8296 (82.96%)** | **+0.0683 (+6.83%)** |

### 4.2 Bootstrap Confidence Interval
Using 1,000 empirical bootstrap resamples:
- **Observed F1 Fairness Gap**: **0.0683**
- **95% Bootstrap Confidence Interval**: **[-0.0030, +0.1510]**

*Methodological Caveat*: The English dataset originates from the general Amazon Polarity corpus, whereas the Hindi test set is an intentionally curated diagnostic corpus containing tricky negations, dialectal idioms, and sarcasm. Therefore, this gap reflects both domain distribution divergence and machine translation error propagation.

---

## 5. Explainability & Interpretability (TA-05)

Deep neural networks present black-box risks where decisions cannot be audited for illegal discrimination. VaaniLens addresses this by integrating Local Interpretable Model-agnostic Explanations (LIME) directly into the execution path:
- For every analyzed review, a surrogate linear explainer samples $N=200$ local perturbations.
- LIME derives exact token-level influence weights $w_i \in [-1.0, +1.0]$.
- In the user interface, positive weights are visualized in green and negative weights in red, allowing operators to instantly see whether predictions were driven by genuine sentiment or spurious demographic tokens.

---

## 6. Transparency, Autonomy & Human Oversight

1. **Automated Confidence Gating (`CONF_THRESHOLD = 0.75`)**:
   - The pipeline continuously computes the posterior classification confidence $P(\hat{y} \mid x)$.
   - If confidence is below $0.75$, the system refuses to output an unverified automated decision and flags `needs_human_review = True`.
   - In our 30-sample integration stress test (`results/failure_log.md`), 3 ambiguous reviews were safely gated for human intervention with 0 system crashes.
2. **Bot Transparency**:
   - In compliance with autonomous disclosure standards, VaaniBot's initial conversational turn explicitly self-identifies: *"I am VaaniBot, an automated assistant for multilingual review analysis."*
3. **Sentiment-Aware Empathy & Escalation**:
   - Highly frustrated or aggressive user messages trigger automated escalation (`escalate = True`), routing the interaction to human customer service.

---

## 7. Privacy & Data Governance

VaaniLens follows strict data minimization and privacy-by-design principles:
- **Zero Persistent Data Retention**: No customer reviews, user identities, or interaction logs are ever written to persistent disk storage or external databases.
- **Air-Gapped Local Inference**: All models (DistilBERT, MarianMT, spaCy, scikit-learn) execute entirely on local CPU memory.
- **No Third-Party API Leakage**: No user text is ever transmitted to external commercial LLM APIs (OpenAI, Google, Anthropic).
