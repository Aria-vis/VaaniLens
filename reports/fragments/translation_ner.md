# Machine Translation & Named Entity Recognition Report Fragment (KA-09)

**Author**: Kaushik (executed by Swapnil)  
**Scope**: Machine translation paradigm benchmark (Rule-Based vs SMT vs NMT), translation gender bias audit, and spaCy Named Entity Recognition evaluation.

---

## 1. Machine Translation Paradigm Benchmark

We implemented and compared three generations of Machine Translation technology on an identical test benchmark of 50 parallel Hindi-English sentence pairs (`data/hi_en_test_manual.csv`):
1. **Rule-Based Machine Translation (RBMT)**: Bilingual dictionary lookup (220+ curated vocabulary items, `data/hi_en_dict.json`) with hand-crafted syntactic transformation rules: SOV to SVO reordering, Hindi postposition inversion (*का/की/के* $\to$ *"of"*), negation particle preservation (*नहीं* $\to$ *"not"*), and unknown word bracket retention.
2. **Statistical Machine Translation (Mini-SMT)**: IBM Model 1 lexical translation table trained via 5 iterations of Expectation-Maximization (EM) over 500 parallel sentence pairs (`models/smt_ibm1.pkl`) with monotone decoding.
3. **Neural Machine Translation (NMT)**: MarianMT (`Helsinki-NLP/opus-mt-hi-en`), a 6-layer sequence-to-sequence Transformer with 74M parameters using beam search decoding (`num_beams=4`).

### Quantitative MT Evaluation

All scores are strictly traceable to `results/mt_rule_based.json`, `results/mt_smt.json`, `results/mt_nmt.json`, and `results/mt_comparison.md`.

| Paradigm | Architecture | BLEU Score | chrF Score | Inference Speed | Model Size |
|---|---|---|---|---|---|
| **Rule-Based MT** | Dictionary + Syntactic Reordering | 1.88 | 22.65 | **0.02 ms / sent** | **15 KB** |
| **Statistical MT** | IBM Model 1 EM Alignment | 4.68 | 32.56 | 0.05 ms / sent | 45 KB |
| **Neural MT** | MarianMT Seq2Seq Transformer | **29.84** | **52.93** | 680 ms / sent | 304 MB |

### Paradigm Analysis:
- **Rule-Based MT (BLEU 1.88)**: Hand-crafted transfer rules maintain rigid structure but suffer drastic vocabulary attrition whenever domain words fall outside the static dictionary.
- **Mini-SMT (BLEU 4.68)**: IBM Model 1 successfully learns statistical word alignments (e.g. *उत्पाद* $\to$ *product*), but without a phrase table or target language model, output sequences suffer from monotone word-for-word word order distortion.
- **Neural MT (BLEU 29.84)**: MarianMT achieves human-readable fluency, capturing morphological variations, compound verb groups, and idiomatic phrasing, at the cost of higher latency (680 ms per sentence on CPU).

---

## 2. Gender Translation Bias Audit

To assess representational harm in translation, we constructed 60 gender-neutral Hindi sentences crossing 20 occupations across 3 syntactic frames avoiding gender-marking verb forms.

### Empirical Findings:
MarianMT exhibited systematic gender skew aligned with historical societal stereotypes (`results/bias_translation_summary.md`):
- **Male-Stereotyped Professions**: *Doctor*, *Driver*, *Lawyer*, *Manager*, *Professor*, *Scientist*, and *Security guard* were consistently assigned male pronouns (*he/his*).
- **Female-Stereotyped Professions**: *Nurse*, *Dancer*, *Cook*, and *Cleaner* were assigned female pronouns (*she/her*).
- **Active Safety Harness**: Whenever the source text uses gender-neutral Hindi pronouns (*वह*, *वो*) and the translation introduces gendered English pronouns, VaaniLens automatically attaches the prominent alert:  
  > *"Pronoun gender was inferred by the translation model, not stated in the source."*

---

## 3. Named Entity Recognition Evaluation

We evaluated spaCy's `en_core_web_sm` model on a gold-standard diagnostic dataset of 40 annotated consumer review sentences (`data/ner_gold.json`), containing 85 labeled entity spans across ORG, PRODUCT, GPE, PERSON, and DATE.

### Quantitative NER Performance:
All figures originate from `results/ner_eval.json`:
- **True Positives**: 49
- **False Positives**: 17
- **False Negatives**: 36
- **Strict Precision**: **0.7424 (74.24%)**
- **Strict Recall**: **0.5765 (57.65%)**
- **Strict Macro-F1**: **0.6490 (64.90%)**

### Key Findings & Limitations:
1. **High Precision on Standard Entities**: The model accurately identifies well-known geographic entities (GPE: *Patna*, *Bangalore*, *Mumbai*) and prominent corporate organizations (ORG: *Amazon*, *Dell*, *Apple*).
2. **Product Name Vulnerability**: Small general-domain models frequently miss or misclassify specific product names (e.g. *Redmi Note*, *Kindle Paperwhite*, *Jawbone headset*), often splitting compound trademarks or labeling them as generic nouns.
3. **Cross-Lingual Pipeline Impact**: Because NER runs on the translated English text, translation inaccuracies in technical terms propagate directly into downstream entity extraction.
