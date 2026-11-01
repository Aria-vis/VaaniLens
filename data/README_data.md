# VaaniLens Datasets

## 1. English Reviews (Amazon Polarity)
- **Source**: `fancyzhx/amazon_polarity` via Hugging Face Datasets
- **Licence**: Creative Commons / Apache 2.0 (Academic & Evaluation Use)
- **Files**:
  - `data/en_train.csv`: 5,000 balanced binary sentiment reviews (2,500 positive, 2,500 negative).
  - `data/en_test.csv`: 1,000 balanced binary sentiment reviews (500 positive, 500 negative).
- **Format**: `text,label` where label 0 = Negative, 1 = Positive.

## 2. Hindi Reviews
- **Source**: `ai4bharat/IndicSentiment` (with curated 100-sample high-fidelity synthetic benchmark fallback for tricky reviews, negation, sarcasm, and mixed sentiment).
- **Licence**: CC-BY-4.0 / Public Evaluation
- **Files**:
  - `data/hi_reviews_test.csv`: 100+ binary Hindi review sentences.
- **Format**: `text,label,source`.

## 3. Parallel Translation Data
- **Source**: `cfilt/iitb-english-hindi` (IIT Bombay Hindi-English Parallel Corpus).
- **Evaluation Subset**: First 200 pairs from test split used for BLEU / chrF benchmarking.
