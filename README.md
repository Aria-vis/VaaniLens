# VaaniLens: Multilingual, Explainable Sentiment, Entity and Translation Analyzer

VaaniLens is a comprehensive NLP system supporting Hindi and English review analysis with sentiment classification, named entity recognition, machine translation comparison, LIME explainability, fairness and bias audits, and an interactive dialog system (VaaniBot).

## Project Overview
- **Languages Supported**: Hindi and English
- **Machine Translation**: MarianMT (Neural), Rule-based, and Statistical (IBM Model 1)
- **Sentiment Classification**: DistilBERT SST-2 vs. TF-IDF + Logistic Regression
- **Named Entity Recognition**: spaCy `en_core_web_sm`
- **Explainability**: LIME feature attribution
- **Safety & Ethics**: Demographic bias audit, gender translation bias audit, low-confidence review flags

## Setup & Quickstart
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app/streamlit_app.py
```
