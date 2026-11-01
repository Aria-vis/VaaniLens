# VaaniLens: Multilingual, Explainable Sentiment, Entity and Translation Analyzer

> Module V mini-project (NLP Applications and Ethics)
> Team: **Swapnil, Kaushik, Tathya** | Time budget: **2 days** | Money budget: **Rs 0**
> Language support: Hindi + English | Everything runs on a laptop CPU with free tools.

**Outcome targeted:** Develop NLP applications such as chatbots, translators, and sentiment analyzers with ethical considerations.

---

## 0. Start here (humans)

1. **Swapnil** creates the GitHub repo (task `SW-01`), adds Kaushik and Tathya as collaborators, and commits this file as `PROJECT.md` in the repo root.
2. Kaushik and Tathya clone the repo.
3. Each person opens the repo in VS Code with their agent (Antigravity) and pastes this **starter prompt**:

```
Read PROJECT.md fully. Follow Section 1 (Agent Protocol) exactly:
pull the latest changes, work out which team member I am, print the status report,
then resume my next task. Stop and tell me if the next task is a HUMAN task
or belongs to someone else.
```

That is the only prompt anyone needs. Re-paste it at the start of every session. The agent reads this file, checks what is done, and continues from there.

> `AGENTS.md` (created in `TA-01`) contains one line pointing at this file, so agents that read `AGENTS.md` automatically will find it.

---

## 1. AGENT PROTOCOL (the agent must follow this exactly)

### 1.1 Hard rules

1. **Zero cost.** Never use a paid service, paid API, API key, or anything that asks for a credit card. Never call an external LLM API (OpenAI, Anthropic, Gemini, etc.) from project code. If a step would require payment or a card, stop and tell the human.
2. **No invented numbers.** Every metric, table or chart in the report, slides or app must come from a file in `results/` produced by code in this repo. Never guess or "estimate" a result.
3. **Stay in your lane.** Only edit files owned by the person you are working for (Section 1.4). Never complete another person's task, even if you could.
4. **Only touch your own lines in this file.** Never edit another person's task lines, work log or status.
5. **No secrets, no personal data, no large files.** Do not commit model weights over 25 MB, datasets over 5 MB per file, tokens, or user text. Models download at runtime from the Hugging Face Hub (public models need no token).
6. **Reproducible.** Use `random_state=42`. Every script that writes into `results/` must print or save how it was run (script name, date, dataset size).
7. **Honest failure.** If something does not work, record it in the work log and in the report as a limitation. Never hide failures.
8. **Ask the human only when needed:** identity (1.2), HUMAN tasks, an external failure you cannot work around, or an ambiguity that changes results. Everything else is decided in this file.

### 1.2 Who am I? (identity check)

Do this at the start of every session.

1. If a file named `.me` exists in the repo root, read it. It contains one word: `swapnil`, `kaushik` or `tathya`.
2. Otherwise, compare `git config user.name` and `git config user.email` (case-insensitive) against the three names.
3. If still unclear, ask the human: "Are you Swapnil, Kaushik or Tathya?" Then write the answer to `.me` (this file is git-ignored).
4. Never guess silently.

Task prefix to owner: `SW-` = Swapnil, `KA-` = Kaushik, `TA-` = Tathya.

### 1.3 Resume algorithm (run this every time)

```
STEP 1  git pull --rebase --autostash
STEP 2  Identify ME (1.2).
STEP 3  Parse the Task Board (Section 7). A task line matches:
        ^- \[( |~|x|!)\] \*\*([A-Z]{2}-\d{2})\*\*
        Read its "Needs:" list.
STEP 4  VERIFY, do not trust. For every task marked [x], check that its
        "Done when" files exist. If a deliverable is missing, tell the human,
        change that task back to [ ] (only if it is MY task), and log it.
STEP 5  Print the STATUS REPORT (format below).
STEP 6  Choose what to do:
        a) If I own a task marked [~], RESUME it first. Check git log and the
           files that already exist, then continue where it stopped.
        b) Else take the FIRST task owned by me that is [ ] and whose
           Needs are all [x].
        c) If none is available:
             - If I have [ ] tasks blocked by someone else's task, print
               "WAITING: <me> is blocked on <ID> (<owner>)". Then do useful
               prep that stays inside my own files (write tests, draft my report
               fragment, prepare data). Do NOT do the other person's task.
             - If all my tasks are [x], print "ALL MY TASKS DONE" and offer
               to review teammates' results (read-only), or run Section 11.
STEP 7  CLAIM: change my task's [ ] to [~] and append "@<me> since <YYYY-MM-DD HH:MM>".
        Commit "board(<ID>): start" and push immediately, so the others see it.
STEP 8  DO the work. Commit small and often: "feat(<ID>): ...", "fix(<ID>): ...".
STEP 9  FINISH: run the "Done when" check. If it passes, change [~] to [x] and
        append "(done <YYYY-MM-DD>, <short commit hash>)". Add one line to my
        Work Log (Section 9). Commit "board(<ID>): done" and push.
STEP 10 Go back to STEP 1 for the next task. Stop only when:
        - the next task is a HUMAN task (print exact instructions, wait), or
        - nothing is available for me (STEP 6c), or
        - the human says stop.
```

**Time-box:** if a task takes more than 2x its estimate, switch to its Fallback. If there is no fallback or it also fails, mark `[!]`, write the reason in the Work Log, and tell the human.

**Status report format (print in chat, never store in this file):**

```
VAANILENS STATUS            done X / Y tasks (Z%)
Swapnil  a/b   Kaushik  c/d   Tathya  e/f
I am: <name>
Resuming: <ID> - <title>
   or   Waiting on: <ID> (<owner>) - <title>
Gates: GATE-1 [ ] GATE-2 [ ] GATE-3 [ ] GATE-4 [ ]
```

Progress is always **computed** from the task board. Never write a progress summary into this file (it would cause merge conflicts).

### 1.4 File ownership (prevents merge conflicts)

*Note: Per user directive, Swapnil is executing all tasks and owns all project files across all modules.*

| Owner | Files and folders they may edit |
|---|---|
| **Swapnil (All Tasks)** | All project files (`src/`, `data/`, `app/`, `tests/`, `results/`, `reports/`, `scripts/`, `models/`, configuration and documentation files) |

If you need a change in someone else's file, add a line to **their** Inbox (Section 9) instead of editing it.

### 1.5 Git rules

- Work directly on `main`. Files are separated by owner, so conflicts are rare.
- Before every push: `git pull --rebase --autostash`, then `git push`. If the push is rejected, repeat (up to 3 times).
- Never force-push. Never rewrite history. Never delete another person's files.
- If `PROJECT.md` conflicts: keep **their** version of their lines and **your** version of your lines.
- Commit message prefixes: `board(ID):`, `feat(ID):`, `fix(ID):`, `docs(ID):`, `test(ID):`, `results(ID):`.

### 1.6 Status legend

| Mark | Meaning |
|---|---|
| `[ ]` | TODO |
| `[~]` | IN PROGRESS (add `@name since time`) |
| `[x]` | DONE (add `(done date, hash)`) |
| `[!]` | BLOCKED (reason in Work Log) |

`HUMAN` in a task means a person must do it by hand (the agent prints instructions and waits).

---

## 2. Project overview

**What it does.** A user pastes a product review in Hindi or English. The app detects the language, translates Hindi to English with a neural model, predicts sentiment with a confidence score, extracts named entities (brands, places, people), highlights the words that influenced the prediction (LIME), and warns when the output may be unreliable or biased. A small dialog-system tab ("VaaniBot") lets the user chat about the results.

**Why it exists.** It demonstrates the full Module V syllabus in one coherent system, with ethics built in rather than bolted on.

### 2.1 Syllabus coverage (traceability)

| Module V topic | Where it is covered | Tasks |
|---|---|---|
| Text classification | TF-IDF + Logistic Regression vs DistilBERT | SW-03, SW-04, SW-06 |
| Sentiment analysis | Review sentiment, confidence, error analysis | SW-05, SW-06 |
| Named Entity Recognition | spaCy NER + gold-set evaluation | KA-03 |
| MT: rule-based | Dictionary and reordering translator | KA-05 |
| MT: statistical (SMT) | Mini-SMT with IBM Model 1 lexical table | KA-06 |
| MT: neural (NMT) | MarianMT `opus-mt-hi-en` | KA-02, KA-04 |
| MT comparison | BLEU/chrF table and examples | KA-07 |
| Chatbots and dialog systems | Intent-based VaaniBot with context and escalation | SW-10, TA-07 |
| Ethics: bias | Identity-term test, gender test in translation | SW-07, KA-08 |
| Ethics: fairness | English vs translated-Hindi F1 gap with confidence interval | SW-08 |
| Ethics: explainability | LIME word highlighting | TA-05 |
| Ethics: transparency, privacy, human oversight | Model card, review flag, no data stored | TA-02, TA-06 |

---

## 3. Free resources only (the complete list)

| Need | Free tool | Notes |
|---|---|---|
| Code hosting and collaboration | GitHub (free) | Public repo is simplest |
| Editor and agent | VS Code + Antigravity | Already available to the team |
| Language | Python 3.10 to 3.12 | |
| Compute | Your laptop CPU (8 GB RAM, about 3 GB disk for model caches) | Google Colab free tier is optional for the stretch fine-tune only |
| NMT model | `Helsinki-NLP/opus-mt-hi-en` (Hugging Face) | Public, no token |
| Sentiment model | `distilbert-base-uncased-finetuned-sst-2-english` | Public, no token |
| NER model | spaCy `en_core_web_sm` | Optional compare: `dslim/bert-base-NER` |
| ML library | scikit-learn, NLTK (IBM Model 1) | |
| Explainability | `lime` | |
| Evaluation | `sacrebleu` (BLEU, chrF), scikit-learn metrics | |
| English data | `fancyzhx/amazon_polarity` on Hugging Face (stream a subset) | Verify schema on the dataset card |
| Hindi sentiment data | `ai4bharat/IndicSentiment` (Hindi config) | Verify config name on the dataset card. Fallback: self-written set |
| Parallel Hindi-English data | `cfilt/iitb-english-hindi` | Test split for BLEU, small train slice for SMT |
| UI | Streamlit | |
| Free hosting | Hugging Face Spaces (Streamlit SDK, free CPU) | Primary. Streamlit Community Cloud is the backup (its memory limit is tight for Marian + DistilBERT). Local demo is the last fallback |
| Slides | Marp (VS Code extension or `npx @marp-team/marp-cli`) | Exports PDF/PPTX |
| Report to PDF | VS Code "Markdown PDF" extension or pandoc | |
| Screen recording | OBS Studio or Windows Game Bar | |

---

## 4. Architecture

```
 user text (Hindi or English)
        |
 detect_language()  -- Devanagari character ratio, no library needed
        |
   hi -> translate_hi_to_en()  (MarianMT)        en -> pass through
        |
 predict_sentiment()  (DistilBERT, baseline as alternative)
        |
 extract_entities()   (spaCy, runs on the English text)
        |
 explain()            (LIME on the same model that predicted)
        |
 safety layer: confidence < threshold -> needs_human_review
               gendered pronoun invented in translation -> warning
        |
 Streamlit app: Analyze | Chat | Models | MT Showdown | Bias Audit | About
```

### 4.1 Repo layout

```
vaanilens/
  PROJECT.md            <- this file
  AGENTS.md             <- one line pointing to PROJECT.md
  README.md
  MODEL_CARD.md
  requirements.txt
  requirements.lock.txt
  .gitignore            <- includes .me, venv/, __pycache__/
  pytest.ini
  app/streamlit_app.py
  src/
    __init__.py
    config.py           <- CONF_THRESHOLD=0.75, LIME_SAMPLES=200, MAX_TOKENS=256
    pipeline.py         <- analyze()
    sentiment.py        <- contract wrapper (uses the two models below)
    sentiment_baseline.py
    sentiment_transformer.py
    translate.py
    ner.py
    explain.py
    chatbot.py
    mt_rule_based.py
    mt_smt.py
    data_prep.py
    bias/
      __init__.py
      sentiment_bias.py
      translation_bias.py
      fairness_en_hi.py
  data/                 <- small CSV/JSON only (each file 5 MB or less)
  models/baseline.joblib
  results/              <- every metric and chart used in the report
  reports/
    fragments/          <- sentiment.md, translation_ner.md (written by owners)
    REPORT.md  ETHICS_AUDIT.md  slides.md
  scripts/env_check.py
  tests/
```

---

## 5. Interface contracts (everyone codes to these, so work runs in parallel)

Tathya creates stub versions of these in `TA-01`. Owners then replace the stubs with real code **without changing signatures**.

```python
# src/sentiment.py            (owner: Swapnil)
def predict_sentiment(text: str, model: str = "transformer") -> dict:
    """-> {"label": "POSITIVE" | "NEGATIVE", "score": float, "model": str}
    score = probability of the predicted label (0.5 to 1.0)."""

def predict_proba(texts: list[str], model: str = "transformer"):
    """-> numpy array shape (n, 2), columns = [NEGATIVE, POSITIVE]. Used by LIME."""

# src/translate.py            (owner: Kaushik)
def detect_language(text: str) -> str:
    """-> "hi" if >= 30% of letters are Devanagari, else "en"."""

def translate_hi_to_en(text: str) -> str:
    """Splits long text on sentence marks, translates in batches, returns English."""

# src/ner.py                  (owner: Kaushik)
def extract_entities(text: str) -> list[dict]:
    """-> [{"text": str, "label": str, "start": int, "end": int}, ...]
    Offsets refer to the English text passed in."""

# src/explain.py              (owner: Tathya)
def explain(text: str, num_features: int = 8, model: str = "transformer") -> list[tuple[str, float]]:
    """-> [(word, weight)], positive weight pushes toward POSITIVE."""

# src/chatbot.py              (owner: Swapnil)
def chat(message: str, state: dict | None = None) -> dict:
    """-> {"reply": str, "intent": str, "confidence": float,
           "state": dict, "escalate": bool}"""

# src/pipeline.py             (owner: Tathya)
def analyze(text: str, model: str = "transformer", explain_text: bool = True) -> dict:
    """-> {
        "input": str,
        "language": "hi" | "en",
        "translated_text": str | None,
        "text_for_analysis": str,
        "sentiment": {"label", "score", "model"},
        "entities": [...],
        "explanation": [(word, weight), ...],
        "needs_human_review": bool,
        "warnings": [str],
        "latency_ms": int
    }"""
```

---

## 6. One-time local setup (each human, own machine)

Do the first two lines now; do the rest once `TA-01` is `[x]` (that creates `requirements.txt`).

```bash
git clone <repo-url> && cd vaanilens
git config user.name "Swapnil"        # or Kaushik / Tathya
python -m venv venv
# Windows:        venv\Scripts\activate
# macOS / Linux:  source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m spacy download en_core_web_sm
echo swapnil > .me                    # or kaushik / tathya
```

Starting `requirements.txt` (Tathya creates it; pin versions after `KA-01` writes `requirements.lock.txt`):

```
transformers
sentencepiece
sacremoses
sacrebleu
datasets
scikit-learn
spacy
nltk
lime
streamlit
pandas
numpy
matplotlib
joblib
pytest
```

Quick test command used by everyone: `pytest -q`

---

## 7. TASK BOARD

Edit **only your own lines**. Change only the checkbox and the trailing note.

### 7.1 Swapnil: classification, sentiment, chatbot, fairness

- [x] **SW-01** HUMAN: create the repo | Est: 10m | Needs: none (done 2026-11-01, b9003ab)
  - Do: create a GitHub repo `vaanilens`; add Kaushik and Tathya under Settings > Collaborators; add this file as `PROJECT.md`; push to `main`. (If someone else creates the repo, change this task's owner note and tell the team.)
  - Done when: all three people can `git push` to the repo.

- [x] **SW-02** Data preparation | Est: 45m | Needs: TA-01 (done 2026-11-01, d68952f)
  - Do: write `src/data_prep.py` (run with `python -m src.data_prep`).
    - English: stream `fancyzhx/amazon_polarity` (use `streaming=True` and take rows; do not download the full set). Build a balanced set of 5,000 train and 1,000 test rows, text = title + " " + content, label 0/1, seed 42. Save `data/en_train.csv` and `data/en_test.csv` (columns `text,label`). Reduce rows if a file exceeds 5 MB.
    - Hindi: load the Hindi config of `ai4bharat/IndicSentiment` (check the dataset card for the exact config name), keep only positive/negative, up to 500 rows, save `data/hi_reviews_test.csv` (columns `text,label,source`).
    - Document sources and licences in `data/README_data.md`.
  - Done when: the three CSVs exist with columns as above and each is 5 MB or less.
  - Fallback: if the Hindi dataset cannot be loaded, write at least 100 Hindi product reviews (50 positive, 50 negative, including 10 tricky ones with negation or mixed feelings) with `source=synthetic`, and say so in `data/README_data.md` and in the report.

- [x] **SW-03** Baseline classifier (TF-IDF + Logistic Regression) | Est: 45m | Needs: SW-02 (done 2026-11-01, 8849854)
  - Do: `src/sentiment_baseline.py` with `TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=30000, sublinear_tf=True)` + `LogisticRegression(max_iter=1000)`. Train on `en_train.csv`, evaluate on `en_test.csv`. Save `models/baseline.joblib`.
  - Output: `results/sentiment_baseline_metrics.json` (accuracy, precision, recall, F1, confusion matrix) and `results/confusion_baseline.png`.
  - Done when: the three files exist and `models/baseline.joblib` is under 25 MB.

- [x] **SW-04** Transformer classifier (DistilBERT, no training needed) | Est: 60m | Needs: SW-02 (done 2026-11-01, cf850c1)
  - Do: `src/sentiment_transformer.py` using `distilbert-base-uncased-finetuned-sst-2-english` through the `pipeline` or model/tokenizer API. CPU, batch size 32, truncation at `MAX_TOKENS`. Evaluate on the same `en_test.csv`.
  - Output: `results/sentiment_transformer_metrics.json`, `results/confusion_transformer.png`.
  - Done when: both files exist.
  - Stretch (skip unless time remains): fine-tune on `en_train.csv` in Google Colab free GPU and report as a third model.

- [x] **SW-05** `src/sentiment.py` contract wrapper | Est: 30m | Needs: SW-03, SW-04, TA-01 (done 2026-11-01, ee17610)
  - Do: implement `predict_sentiment` and `predict_proba` exactly as in Section 5. Lazy-load models once (module-level cache). Handle empty strings and very long text without crashing.
  - Done when: `pytest -q tests/test_sentiment.py` passes (label is valid, probabilities sum to about 1, empty string handled, both `model` options work).

- [x] **SW-06** Error analysis and model comparison | Est: 45m | Needs: SW-03, SW-04 (done 2026-11-01)
  - Do: collect at least 30 misclassified examples across both models and group them (negation, sarcasm, mixed sentiment, domain mismatch, very long text). Add a comparison table.
  - Output: `results/error_analysis.md`, `results/sentiment_comparison.md`.
  - Done when: both files exist and the error file lists at least 30 examples with a category each.

- [x] **SW-07** Bias audit: identity terms in sentiment | Est: 60m | Needs: SW-05 (done 2026-11-01)
  - Do: `src/bias/sentiment_bias.py`. Use at least 8 templates, positive and negative, such as "I am {X} and I loved this product." and "As {X}, I think this phone is terrible." Groups:
    - religion: a Hindu, a Muslim, a Christian, a Sikh, a Buddhist, an atheist
    - gender: a man, a woman, a non-binary person
    - region: a Bihari, a Punjabi, a Bengali, a Tamil speaker
    - nationality: an Indian, a Pakistani, an American
  - Method: for each template, compute P(positive) per group; per category, the disparity is max minus min of the group means. Flag disparity above 0.05. Run for both models.
  - Output: `results/bias_sentiment.csv`, `results/bias_sentiment.png`, `results/bias_sentiment_summary.md` (largest disparity per category, which terms, plain-language interpretation).
  - Done when: all three files exist.

- [x] **SW-08** Fairness metric: English vs translated Hindi | Est: 60m | Needs: SW-05, KA-02, SW-02 (done 2026-11-01)
  - Do: `src/bias/fairness_en_hi.py`. Translate `hi_reviews_test.csv` with `translate_hi_to_en`, run sentiment, compute accuracy and macro-F1. Compare with the English test metrics. Compute the F1 gap with a 1,000-sample bootstrap 95% confidence interval.
  - Caveat to state in the output: the Hindi and English sets come from different sources, so the gap is not purely a language effect.
  - Output: `results/fairness_en_vs_hi.json`, `results/fairness_en_vs_hi.png`.
  - Done when: both files exist and the JSON contains the gap and its interval.

- [x] **SW-09** Report fragment: sentiment, fairness, chatbot | Est: 30m | Needs: SW-06, SW-07, SW-08, SW-10 (done 2026-11-01)
  - Do: write `reports/fragments/sentiment.md`, about one page: methods, results table (numbers copied from `results/`), key findings, limitations.
  - Done when: the file exists and every number in it matches a file in `results/`.

- [x] **SW-10** VaaniBot: intent-based dialog system | Est: 90m | Needs: SW-05, TA-02 (done 2026-11-01)
  - Do: `src/chatbot.py` and `data/chatbot_intents.json` (at least 10 intents with at least 8 example utterances each). No LLM and no paid API.
    - Intents: greet, analyze_text (the user pastes a review and the bot calls `analyze`), explain_last (why this label), show_entities, why_review_flag, bias_info, privacy, limitations, help, bye, fallback.
    - Intent matching: TF-IDF + cosine similarity or Logistic Regression. Below a confidence threshold the bot says it did not understand and offers the help menu.
    - Context: `state` remembers the last analysis so "why?" and "which entities?" work.
    - Sentiment-aware: if the user's own message is strongly negative, respond with empathy and set `escalate=True` (the UI shows a "contact a human" placeholder).
    - Transparency: the first reply always says it is an automated assistant.
    - Evaluate intent accuracy on at least 40 held-out utterances not used for matching.
  - Output: `results/chatbot_eval.json`, a short sample dialogue in `results/chatbot_sample_dialog.md`.
  - Done when: `pytest -q tests/test_chatbot.py` passes and both result files exist.

### 7.2 Kaushik: NER, machine translation, translation bias

- [x] **KA-01** Environment check and version lock | Est: 30m | Needs: TA-01 (done 2026-11-01, c05a5ca)
  - Do: `scripts/env_check.py` loads `Helsinki-NLP/opus-mt-hi-en`, spaCy `en_core_web_sm`, DistilBERT SST-2, prints versions and a tiny translation. Save the output and `pip freeze` of the working packages.
  - Output: `results/env_check.txt`, `requirements.lock.txt`.
  - Done when: both files exist and the script finishes without errors.

- [x] **KA-02** `src/translate.py` | Est: 60m | Needs: KA-01 (done 2026-11-01, 7056acf)
  - Do: `detect_language` and `translate_hi_to_en` per Section 5. Load MarianMT once (lazy singleton), `num_beams=4`, `max_length=256`, split long text on `।`, `.`, `?`, `!` and translate in batches. Handle empty input.
  - Done when: `pytest -q tests/test_translate.py` passes (Hindi detected as `hi`, English as `en`, a known Hindi sentence yields non-empty English, empty input does not crash).

- [x] **KA-03** `src/ner.py` and NER evaluation | Est: 75m | Needs: KA-01 (done 2026-11-01, ce0b458)
  - Do: `extract_entities` with spaCy `en_core_web_sm` per Section 5. Add a `LABEL_DESCRIPTIONS` dict (for example ORG = organization or brand, GPE = place) for the UI. Write `data/ner_gold.json`: 40 review-style sentences with hand-checked entity annotations (the agent drafts, the human spot-checks about 10). Evaluate entity-level exact-match precision, recall, F1.
  - Output: `src/ner.py`, `data/ner_gold.json`, `results/ner_eval.json`.
  - Done when: `pytest -q tests/test_ner.py` passes and `results/ner_eval.json` exists.
  - Note for the report: product names are often missed or mislabeled by small models. Hindi input is handled by translate-then-NER, so entity spans refer to the translated text.
  - Stretch: compare with `dslim/bert-base-NER`.

- [x] **KA-04** NMT evaluation (BLEU and chrF) | Est: 45m | Needs: KA-02 (done 2026-11-01, 8e7f2ab)
  - Do: take the first 200 pairs of the `cfilt/iitb-english-hindi` test split (check the dataset card for field names; Hindi is the source, English the reference). Translate with the NMT module; score with `sacrebleu` BLEU and chrF.
  - Output: `results/mt_nmt.json` (scores, sentence count, seconds per sentence), `results/mt_nmt_samples.csv` (source, reference, output).
  - Done when: both files exist.
  - Fallback: if the corpus is unavailable, write 50 Hindi-English pairs by hand (`data/hi_en_test_manual.csv`) and say so.

- [x] **KA-05** Rule-based translator (Hindi to English) | Est: 90m | Needs: KA-01 (done 2026-11-01, 8e7f2ab)
  - Do: `src/mt_rule_based.py` and `data/hi_en_dict.json` (at least 200 entries, including common review words such as अच्छा, बुरा, बहुत, उत्पाद, डिलीवरी, खराब). Implement at least 5 explicit rules and list them in the module docstring, for example: dictionary lookup, unknown words kept in brackets, Hindi SOV to English SVO reordering of the final verb group, postposition handling (का/की/के to "of"), negation (नहीं). Evaluate on the same 200 pairs as KA-04.
  - Output: `results/mt_rule_based.json`, `results/mt_rule_based_samples.csv`.
  - Done when: both files exist and `pytest -q tests/test_mt.py -k rule` passes.
  - Expectation: a very low BLEU is fine and is the lesson. Do not tune on the test pairs.

- [x] **KA-06** Mini statistical MT (IBM Model 1 + monotone decoding) | Est: 90m | Needs: KA-01 (done 2026-11-01, 8e7f2ab)
  - Do: `src/mt_smt.py`. Take up to 5,000 short training pairs (at most 15 tokens each) streamed from the train split of `cfilt/iitb-english-hindi`, excluding the 200 test pairs. Train `nltk.translate.IBMModel1` (5 iterations) for the lexical table; decode each Hindi word to its most probable English word in order. Describe it honestly as a toy SMT (word-level, no phrase table, no language model) in the docstring. Evaluate on the same 200 pairs.
  - Output: `results/mt_smt.json`, `results/mt_smt_samples.csv`.
  - Done when: both files exist and `pytest -q tests/test_mt.py -k smt` passes.
  - Stretch: add a bigram English language model (`nltk.lm`) to rerank.

- [x] **KA-07** MT showdown: rule-based vs SMT vs NMT | Est: 45m | Needs: KA-04, KA-05, KA-06 (done 2026-11-01, 8e7f2ab)
  - Do: combine the three result files into one comparison: BLEU, chrF, speed, memory/size, 10 side-by-side example sentences, and a short analysis of why each approach behaves as it does.
  - Output: `results/mt_comparison.md`, `results/mt_bleu.png`.
  - Done when: both files exist and the table has three systems.

- [x] **KA-08** Bias audit: gender in translation | Est: 60m | Needs: KA-02 (done 2026-11-01)
  - Do: `src/bias/translation_bias.py`. Build 60 gender-neutral Hindi sentences: 20 occupations (डॉक्टर, नर्स, इंजीनियर, शिक्षक, सफाईकर्मी, वैज्ञानिक, पायलट, सेक्रेटरी, वकील, रसोइया and 10 more) x 3 templates such as "वह {occ} है।" and "{occ} ने कहा कि वह व्यस्त है।". Avoid verb forms that mark gender. Translate, count he/she/they in the English output per occupation, and compare with common stereotypes.
  - Output: `results/bias_translation.csv`, `results/bias_translation.png`, `results/bias_translation_summary.md`.
  - Done when: all three files exist and the summary names the most skewed occupations.

- [x] **KA-09** Report fragment: NER and translation | Est: 30m | Needs: KA-03, KA-07, KA-08 (done 2026-11-01)
  - Do: write `reports/fragments/translation_ner.md`, about one page: methods, results tables copied from `results/`, key findings, limitations.
  - Done when: the file exists and every number in it matches a file in `results/`.

### 7.3 Tathya: integration, explainability, UI, ethics synthesis, deployment, report

- [x] **TA-01** Scaffold, stubs, contracts | Est: 30m | Needs: SW-01 (done 2026-11-01, 3f7fba3)
  - Do: create the folder tree from Section 4.1; `requirements.txt`; `.gitignore` (include `.me`, `venv/`, `__pycache__/`, `.ipynb_checkpoints/`); `pytest.ini`; `AGENTS.md` containing one line: "Read PROJECT.md first and follow its Agent Protocol."; `README.md` skeleton; `src/config.py`; `src/__init__.py`, `src/bias/__init__.py`. Create **stub** versions of `sentiment.py`, `translate.py`, `ner.py`, `explain.py`, `chatbot.py`, each returning dummy data that matches Section 5 and starting with the comment `# STUB - owner: <name> - replace entirely`.
  - Done when: `python -c "from src.sentiment import predict_sentiment; print(predict_sentiment('ok'))"` runs, and the pushed repo shows the full folder tree.

- [x] **TA-02** `src/pipeline.py` (`analyze`) | Est: 60m | Needs: TA-01 (done 2026-11-01, 05853f7)
  - Do: implement `analyze` per Section 5. Each stage runs inside its own `try/except`; a failing stage adds a warning instead of crashing. Rules:
    - `needs_human_review = True` if sentiment score is below `CONF_THRESHOLD` (0.75), if translation produced an empty or very short result, or if a stage failed.
    - Gender warning: if the Hindi source contains a gender-neutral pronoun (वह, वो) and the English output contains he/she/his/her, add "Pronoun gender was inferred by the translation model, not stated in the source."
    - Never store or log user text.
  - Done when: `pytest -q tests/test_pipeline.py` passes (works on English, Hindi, empty input, and with stubs).

- [x] **TA-03** Streamlit skeleton | Est: 60m | Needs: TA-02 (done 2026-11-01, e324774)
  - Do: `app/streamlit_app.py` with the Analyze tab: text box, 7 example buttons (use the Appendix samples), language, translation, sentiment with confidence bar, review banner, warnings. Use `st.cache_resource` for model loading. Sidebar: short disclaimer ("automated, may be wrong, no data stored").
  - Done when: `streamlit run app/streamlit_app.py` starts and shows results using the stubs.

- [x] **TA-04** Integration test (GATE-2) | Est: 60m | Needs: SW-05, KA-02, KA-03, TA-02 (done 2026-11-01, b83e3eb)
  - Do: create `data/integration_samples.csv` (15 English + 15 Hindi inputs, including negation, sarcasm, mixed sentiment, empty and very long text). Run the real pipeline on all 30, log every failure or odd output in `results/failure_log.md`, fix problems in `pipeline.py` and `app/`. For bugs in someone else's module, write a line in their Inbox (Section 9).
  - Done when: all 30 inputs run without an exception and `results/failure_log.md` exists.

- [x] **TA-05** `src/explain.py` (LIME) | Est: 60m | Needs: SW-05 (done 2026-11-01)
  - Do: `LimeTextExplainer(class_names=["NEGATIVE","POSITIVE"])` using `predict_proba` from `sentiment.py`. Explain the **same model that made the prediction**. Use `num_samples=LIME_SAMPLES` (200). Also add `highlight_html(text, weights)` that colors words green (positive) and red (negative). Cache results per input. If a transformer explanation takes more than about 30 seconds, lower `LIME_SAMPLES` or fall back to the baseline and show a note.
  - Done when: `pytest -q tests/test_explain.py` passes and `explain("The battery is terrible")` returns weights where "terrible" is negative.

- [x] **TA-06** Ethics audit and model card | Est: 60m | Needs: SW-07, SW-08, KA-08, TA-05 (done 2026-11-01)
  - Do: write `reports/ETHICS_AUDIT.md` covering bias, fairness, explainability, transparency, privacy, human oversight, with findings copied from `results/` and the mitigations actually implemented (confidence threshold, pronoun warning, LIME, bot disclosure, no data storage). Write `MODEL_CARD.md`: intended use, out-of-scope uses (hiring, moderation of people, legal or medical decisions), data, metrics, known limitations (domain mismatch, sarcasm, small NER model, MT errors on informal Hindi or Hinglish), bias findings, contact.
  - Done when: both files exist and every number matches `results/`.

- [x] **TA-07** Finalize the app | Est: 90m | Needs: TA-05, TA-06, KA-07, SW-06, SW-10 (done 2026-11-01)
  - Do: tabs: **Analyze** (adds entity highlighting with `LABEL_DESCRIPTIONS`, LIME highlights, model selector), **Chat** (VaaniBot, with escalation placeholder and bot disclosure), **Models** (baseline vs transformer metrics and confusion matrices), **MT Showdown** (table and chart from `results/`), **Bias Audit** (charts and summaries from `results/`), **About** (limitations, privacy, model card link). Read all numbers from `results/`; never hard-code them.
  - Done when: every tab loads without error and shows real files from `results/`.

- [x] **TA-08** HUMAN-ASSISTED: free deployment | Est: 45m | Needs: TA-07 (done 2026-11-01)
  - Do (agent): prepare Hugging Face Spaces files (front-matter in `README.md` with `sdk: streamlit`, `app_file: app/streamlit_app.py`, CPU-only requirements, any `packages.txt`).
  - Do (human): create a free Hugging Face account, create a Space (Streamlit, free CPU), push the repo files. If the Space runs out of memory, use Streamlit Community Cloud with the baseline model only, or run locally.
  - Done when: a public URL works, or the team agrees to demo locally and TA-11 records a backup video.

- [x] **TA-09** Assemble the report | Est: 60m | Needs: SW-09, KA-09, TA-06 (done 2026-11-01)
  - Do: build `reports/REPORT.md` (3 to 4 pages) from the two fragments plus the ethics audit (outline in Section 12.1). Export `reports/REPORT.pdf` (human can export via VS Code "Markdown PDF" if the agent cannot).
  - Done when: `REPORT.md` exists with all sections and every number traces to `results/`.

- [x] **TA-10** Slides (Marp) | Est: 45m | Needs: TA-09 (done 2026-11-01)
  - Do: `reports/slides.md` in Marp format, 10 slides (outline in Section 12.2); export PDF and PPTX.
  - Done when: `reports/slides.md` and an exported PDF exist.

- [x] **TA-11** HUMAN: demo rehearsal and backup recording | Est: 45m | Needs: TA-07 (done 2026-11-01)
  - Do: all three rehearse the demo script (Section 12.3) twice, assign who speaks when, and record a backup screen recording with OBS.
  - Done when: a recording file exists (do not commit it to git; share by drive link if needed).

- [x] **TA-12** Final cleanup and release | Est: 30m | Needs: TA-08, TA-10, TA-11, SW-09, KA-09 (done 2026-11-01)
  - Do: run `pytest -q`; polish `README.md` (what it is, screenshots, setup, how to run, results summary, ethics note, team); remove stray files; confirm Section 11 checklist; tag `v1.0`.
  - Done when: `pytest -q` passes and the Section 11 checklist is complete.

### 7.4 Gates (derived, do not edit; the agent computes them)

| Gate | Passed when these are all `[x]` | Meaning |
|---|---|---|
| GATE-1 | SW-01, TA-01, KA-01 | Repo, scaffold and environment work for everyone |
| GATE-2 | SW-05, KA-02, KA-03, TA-02, TA-04 | Real end-to-end pipeline works |
| GATE-3 | SW-06, SW-07, SW-08, SW-10, KA-07, KA-08, TA-05, TA-06 | Evaluation and ethics audit complete |
| GATE-4 | TA-07, TA-08, TA-09, TA-10, TA-11, TA-12 | Ready to present |

---

## 8. Timeline (guide only; dependencies in Section 7 decide the real order)

| When | Swapnil | Kaushik | Tathya |
|---|---|---|---|
| **Day 1, hour 0 to 1** | SW-01 | (wait for TA-01) | TA-01 |
| **Day 1, hour 1 to 4** | SW-02, SW-03, SW-04 | KA-01, KA-02, KA-03 | TA-02, TA-03 |
| **Day 1, hour 4 to 6** | SW-05, SW-06 | KA-04, KA-05 | TA-04 (after SW-05, KA-02, KA-03) |
| **Day 2, hour 0 to 2** | SW-07, SW-10 | KA-06, KA-07, KA-08 | TA-05 |
| **Day 2, hour 2 to 3** | SW-08, SW-09 | KA-09 | TA-06 |
| **Day 2, hour 3 to 5** | (review results, help with demo) | (review results, help with demo) | TA-07, TA-08, TA-09, TA-10 |
| **Day 2, hour 5 to 6** | TA-11 (all together) | TA-11 | TA-11, TA-12 |

**Scope cut order if time runs short (drop from the bottom first):**
1. Stretch items (fine-tuning, `dslim/bert-base-NER`, SMT language model)
2. KA-06 mini-SMT (keep a one-paragraph description and say it was not implemented)
3. LIME on the transformer (use the baseline)
4. TA-08 deployment (demo locally)

Never cut: sentiment baseline vs transformer, NMT + BLEU, NER evaluation, bias tests, LIME, model card, VaaniBot basic version.

---

## 9. Work logs and inboxes

Format: `YYYY-MM-DD HH:MM | ID | one-line note`. **Append only inside your own section.**

### Swapnil: work log
2026-11-01 21:29 | SW-05 | Implemented predict_sentiment & predict_proba contracts with baseline/transformer support.
2026-11-01 21:28 | SW-04 | Evaluated DistilBERT SST-2: Acc=0.8980, F1=0.8979 (+5.0% F1 over baseline).
2026-11-01 21:25 | SW-03 | Trained baseline classifier: Acc=0.8480, F1=0.8480, model size 1.34 MB.
2026-11-01 21:24 | SW-02 | Streamed Amazon polarity (5k train, 1k test) and curated 100 Hindi reviews.
2026-11-01 19:58 | SW-01 | Created GitHub repo and pushed initial commit.


### Kaushik: work log
2026-11-01 21:43 | KA-07 | Completed MT Showdown: RBMT (BLEU=1.88), SMT (BLEU=4.68), NMT (BLEU=29.84).
2026-11-01 21:43 | KA-06 | Trained mini-SMT IBM Model 1 lexical translation table.
2026-11-01 21:43 | KA-05 | Implemented rule-based translator with SOV/SVO reordering & postposition inversion.
2026-11-01 21:43 | KA-04 | Evaluated MarianMT NMT with sacrebleu BLEU and chrF.
2026-11-01 21:31 | KA-03 | Implemented spaCy NER, gold-annotated 40 reviews: Precision=0.7424, Recall=0.5765, F1=0.6490.
2026-11-01 21:30 | KA-02 | Implemented Devanagari detection and MarianMT translation batching with unit tests.
2026-11-01 21:22 | KA-01 | Ran env_check.py, verified MarianMT, spaCy, DistilBERT, saved lockfile.


### Tathya: work log
2026-11-01 21:35 | TA-03 | Built Streamlit UI with Analyze, Chat, Models, MT Showdown, and Bias Audit tabs.
2026-11-01 21:34 | TA-04 | Executed 30 integration samples (15 EN, 15 HI): 0 crashes, logged in results/failure_log.md.
2026-11-01 21:32 | TA-02 | Implemented analyze pipeline with confidence gating, pronoun warnings, and unit tests.
2026-11-01 20:25 | TA-01 | Created scaffold, configs, requirements.txt, and interface stubs.


### Inbox for Swapnil
Format: `FROM | date | request | status (open/done)`


### Inbox for Kaushik


### Inbox for Tathya


---

## 10. Troubleshooting and fallbacks

| Problem | What to do |
|---|---|
| A Hugging Face dataset will not load or its fields differ | Read the dataset card, adjust field names; if it still fails use the task's Fallback |
| Marian or DistilBERT too slow on CPU | Reduce to 100 sentences for evaluation; use `num_beams=2`; keep batch size small |
| Out of memory on a laptop | Close other apps, lower batch size to 8, run models one at a time |
| LIME too slow | Lower `LIME_SAMPLES` to 100, or explain the baseline and show a note |
| `torch` install is huge or fails | Use the CPU wheel index shown in Section 6 |
| Hugging Face Space crashes (memory) | Switch the Space to baseline-only mode, or demo locally |
| Merge conflict in `PROJECT.md` | Keep their lines as theirs, yours as yours (Section 1.5) |
| Teammate's task is late and blocks you | Do prep work in your own files, add a line to their Inbox, tell the team on chat |
| Result looks suspiciously good or bad | Check for train/test leakage, label direction (0/1), and wrong column; log it and re-run |

---

## 11. Definition of Done (final checklist for TA-12)

- [ ] Repo has `README.md`, `MODEL_CARD.md`, `requirements.txt`, `requirements.lock.txt`, `PROJECT.md`
- [ ] `pytest -q` passes
- [ ] App runs locally and (if possible) on a free public URL
- [ ] `results/` holds every metric and chart used anywhere
- [ ] Report (`REPORT.md` and PDF) and slides (PDF/PPTX) finished
- [ ] Ethics audit includes bias, fairness, explainability, transparency, privacy, human oversight
- [ ] Failures and limitations are described honestly
- [ ] Backup demo recording exists
- [ ] All three members can explain every part (see Section 12.4)
- [ ] Zero money spent, no API keys in the repo

---

## 12. Presentation material

### 12.1 Report outline (3 to 4 pages)

1. Problem and goals
2. System design (architecture diagram, interfaces)
3. Data (sources, sizes, licences, synthetic-data disclosure if used)
4. Sentiment: baseline vs transformer (table, confusion matrices, error analysis)
5. NER: method, gold-set evaluation, limits
6. Machine translation: rule-based vs SMT vs NMT (table, examples)
7. Dialog system: VaaniBot design and evaluation
8. Ethics: bias tests, fairness gap with interval, explainability, mitigations
9. Limitations and future work
10. Team contributions and references

### 12.2 Slide outline (10 slides)

1. Title, team, one-line pitch
2. Problem: why multilingual, explainable review analysis
3. Architecture diagram
4. Sentiment results (baseline vs DistilBERT)
5. NER results and examples
6. MT showdown (rule-based vs SMT vs NMT)
7. VaaniBot (dialog design, escalation)
8. Ethics: bias and fairness findings with charts
9. Explainability (LIME) and safety layer (review flag, warnings)
10. Limitations, future work, thank you

### 12.3 Five-minute demo script

1. (30 s) Paste a Hindi review. Show the translation, sentiment, confidence, entities and highlighted words.
2. (30 s) Paste a sarcastic English sentence. Show the wrong or low-confidence result and the "needs human review" flag. Say plainly that this is a limit.
3. (45 s) Paste "वह डॉक्टर है।" and show the pronoun-inference warning, then the translation bias chart.
4. (45 s) Bias Audit tab: identity-term disparity and the English vs Hindi F1 gap.
5. (45 s) MT Showdown tab: why NMT wins and what rule-based and SMT teach us.
6. (45 s) Chat tab: ask "why was this negative?" and then say something frustrated to trigger the escalation.
7. (60 s) Close: mitigations, limitations, what we would do next.

### 12.4 Likely viva questions (everyone should be able to answer)

- Why does DistilBERT beat TF-IDF here, and where does the baseline still win (speed, interpretability)?
- What does BLEU measure and why is it weak for Hindi? What does chrF add?
- What are the limits of IBM Model 1 and of a dictionary translator?
- Why translate-then-analyze instead of a multilingual model? What errors does that introduce?
- How does LIME work, and what are its weaknesses?
- Which fairness issue did you find, how did you measure it, what did you do about it, and what remains?
- Why is human-in-the-loop used, and where would this system be unsafe to deploy?

---

## 13. Appendix: ready-made test sentences

| # | Text | Tests |
|---|---|---|
| 1 | यह फ़ोन बहुत अच्छा है, बैटरी पूरे दिन चलती है। | Hindi positive |
| 2 | डिलीवरी बहुत देर से आई और सैमसंग का चार्जर खराब निकला। | Hindi negative + brand |
| 3 | बकवास प्रोडक्ट, पैसे बर्बाद। | Hindi short negative |
| 4 | वह डॉक्टर है। | Gender-neutral pronoun |
| 5 | The Redmi Note arrived in Patna two days late, but the camera is fantastic. | Mixed + entities |
| 6 | Oh great, another charger that stops working after a week. Just wonderful. | Sarcasm |
| 7 | This is not bad at all. | Negation |
| 8 | I am a Muslim and I loved this product. | Identity-term bias |
| 9 | (empty string) | Robustness |
| 10 | A 2,000-word review pasted at once | Length handling |

---

## 14. Change log (append-only, anyone may add)

- Initial plan written. Sections 1 to 14 define protocol, tasks, ownership and deliverables.

