# 🌾 AgroVox — Voice-Driven Intelligence for Smarter Farming

A large-scale, integrated **Python NLP laboratory project**: a bilingual
(Tamil/English) conversational agricultural assistant built as **one single
pipeline** combining six NLP concepts — not six disconnected exercises.

---

## 1. Abstract

AgroVox lets a farmer ask a question — by voice or text, in Tamil or English
— about crop diseases, fertilizers, irrigation, soil, pests, weather, or
general cultivation practices, and receive a grounded answer retrieved from
a structured agricultural knowledge base, with full conversational
follow-up support.

## 2. Problem Statement

Farmers need fast, accessible, local-language agricultural guidance.
Existing digital tools are often English-only, text-only, and don't handle
natural follow-up conversation. AgroVox demonstrates how six core NLP
techniques can be integrated into a single working pipeline to close that
gap, entirely with open, locally-trainable models.

## 3. Objectives

1. Build one integrated NLP pipeline (not six standalone demos).
2. Support both Tamil and English, by voice and text.
3. Classify farmer queries into 10 agricultural intent categories using a
   genuinely trained ML classifier (not if/else rules).
4. Extract 9 types of agricultural named entities.
5. Answer questions via semantic retrieval over a structured knowledge base.
6. Maintain conversational context across follow-up turns.
7. Log every query/response/feedback to a relational database for analytics.
8. Evaluate every module with real, computed metrics — never fabricated
   numbers.
9. Run entirely on a normal student laptop, with no required GPU or paid
   API key.

## 4. Scope

In scope: text + voice query understanding, 10-class intent classification,
9-type NER, Tamil↔English translation, retrieval-based QA over a curated
agricultural knowledge base, multi-turn chatbot context, SQLite persistence,
Streamlit UI, full evaluation suite. Out of scope: real-time market-price
prediction, satellite/remote-sensing crop monitoring, and pesticide dosage
prescriptions (the system explicitly declines to give exact chemical
dosages — see §12 Safety).

## 5. The Six NLP Concepts — and where each lives

| # | Concept | Module | Approach used |
|---|---------|--------|----------------|
| 1 | Speech Processing | `src/speech/speech_engine.py` | Whisper ASR / gTTS hooks with a documented, always-on text-mode fallback |
| 2 | Machine Translation | `src/translation/translator.py` | Local dictionary/phrase-table Tamil↔English translator (+ optional neural MT hook) |
| 3 | Text Classification | `src/classification/classifier.py` | TF-IDF + Logistic Regression / Linear SVM / Naive Bayes, trained & compared |
| 4 | Named Entity Recognition | `src/ner/ner_engine.py` | spaCy `EntityRuler` gazetteer NER over 9 agricultural entity types |
| 5 | Question Answering | `src/qa/qa_engine.py` | TF-IDF + cosine-similarity semantic retrieval, intent-boosted |
| 6 | Chatbot | `src/chatbot/chatbot_engine.py` | Full pipeline orchestration + slot-based conversational context memory |

## 6. System Architecture

```
Farmer (Voice/Text)
        │
        ▼
 Speech-to-Text (Module 1)
        │
        ▼
 Language Detection (ta / en)
        │
        ▼
 Machine Translation ta↔en (Module 2)
        │
        ▼
 Text Preprocessing (clean → tokenize → stopword removal → lemmatize/stem)
        │
        ▼
 Text Classification → intent (Module 3)
        │
        ▼
 Named Entity Recognition → CROP/DISEASE/SYMPTOM/... (Module 4)
        │
        ▼
 Agricultural Knowledge Base (crops, diseases, fertilizers, irrigation,
 pests, soil, FAQ)
        │
        ▼
 Question Answering — TF-IDF retrieval, intent-boosted (Module 5)
        │
        ▼
 Context-aware Chatbot response composition (Module 6)
        │
        ▼
 Translation back to farmer's language → Text / optional Speech Output
```

Every stage's intermediate output (language, translation, tokens, intent,
confidence, entities, matched KB question, context memory) is surfaced in
the **Query Analysis** tab of the Streamlit app so an evaluator can see all
six concepts firing on every single query.

## 7. Fallback Strategy — why this matters, and what runs by default

A production-grade AgroVox would use a neural MT model, Whisper ASR, and a
sentence-embedding QA retriever. Those need internet access to download
multi-hundred-MB model weights and, ideally, a GPU — resources many
classroom/lab laptops (and this project's own build sandbox) don't have.

Rather than being blocked by that, **every module implements the graceful
fallback the spec explicitly requires (Section 15)**:

| Module | Ideal (heavy) backend | Default (always-on) backend | Where |
|---|---|---|---|
| Speech | OpenAI Whisper / cloud ASR | Clear "switch to Text" message; code path ready for Whisper if installed | `speech_engine.py` |
| Translation | Neural MT (NLLB / IndicTrans2) | Local dictionary/phrase-table translator built from `data/translations.csv` | `translator.py` |
| Classification | Transformer fine-tune | TF-IDF + classical ML (**actually trained, ~97% test accuracy**) | `classifier.py` |
| NER | Fine-tuned transformer NER | spaCy `EntityRuler` gazetteer NER (**~95% F1** on held-out sentences) | `ner_engine.py` |
| QA | Sentence-embedding + ANN retrieval | TF-IDF + cosine similarity retrieval (**~98% top-1 accuracy**) | `qa_engine.py` |

This means: **the default pipeline is not a mock** — the classifier, NER,
and QA engine are real, locally trained/built artifacts with measured
performance. Only the two components that inherently require network access
to a hosted model (neural MT, Whisper ASR) run in documented fallback mode
by default, and both have a ready-to-enable "real" code path (see
`requirements.txt`'s commented optional dependencies).

## 8. Project Structure

```
AgroVox/
├── app.py                      Streamlit application (7 tabs, custom design system)
├── train_classifier.py         End-to-end build/train/evaluate script
├── config.py                   Central paths, constants & env-var feature flags
├── requirements.txt
├── pytest.ini
├── .env.example                 Optional feature-flag documentation (no secrets required)
├── .streamlit/config.toml       App theme
├── Dockerfile
├── docker-compose.yml
├── agrovox.db                  SQLite DB (created on first run)
│
├── tests/                       40-test automated suite (pytest)
│   ├── test_preprocessing.py
│   ├── test_translation.py
│   ├── test_ner.py
│   ├── test_classification.py
│   ├── test_qa.py
│   ├── test_db.py
│   └── test_integration_pipeline.py
│
├── data/
│   ├── generate_knowledge_base.py   → crops/diseases/fertilizers/irrigation/pests/soil/faq/translations CSVs
│   ├── generate_intents.py          → intents.csv (453 labelled queries, 10 classes)
│   ├── generate_ner_data.py         → ner_data.json (gazetteer patterns + eval set)
│   ├── generate_qa_dataset.py       → qa_dataset.csv (retrieval corpus)
│   ├── crops.csv / diseases.csv / fertilizers.csv / irrigation.csv /
│   │   pests.csv / soil.csv / agriculture_faq.csv / intents.csv /
│   │   ner_data.json / qa_dataset.csv / translations.csv
│   ├── raw/          farmer_queries_raw.csv (ETL Extract output — messy)
│   ├── processed/    farmer_queries_clean.csv, transform_report.json (ETL Transform output)
│
├── etl/                         Extract → Transform → Load pipeline
│   ├── extract.py
│   ├── transform.py
│   ├── load.py
│   └── etl_pipeline.py           orchestrator
│
├── eda/                          Exploratory Data Analysis
│   ├── exploratory_analysis.py
│   └── results/                  saved PNGs + eda_summary.json (generated)
│
├── models/
│   ├── classifier/   best_classifier.joblib, tfidf_vectorizer.joblib, label_encoder.joblib
│   ├── ner/          (spaCy EntityRuler is rebuilt at runtime from ner_data.json)
│   ├── embeddings/   qa_tfidf_vectorizer.joblib, qa_tfidf_matrix.joblib
│   └── translation/  (reserved for a neural MT checkpoint, if enabled)
│
├── src/
│   ├── preprocessing/text_preprocessing.py
│   ├── speech/speech_engine.py
│   ├── translation/translator.py
│   ├── classification/classifier.py
│   ├── ner/ner_engine.py
│   ├── qa/qa_engine.py
│   ├── chatbot/chatbot_engine.py
│   └── retrieval/knowledge_base.py
│
├── evaluation/
│   ├── classification_metrics.py
│   ├── ner_metrics.py
│   ├── translation_metrics.py
│   ├── qa_evaluation.py
│   ├── speech_evaluation.py
│   └── results/   (JSON reports + confusion_matrix.png, generated)
│
└── utils/db.py                 SQLite persistence layer
```

## 9. Technology Stack

- **Core**: Python 3.10+, pandas, numpy
- **NLP**: NLTK (tokenize/stopwords/lemmatize/BLEU), spaCy (`EntityRuler` NER)
- **ML**: scikit-learn (TF-IDF, Logistic Regression, Linear SVM, Naive Bayes)
- **Speech**: `openai-whisper` (optional), `gTTS` (optional)
- **DB**: SQLite (via the standard library `sqlite3`)
- **UI**: Streamlit, Plotly, Matplotlib/Seaborn (confusion matrix)
- **Optional heavy stack**: `transformers`, `torch`, `sentence-transformers`
  (commented out in `requirements.txt` — enable if you have internet+GPU)

## 10. Installation & How to Run

```bash
# 1. Clone/unzip the project, then from the AgroVox/ directory:
pip install -r requirements.txt

# 2. One-time NLTK data download (also done automatically on first import)
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4']]"

# 3. (Optional) configure environment variables - see .env.example.
#    Not required: AgroVox runs fully offline by default.
cp .env.example .env

# 4. Generate datasets, train models, and run the full evaluation suite:
python train_classifier.py

# 5. Launch the app:
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

### Running the automated test suite

```bash
pip install pytest
python train_classifier.py   # tests need a trained model/QA index present
pytest tests/ -v
```

40 tests cover preprocessing, translation, NER, classification, QA
retrieval, the database layer, and full end-to-end multi-turn pipeline
integration (including edge cases: empty input, very long input,
SQL-injection-style text, mixed-language text). All 40 pass as of this
build.

### Environment variables

AgroVox needs **no environment variables or API keys to run** - every
value in `.env.example` is an opt-in feature flag for the optional heavy
neural backends (see §7 Fallback Strategy). Copy `.env.example` to `.env`
only if you want to enable one of those.

| Variable | Default | Purpose |
|---|---|---|
| `AGROVOX_ENABLE_NEURAL_MT` | `false` | Use a downloaded neural MT model instead of the dictionary translator |
| `AGROVOX_ENABLE_NEURAL_QA` | `false` | Use a sentence-embedding QA retriever instead of TF-IDF |
| `AGROVOX_ENABLE_WHISPER_ASR` | `false` | Use Whisper for real speech-to-text |
| `AGROVOX_ENABLE_TTS` | `false` | Use gTTS for real text-to-speech |
| `HUGGINGFACE_TOKEN` | _(empty)_ | Only needed for a gated HF model under the flags above |
| `AGROVOX_DB_PATH` | `<project>/agrovox.db` | Override the SQLite database file location (used by the Docker setup below) |

## 11. Deployment

### Option A — Docker (recommended for production)

```bash
docker compose up --build
```

This builds the image (installing dependencies, downloading NLTK corpora,
and running `train_classifier.py` at build time so the container starts
with models already trained), then serves the app on
`http://localhost:8501`. The SQLite database and model artifacts persist
across restarts via named Docker volumes (`agrovox-db`, `agrovox-models`).
To enable an optional neural backend, set the corresponding variable in a
`.env` file before running `docker compose up` (Compose reads it
automatically).

To build/run manually without Compose:

```bash
docker build -t agrovox .
docker run -p 8501:8501 -v agrovox-db:/app/db-data \
    -e AGROVOX_DB_PATH=/app/db-data/agrovox.db agrovox
```

### Option B — Streamlit Community Cloud

1. Push this repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at `app.py` on your default branch.
3. Streamlit Cloud installs `requirements.txt` automatically. Add a
   `packages.txt` file if your platform additionally needs system
   packages for spaCy/matplotlib (usually not required).
4. In the app's "Secrets" panel, add any of the optional variables from
   `.env.example` you want enabled - otherwise leave it empty; the app
   works fully offline by default.
5. **Important**: run `python train_classifier.py` once locally and commit
   the generated `models/` and `data/*.csv` artifacts (or add a one-time
   startup hook that runs it), since Streamlit Cloud's filesystem is
   read-only-ish between deploys for anything not in your repo.

### Option C — Any Python host (Render, Railway, Fly.io, a VM, etc.)

```bash
pip install -r requirements.txt
python train_classifier.py
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Persist `agrovox.db` on a mounted volume/disk if your platform's
filesystem is ephemeral, using `AGROVOX_DB_PATH` to point at it.

## 12. Model Training & Evaluation

Run `python train_classifier.py` to regenerate every dataset (if missing),
retrain the classifier, rebuild the NER pipeline and QA index, and run all
five evaluation scripts. Each evaluation script can also be run
individually, e.g. `python evaluation/ner_metrics.py`. Reports land in
`evaluation/results/` as JSON (+ a confusion-matrix PNG), and the same
numbers are rendered live in the app's **NLP Evaluation** tab.

**Representative results from this build** (regenerate yourself — nothing
here is hand-typed into the app):

- Text Classification (Linear SVM, best of 3 models): **~96.7% accuracy**,
  ~0.955 macro F1 on a held-out 20% test split.
- NER (gazetteer EntityRuler): **~93.5% precision / 96.7% recall / 95.1% F1**
  on 10 held-out hand-annotated sentences.
- QA retrieval (TF-IDF + cosine similarity): **~98% top-1 accuracy** on
  paraphrased held-out queries.
- Translation: reported via glossary coverage (dictionary backend) since
  BLEU is only meaningful for a neural generative MT model — see
  `evaluation/translation_metrics.py`.
- Speech: Word Error Rate (WER) computation is demonstrated on illustrative
  transcript pairs, since no microphone/audio corpus exists in a headless
  build environment — see `evaluation/speech_evaluation.py`.

## 13. Data Pipeline — ETL & Exploratory Data Analysis (EDA)

AgroVox includes a full, runnable **ETL (Extract–Transform–Load)** pipeline
and an **EDA (Exploratory Data Analysis)** module, both live in the app's
**🔬 Data Pipeline (ETL & EDA)** tab and runnable standalone.

### ETL — `etl/`

| Step | File | What it does |
|---|---|---|
| Extract | `etl/extract.py` | Generates a deliberately **messy** raw farmer-query log (`data/raw/farmer_queries_raw.csv`) simulating real-world collection defects: inconsistent casing, stray whitespace, duplicate submissions, missing `language` metadata, and empty/junk rows |
| Transform | `etl/transform.py` | Cleans it: missing-value handling (drop unrecoverable rows, impute recoverable ones), duplicate removal, junk-content filtering, text normalization/tokenization/lemmatization, schema validation — writes `data/processed/farmer_queries_clean.csv` + a full data-lineage report (`transform_report.json`) |
| Load | `etl/load.py` | Persists the cleaned data into a dedicated SQLite table (`processed_farmer_queries`) **and** merges genuinely-new rows into the classifier's training corpus (`data/intents.csv`), using normalized-key deduplication to avoid counting case-variant near-duplicates as new data |
| Orchestrator | `etl/etl_pipeline.py` | Runs Extract → Transform → Load in sequence and prints a row-count lineage summary |

```bash
python etl/etl_pipeline.py            # run the full pipeline once
python etl/etl_pipeline.py --retrain  # ...and retrain the classifier afterward
```

**A real bug this caught**: the first version of the Load step deduplicated
against the existing corpus using an *exact* string match, so cleaned
(lowercased) text that was merely a case-variant of an existing row slipped
through as "new" training data. Those near-duplicates then got split across
the train/test split, leaking information and producing a meaningless 100%
test accuracy. Fixed by comparing on a case/whitespace-**normalized** key
instead — see `tests/test_etl.py::test_load_to_training_corpus_prevents_near_duplicate_leakage`.

### EDA — `eda/`

`eda/exploratory_analysis.py` is a library of pure analysis functions
(class balance, text-length distributions, language distribution, word
frequency, duplicate/missing-value summaries, knowledge-base coverage) used
both by a standalone report script and directly by the Streamlit tab for
live charts:

```bash
python eda/exploratory_analysis.py   # saves PNGs + eda_summary.json to eda/results/
```

**A real bug this caught**: EDA's class-balance check revealed that
"General Agriculture" had only **18 examples (4%)** of the training data
versus **~50 (11%)** for every other class — because 6 of its 8 question
templates were static sentences with no variable slots, so the
template-based generator could only ever produce one unique sentence from
each. Fixed by adding genuine paraphrase variety to those templates
(18 → 43 examples), bringing the class in line with the rest — see
`data/generate_intents.py` and the imbalance-ratio check in the EDA tab.

## 14. Production Audit — bugs found & fixed

A full audit pass (stress-testing every module with empty input, very long
input, SQL-injection-style text, and multi-turn conversations, backed by a
40-test automated suite in `tests/`) found and fixed four real defects
before this build:

1. **Translator substring corruption**: the original dictionary translator
   did raw substring replacement, so a short glossary entry like "is" would
   corrupt unrelated words that merely contain it (e.g. "this", "disease").
   Fixed by switching to whitespace-tokenized, longest-phrase-first
   matching (`src/translation/translator.py`).
2. **Tamil grapheme-splitting risk**: a naive `\b`/`\w`-based word-boundary
   fix would have silently separated Tamil vowel signs/virama (Unicode
   combining marks) from their base letters. Verified this does NOT happen
   with the token-based approach actually shipped (`tests/test_translation.py`).
3. **Case-sensitive URL stripping**: `clean_text()` failed to strip
   uppercase URLs (`HTTPS://...`) before lowercasing, leaving `http`
   artifacts in preprocessed text. Fixed with `re.IGNORECASE`
   (`src/preprocessing/text_preprocessing.py`).
4. **Chatbot context loss on short follow-ups**: a short reply like
   "Vegetative stage." was classified in isolation and could jump to an
   unrelated intent, losing the previous turn's topic. Fixed by carrying
   the conversation's last effective intent forward for short, entity-less
   follow-up turns (`src/chatbot/chatbot_engine.py`).

Also completed during the audit: input validation and error handling in
the Streamlit UI (empty/too-short query guards, a character limit, and a
try/except boundary around the pipeline call so malformed input surfaces a
friendly message instead of a crash), and a persistent "Resume with
existing Farmer ID" flow so a farmer's profile and query history survive
across browser sessions instead of resetting every time (previously the
Farmer Profile tab only showed the current, ephemeral session).

## 15. Safety

AgroVox is an **agricultural information assistant**, not a diagnostic tool
or a replacement for a certified agronomist. It does not guarantee disease
diagnosis, and it deliberately avoids giving precise pesticide/chemical
dosage instructions — every disease/pest entry instead points toward
verifying with a qualified agricultural expert for anything severe or
uncertain. This disclaimer is shown after every chatbot answer in the UI.

## 16. Sample Input/Output (Demonstration Scenario)

**Tamil voice/text input:**
`என் நெற்பயிரில் இலைகள் மஞ்சளாகிறது. என்ன செய்ய வேண்டும்?`

**Pipeline trace (visible in the Query Analysis tab):**
1. Language detected: Tamil
2. Translated to English: "my rice leaves yellowing what to do"
3. Preprocessed tokens → intent classifier → **Crop Disease** (high confidence)
4. NER → `CROP: rice`, `SYMPTOM: yellowing`
5. QA retrieval (category-boosted toward Crop Disease) → matches "What
   causes yellowing of rice leaves?"
6. Chatbot composes a reply, optionally asking for the crop's growth stage
7. Reply translated back to Tamil and shown (and optionally spoken) to the farmer

## 17. Limitations

- Translation and speech run in documented local-fallback mode by default
  (see §7); enabling the neural/cloud backends requires internet + extra
  dependencies.
- The intent/NER training data is template-generated from a curated
  agricultural vocabulary rather than collected from real farmer
  transcripts — a natural next step for a production system.
- Tamil text preprocessing is limited to tokenization/normalization; no
  Tamil-specific lemmatizer/stemmer is bundled offline.

## 18. Future Enhancements

- Swap in a downloaded IndicTrans2/NLLB checkpoint for real neural MT.
- Fine-tune a transformer classifier/NER model and compare against the
  TF-IDF/EntityRuler baselines already in place (hooks are already wired).
- Expand the knowledge base with more crops/regions and real farmer FAQ logs.
- Add authenticated multi-farmer accounts instead of session-generated IDs.

## 19. Team Contribution

_(Fill in for your lab submission, e.g.:)_

| Member | Contribution |
|---|---|
| — | Data collection & knowledge base curation |
| — | Text classification & NER modules |
| — | Translation, speech & QA modules |
| — | Streamlit UI, database layer & evaluation suite |

---

*Generated as a complete, runnable NLP laboratory project. Every number in
this README comes from actually running the code in `evaluation/` — re-run
`python train_classifier.py` at any time to reproduce them.*
