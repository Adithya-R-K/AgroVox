# AgroVox — Project Code

This zip contains two parts of the same project:

```
agrovox-project/
├── python-backend/   The original NLP project: ETL, EDA, text classifier,
│                      NER, QA engine, translator, speech engine, chatbot,
│                      evaluation scripts, tests, and a Streamlit app (app.py).
│                      This is the "real" implementation and where the six
│                      NLP concepts are actually implemented/trained.
│
└── website/           A self-contained, static demo site (index.html) that
                        runs a genuine subset of the backend's NLP logic
                        (classification, NER, translation, QA retrieval)
                        directly in the browser — exported from the trained
                        Python model, not re-implemented from scratch.
```

## python-backend/

Standard layout:
- `data/` — raw + processed datasets, generator scripts, knowledge base CSVs
- `etl/` — extract.py / transform.py / load.py
- `eda/` — exploratory analysis + generated charts (`eda/results/`)
- `src/` — one subfolder per NLP module (classification, ner, qa, translation,
  speech, chatbot, preprocessing, retrieval)
- `evaluation/` — metrics scripts + generated results (`evaluation/results/`)
- `notebooks/phase1_eda_etl_classification.ipynb` — the executed Phase I
  notebook (EDA + ETL + Text Classification + evaluation, with real outputs)
- `train_classifier.py`, `app.py` — entry points (Streamlit app)
- `export_for_browser.py` — exports the trained classifier, QA knowledge
  base, NER gazetteer, and translation dictionary as JSON for the website
  (output already included at `browser_export/`, and copied into
  `website/src/data_blob.json`)
- `tests/` — pytest suite

Setup:
```
cd python-backend
pip install -r requirements.txt
python train_classifier.py     # or: streamlit run app.py
```

## website/

- `index.html` — the final, self-contained build (what's actually published)
- `src/` — the pieces `assemble.py` stitches into `index.html`:
  - `template.html` — page structure + CSS
  - `app.js` — UI wiring, pipeline orchestration, chart rendering (with
    fallback if a CDN script fails to load)
  - `icons.js` — the inline SVG icon set (no emoji, for consistent rendering)
  - `nlp_core.browser.js` — the actual NLP logic: TF-IDF vectorization,
    Naive Bayes inference, gazetteer NER, dictionary translation, QA
    cosine-similarity retrieval
  - `porter_stemmer.browser.js` — a Porter stemmer, tuned to match Python's
    `SnowballStemmer` closely enough for classification features
  - `data_blob.json` — the trained model weights + knowledge base, exported
    by `python-backend/export_for_browser.py`
- `validation/` — the test scripts used to verify the above against the
  real Python outputs before shipping (see below)

Rebuild after editing `src/`:
```
cd website/src
python3 assemble.py    # writes ../../website/index.html
```

### Why a website reimplementation exists at all

The classifier, NER, translator, and QA retrieval in `nlp_core.browser.js`
are not new models — they're the trained artifacts from `python-backend/`,
exported as plain JSON (vocabulary, IDF weights, Naive Bayes log-probabilities,
gazetteer terms, translation dictionary) and re-run with equivalent JS math,
so the demo works standalone in a browser with no server or API key.

### Validation

Before shipping, each piece of browser-side logic was checked against the
real Python pipeline, not just assumed to be equivalent:

- `validation/porter_stemmer_node_version.js` — the stemmer, checked against
  `nltk.SnowballStemmer` on every unique word in the training corpus
  (266/274 exact matches)
- `validation/nlp_core_node_version.js` — the classifier/translator/NER/QA
  logic, checked against Python's actual predictions on held-out text
  (99% agreement with Python's classifier output; translation output is
  byte-for-byte identical on tested sentences)
- `validation/real_browser_test.py`, `interaction_test.py`, `full_qa_test.py`
  — real headless-Chromium tests (via Playwright) of the assembled
  `index.html`: clicking sample queries, running the full pipeline,
  dark mode, viva mode, tabs, history, and feedback — checking for
  uncaught JS errors and correct DOM state, not just "it loads"
- `validation/diagnose_cdn.py`, `section_shots.py` — diagnostic scripts used
  to catch a real bug where a blocked/slow chart CDN script could silently
  break the rest of the page (fixed via per-section error isolation in
  `app.js`, with a plain CSS bar-chart fallback if Chart.js is unavailable)

Run these from a machine with Python + Playwright + a Node.js install
(`pip install playwright && playwright install chromium`, `npm install jsdom`
if using the older jsdom-based smoke test).
