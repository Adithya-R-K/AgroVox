"""
Exports real, trained AgroVox artifacts as compact JSON so the same
TF-IDF + Naive Bayes classifier, TF-IDF QA retrieval, and gazetteer NER
that run in Python can run as genuine (not fabricated) client-side
inference inside a static, published demo page — this is exactly the
"Demo Mode / local dataset + lightweight model, no API keys, works
immediately" requirement in the project's own spec.

Run: python export_for_browser.py
"""
import sys, os, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.preprocessing.text_preprocessing import preprocess
from src.translation.translator import Translator

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "browser_export")
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------------------- 1. CLASSIFIER --
def stemmed_features(text, language=None):
    info = preprocess(text, language=language)
    lang = info["language"]
    if lang == "ta":
        translator = Translator()
        en_text = translator.translate(info["cleaned_text"], "ta", "en")["translated_text"]
        info = preprocess(en_text, language="en")
    return info["stemmed_text"]


def build_classifier_export():
    df = pd.read_csv(config.INTENTS_CSV).dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    df["features_text"] = df.apply(lambda r: stemmed_features(r["text"], r.get("language")), axis=1)
    df = df[df["features_text"].str.strip() != ""].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["features_text"], df["label"], test_size=0.2,
        random_state=config.RANDOM_SEED, stratify=df["label"])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    candidates = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced"),
    }
    scored = {}
    for name, model in candidates.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        scored[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, preds),
            "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
        }
        print(f"[stemmed-feature classifier] {name}: "
              f"accuracy={scored[name]['accuracy']:.3f}  f1_macro={scored[name]['f1_macro']:.3f}")

    best_name = max(scored, key=lambda n: scored[n]["f1_macro"])
    best_model = scored[best_name]["model"]
    print(f"[stemmed-feature classifier] Exporting best model: {best_name}")

    vocab = vectorizer.vocabulary_                      # term -> column index
    idf = vectorizer.idf_.tolist()                       # column index -> idf weight
    classes = list(best_model.classes_)

    export = {
        "model_type": best_name,
        "vocabulary": vocab,
        "idf": idf,
        "ngram_range": [1, 2],
        "classes": classes,
        "held_out_accuracy": round(scored[best_name]["accuracy"], 4),
        "held_out_f1_macro": round(scored[best_name]["f1_macro"], 4),
    }
    if best_name == "naive_bayes":
        export["class_log_prior"] = best_model.class_log_prior_.tolist()
        export["feature_log_prob"] = best_model.feature_log_prob_.tolist()  # [n_classes][n_features]
    else:
        export["coef"] = best_model.coef_.tolist()
        export["intercept"] = best_model.intercept_.tolist()

    with open(os.path.join(OUT_DIR, "classifier.json"), "w", encoding="utf-8") as f:
        json.dump(export, f)
    print(f"[EXPORT] classifier.json  (vocab={len(vocab)}, classes={len(classes)})")

    # also dump a handful of ground-truth (text -> stemmed -> prediction) pairs
    # for JS-side validation
    sanity = []
    for txt, true_label in list(zip(X_test, y_test))[:15]:
        pred = best_model.predict(vectorizer.transform([txt]))[0]
        sanity.append({"stemmed_text": txt, "true_label": true_label, "python_pred": pred})
    with open(os.path.join(OUT_DIR, "classifier_sanity_check.json"), "w", encoding="utf-8") as f:
        json.dump(sanity, f, indent=2)


# --------------------------------------------------------------- 2. QA RETRIEVAL --
def build_qa_export():
    df = pd.read_csv(config.QA_CSV).dropna(subset=["question", "answer"]).reset_index(drop=True)

    def _feat(text):
        info = preprocess(text, language="en")
        return info["stemmed_text"]

    df["features_text"] = df["question"].apply(_feat)
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1, sublinear_tf=True)
    doc_vecs = vectorizer.fit_transform(df["features_text"]).toarray()

    export = {
        "vocabulary": vectorizer.vocabulary_,
        "idf": vectorizer.idf_.tolist(),
        "questions": df["question"].tolist(),
        "answers": df["answer"].tolist(),
        "categories": df["category"].tolist(),
        "doc_vectors": np.round(doc_vecs, 6).tolist(),
    }
    with open(os.path.join(OUT_DIR, "qa_knowledge_base.json"), "w", encoding="utf-8") as f:
        json.dump(export, f)
    print(f"[EXPORT] qa_knowledge_base.json  ({len(df)} QA pairs, vocab={len(vectorizer.vocabulary_)})")


# --------------------------------------------------------------- 3. NER GAZETTEER --
def build_ner_export():
    gaz = {"CROP": [], "DISEASE": [], "PEST": [], "FERTILIZER": [], "SOIL_TYPE": []}

    crops = pd.read_csv(config.CROPS_CSV)
    gaz["CROP"] = sorted(set(crops["crop_name_en"].str.lower()) | set(crops["crop_name_ta"]))

    diseases = pd.read_csv(config.DISEASES_CSV)
    gaz["DISEASE"] = sorted(set(diseases["disease_name_en"].str.lower()) | set(diseases["disease_name_ta"]))

    pests = pd.read_csv(config.PESTS_CSV)
    pest_name_col = "pest_name_en" if "pest_name_en" in pests.columns else pests.columns[1]
    gaz["PEST"] = sorted(set(pests[pest_name_col].str.lower().dropna()))

    ferts = pd.read_csv(config.FERTILIZERS_CSV)
    fert_col = "fertilizer_name_en" if "fertilizer_name_en" in ferts.columns else ferts.columns[1]
    gaz["FERTILIZER"] = sorted(set(ferts[fert_col].str.lower().dropna()))

    soils = pd.read_csv(config.SOIL_CSV)
    soil_col = "soil_name_en" if "soil_name_en" in soils.columns else soils.columns[1]
    gaz["SOIL_TYPE"] = sorted(set(soils[soil_col].str.lower().dropna()))

    # common Tamil Nadu locations (small curated list — LOCATION isn't in a CSV)
    gaz["LOCATION"] = sorted([
        "coimbatore", "thanjavur", "madurai", "salem", "trichy", "tiruchirappalli",
        "erode", "vellore", "tirunelveli", "chennai", "kanyakumari", "dindigul",
        "cuddalore", "villupuram", "namakkal", "karur", "tamil nadu", "tanjore",
    ])

    with open(os.path.join(OUT_DIR, "ner_gazetteer.json"), "w", encoding="utf-8") as f:
        json.dump(gaz, f, ensure_ascii=False, indent=2)
    total = sum(len(v) for v in gaz.values())
    print(f"[EXPORT] ner_gazetteer.json  ({total} terms across {len(gaz)} entity types)")


# --------------------------------------------------------------- 4. TRANSLATION --
def build_translation_export():
    rows = list(csv.DictReader(open(config.TRANSLATIONS_CSV, encoding="utf-8")))
    en_to_ta = {r["term_en"].strip().lower(): r["term_ta"].strip() for r in rows}
    ta_to_en = {v: k for k, v in en_to_ta.items()}
    with open(os.path.join(OUT_DIR, "translation_dict.json"), "w", encoding="utf-8") as f:
        json.dump({"en_to_ta": en_to_ta, "ta_to_en": ta_to_en}, f, ensure_ascii=False, indent=2)
    print(f"[EXPORT] translation_dict.json  ({len(en_to_ta)} terms)")


# --------------------------------------------------------------- 5. STATS --
def build_stats_export():
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold
    from src.classification.classifier import load_dataset, build_features, MODEL_FACTORIES

    df = pd.read_csv(config.INTENTS_CSV)
    class_counts = df["label"].value_counts().to_dict()

    df_feat = build_features(load_dataset())
    X_train, X_test, y_train, y_test = train_test_split(
        df_feat["features_text"], df_feat["label"], test_size=0.2,
        random_state=config.RANDOM_SEED, stratify=df_feat["label"])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, preds, labels=labels).tolist()
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    # 5-fold CV (reliability)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    X_all, y_all = df_feat["features_text"].values, df_feat["label"].values
    cv_acc = []
    for train_idx, test_idx in skf.split(X_all, y_all):
        vec_fold = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)
        Xtr = vec_fold.fit_transform(X_all[train_idx])
        Xte = vec_fold.transform(X_all[test_idx])
        m = MultinomialNB().fit(Xtr, y_all[train_idx])
        cv_acc.append(accuracy_score(y_all[test_idx], m.predict(Xte)))

    stats = {
        "total_intent_records": int(len(df)),
        "class_counts": class_counts,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "vocab_size": len(vectorizer.vocabulary_),
        "held_out_accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "held_out_f1_macro": round(float(f1_score(y_test, preds, average="macro", zero_division=0)), 4),
        "held_out_precision_macro": round(float(report["macro avg"]["precision"]), 4),
        "held_out_recall_macro": round(float(report["macro avg"]["recall"]), 4),
        "cv_accuracy_mean": round(float(np.mean(cv_acc)), 4),
        "cv_accuracy_std": round(float(np.std(cv_acc)), 4),
        "confusion_matrix": cm,
        "confusion_labels": labels,
        "per_class_report": {k: v for k, v in report.items()
                              if k not in ("accuracy", "macro avg", "weighted avg")},
        "qa_pairs": None,   # filled below
        "translation_terms": None,
        "ner_terms_by_type": None,
    }

    qa_df = pd.read_csv(config.QA_CSV)
    stats["qa_pairs"] = int(len(qa_df))
    trans_rows = list(csv.DictReader(open(config.TRANSLATIONS_CSV, encoding="utf-8")))
    stats["translation_terms"] = len(trans_rows)

    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f)
    print(f"[EXPORT] stats.json  (held-out acc={stats['held_out_accuracy']}, "
          f"cv acc={stats['cv_accuracy_mean']}±{stats['cv_accuracy_std']})")


def build_data_blob():
    blob = {
        "classifier": json.load(open(os.path.join(OUT_DIR, "classifier.json"), encoding="utf-8")),
        "qa": json.load(open(os.path.join(OUT_DIR, "qa_knowledge_base.json"), encoding="utf-8")),
        "ner": json.load(open(os.path.join(OUT_DIR, "ner_gazetteer.json"), encoding="utf-8")),
        "translations": json.load(open(os.path.join(OUT_DIR, "translation_dict.json"), encoding="utf-8")),
        "stats": json.load(open(os.path.join(OUT_DIR, "stats.json"), encoding="utf-8")),
    }
    blob_path = os.path.join(OUT_DIR, "data_blob.json")
    with open(blob_path, "w", encoding="utf-8") as f:
        json.dump(blob, f)
    print(f"[EXPORT] Bundled data_blob.json -> {blob_path}")

    website_blob_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "website", "src", "data_blob.json")
    if os.path.exists(os.path.dirname(website_blob_path)):
        with open(website_blob_path, "w", encoding="utf-8") as f:
            json.dump(blob, f)
        print(f"[EXPORT] Copied data_blob.json -> {website_blob_path}")

        # Assemble website/index.html
        assemble_py = os.path.join(os.path.dirname(website_blob_path), "assemble.py")
        if os.path.exists(assemble_py):
            import subprocess
            subprocess.run([sys.executable, "assemble.py"], cwd=os.path.dirname(assemble_py), check=True)


if __name__ == "__main__":
    build_classifier_export()
    build_qa_export()
    build_ner_export()
    build_translation_export()
    build_stats_export()
    build_data_blob()
    print("\nAll browser export artifacts written to:", OUT_DIR)
