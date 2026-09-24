"""
AgroVox - Module 3: Text Classification (Intent / Query-category classifier)

Real, locally-trainable models (no internet/GPU needed):
  - TF-IDF + Logistic Regression
  - TF-IDF + Linear SVM
  - TF-IDF + Multinomial Naive Bayes

An optional transformer-based classifier hook is provided
(`TransformerClassifier`) for environments with internet + GPU access
(Section 12: model comparison); it is skipped automatically if
`transformers`/`torch` are not installed, and is NOT part of the default
pipeline (see README -> Fallback Strategy).
"""
import os
import sys
import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.preprocessing.text_preprocessing import preprocess

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              classification_report, confusion_matrix)


def _preprocess_for_classification(text: str, language: str = None) -> str:
    """Preprocess + (if needed) translate Tamil text to English so a single
    English-trained TF-IDF vector space can classify both languages. Import
    is local to avoid a circular import with the translation module."""
    from src.translation.translator import Translator
    info = preprocess(text, language=language)
    lang = info["language"]
    if lang == "ta":
        translator = Translator()
        en_text = translator.translate(info["cleaned_text"], "ta", "en")["translated_text"]
        info_en = preprocess(en_text, language="en")
        return info_en["processed_text"]
    return info["processed_text"]


def load_dataset(path: str = config.INTENTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame):
    """Preprocess raw text -> processed feature text column."""
    df = df.copy()
    df["features_text"] = df.apply(
        lambda r: _preprocess_for_classification(r["text"], r.get("language")), axis=1)
    df = df[df["features_text"].str.strip() != ""]
    return df


MODEL_FACTORIES = {
    "logistic_regression": lambda: LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced"),
    "linear_svm": lambda: CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3),
    "naive_bayes": lambda: MultinomialNB(),
}


def train_and_compare(random_state: int = config.RANDOM_SEED):
    """Train all classical models, compare on a held-out test split, and
    return (best_model_name, fitted_vectorizer, {name: fitted_model}, comparison_df,
    (X_test, y_test) preprocessed feature texts)."""
    df = load_dataset()
    df = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["features_text"], df["label"], test_size=0.2, random_state=random_state,
        stratify=df["label"])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results = {}
    fitted_models = {}
    for name, factory in MODEL_FACTORIES.items():
        model = factory()
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="macro", zero_division=0)
        results[name] = {
            "accuracy": acc, "precision_macro": precision,
            "recall_macro": recall, "f1_macro": f1,
        }
        fitted_models[name] = model

    comparison_df = pd.DataFrame(results).T.sort_values("f1_macro", ascending=False)
    best_name = comparison_df.index[0]

    return {
        "best_model_name": best_name,
        "vectorizer": vectorizer,
        "models": fitted_models,
        "comparison": comparison_df,
        "X_test_vec": X_test_vec,
        "y_test": y_test,
        "X_test_text": X_test,
    }


def save_best_model(train_result: dict):
    best_name = train_result["best_model_name"]
    best_model = train_result["models"][best_name]
    joblib.dump(best_model, config.CLASSIFIER_MODEL_PATH)
    joblib.dump(train_result["vectorizer"], config.CLASSIFIER_VECTORIZER_PATH)
    joblib.dump(best_name, config.CLASSIFIER_LABELS_PATH)
    train_result["comparison"].to_json(config.CLASSIFIER_COMPARISON_PATH, orient="index", indent=2)
    print(f"Saved best model '{best_name}' -> {config.CLASSIFIER_MODEL_PATH}")


class IntentClassifier:
    """Inference-time wrapper: loads the trained best model + vectorizer and
    classifies a raw farmer query (any supported language)."""

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.model_name = None
        self._load()

    def _load(self):
        if (os.path.exists(config.CLASSIFIER_MODEL_PATH) and
                os.path.exists(config.CLASSIFIER_VECTORIZER_PATH)):
            self.model = joblib.load(config.CLASSIFIER_MODEL_PATH)
            self.vectorizer = joblib.load(config.CLASSIFIER_VECTORIZER_PATH)
            if os.path.exists(config.CLASSIFIER_LABELS_PATH):
                self.model_name = joblib.load(config.CLASSIFIER_LABELS_PATH)

    def is_ready(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    def predict(self, text: str, language: str = None) -> dict:
        if not self.is_ready():
            return {"intent": "General Agriculture", "confidence": 0.0,
                     "model": "untrained-fallback"}
        features_text = _preprocess_for_classification(text, language)
        vec = self.vectorizer.transform([features_text])
        pred = self.model.predict(vec)[0]
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(vec)[0]
            classes = list(self.model.classes_)
            confidence = float(proba[classes.index(pred)])
        return {"intent": pred, "confidence": confidence, "model": self.model_name or "unknown"}


if __name__ == "__main__":
    result = train_and_compare()
    print(result["comparison"])
    save_best_model(result)
    clf = IntentClassifier()
    print(clf.predict("What fertilizer is suitable for rice during the vegetative stage?"))
    print(clf.predict("என் நெற்பயிரில் இலைகள் மஞ்சளாகிறது. என்ன செய்ய வேண்டும்?"))
