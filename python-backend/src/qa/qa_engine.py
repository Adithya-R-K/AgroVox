"""
AgroVox - Module 5: Question Answering (semantic retrieval)

STRATEGY
--------
Real semantic retrieval-based QA: builds a TF-IDF vector space over the
question+answer text of every row in data/qa_dataset.csv (derived from the
structured knowledge base) and answers a new question by cosine-similarity
nearest-neighbour retrieval - a standard, fully local, Retrieval-based QA
approach (a lightweight stand-in for a sentence-embedding + ANN retriever
when `sentence-transformers` and internet access to download an embedding
model are not available; see `SentenceEmbeddingQA` below for the optional
neural upgrade path documented for environments that do have that access).
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.preprocessing.text_preprocessing import preprocess

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFQAEngine:
    def __init__(self, qa_csv: str = config.QA_CSV):
        self.df = pd.read_csv(qa_csv)
        self.vectorizer = None
        self.matrix = None
        self._build_or_load_index()

    def _corpus_text(self, row):
        # Index on STEMMED text (question weighted x3, answer x1) so
        # inflectional variants like "yellow"/"yellowing" or
        # "irrigate"/"irrigation" still match via TF-IDF bag-of-words
        # retrieval, while question terms dominate the similarity signal.
        q_stem = preprocess(row["question"])["stemmed_text"]
        a_stem = preprocess(str(row["answer"]))["stemmed_text"]
        return (q_stem + " ") * 3 + a_stem

    def _build_or_load_index(self):
        if os.path.exists(config.QA_VECTORIZER_PATH) and os.path.exists(config.QA_MATRIX_PATH):
            self.vectorizer = joblib.load(config.QA_VECTORIZER_PATH)
            self.matrix = joblib.load(config.QA_MATRIX_PATH)
            if self.matrix.shape[0] == len(self.df):
                return
        self.build_index()

    def build_index(self):
        corpus = self.df.apply(self._corpus_text, axis=1)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(corpus)
        joblib.dump(self.vectorizer, config.QA_VECTORIZER_PATH)
        joblib.dump(self.matrix, config.QA_MATRIX_PATH)

    def answer(self, question: str, top_k: int = 3, language: str = "en",
               category_hint: str = None, category_boost: float = 0.15) -> dict:
        """Retrieve the best-matching QA pair. If `category_hint` (typically
        the text-classification module's predicted intent) is supplied, rows
        whose category matches get a small similarity boost - this is how
        the Classification module (upstream) improves the QA module
        (downstream) inside AgroVox's single integrated pipeline, rather
        than QA running as an isolated component."""
        query_text = preprocess(question, language="en")["stemmed_text"]
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.matrix).flatten()

        if category_hint:
            boost_mask = (self.df["category"] == category_hint).to_numpy()
            sims = sims + boost_mask * category_boost

        top_idx = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_idx:
            row = self.df.iloc[idx]
            results.append({
                "question": row["question"], "answer": row["answer"],
                "category": row["category"], "source": row["source"],
                "similarity": float(sims[idx]),
            })

        best = results[0] if results else None
        return {
            "query": question,
            "answer": best["answer"] if best else
                "I don't have specific information on that yet. Please consult your local "
                "agricultural extension office for expert guidance.",
            "matched_question": best["question"] if best else None,
            "category": best["category"] if best else None,
            "source": best["source"] if best else None,
            "confidence": best["similarity"] if best else 0.0,
            "alternatives": results[1:],
        }


class SentenceEmbeddingQA:
    """Optional neural-embedding upgrade (requires `sentence-transformers` +
    internet access to download an embedding checkpoint). Not used by
    default - see README -> Fallback Strategy. Kept here so the codebase
    demonstrates the full RAG-style architecture referenced in the spec."""

    _MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, qa_csv: str = config.QA_CSV):
        self.df = pd.read_csv(qa_csv)
        self._model = None
        self._embeddings = None
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            self._available = True
        except Exception:
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def build_index(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._MODEL_NAME)
        self._embeddings = self._model.encode(self.df["question"].tolist(), normalize_embeddings=True)

    def answer(self, question: str, top_k: int = 3) -> dict:
        if self._model is None:
            self.build_index()
        q_emb = self._model.encode([question], normalize_embeddings=True)
        sims = (self._embeddings @ q_emb.T).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = [{
            "question": self.df.iloc[i]["question"], "answer": self.df.iloc[i]["answer"],
            "category": self.df.iloc[i]["category"], "source": self.df.iloc[i]["source"],
            "similarity": float(sims[i]),
        } for i in top_idx]
        best = results[0]
        return {"query": question, "answer": best["answer"], "matched_question": best["question"],
                 "category": best["category"], "source": best["source"],
                 "confidence": best["similarity"], "alternatives": results[1:]}


def get_qa_engine():
    """Factory: uses the neural sentence-embedding engine only if the
    AGROVOX_ENABLE_NEURAL_QA env flag is set (see config.py / .env.example)
    AND `sentence-transformers` + a downloadable checkpoint are available;
    otherwise (the default) uses the always-available TF-IDF retrieval
    engine, which needs no network access or extra dependencies."""
    if config.ENABLE_NEURAL_QA:
        neural = SentenceEmbeddingQA()
        if neural.is_available():
            try:
                neural.build_index()
                return neural
            except Exception:
                pass
    return TFIDFQAEngine()


if __name__ == "__main__":
    qa = TFIDFQAEngine()
    r = qa.answer("What fertilizer is suitable for rice during the vegetative stage?")
    print(r["answer"])
    print("confidence:", r["confidence"])
