"""AI_SUMMARIZE — IRISModel that produces an extractive summary of input text.

SQL usage:

    CREATE MODEL ReportSummarizer PREDICTING (executive_summary)
    FROM AIFunctions.MarketResearch
    USING {
        "pathtoclassifiers": "/opt/irisapp/demos/ai_functions/iris_models",
        "iscmodelsdisabled": 1,
        "userparams": {"max_sentences": 3}
    };
    TRAIN MODEL ReportSummarizer;
    SELECT report_id, PREDICT(ReportSummarizer) AS executive_summary
    FROM AIFunctions.MarketResearch;

The summary is produced by ranking sentences via TF-IDF cosine similarity to
the document centroid (a lightweight TextRank approximation) and returning
the top-K sentences in their original order. This means the demo runs without
any external LLM call while keeping the SQL surface area identical to
SingleStore's `aura.AI_SUMMARIZE(text, length)`.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_sentences(text: str) -> list:
    if not isinstance(text, str) or not text.strip():
        return []
    parts = _SENT_RE.split(text.strip())
    # Collapse trailing whitespace/newlines.
    return [p.strip() for p in parts if p and p.strip()]


def _extract_text_column(X) -> list:
    if isinstance(X, pd.DataFrame):
        for cand in ("text", "full_report", "article_body", "body",
                     "content", "document", "ticket_body", "report_text"):
            if cand in X.columns:
                return X[cand].astype(str).tolist()
        for col in X.columns:
            if X[col].dtype == object:
                return X[col].astype(str).tolist()
        return X.iloc[:, 0].astype(str).tolist()
    arr = np.asarray(X)
    if arr.ndim == 1:
        return [str(v) for v in arr.tolist()]
    return [str(v) for v in arr[:, 0].tolist()]


def _extract_length_column(X, default: int) -> list:
    if isinstance(X, pd.DataFrame):
        for cand in ("length", "max_sentences", "summary_length",
                     "n_sentences"):
            if cand in X.columns:
                vals = []
                for v in X[cand].tolist():
                    try:
                        vals.append(max(1, int(v)))
                    except (TypeError, ValueError):
                        vals.append(default)
                return vals
    return None


def _summarize_one(text: str, max_sentences: int) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True,
                                 ngram_range=(1, 2), min_df=1)
    try:
        mat = vectorizer.fit_transform(sentences)
    except ValueError:
        # Empty vocabulary (e.g. pure punctuation). Take the first K sentences.
        return " ".join(sentences[:max_sentences])

    centroid = np.asarray(mat.mean(axis=0))
    sims = cosine_similarity(mat, centroid).ravel()
    top_idx = np.argsort(-sims)[:max_sentences]
    top_idx_sorted = sorted(top_idx.tolist())
    return " ".join(sentences[i] for i in top_idx_sorted)


class IRISModel:
    """Extractive summarizer."""

    name = "ai_summarize"

    def __init__(self, max_sentences: int = 3, **kwargs):
        self.max_sentences = int(max_sentences)
        self.model = self

    def get_params(self, deep=True):
        return {"max_sentences": self.max_sentences}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X):
        texts = _extract_text_column(X)
        per_row_lengths = _extract_length_column(X, self.max_sentences)
        out = []
        for i, t in enumerate(texts):
            k = per_row_lengths[i] if per_row_lengths else self.max_sentences
            out.append(_summarize_one(t, k))
        return np.array(out, dtype=object)
