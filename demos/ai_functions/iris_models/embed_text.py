"""EMBED_TEXT — IRISModel that turns text into a fixed-dimensional vector.

SQL usage:

    CREATE MODEL ArticleEmbedder PREDICTING (content_vector)
    FROM AIFunctions.Articles
    USING {
        "pathtoclassifiers": "/opt/irisapp/demos/ai_functions/iris_models",
        "iscmodelsdisabled": 1,
        "userparams": {"dim": 128}
    };
    TRAIN MODEL ArticleEmbedder;
    SELECT article_id, PREDICT(ArticleEmbedder) AS content_vector
    FROM AIFunctions.Articles;

The default backend is sklearn's HashingVectorizer wrapped in a TF-IDF-style
sublinear weighting and L2-normalised — a stable, dependency-free embedding
that can power semantic-search demos without any external API. Vectors are
returned as JSON-encoded float lists so they fit a VARCHAR column. Use
`SELECT JSON_TABLE(...)` or a small UDF to materialise them as IRIS VECTORs
when you wire this into a real RAG pipeline.

Self-contained for irispython: only standard library + sklearn + numpy + pandas.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


def _extract_text_column(X) -> list:
    if isinstance(X, pd.DataFrame):
        for cand in ("text", "article_body", "body", "content", "review_text",
                     "comment_text", "description", "document"):
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


class IRISModel:
    """Hashing-trick text embedder."""

    name = "embed_text"

    def __init__(self, dim: int = 128, ngram_range=(1, 2), **kwargs):
        self.dim = int(dim)
        if isinstance(ngram_range, str):
            # IRIS may stringify tuple params; accept "1,2" too.
            parts = [p.strip() for p in ngram_range.split(",")]
            ngram_range = (int(parts[0]), int(parts[1]))
        self.ngram_range = (int(ngram_range[0]), int(ngram_range[1]))
        self._vectorizer = HashingVectorizer(
            n_features=self.dim,
            ngram_range=self.ngram_range,
            alternate_sign=False,
            norm=None,
            lowercase=True,
        )
        self.model = self

    def get_params(self, deep=True):
        return {"dim": self.dim, "ngram_range": self.ngram_range}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        # Recompute the underlying vectorizer when params change.
        self._vectorizer = HashingVectorizer(
            n_features=self.dim,
            ngram_range=self.ngram_range,
            alternate_sign=False,
            norm=None,
            lowercase=True,
        )
        return self

    def fit(self, X, y=None, **kwargs):
        return self

    def embed(self, X) -> np.ndarray:
        texts = _extract_text_column(X)
        if not texts:
            return np.zeros((0, self.dim), dtype=float)
        mat = self._vectorizer.transform(texts).toarray().astype(float)
        # Sublinear TF (log scaling) to dampen frequent-token dominance.
        np.log1p(mat, out=mat)
        # L2-normalise so cosine similarity = dot product.
        mat = normalize(mat, norm="l2", axis=1)
        return mat

    def predict(self, X):
        mat = self.embed(X)
        # IRIS columns are scalars per row, so serialise vectors to JSON
        # strings. Callers can deserialise with JSON_TABLE or %JSON.Object.
        out = [json.dumps([round(float(v), 6) for v in row]) for row in mat]
        return np.array(out, dtype=object)
