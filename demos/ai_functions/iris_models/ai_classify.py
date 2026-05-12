"""AI_CLASSIFY — IRISModel for zero/few-shot text classification.

SQL usage:

    CREATE MODEL TicketRouter PREDICTING (department)
    FROM AIFunctions.SupportTickets
    USING {
        "pathtoclassifiers": "/opt/irisapp/demos/ai_functions/iris_models",
        "iscmodelsdisabled": 1
    };
    TRAIN MODEL TicketRouter;
    SELECT ticket_id, PREDICT(TicketRouter) AS department
    FROM AIFunctions.SupportTickets;

This model fits an in-memory TF-IDF vectorizer over both the training texts
and a small set of category seed phrases, then routes each new row to the
nearest category by cosine similarity. When labelled training rows are
available, AI_CLASSIFY leans on them; otherwise it falls back to seed phrases
embedded in the model itself, which is the SingleStore "AI_CLASSIFY(text,
categories)" zero-shot behaviour.

Self-contained for irispython: only sklearn / numpy / pandas.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Seed phrases per category. Used when no training labels are present, or as
# extra anchors when they are. Kept small so behaviour is predictable in tests.
_DEFAULT_SEEDS = {
    "billing": [
        "I was charged twice for my subscription",
        "refund my payment please",
        "invoice and credit card statement",
        "cancel my plan and stop billing",
        "unexpected fee on my account",
    ],
    "technical": [
        "the app keeps crashing on startup",
        "I see an error when I click the button",
        "page does not load and shows a 500",
        "login is broken and I cannot reset password",
        "feature is not working as expected",
    ],
    "sales": [
        "interested in upgrading to the enterprise plan",
        "can I get a quote for my team",
        "questions about pricing and discounts",
        "request a product demo",
        "compare your tiers with the competition",
    ],
    "returns": [
        "I want to return this item it is the wrong size",
        "package arrived damaged and I need a replacement",
        "where do I ship the return label",
        "exchange my order for a different colour",
        "the product is defective please refund",
    ],
}


_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def _extract_text_column(X) -> list:
    if isinstance(X, pd.DataFrame):
        for cand in ("text", "ticket_body", "message_text", "message",
                     "body", "review_text", "comment_text", "content",
                     "article_body"):
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
    """Zero/few-shot text classifier."""

    name = "ai_classify"

    def __init__(self, categories=None, seeds=None, min_seed_weight: float = 0.3,
                 **kwargs):
        self.categories = list(categories) if categories else None
        self.seeds = dict(seeds) if seeds else dict(_DEFAULT_SEEDS)
        self.min_seed_weight = float(min_seed_weight)

        self._vectorizer = None
        self._category_vectors = None
        self._categories_in_use = None
        self.classes_ = None
        self.model = self

    def get_params(self, deep=True):
        return {
            "categories": list(self.categories) if self.categories else None,
            "min_seed_weight": self.min_seed_weight,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _resolve_categories(self, y):
        if self.categories:
            return list(self.categories)
        if y is not None:
            labels = pd.Series(y).dropna().astype(str).unique().tolist()
            if labels:
                return labels
        return list(self.seeds.keys())

    def fit(self, X, y=None, **kwargs):
        texts = _extract_text_column(X)
        cats = self._resolve_categories(y)
        self._categories_in_use = cats
        self.classes_ = np.array(cats, dtype=object)

        # Build per-category training corpus: labelled rows (if any) +
        # seed phrases as a fallback so unseen categories still have anchors.
        per_cat_docs = defaultdict(list)
        if y is not None:
            y_arr = np.asarray(y, dtype=object)
            for txt, lbl in zip(texts, y_arr):
                if lbl is None:
                    continue
                key = str(lbl)
                if key in cats:
                    per_cat_docs[key].append(str(txt))
        for cat in cats:
            for seed in self.seeds.get(cat, []):
                per_cat_docs[cat].append(seed)
            if not per_cat_docs[cat]:
                # Last resort: use the category name itself as an anchor.
                per_cat_docs[cat].append(cat)

        all_docs = []
        for cat in cats:
            all_docs.extend(per_cat_docs[cat])

        # Add the unlabelled training texts so vocabulary covers them too.
        if texts:
            all_docs.extend(texts)

        self._vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=1, stop_words="english"
        )
        self._vectorizer.fit(all_docs)

        cat_vectors = []
        for cat in cats:
            docs = per_cat_docs[cat]
            mat = self._vectorizer.transform(docs)
            # Mean of doc vectors as the category centroid.
            centroid = np.asarray(mat.mean(axis=0))
            cat_vectors.append(centroid.reshape(-1))
        self._category_vectors = np.vstack(cat_vectors)
        return self

    def _ensure_fitted(self):
        if self._vectorizer is None or self._category_vectors is None:
            # Allow predict-only flows by lazily initialising on the seed bank.
            self.fit(pd.DataFrame({"text": []}), y=None)

    def predict(self, X):
        self._ensure_fitted()
        texts = _extract_text_column(X)
        if not texts:
            return np.array([], dtype=object)
        mat = self._vectorizer.transform(texts)
        sims = cosine_similarity(mat, self._category_vectors)
        idx = sims.argmax(axis=1)
        return np.array(self._categories_in_use, dtype=object)[idx]

    def predict_proba(self, X):
        self._ensure_fitted()
        texts = _extract_text_column(X)
        if not texts:
            return np.zeros((0, len(self._categories_in_use)))
        mat = self._vectorizer.transform(texts)
        sims = cosine_similarity(mat, self._category_vectors)
        # Softmax-ish normalisation over similarity scores.
        sims = np.clip(sims, 0.0, None)
        row_sums = sims.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return sims / row_sums
