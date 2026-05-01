"""AI_SENTIMENT — IRISModel that classifies text as positive / negative / neutral.

SQL usage:

    CREATE MODEL AISentiment PREDICTING (sentiment)
    FROM AIFunctions.ProductReviews
    USING {
        "pathtoclassifiers": "/opt/irisapp/demos/ai_functions/iris_models",
        "iscmodelsdisabled": 1
    };
    TRAIN MODEL AISentiment;
    SELECT review_id, PREDICT(AISentiment) AS sentiment
    FROM AIFunctions.ProductReviews;

This file is loaded inside IRIS's embedded Python interpreter (irispython),
so it must be self-contained: no imports from this repo's `shared` package
and only sklearn / numpy / pandas.

The default implementation is a deterministic VADER-style lexicon scorer so
the demo runs offline. It is not a state-of-the-art sentiment model — its job
is to make the SQL pattern work end-to-end without external dependencies.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


_POSITIVE_TERMS = {
    "amazing": 2.5, "awesome": 2.5, "excellent": 2.5, "fantastic": 2.5,
    "great": 2.0, "love": 2.5, "loved": 2.0, "loving": 2.0, "perfect": 2.5,
    "wonderful": 2.5, "best": 2.0, "good": 1.5, "happy": 2.0, "delight": 2.0,
    "delighted": 2.0, "delightful": 2.0, "recommend": 1.5, "recommended": 1.5,
    "satisfied": 1.5, "pleased": 1.5, "nice": 1.0, "fast": 1.0, "easy": 1.0,
    "smooth": 1.0, "reliable": 1.5, "intuitive": 1.5, "beautiful": 2.0,
    "brilliant": 2.5, "outstanding": 2.5, "superb": 2.5, "thanks": 1.0,
    "thank": 1.0, "helpful": 1.5, "impressed": 2.0, "enjoyed": 1.5,
    "enjoy": 1.5, "works": 1.0, "worked": 1.0,
}

_NEGATIVE_TERMS = {
    "terrible": -2.5, "awful": -2.5, "horrible": -2.5, "worst": -2.5,
    "bad": -1.5, "poor": -1.5, "hate": -2.5, "broken": -2.0, "buggy": -2.0,
    "slow": -1.5, "useless": -2.5, "annoying": -2.0, "frustrating": -2.0,
    "frustrated": -2.0, "disappointed": -2.0, "disappointing": -2.0,
    "garbage": -2.5, "trash": -2.5, "fail": -2.0, "failed": -2.0,
    "failure": -2.0, "error": -1.0, "errors": -1.0, "crash": -2.0,
    "crashes": -2.0, "crashed": -2.0, "freezes": -1.5, "lag": -1.5,
    "laggy": -1.5, "expensive": -1.0, "overpriced": -1.5, "refund": -1.5,
    "scam": -2.5, "wasted": -2.0, "regret": -2.0, "stuck": -1.0,
    "issue": -0.5, "issues": -0.5, "problem": -0.5, "problems": -0.5,
    "complaint": -1.5, "unhappy": -2.0, "angry": -2.0,
}

_NEGATIONS = {"not", "no", "never", "nothing", "none", "n't", "without"}
_INTENSIFIERS = {"very": 1.4, "really": 1.3, "extremely": 1.6, "super": 1.4,
                 "absolutely": 1.5, "totally": 1.4, "highly": 1.3}
_DIMINISHERS = {"slightly": 0.6, "somewhat": 0.7, "kinda": 0.7, "barely": 0.4}

_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def _score_text(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if not tokens:
        return 0.0

    score = 0.0
    n_terms = 0
    for i, tok in enumerate(tokens):
        base = _POSITIVE_TERMS.get(tok, 0.0) + _NEGATIVE_TERMS.get(tok, 0.0)
        if base == 0.0:
            continue
        # Look back up to 2 tokens for negation / intensification.
        modifier = 1.0
        negated = False
        for back in (1, 2):
            if i - back < 0:
                break
            prev = tokens[i - back]
            if prev in _NEGATIONS:
                negated = True
            if prev in _INTENSIFIERS:
                modifier *= _INTENSIFIERS[prev]
            if prev in _DIMINISHERS:
                modifier *= _DIMINISHERS[prev]
        contribution = base * modifier * (-1.0 if negated else 1.0)
        # "!" boosts the magnitude a touch.
        if "!" in text:
            contribution *= 1.1
        score += contribution
        n_terms += 1

    if n_terms == 0:
        return 0.0
    # Normalize roughly into [-1, 1].
    norm = score / (n_terms * 2.5)
    if norm > 1.0:
        norm = 1.0
    elif norm < -1.0:
        norm = -1.0
    return float(norm)


def _label_for_score(score: float) -> str:
    if score >= 0.15:
        return "positive"
    if score <= -0.15:
        return "negative"
    return "neutral"


def _extract_text_column(X) -> list:
    if isinstance(X, pd.DataFrame):
        # Prefer a column literally named text/review/comment/message/body.
        for cand in ("text", "review_text", "review", "comment_text",
                     "comment", "message_text", "message", "body",
                     "ticket_body", "content", "article_body"):
            if cand in X.columns:
                return X[cand].astype(str).tolist()
        # Fall back to the first object/string column.
        for col in X.columns:
            if X[col].dtype == object:
                return X[col].astype(str).tolist()
        # Last resort: stringify the first column.
        return X.iloc[:, 0].astype(str).tolist()

    arr = np.asarray(X)
    if arr.ndim == 1:
        return [str(v) for v in arr.tolist()]
    return [str(v) for v in arr[:, 0].tolist()]


class IRISModel:
    """Sentiment classifier exposed through the IntegratedML Custom Models API."""

    name = "ai_sentiment"

    def __init__(self, **kwargs):
        # No trainable state — sentiment is a deterministic lexicon scorer.
        self.classes_ = np.array(["negative", "neutral", "positive"])
        self.model = self  # IRIS introspects `model` for sklearn shape

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X, y=None, **kwargs):
        # Lexicon model is stateless — fit is a no-op so TRAIN MODEL succeeds.
        return self

    def predict(self, X):
        texts = _extract_text_column(X)
        return np.array([_label_for_score(_score_text(t)) for t in texts],
                        dtype=object)

    def predict_proba(self, X):
        texts = _extract_text_column(X)
        rows = []
        for t in texts:
            s = _score_text(t)
            # Map score in [-1, 1] to soft probabilities over the 3 classes.
            pos = max(s, 0.0)
            neg = max(-s, 0.0)
            neu = max(1.0 - (pos + neg), 0.0)
            total = pos + neg + neu
            if total == 0:
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([neg / total, neu / total, pos / total])
        return np.asarray(rows, dtype=float)
