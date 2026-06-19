"""AI_EXTRACT — IRISModel that pulls structured info from unstructured text.

SQL usage:

    CREATE MODEL RenewalDateExtractor PREDICTING (renewal_date)
    FROM AIFunctions.LegalDocuments
    USING {
        "pathtoclassifiers": "/opt/irisapp/demos/ai_functions/iris_models",
        "iscmodelsdisabled": 1,
        "userparams": {"question": "What is the contract renewal date?"}
    };
    TRAIN MODEL RenewalDateExtractor;
    SELECT contract_id, PREDICT(RenewalDateExtractor) AS renewal_date
    FROM AIFunctions.LegalDocuments;

The extractor parses the question for an entity hint (date / amount / email /
party / merchant / id / phone / url / percent) and runs a corresponding regex
pass over the source text. If multiple candidates match, the one closest to a
question keyword in the document is returned. This is intentionally
deterministic so the demo is reproducible — production deployments can swap
in an LLM-backed implementation behind the same `IRISModel` interface.

Self-contained for irispython: only standard library + numpy + pandas.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


_DATE_RE = re.compile(
    r"\b("
    r"\d{4}-\d{2}-\d{2}"  # ISO
    r"|\d{1,2}/\d{1,2}/\d{2,4}"  # US/EU slash
    r"|(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}(?:,\s*\d{4})?"
    r"|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)(?:\s+\d{4})?"
    r")\b",
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(
    r"(?:USD|EUR|GBP|\$|€|£)\s?\d[\d,]*(?:\.\d+)?"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:dollars|euros|pounds|USD|EUR|GBP)\b",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(
    r"\+?\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}"
)
_URL_RE = re.compile(r"https?://[^\s]+")
_PERCENT_RE = re.compile(r"\b\d{1,3}(?:\.\d+)?\s?%")
_ID_RE = re.compile(r"\b[A-Z]{2,}-?\d{2,}\b|\b\d{6,}\b")
_PARTY_RE = re.compile(
    r"between\s+(?P<a>[A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*)*)"
    r"\s+and\s+(?P<b>[A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*)*)",
)
_MERCHANT_HINT_RE = re.compile(
    r"\b(?:at|from|to)\s+([A-Z][\w&'.,\- ]{2,40}?)\b",
)


_HINT_KEYWORDS = {
    "date": ("date", "renewal", "expir", "deadline", "starts", "ends",
             "effective"),
    "amount": ("amount", "fee", "price", "cost", "total", "charge",
               "payment", "value"),
    "email": ("email", "contact", "address"),
    "phone": ("phone", "tel", "call"),
    "url": ("url", "link", "website"),
    "percent": ("percent", "%", "rate", "interest"),
    "id": ("id", "identifier", "reference", "ticket", "order", "case"),
    "party": ("party", "parties", "between", "company", "counterparty",
              "client", "vendor", "buyer", "seller"),
    "merchant": ("merchant", "store", "vendor", "shop", "retailer"),
}


def _classify_question(question: str) -> str:
    if not question:
        return "snippet"
    q = question.lower()
    if any(k in q for k in ("when", "date", "renewal", "expir", "deadline")):
        return "date"
    if any(k in q for k in ("how much", "amount", "fee", "cost", "price",
                            "value", "total", "charge")):
        return "amount"
    if "email" in q:
        return "email"
    if "phone" in q or "telephone" in q:
        return "phone"
    if "url" in q or "link" in q or "website" in q:
        return "url"
    if "%" in q or "percent" in q or "rate" in q:
        return "percent"
    if any(k in q for k in ("ticket", "id", "identifier", "reference",
                            "case number")):
        return "id"
    if any(k in q for k in ("who", "party", "parties", "counterparty",
                            "company", "between")):
        return "party"
    if "merchant" in q or "vendor" in q or "store" in q:
        return "merchant"
    return "snippet"


def _question_keywords(question: str, kind: str) -> tuple:
    """Return (priority_keywords, fallback_keywords).

    Priority keywords come straight from the question text (minus stopwords);
    fallback keywords are the generic per-kind hints. The matcher tries
    priority first and only falls back to generic hints if nothing in the
    document matches a priority keyword.
    """
    base = list(_HINT_KEYWORDS.get(kind, ()))
    if not question:
        return ([], base)
    extras = re.findall(r"[a-zA-Z]{4,}", question.lower())
    skip = {"what", "when", "where", "which", "this", "that", "from",
            "the", "with", "have", "your", "their", "into", "onto",
            "contract", "value", "amount", "fees",
            "is", "are"}
    extras = [e for e in extras if e not in skip]
    return (extras, base)


def _best_match(text: str, regex: re.Pattern, keywords) -> str:
    matches = list(regex.finditer(text))
    if not matches:
        return ""
    if isinstance(keywords, tuple) and len(keywords) == 2:
        priority, fallback = keywords
    else:
        priority, fallback = [], list(keywords)

    text_lower = text.lower()

    def _positions(kws):
        out = []
        for kw in kws:
            for m in re.finditer(re.escape(kw.lower()), text_lower):
                out.append(m.start())
        return out

    keyword_positions = _positions(priority)
    if not keyword_positions:
        keyword_positions = _positions(fallback)

    if not keyword_positions:
        return matches[0].group(0).strip()

    # Prefer matches that appear *after* a keyword within ~80 chars (the
    # typical "<keyword> is <answer>" pattern). Only fall back to absolute
    # distance when no match satisfies the after-keyword heuristic.
    best_after = None
    best_after_gap = float("inf")
    for m in matches:
        for kp in keyword_positions:
            gap = m.start() - kp
            if 0 <= gap <= 80 and gap < best_after_gap:
                best_after_gap = gap
                best_after = m
    if best_after is not None:
        return best_after.group(0).strip()

    best = matches[0]
    best_dist = float("inf")
    for m in matches:
        center = (m.start() + m.end()) / 2
        dist = min(abs(center - kp) for kp in keyword_positions)
        if dist < best_dist:
            best_dist = dist
            best = m
    return best.group(0).strip()


def _extract_one(text: str, question: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    kind = _classify_question(question)
    keywords = _question_keywords(question, kind)

    if kind == "date":
        return _best_match(text, _DATE_RE, keywords)
    if kind == "amount":
        return _best_match(text, _AMOUNT_RE, keywords)
    if kind == "email":
        m = _EMAIL_RE.search(text)
        return m.group(0) if m else ""
    if kind == "phone":
        m = _PHONE_RE.search(text)
        return m.group(0) if m else ""
    if kind == "url":
        m = _URL_RE.search(text)
        return m.group(0) if m else ""
    if kind == "percent":
        return _best_match(text, _PERCENT_RE, keywords)
    if kind == "id":
        return _best_match(text, _ID_RE, keywords)
    if kind == "party":
        m = _PARTY_RE.search(text)
        if m:
            return f"{m.group('a').strip()} / {m.group('b').strip()}"
        # Fall back to first capitalised noun phrase.
        m2 = re.search(r"\b([A-Z][\w&]{2,}(?:\s+[A-Z][\w&]+)*)", text)
        return m2.group(1).strip() if m2 else ""
    if kind == "merchant":
        m = _MERCHANT_HINT_RE.search(text)
        return m.group(1).strip() if m else ""

    # Default: return the sentence that mentions the most question keywords.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if isinstance(keywords, tuple) and len(keywords) == 2:
        flat = list(keywords[0]) + list(keywords[1])
    else:
        flat = list(keywords)
    best_sent, best_score = "", 0
    for sent in sentences:
        sl = sent.lower()
        score = sum(1 for kw in flat if kw in sl)
        if score > best_score:
            best_score = score
            best_sent = sent
    return best_sent.strip() or sentences[0].strip()


def _extract_text_column(X) -> list:
    if isinstance(X, pd.DataFrame):
        for cand in ("text", "contract_text", "document", "body",
                     "description", "content", "article_body"):
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


def _extract_question_column(X, default: str) -> list:
    if isinstance(X, pd.DataFrame):
        for cand in ("question", "query", "prompt"):
            if cand in X.columns:
                return [str(v) if v is not None else default
                        for v in X[cand].tolist()]
    return None


class IRISModel:
    """Question-aware information extractor."""

    name = "ai_extract"

    def __init__(self, question: str = "", **kwargs):
        self.question = str(question) if question is not None else ""
        self.model = self

    def get_params(self, deep=True):
        return {"question": self.question}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X):
        texts = _extract_text_column(X)
        per_row = _extract_question_column(X, self.question)
        out = []
        for i, t in enumerate(texts):
            q = per_row[i] if per_row else self.question
            out.append(_extract_one(t, q))
        return np.array(out, dtype=object)
