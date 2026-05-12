"""AI_COMPLETE — IRISModel that performs LLM text completion.

SQL usage:

    CREATE MODEL ProductCopywriter PREDICTING (product_copy)
    FROM AIFunctions.ProductPrompts
    USING {
        "pathtoclassifiers": "/opt/irisapp/demos/ai_functions/iris_models",
        "iscmodelsdisabled": 1,
        "userparams": {
            "model_id": "claude-haiku-4-5",
            "max_tokens": 256,
            "system_prompt": "You are a concise marketing copywriter."
        }
    };
    TRAIN MODEL ProductCopywriter;
    SELECT prompt_id, PREDICT(ProductCopywriter) AS product_copy
    FROM AIFunctions.ProductPrompts;

Provider selection:

* If the `anthropic` Python package is importable and `ANTHROPIC_API_KEY` is
  set in the IRIS process environment, this model calls Claude via the
  Messages API. Defaults to `claude-haiku-4-5` for low latency, configurable
  via `userparams.model_id`.
* Otherwise it falls back to a deterministic template-based completion so the
  demo works fully offline. The fallback is clearly marked in its output so
  no caller mistakes it for a real LLM response.

Self-contained for irispython: only standard library + numpy + pandas, with
an optional `anthropic` import.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd


_FALLBACK_PREFIX = "[offline-stub] "


def _have_anthropic() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False
    return True


def _stub_complete(prompt: str) -> str:
    """Deterministic offline completion used when no LLM is available.

    The goal is not to imitate a real model — it's to produce a useful,
    inspectable string so SQL pipelines run end-to-end during the demo.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return _FALLBACK_PREFIX + "(empty prompt)"

    p = prompt.strip()
    p_lower = p.lower()

    if "product description" in p_lower or "marketing" in p_lower:
        kw = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", p)
        anchor = " ".join(kw[-4:]) if len(kw) >= 4 else p
        return (_FALLBACK_PREFIX +
                f"Discover the all-new {anchor}. Designed for everyday "
                "performance and built to last, it brings together comfort, "
                "quality and craft in a single product. Available now.")

    if "summarize" in p_lower or "summary" in p_lower:
        # Take the first sentence of the prompt as a stand-in summary.
        first = re.split(r"(?<=[.!?])\s+", p, maxsplit=1)[0]
        return _FALLBACK_PREFIX + first

    if "fraudulent" in p_lower or "fraud" in p_lower:
        # Extract one signal-per-keyword, then score them.
        signals = []
        seen_keys = set()
        for kw in ("similarity", "merchant", "merchant_category", "time",
                   "amount", "country", "device"):
            if kw in seen_keys:
                continue
            m = re.search(rf"{kw}=[^,;\n]*", p)
            if m:
                signals.append(m.group(0).strip())
                seen_keys.add(kw)

        risk = 0
        risk_reasons = []
        if "gambling" in p_lower or "casino" in p_lower:
            risk += 2
            risk_reasons.append("merchant category indicates gambling")
        if "overseas" in p_lower or "wire" in p_lower:
            risk += 2
            risk_reasons.append("overseas / wire transfer")
        if "low similarity" in p_lower:
            risk += 1
            risk_reasons.append("vector similarity to user baseline is low")

        amt_match = re.search(r"amount=(\d+(?:\.\d+)?)", p_lower)
        if amt_match and float(amt_match.group(1)) >= 1000:
            risk += 1
            risk_reasons.append("amount above $1000")
        time_match = re.search(r"time=\S*?(\d{1,2}):(\d{2})", p)
        if time_match:
            hour = int(time_match.group(1))
            if 0 <= hour <= 5:
                risk += 1
                risk_reasons.append("transaction occurred overnight")

        verdict = "Yes" if risk >= 2 else "No"
        why = ("; ".join(risk_reasons) if risk_reasons
               else "no anomalous signals detected")
        sig_line = "; ".join(signals) if signals else "no signals provided"
        return (_FALLBACK_PREFIX
                + f"{verdict}. Reason: {why}. Signals seen: {sig_line}.")

    if "churn" in p_lower:
        login_match = re.search(r"login_freq=(-?\d+)", p_lower)
        sent_match = re.search(r"sentiment=(-?\d+\.?\d*)", p_lower)
        days_match = re.search(r"days_since_purchase=(-?\d+)", p_lower)

        risk = 0
        reasons = []
        if login_match and int(login_match.group(1)) <= 5:
            risk += 1
            reasons.append("low login frequency")
        if sent_match and float(sent_match.group(1)) < -0.3:
            risk += 1
            reasons.append("negative sentiment")
        if days_match and int(days_match.group(1)) > 60:
            risk += 1
            reasons.append("long time since last purchase")

        verdict = "Yes" if risk >= 2 else "No"
        why = "; ".join(reasons) if reasons else (
            "engagement and sentiment look healthy")
        return (_FALLBACK_PREFIX
                + f"{verdict}. Reason: {why}.")

    if "translate" in p_lower:
        return _FALLBACK_PREFIX + "(translation not performed by stub)"

    # Generic completion: echo the prompt with a closing sentence.
    return _FALLBACK_PREFIX + p + " — (completion stub)"


def _claude_complete(prompt: str, model_id: str, max_tokens: int,
                     system_prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = []
    for block in msg.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts).strip()


def _extract_prompt_column(X) -> list:
    if isinstance(X, pd.DataFrame):
        for cand in ("prompt", "input", "text", "question", "user_message",
                     "request"):
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
    """LLM completion via Anthropic Claude (with deterministic fallback)."""

    name = "ai_complete"

    def __init__(self, model_id: str = "claude-haiku-4-5",
                 max_tokens: int = 256,
                 system_prompt: str = "You are a helpful assistant.",
                 force_offline: bool = False, **kwargs):
        self.model_id = str(model_id)
        self.max_tokens = int(max_tokens)
        self.system_prompt = str(system_prompt)
        self.force_offline = bool(force_offline)
        self.model = self

    def get_params(self, deep=True):
        return {
            "model_id": self.model_id,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "force_offline": self.force_offline,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y=None, **kwargs):
        return self

    def _complete(self, prompt: str) -> str:
        if self.force_offline or not _have_anthropic():
            return _stub_complete(prompt)
        try:
            return _claude_complete(prompt, self.model_id, self.max_tokens,
                                    self.system_prompt)
        except Exception as exc:  # pragma: no cover — depends on network
            return _FALLBACK_PREFIX + f"(claude error: {exc!s}) " + _stub_complete(prompt)

    def predict(self, X):
        prompts = _extract_prompt_column(X)
        return np.array([self._complete(p) for p in prompts], dtype=object)
