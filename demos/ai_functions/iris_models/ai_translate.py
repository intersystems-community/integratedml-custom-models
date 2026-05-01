"""AI_TRANSLATE — IRISModel that translates short text between languages.

SQL usage:

    CREATE MODEL CommentTranslator PREDICTING (translated_comment)
    FROM AIFunctions.UserComments
    USING {
        "pathtoclassifiers": "/opt/irisapp/demos/ai_functions/iris_models",
        "iscmodelsdisabled": 1,
        "userparams": {"target_lang": "English"}
    };
    TRAIN MODEL CommentTranslator;
    SELECT comment_id, PREDICT(CommentTranslator) AS translated_comment
    FROM AIFunctions.UserComments;

The default implementation is a deterministic phrase + word dictionary
covering Spanish, French, German and Portuguese -> English. It's intentionally
small — production use should swap in a real MT model — but it lets the SQL
pattern run end-to-end inside IRIS without any external API calls.

Self-contained for irispython: only standard library + numpy + pandas.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


# Each entry is "source_lang -> target_lang".
_PHRASE_BOOK = {
    ("spanish", "english"): {
        "buenos días": "good morning",
        "buenas noches": "good night",
        "muchas gracias": "thank you very much",
        "por favor": "please",
        "lo siento": "I'm sorry",
        "no funciona": "it does not work",
        "muy bueno": "very good",
        "muy malo": "very bad",
        "me encanta": "I love it",
        "no me gusta": "I do not like it",
    },
    ("french", "english"): {
        "bonjour": "hello",
        "bonsoir": "good evening",
        "merci beaucoup": "thank you very much",
        "s'il vous plaît": "please",
        "je suis désolé": "I am sorry",
        "ne fonctionne pas": "does not work",
        "très bon": "very good",
        "très mauvais": "very bad",
        "j'adore": "I love it",
        "je n'aime pas": "I do not like it",
    },
    ("german", "english"): {
        "guten tag": "good day",
        "guten morgen": "good morning",
        "vielen dank": "thank you very much",
        "bitte": "please",
        "es tut mir leid": "I am sorry",
        "funktioniert nicht": "does not work",
        "sehr gut": "very good",
        "sehr schlecht": "very bad",
        "ich liebe es": "I love it",
        "ich mag es nicht": "I do not like it",
    },
    ("portuguese", "english"): {
        "bom dia": "good morning",
        "boa noite": "good night",
        "muito obrigado": "thank you very much",
        "por favor": "please",
        "desculpe": "sorry",
        "não funciona": "does not work",
        "muito bom": "very good",
        "muito ruim": "very bad",
        "eu adoro": "I love it",
        "eu não gosto": "I do not like it",
    },
}


_WORD_BOOK = {
    ("spanish", "english"): {
        "hola": "hello", "adiós": "goodbye", "sí": "yes", "no": "no",
        "gracias": "thanks", "amigo": "friend", "casa": "house",
        "comida": "food", "agua": "water", "tiempo": "time",
        "bueno": "good", "malo": "bad", "grande": "big", "pequeño": "small",
        "rápido": "fast", "lento": "slow", "feliz": "happy", "triste": "sad",
        "producto": "product", "servicio": "service", "cliente": "customer",
        "compra": "purchase", "pago": "payment", "envío": "shipping",
        "calidad": "quality", "precio": "price", "tienda": "store",
        "es": "is", "está": "is", "esta": "this", "el": "the", "la": "the",
        "un": "a", "una": "a", "y": "and", "pero": "but", "muy": "very",
        "porque": "because", "cuando": "when", "como": "like",
    },
    ("french", "english"): {
        "bonjour": "hello", "salut": "hi", "au revoir": "goodbye",
        "oui": "yes", "non": "no", "merci": "thanks", "ami": "friend",
        "maison": "house", "nourriture": "food", "eau": "water",
        "temps": "time", "bon": "good", "mauvais": "bad", "grand": "big",
        "petit": "small", "rapide": "fast", "lent": "slow", "heureux": "happy",
        "triste": "sad", "produit": "product", "service": "service",
        "client": "customer", "achat": "purchase", "paiement": "payment",
        "livraison": "shipping", "qualité": "quality", "prix": "price",
        "magasin": "store", "le": "the", "la": "the", "un": "a", "une": "a",
        "et": "and", "mais": "but", "très": "very", "parce": "because",
        "quand": "when", "comme": "like", "est": "is",
    },
    ("german", "english"): {
        "hallo": "hello", "tschüss": "bye", "ja": "yes", "nein": "no",
        "danke": "thanks", "freund": "friend", "haus": "house",
        "essen": "food", "wasser": "water", "zeit": "time", "gut": "good",
        "schlecht": "bad", "groß": "big", "klein": "small",
        "schnell": "fast", "langsam": "slow", "glücklich": "happy",
        "traurig": "sad", "produkt": "product", "dienst": "service",
        "kunde": "customer", "kauf": "purchase", "zahlung": "payment",
        "versand": "shipping", "qualität": "quality", "preis": "price",
        "laden": "store", "der": "the", "die": "the", "das": "the",
        "ein": "a", "eine": "a", "und": "and", "aber": "but", "sehr": "very",
        "weil": "because", "wann": "when", "wie": "like", "ist": "is",
    },
    ("portuguese", "english"): {
        "olá": "hello", "tchau": "bye", "sim": "yes", "não": "no",
        "obrigado": "thanks", "amigo": "friend", "casa": "house",
        "comida": "food", "água": "water", "tempo": "time", "bom": "good",
        "ruim": "bad", "grande": "big", "pequeno": "small",
        "rápido": "fast", "lento": "slow", "feliz": "happy",
        "triste": "sad", "produto": "product", "serviço": "service",
        "cliente": "customer", "compra": "purchase", "pagamento": "payment",
        "envio": "shipping", "qualidade": "quality", "preço": "price",
        "loja": "store", "o": "the", "a": "the", "um": "a", "uma": "a",
        "e": "and", "mas": "but", "muito": "very", "porque": "because",
        "quando": "when", "como": "like", "é": "is", "está": "is",
    },
}


_TOKEN_RE = re.compile(r"[\w'À-ſ]+|[^\w\s]", re.UNICODE)


def _normalize_lang(lang) -> str:
    if not isinstance(lang, str):
        return "english"
    return lang.strip().lower()


def _translate_one(text: str, source_lang: str, target_lang: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    src = _normalize_lang(source_lang)
    tgt = _normalize_lang(target_lang)
    if src == tgt:
        return text

    pair = (src, tgt)
    phrase_book = _PHRASE_BOOK.get(pair, {})
    word_book = _WORD_BOOK.get(pair, {})

    working = text
    # First substitute multi-word phrases (case-insensitive).
    for phrase, replacement in sorted(phrase_book.items(),
                                      key=lambda kv: -len(kv[0])):
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        working = pattern.sub(replacement, working)

    if not word_book and not phrase_book:
        # No dictionary for this pair — return original with a marker so the
        # caller knows nothing happened (better than silently lying).
        return f"[{source_lang}->{target_lang} not in offline dictionary] {text}"

    # Tokenize and translate word-by-word, keeping unknowns as-is.
    out_tokens = []
    for tok in _TOKEN_RE.findall(working):
        lookup = tok.lower()
        if lookup in word_book:
            replaced = word_book[lookup]
            if tok[:1].isupper():
                replaced = replaced[:1].upper() + replaced[1:]
            out_tokens.append(replaced)
        else:
            out_tokens.append(tok)

    # Re-join tokens, gluing punctuation tightly to the previous word.
    result = ""
    for tok in out_tokens:
        if re.fullmatch(r"[^\w\s]", tok):
            result += tok
        else:
            if result and not result.endswith(" "):
                result += " "
            result += tok
    return result.strip()


def _extract_text_column(X) -> list:
    if isinstance(X, pd.DataFrame):
        for cand in ("text", "comment_text", "comment", "review_text",
                     "message_text", "message", "body", "content"):
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


def _extract_lang_columns(X, default_source: str, default_target: str):
    n = len(X) if hasattr(X, "__len__") else 0
    sources = [default_source] * n
    targets = [default_target] * n
    if isinstance(X, pd.DataFrame):
        for cand in ("source_lang", "src_lang", "from_lang", "language"):
            if cand in X.columns:
                sources = [str(v) if v is not None else default_source
                           for v in X[cand].tolist()]
                break
        for cand in ("target_lang", "tgt_lang", "to_lang"):
            if cand in X.columns:
                targets = [str(v) if v is not None else default_target
                           for v in X[cand].tolist()]
                break
    return sources, targets


class IRISModel:
    """Dictionary-based translator."""

    name = "ai_translate"

    def __init__(self, source_lang: str = "Spanish",
                 target_lang: str = "English", **kwargs):
        self.source_lang = str(source_lang)
        self.target_lang = str(target_lang)
        self.model = self

    def get_params(self, deep=True):
        return {"source_lang": self.source_lang, "target_lang": self.target_lang}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X):
        texts = _extract_text_column(X)
        sources, targets = _extract_lang_columns(X, self.source_lang,
                                                 self.target_lang)
        out = []
        for i, t in enumerate(texts):
            src = sources[i] if i < len(sources) else self.source_lang
            tgt = targets[i] if i < len(targets) else self.target_lang
            out.append(_translate_one(t, src, tgt))
        return np.array(out, dtype=object)
