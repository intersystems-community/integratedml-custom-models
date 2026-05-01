import pandas as pd

from iris_models.ai_summarize import IRISModel


_DOC = (
    "The global market for operational AI grew 38 percent year over year. "
    "Adoption is concentrated in financial services, retail and healthcare. "
    "Customers cite data movement and latency as the largest barriers. "
    "Vendors who co-locate inference with data are growing twice as fast. "
    "Open-source foundation models are gaining share in privacy-sensitive "
    "verticals. We expect in-database AI to accelerate through 2026."
)


def test_summary_is_a_subset_of_original_sentences():
    model = IRISModel(max_sentences=2).fit(pd.DataFrame({"text": [_DOC]}))
    out = model.predict(pd.DataFrame({"text": [_DOC]}))[0]
    # Each sentence in the summary must be present in the original document.
    for sent in out.split(". "):
        sent = sent.strip().rstrip(".")
        if sent:
            assert sent in _DOC


def test_summary_respects_max_sentences():
    for k in (1, 2, 3):
        model = IRISModel(max_sentences=k).fit(pd.DataFrame({"text": [_DOC]}))
        out = model.predict(pd.DataFrame({"text": [_DOC]}))[0]
        # Roughly count sentences by terminating punctuation.
        n = sum(out.count(p) for p in (".", "!", "?"))
        assert n <= k


def test_short_text_is_returned_verbatim():
    model = IRISModel(max_sentences=5).fit(pd.DataFrame({"text": ["Hi."]}))
    out = model.predict(pd.DataFrame({"text": ["Hi."]}))[0]
    assert out == "Hi."


def test_per_row_length_column_overrides_default():
    model = IRISModel(max_sentences=5)
    df = pd.DataFrame({"text": [_DOC, _DOC], "length": [1, 3]})
    out = list(model.predict(df))
    n0 = sum(out[0].count(p) for p in (".", "!", "?"))
    n1 = sum(out[1].count(p) for p in (".", "!", "?"))
    assert n0 == 1
    assert n1 <= 3
