import pandas as pd

from iris_models.ai_sentiment import IRISModel


def _predict(texts):
    model = IRISModel().fit(pd.DataFrame({"text": texts}))
    return list(model.predict(pd.DataFrame({"text": texts})))


def test_positive_text_classified_as_positive():
    labels = _predict([
        "I absolutely love this product, it's amazing and works perfectly!",
        "Best purchase I've made all year, highly recommended.",
    ])
    assert labels == ["positive", "positive"]


def test_negative_text_classified_as_negative():
    labels = _predict([
        "This is terrible. The product broke after one day.",
        "Awful experience, totally useless and a complete waste of money.",
    ])
    assert labels == ["negative", "negative"]


def test_neutral_text_classified_as_neutral():
    labels = _predict([
        "The package arrived on Tuesday afternoon.",
        "It is a chair with four legs.",
    ])
    assert labels == ["neutral", "neutral"]


def test_negation_flips_polarity():
    labels = _predict(["This is not good at all."])
    assert labels == ["negative"]


def test_empty_input_is_neutral():
    labels = _predict(["", "   "])
    assert labels == ["neutral", "neutral"]


def test_predict_proba_sums_to_one():
    model = IRISModel()
    proba = model.predict_proba(pd.DataFrame({
        "text": ["I love it!", "I hate it.", "It is fine."]
    }))
    assert proba.shape == (3, 3)
    for row in proba:
        assert abs(row.sum() - 1.0) < 1e-6


def test_works_with_numpy_input():
    import numpy as np
    model = IRISModel()
    arr = np.array(["I love this!", "This is terrible"])
    out = list(model.predict(arr))
    assert out == ["positive", "negative"]
