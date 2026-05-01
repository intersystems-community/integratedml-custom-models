import pandas as pd

from iris_models.ai_translate import IRISModel


def test_spanish_to_english_phrase_book():
    model = IRISModel(source_lang="Spanish", target_lang="English")
    out = model.predict(pd.DataFrame({"text": ["Muy bueno, me encanta el producto"]}))[0]
    assert "very good" in out.lower()
    assert "love" in out.lower()


def test_french_to_english_phrase_book():
    model = IRISModel(source_lang="French", target_lang="English")
    out = model.predict(pd.DataFrame({"text": ["Très bon produit, je recommande"]}))[0]
    assert "very good" in out.lower() or "good" in out.lower()


def test_per_row_lang_columns_override_defaults():
    model = IRISModel()  # defaults are Spanish->English
    df = pd.DataFrame({
        "text": ["Sehr gut, vielen Dank!", "Muito bom"],
        "source_lang": ["German", "Portuguese"],
        "target_lang": ["English", "English"],
    })
    out = list(model.predict(df))
    assert "very good" in out[0].lower()
    assert "very good" in out[1].lower()


def test_same_language_passthrough():
    model = IRISModel(source_lang="English", target_lang="English")
    out = model.predict(pd.DataFrame({"text": ["Hello world"]}))[0]
    assert out == "Hello world"


def test_unsupported_pair_returns_marker():
    model = IRISModel(source_lang="Klingon", target_lang="English")
    out = model.predict(pd.DataFrame({"text": ["nuqneH"]}))[0]
    assert "not in offline dictionary" in out
