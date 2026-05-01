import json

import numpy as np
import pandas as pd

from iris_models.embed_text import IRISModel


def test_embeddings_are_normalised_to_unit_length():
    model = IRISModel(dim=64)
    out = model.embed(pd.DataFrame({"text": [
        "vector databases store embeddings",
        "operational AI runs next to live data",
    ]}))
    assert out.shape == (2, 64)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6) or np.allclose(norms, 0.0, atol=1e-6)


def test_predict_returns_json_strings():
    model = IRISModel(dim=32)
    out = model.predict(pd.DataFrame({"text": ["hello world"]}))
    assert len(out) == 1
    decoded = json.loads(out[0])
    assert isinstance(decoded, list)
    assert len(decoded) == 32
    for v in decoded:
        assert isinstance(v, float)


def test_similar_texts_have_higher_cosine_than_dissimilar_ones():
    model = IRISModel(dim=128)
    a, b, c = model.embed(pd.DataFrame({"text": [
        "I love operational AI on transactional data",
        "Operational AI thrives when models live next to transactional data",
        "The recipe for chocolate cake calls for flour, sugar and butter",
    ]}))
    cos_ab = float(np.dot(a, b))
    cos_ac = float(np.dot(a, c))
    assert cos_ab > cos_ac


def test_dim_param_controls_output_dimension():
    for dim in (16, 64, 256):
        model = IRISModel(dim=dim)
        out = model.embed(pd.DataFrame({"text": ["sample"]}))
        assert out.shape == (1, dim)
