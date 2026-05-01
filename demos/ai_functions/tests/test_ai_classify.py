import pandas as pd

from iris_models.ai_classify import IRISModel


def test_zero_shot_routing_against_default_seed_categories():
    model = IRISModel()
    model.fit(pd.DataFrame({"text": []}))
    texts = pd.DataFrame({"text": [
        "I was charged twice and I want a refund please",
        "The app crashes every time I open it",
        "Can I get a quote for the enterprise plan",
        "The package arrived broken, I need to return it",
    ]})
    labels = list(model.predict(texts))
    assert labels == ["billing", "technical", "sales", "returns"]


def test_explicit_categories_param_overrides_seeds():
    # Restrict the candidate categories. Even strongly billing-flavoured text
    # must be routed to one of the explicit categories.
    model = IRISModel(categories=["urgent", "normal"]).fit(
        pd.DataFrame({"text": []}))
    label = model.predict(pd.DataFrame({"text": [
        "Refund my payment please"
    ]}))[0]
    assert label in {"urgent", "normal"}


def test_few_shot_with_labelled_training_data():
    train = pd.DataFrame({
        "text": [
            "Refund my subscription right now",
            "Why was I billed twice this month",
            "The app crashes constantly",
            "I get a 500 error when I click submit",
        ],
        "department": ["billing", "billing", "technical", "technical"],
    })
    model = IRISModel(categories=["billing", "technical"]).fit(
        train.drop(columns=["department"]),
        y=train["department"].tolist(),
    )
    preds = list(model.predict(pd.DataFrame({"text": [
        "I was charged for a plan I cancelled",
        "Login button does nothing",
    ]})))
    assert preds == ["billing", "technical"]


def test_predict_proba_returns_distribution():
    model = IRISModel().fit(pd.DataFrame({"text": []}))
    proba = model.predict_proba(pd.DataFrame({"text": [
        "Refund the duplicate charge",
    ]}))
    assert proba.shape == (1, 4)
    # Probabilities sum to 1 (or 0 when no signal at all).
    assert abs(proba.sum() - 1.0) < 1e-6
