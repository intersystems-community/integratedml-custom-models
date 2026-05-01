import os

import pandas as pd

from iris_models.ai_complete import IRISModel


def test_offline_stub_handles_marketing_prompt():
    os.environ.pop("ANTHROPIC_API_KEY", None)
    model = IRISModel(force_offline=True)
    out = model.predict(pd.DataFrame({"prompt": [
        "Generate a product description for a new line of carbon fiber running shoes."
    ]}))[0]
    assert "[offline-stub]" in out
    assert "carbon" in out.lower() or "running" in out.lower()


def test_offline_stub_handles_fraud_prompt():
    model = IRISModel(force_offline=True)
    prompts = [
        "Is this transaction likely fraudulent? Signals: low similarity, "
        "merchant_category=Gambling, time=3 AM",
        "Is this transaction likely fraudulent? Signals: similarity=0.92, "
        "merchant_category=Coffee Shop, time=9 AM",
    ]
    out = list(model.predict(pd.DataFrame({"prompt": prompts})))
    assert out[0].startswith("[offline-stub]")
    assert "Yes" in out[0] or "yes" in out[0].lower()
    assert out[1].startswith("[offline-stub]")
    assert "No" in out[1] or "no" in out[1].lower()


def test_offline_stub_handles_churn_prompt():
    model = IRISModel(force_offline=True)
    high_risk = (
        "Act as a churn analyst. Predict churn. Data: "
        "login_freq=0, sentiment=-0.8, days_since_purchase=180."
    )
    healthy = (
        "Act as a churn analyst. Predict churn. Data: "
        "login_freq=28, sentiment=0.42, days_since_purchase=5."
    )
    out = list(model.predict(pd.DataFrame({"prompt": [high_risk, healthy]})))
    assert out[0].startswith("[offline-stub]")
    assert "Yes" in out[0]
    assert "No" in out[1]


def test_offline_stub_does_not_call_anthropic_when_force_offline():
    # Even with the env var set, force_offline should bypass the network path.
    os.environ["ANTHROPIC_API_KEY"] = "sk-test-do-not-use"
    try:
        model = IRISModel(force_offline=True)
        out = model.predict(pd.DataFrame({"prompt": ["Hello"]}))[0]
        assert out.startswith("[offline-stub]")
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)


def test_get_set_params_round_trip():
    model = IRISModel()
    params = model.get_params()
    assert "model_id" in params
    assert "max_tokens" in params
    model.set_params(max_tokens=512)
    assert model.get_params()["max_tokens"] == 512
