import pandas as pd

from demos.ai_functions.iris_models.ai_extract import IRISModel


CONTRACT_A = (
    "This Master Services Agreement is entered into between Acme Corp and "
    "Beta Industries effective on January 15, 2025. The renewal date is "
    "January 15, 2026. Annual fees of USD 48,000 are payable quarterly. "
    "Contact the support team at support@acme.example."
)


def test_extracts_renewal_date():
    model = IRISModel(question="What is the contract renewal date?")
    out = model.predict(pd.DataFrame({"text": [CONTRACT_A]}))[0]
    assert "2026" in out
    assert "January" in out


def test_extracts_amount():
    model = IRISModel(question="What is the annual fee amount?")
    out = model.predict(pd.DataFrame({"text": [CONTRACT_A]}))[0]
    assert "48,000" in out
    assert "USD" in out or "$" in out or "USD" in out.upper()


def test_extracts_email():
    model = IRISModel(question="What is the support email address?")
    out = model.predict(pd.DataFrame({"text": [CONTRACT_A]}))[0]
    assert out == "support@acme.example"


def test_extracts_parties():
    model = IRISModel(question="Who are the parties in this contract?")
    out = model.predict(pd.DataFrame({"text": [CONTRACT_A]}))[0]
    assert "Acme Corp" in out
    assert "Beta Industries" in out


def test_per_row_question_overrides_default():
    model = IRISModel(question="What is the contract renewal date?")
    df = pd.DataFrame({
        "text": [CONTRACT_A, CONTRACT_A],
        "question": [
            "What is the contract renewal date?",
            "What is the support email address?",
        ],
    })
    out = list(model.predict(df))
    assert "2026" in out[0]
    assert "@" in out[1]


def test_handles_empty_text():
    model = IRISModel(question="What is the date?")
    out = model.predict(pd.DataFrame({"text": [""]}))[0]
    assert out == ""
