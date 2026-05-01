"""End-to-end runner for the IRIS IntegratedML AI Functions demo.

Reproduces the three use cases from the SingleStore "Introducing AI
Functions" article (support intelligence, fraud detection, churn prediction)
using the seven IRISModel classes under
`demos/ai_functions/iris_models/`.

The runner imports the IRISModel classes directly into local Python — i.e.
it doesn't require a running IRIS instance. This lets contributors verify
the AI Function behaviour before deploying the .py files into IRIS's
embedded Python interpreter for the real CREATE MODEL / TRAIN MODEL flow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEMO_DIR = Path(__file__).resolve().parent / "demos" / "ai_functions"
DATA_DIR = DEMO_DIR / "data"
sys.path.insert(0, str(DEMO_DIR))

from iris_models.ai_classify import IRISModel as ClassifyModel  # noqa: E402
from iris_models.ai_complete import IRISModel as CompleteModel  # noqa: E402
from iris_models.ai_extract import IRISModel as ExtractModel  # noqa: E402
from iris_models.ai_sentiment import IRISModel as SentimentModel  # noqa: E402
from iris_models.ai_summarize import IRISModel as SummarizeModel  # noqa: E402
from iris_models.ai_translate import IRISModel as TranslateModel  # noqa: E402
from iris_models.embed_text import IRISModel as EmbedModel  # noqa: E402


def _hr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _print_table(df: pd.DataFrame, max_rows: int = 10) -> None:
    with pd.option_context("display.max_colwidth", 90,
                           "display.width", 200,
                           "display.max_columns", None):
        print(df.head(max_rows).to_string(index=False))


def use_case_1_support_intelligence() -> None:
    _hr("Use case 1 — Customer support intelligence")
    tickets = pd.read_csv(DATA_DIR / "support_tickets.csv")

    sentiment = SentimentModel().fit(tickets[["message_text"]])
    classify = ClassifyModel(
        categories=["billing", "technical", "sales", "returns"]
    ).fit(tickets[["message_text"]])
    summarize = SummarizeModel(max_sentences=1).fit(tickets[["message_text"]])

    enriched = pd.DataFrame({
        "ticket_id": tickets["ticket_id"],
        "department": classify.predict(tickets[["message_text"]]),
        "sentiment": sentiment.predict(tickets[["message_text"]]),
        "summary": summarize.predict(tickets[["message_text"]]),
    })

    print("Routed tickets:")
    _print_table(enriched, max_rows=15)

    print()
    print("Routing accuracy vs. labels:")
    correct = (enriched["department"].values == tickets["department"].values).mean()
    print(f"  {correct:.1%} of {len(tickets)} tickets routed to the labelled department")

    escalations = enriched[enriched["sentiment"] == "negative"]
    print()
    print(f"Escalation queue: {len(escalations)} tickets flagged for senior agents")
    _print_table(escalations[["ticket_id", "department", "summary"]],
                 max_rows=5)


def use_case_2_fraud_detection() -> None:
    _hr("Use case 2 — Real-time fraud detection")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")

    extract = ExtractModel(question="What is the merchant category?")
    embed = EmbedModel(dim=64)
    complete = CompleteModel(force_offline=True,
                             system_prompt="You are a fraud analyst.")

    df = transactions.copy()
    df["merchant_category"] = extract.predict(df[["description"]])
    embeddings = embed.embed(df[["description"]])
    print("Transaction embedding shape:", embeddings.shape)

    df["prompt"] = df.apply(
        lambda r: (
            "Is this transaction likely fraudulent? Reply Yes or No and one "
            "sentence of reasoning. Signals: amount=" + str(r["amount"])
            + ", time=" + str(r["timestamp"])
            + ", merchant_category=" + str(r["merchant_category"])
            + ", description=" + str(r["description"])
        ),
        axis=1,
    )
    df["fraud_assessment"] = complete.predict(df[["prompt"]])

    print()
    print("Fraud assessments:")
    _print_table(
        df[["transaction_id", "amount", "merchant_category",
            "fraud_assessment"]],
        max_rows=15,
    )


def use_case_3_churn_prediction() -> None:
    _hr("Use case 3 — Customer churn prediction")
    customers = pd.read_csv(DATA_DIR / "customer_metrics.csv")

    complete = CompleteModel(force_offline=True,
                             system_prompt="You are a churn analyst.")

    customers["prompt"] = customers.apply(
        lambda r: (
            "Act as a churn analyst. Predict if this customer will churn "
            "(Yes/No) and give a reason. Data: "
            "plan=" + str(r["plan"])
            + ", login_freq=" + str(r["login_freq"])
            + ", sentiment=" + str(r["avg_sentiment"])
            + ", days_since_purchase=" + str(r["days_since_purchase"]) + "."
        ),
        axis=1,
    )
    customers["churn_prediction"] = complete.predict(customers[["prompt"]])

    print("Churn predictions:")
    _print_table(
        customers[["user_id", "plan", "login_freq", "avg_sentiment",
                   "days_since_purchase", "churn_prediction"]],
        max_rows=15,
    )

    at_risk = customers[
        (customers["login_freq"] <= 5)
        | (customers["avg_sentiment"] < -0.3)
        | (customers["days_since_purchase"] > 90)
    ]
    print()
    print(f"At-risk segment: {len(at_risk)} customers")


def supplementary_demos() -> None:
    """Show the other AI Functions that weren't part of the 3 main use cases."""
    _hr("Supplementary — AI_TRANSLATE, AI_SUMMARIZE on long text, AI_COMPLETE")
    comments = pd.read_csv(DATA_DIR / "user_comments.csv")
    translator = TranslateModel().fit(comments[["comment_text"]])
    comments["translated"] = translator.predict(comments[[
        "comment_text", "source_lang", "target_lang"
    ]])
    print("AI_TRANSLATE (auto-detect per row):")
    _print_table(
        comments[["comment_id", "source_lang", "comment_text", "translated"]],
        max_rows=10,
    )

    print()
    research = pd.read_csv(DATA_DIR / "market_research.csv")
    summarizer = SummarizeModel(max_sentences=2).fit(research[["full_report"]])
    research["executive_summary"] = summarizer.predict(research[["full_report"]])
    print("AI_SUMMARIZE (executive summaries):")
    _print_table(research[["report_id", "executive_summary"]], max_rows=4)

    print()
    legal = pd.read_csv(DATA_DIR / "legal_documents.csv")
    extract = ExtractModel(question="What is the contract renewal date?")
    legal["renewal_date"] = extract.predict(legal[["contract_text"]])
    print("AI_EXTRACT (renewal dates):")
    _print_table(legal[["contract_id", "renewal_date"]], max_rows=10)

    print()
    prompts = pd.read_csv(DATA_DIR / "product_prompts.csv")
    completer = CompleteModel(force_offline=True)
    prompts["product_copy"] = completer.predict(prompts[["prompt"]])
    print("AI_COMPLETE (offline stub — set ANTHROPIC_API_KEY for real calls):")
    _print_table(prompts[["prompt_id", "product_copy"]], max_rows=10)


def main() -> None:
    use_case_1_support_intelligence()
    use_case_2_fraud_detection()
    use_case_3_churn_prediction()
    supplementary_demos()
    print()
    print("Done. To deploy these models into IRIS, run:")
    print("  python demos/ai_functions/scripts/deploy_models.py")
    print("  iris session iris -U USER < demos/ai_functions/sql/01_setup_tables.sql")
    print("  ... then load the data and execute 02_create_models.sql")


if __name__ == "__main__":
    main()
