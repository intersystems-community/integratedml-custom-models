"""End-to-end runner for the TabPFN-3 foundation-model demo.

Trains the TabPFN classifier on the patient-screening dataset and the TabPFN
regressor on the building-energy dataset, then prints predictions, metrics,
and a hardest-cases / largest-errors slice for each. The runner imports the
IRISModel classes directly from local Python — no IRIS instance required.

If `tabpfn` is installed in the active Python environment, both models use
the real TabPFN-3 backend; otherwise they fall back to scikit-learn and
clearly report the active backend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

DEMO_DIR = Path(__file__).resolve().parent / "demos" / "tabpfn_foundation"
DATA_DIR = DEMO_DIR / "data"
sys.path.insert(0, str(DEMO_DIR))

from iris_models.tabpfn_classifier import IRISModel as TabPFNClassifier  # noqa: E402
from iris_models.tabpfn_regressor import IRISModel as TabPFNRegressor  # noqa: E402


def _hr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _print_table(df: pd.DataFrame, max_rows: int = 10) -> None:
    with pd.option_context("display.max_colwidth", 80,
                           "display.width", 200,
                           "display.max_columns", None):
        print(df.head(max_rows).to_string(index=False))


def _split(df: pd.DataFrame, target: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(0.75 * len(df))
    train, test = df.iloc[idx[:cut]].copy(), df.iloc[idx[cut:]].copy()
    feature_cols = [c for c in df.columns if c not in (target,)
                    and c not in ("patient_id", "building_id")]
    return train, test, feature_cols


def run_classification() -> None:
    _hr("TabPFN-3 classification — Patient risk screening")
    df = pd.read_csv(DATA_DIR / "patient_screening.csv")
    train, test, fc = _split(df, target="needs_followup")

    model = TabPFNClassifier(device="cpu", n_estimators=8)
    model.fit(train[fc], train["needs_followup"])

    print(f"Active backend: {model.backend}")
    print(f"Train size: {len(train)}, test size: {len(test)}")
    print(f"Positive rate (train): "
          f"{train['needs_followup'].mean():.1%}")

    proba = model.predict_proba(test[fc])[:, 1]
    preds = model.predict(test[fc])

    acc = accuracy_score(test["needs_followup"], preds)
    auc = roc_auc_score(test["needs_followup"], proba)
    print()
    print(f"Accuracy: {acc:.3f}   AUC: {auc:.3f}")
    print()
    print(classification_report(test["needs_followup"], preds,
                                target_names=["no_followup", "needs_followup"]))

    # Show a couple of confident positives and confident negatives.
    test = test.assign(
        prob_positive=proba,
        predicted=preds,
    )
    print("Most-confident high-risk patients:")
    _print_table(
        test.sort_values("prob_positive", ascending=False)[
            ["patient_id", "age", "sex", "bmi", "fasting_glucose", "smoker",
             "needs_followup", "prob_positive"]
        ],
        max_rows=5,
    )
    print()
    print("Most-confident low-risk patients:")
    _print_table(
        test.sort_values("prob_positive", ascending=True)[
            ["patient_id", "age", "sex", "bmi", "fasting_glucose", "smoker",
             "needs_followup", "prob_positive"]
        ],
        max_rows=5,
    )


def run_regression() -> None:
    _hr("TabPFN-3 regression — Building energy consumption")
    df = pd.read_csv(DATA_DIR / "building_energy.csv")
    train, test, fc = _split(df, target="kwh_day")

    model = TabPFNRegressor(device="cpu", n_estimators=8)
    model.fit(train[fc], train["kwh_day"])

    print(f"Active backend: {model.backend}")
    print(f"Train size: {len(train)}, test size: {len(test)}")

    preds = model.predict(test[fc])
    mae = mean_absolute_error(test["kwh_day"], preds)
    rmse = mean_squared_error(test["kwh_day"], preds) ** 0.5
    r2 = r2_score(test["kwh_day"], preds)
    mape = float(np.mean(np.abs((test["kwh_day"] - preds) / test["kwh_day"])))

    print()
    print(f"MAE: {mae:.2f} kWh/day   RMSE: {rmse:.2f} kWh/day")
    print(f"R²:  {r2:.3f}            MAPE: {mape:.1%}")
    print()

    test = test.assign(
        predicted=np.round(preds, 1),
        abs_error=np.round(np.abs(test["kwh_day"] - preds), 1),
    )
    print("Largest absolute errors:")
    _print_table(
        test.sort_values("abs_error", ascending=False)[
            ["building_id", "floor_area", "occupants", "hvac_type",
             "outdoor_temp_c", "insulation_rating", "kwh_day", "predicted",
             "abs_error"]
        ],
        max_rows=5,
    )
    print()
    print("Best predictions:")
    _print_table(
        test.sort_values("abs_error", ascending=True)[
            ["building_id", "floor_area", "occupants", "hvac_type",
             "outdoor_temp_c", "kwh_day", "predicted", "abs_error"]
        ],
        max_rows=5,
    )


def main() -> None:
    run_classification()
    run_regression()
    print()
    print("Done. To deploy these models into IRIS:")
    print("  python demos/tabpfn_foundation/scripts/deploy_models.py")
    print("  iris session iris -U USER < demos/tabpfn_foundation/sql/01_setup_tables.sql")
    print("  ... load CSVs, then execute 02_create_models.sql / 03_evaluation.sql")


if __name__ == "__main__":
    main()
