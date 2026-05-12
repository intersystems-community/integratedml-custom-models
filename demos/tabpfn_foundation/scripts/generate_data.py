"""Generate the small tabular CSVs that ship with the TabPFN demo.

Two synthetic datasets, both sized for TabPFN's sweet spot of small-to-medium
tabular learning:

* `patient_screening.csv` — 300 patients × 11 features + binary target
  (`needs_followup`). The label is a noisy logistic mix of biometric
  signals (age, BMI, blood pressure, fasting glucose, etc.) so a tabular
  model has a clear learnable signal but a hand-tuned threshold rule would
  miss interactions.

* `building_energy.csv` — 300 buildings × 9 features + continuous target
  (`kwh_day`). Energy is a function of floor area, occupancy, HVAC type,
  outdoor temperature, insulation rating, weekend flag and time of year,
  with interaction terms (insulation×temperature) that reward a model that
  can capture non-linearities.

Re-run this script if you want fresh data; the seed is fixed so the
checked-in CSVs are reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _bool(p: float, rng: np.random.Generator, n: int) -> np.ndarray:
    return (rng.random(n) < p).astype(int)


def make_patient_screening(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(20, 85, size=n)
    bmi = rng.normal(27.5, 5.2, size=n).clip(15, 55)
    systolic_bp = rng.normal(128, 18, size=n).clip(80, 220)
    diastolic_bp = (systolic_bp * 0.65 + rng.normal(0, 6, size=n)).clip(50, 130)
    fasting_glucose = rng.normal(98, 22, size=n).clip(60, 280)
    hdl = rng.normal(52, 14, size=n).clip(15, 110)
    ldl = rng.normal(118, 32, size=n).clip(40, 260)
    smoker = _bool(0.28, rng, n)
    family_history = _bool(0.34, rng, n)
    exercise_days = rng.integers(0, 8, size=n)
    sex = rng.choice(["F", "M"], size=n, p=[0.52, 0.48])

    # Latent risk score: a sum of standardised signals + interactions.
    # Coefficients chosen so the resulting class is meaningfully learnable
    # (target AUC ~0.90) while keeping a ~25-30% positive rate.
    logit = (
        0.075 * (age - 50)
        + 0.18 * (bmi - 25)
        + 0.025 * (systolic_bp - 120)
        + 0.045 * (fasting_glucose - 100)
        - 0.045 * (hdl - 50)
        + 0.018 * (ldl - 120)
        + 1.60 * smoker
        + 1.05 * family_history
        - 0.32 * exercise_days
        + 0.55 * (sex == "M").astype(float)
        # Interaction: glucose × bmi makes diabetic-range patients much riskier.
        + 0.040 * np.maximum(fasting_glucose - 110, 0) * np.maximum(bmi - 28, 0) / 10.0
        - 2.9
    )
    proba = 1.0 / (1.0 + np.exp(-logit))
    needs_followup = (rng.random(n) < proba).astype(int)

    return pd.DataFrame({
        "patient_id": np.arange(1, n + 1),
        "age": age,
        "sex": sex,
        "bmi": np.round(bmi, 1),
        "systolic_bp": np.round(systolic_bp, 0).astype(int),
        "diastolic_bp": np.round(diastolic_bp, 0).astype(int),
        "fasting_glucose": np.round(fasting_glucose, 0).astype(int),
        "hdl": np.round(hdl, 0).astype(int),
        "ldl": np.round(ldl, 0).astype(int),
        "smoker": smoker,
        "family_history": family_history,
        "exercise_days": exercise_days,
        "needs_followup": needs_followup,
    })


def make_building_energy(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    floor_area = rng.integers(120, 4800, size=n)  # square metres
    occupants = rng.integers(1, 120, size=n)
    hvac_type = rng.choice(
        ["heat_pump", "gas_furnace", "electric_resistance", "geothermal"],
        size=n,
        p=[0.35, 0.30, 0.20, 0.15],
    )
    outdoor_temp_c = rng.normal(11.0, 9.0, size=n)
    insulation_rating = rng.integers(1, 6, size=n)  # 1=poor … 5=excellent
    is_weekend = _bool(2 / 7, rng, n)
    month = rng.integers(1, 13, size=n)
    appliance_density = rng.normal(0.45, 0.18, size=n).clip(0.05, 1.2)

    # Base load per m² per day in kWh, adjusted by occupancy + climate.
    base = 0.085 * floor_area + 0.65 * occupants

    # Heating/cooling: each degree C below 18 or above 24 raises consumption,
    # damped by insulation rating.
    heating_load = np.maximum(18 - outdoor_temp_c, 0)
    cooling_load = np.maximum(outdoor_temp_c - 24, 0)
    thermal = (heating_load + cooling_load) * (1.0 - 0.12 * insulation_rating)
    thermal_kwh = thermal * (0.30 * floor_area / 100.0)

    hvac_multiplier = np.where(
        hvac_type == "heat_pump", 0.85,
        np.where(hvac_type == "geothermal", 0.70,
                 np.where(hvac_type == "gas_furnace", 1.10, 1.40)),
    )

    seasonal_bump = np.where((month <= 2) | (month >= 11), 1.15,
                             np.where((month >= 6) & (month <= 8), 1.10, 1.0))

    kwh_day = (
        (base + thermal_kwh * hvac_multiplier)
        * seasonal_bump
        * (0.85 + appliance_density * 0.4)
        * (0.90 if False else 1.0)
    )
    # Weekends drop commercial buildings a bit.
    kwh_day = kwh_day * np.where(is_weekend.astype(bool), 0.78, 1.0)
    kwh_day = kwh_day + rng.normal(0, kwh_day * 0.05)

    return pd.DataFrame({
        "building_id": np.arange(1, n + 1),
        "floor_area": floor_area,
        "occupants": occupants,
        "hvac_type": hvac_type,
        "outdoor_temp_c": np.round(outdoor_temp_c, 1),
        "insulation_rating": insulation_rating,
        "is_weekend": is_weekend,
        "month": month,
        "appliance_density": np.round(appliance_density, 3),
        "kwh_day": np.round(kwh_day, 1),
    })


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    patients = make_patient_screening()
    patients_path = DATA_DIR / "patient_screening.csv"
    patients.to_csv(patients_path, index=False)
    print(f"Wrote {patients_path}: {len(patients)} rows, "
          f"{patients['needs_followup'].mean():.1%} positives")

    energy = make_building_energy()
    energy_path = DATA_DIR / "building_energy.csv"
    energy.to_csv(energy_path, index=False)
    print(f"Wrote {energy_path}: {len(energy)} rows, "
          f"kwh_day mean={energy['kwh_day'].mean():.1f}")


if __name__ == "__main__":
    main()
