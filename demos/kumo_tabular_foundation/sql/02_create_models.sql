-- ----------------------------------------------------------------------------
-- Kumo Tabular demo — register classifier + regressor as IntegratedML models
--
-- Each model points at a per-function staging directory created by
-- scripts/deploy_models.py. The userparams block is forwarded to the
-- IRISModel wrapper:
--   * num_estimators     ensemble size passed to KumoTabular at inference
--                        (the model card example uses 8)
--   * max_context_rows   cap on labelled rows used as in-context examples;
--                        larger training sets are sampled down (Kumo Tabular
--                        was trained on tables of up to 60k rows)
--   * device             "auto" (default), "cuda" or "cpu"
--   * model_kwargs       extra kwargs for sdm.models.KumoTabular(...), e.g. to
--                        pick a model size — see structured-data-models docs
--   * force_fallback     true -> use the sklearn fallback even when sdm is
--                        installed (useful for A/B testing)
-- ----------------------------------------------------------------------------

-- ----- Classification: who needs follow-up testing? -----
DROP MODEL IF EXISTS KumoPatientRiskScreener;
CREATE MODEL KumoPatientRiskScreener
PREDICTING (needs_followup)
FROM (
    SELECT age, sex, bmi, systolic_bp, diastolic_bp, fasting_glucose, hdl, ldl,
           smoker, family_history, exercise_days, needs_followup
    FROM KumoTab.PatientScreening
    WHERE split = 'train'
)
USING {
    "pathtoclassifiers": "/opt/irisapp/demos/kumo_tabular_foundation/iris_models/_staging/kumo_classifier",
    "iscmodelsdisabled": 1,
    "userparams": {
        "num_estimators": 8,
        "max_context_rows": 10000
    }
};
TRAIN MODEL KumoPatientRiskScreener;

-- Predictions on the held-out split.
SELECT
    patient_id,
    age,
    sex,
    bmi,
    fasting_glucose,
    needs_followup AS actual,
    PREDICT(KumoPatientRiskScreener) AS predicted
FROM KumoTab.PatientScreening
WHERE split = 'test'
ORDER BY patient_id;


-- ----- Regression: how many kWh/day will the building consume? -----
DROP MODEL IF EXISTS KumoBuildingEnergyForecaster;
CREATE MODEL KumoBuildingEnergyForecaster
PREDICTING (kwh_day)
FROM (
    SELECT floor_area, occupants, hvac_type, outdoor_temp_c, insulation_rating,
           is_weekend, month, appliance_density, kwh_day
    FROM KumoTab.BuildingEnergy
    WHERE split = 'train'
)
USING {
    "pathtoregressors": "/opt/irisapp/demos/kumo_tabular_foundation/iris_models/_staging/kumo_regressor",
    "iscmodelsdisabled": 1,
    "userparams": {
        "num_estimators": 8,
        "max_context_rows": 10000
    }
};
TRAIN MODEL KumoBuildingEnergyForecaster;

-- Predictions + residuals on the held-out split.
SELECT
    building_id,
    floor_area,
    occupants,
    hvac_type,
    outdoor_temp_c,
    kwh_day AS actual,
    PREDICT(KumoBuildingEnergyForecaster) AS predicted,
    kwh_day - PREDICT(KumoBuildingEnergyForecaster) AS residual
FROM KumoTab.BuildingEnergy
WHERE split = 'test'
ORDER BY building_id;
