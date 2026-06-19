-- ----------------------------------------------------------------------------
-- TabPFN-3 demo — register classifier + regressor as IntegratedML models
--
-- Each model points at a per-function staging directory created by
-- scripts/deploy_models.py. The userparams block forwards TabPFN-3
-- constructor kwargs:
--   * device                      "cpu" or "cuda"
--   * n_estimators                ensemble size (8 = paper default)
--   * ignore_pretraining_limits   bypass row/feature guardrails for larger
--                                 datasets (TabPFN-3 supports up to 1M rows)
--   * force_fallback              true → use the sklearn fallback even when
--                                 tabpfn is installed (useful for A/B testing)
-- ----------------------------------------------------------------------------

-- ----- Classification: who needs follow-up testing? -----
DROP MODEL IF EXISTS PatientRiskScreener;
CREATE MODEL PatientRiskScreener
PREDICTING (needs_followup)
FROM TabPFN.PatientScreening
USING {
    "pathtoclassifiers": "/opt/irisapp/demos/tabpfn_foundation/iris_models/_staging/tabpfn_classifier",
    "iscmodelsdisabled": 1,
    "userparams": {
        "device": "cpu",
        "n_estimators": 8
    }
}
WHERE split = 'train';
TRAIN MODEL PatientRiskScreener;

-- Predictions on the held-out split.
SELECT
    patient_id,
    age,
    sex,
    bmi,
    fasting_glucose,
    needs_followup AS actual,
    PREDICT(PatientRiskScreener) AS predicted
FROM TabPFN.PatientScreening
WHERE split = 'test'
ORDER BY patient_id;


-- ----- Regression: how many kWh/day will the building consume? -----
DROP MODEL IF EXISTS BuildingEnergyForecaster;
CREATE MODEL BuildingEnergyForecaster
PREDICTING (kwh_day)
FROM TabPFN.BuildingEnergy
USING {
    "pathtoregressors": "/opt/irisapp/demos/tabpfn_foundation/iris_models/_staging/tabpfn_regressor",
    "iscmodelsdisabled": 1,
    "userparams": {
        "device": "cpu",
        "n_estimators": 8
    }
}
WHERE split = 'train';
TRAIN MODEL BuildingEnergyForecaster;

-- Predictions + residuals on the held-out split.
SELECT
    building_id,
    floor_area,
    occupants,
    hvac_type,
    outdoor_temp_c,
    kwh_day AS actual,
    PREDICT(BuildingEnergyForecaster) AS predicted,
    kwh_day - PREDICT(BuildingEnergyForecaster) AS residual
FROM TabPFN.BuildingEnergy
WHERE split = 'test'
ORDER BY building_id;
