-- ----------------------------------------------------------------------------
-- TabFM demo — register classifier + regressor as IntegratedML models
--
-- Each model points at a per-function staging directory created by
-- scripts/deploy_models.py. The userparams block forwards TabFM
-- constructor kwargs:
--   * n_estimators       ensemble size over different in-context subsets
--   * max_num_rows       max labelled rows used as in-context examples
--                        (TabFM default is 100; raised to 1000 here)
--   * max_num_features   max feature columns (default 500)
--   * force_fallback     true -> use the sklearn fallback even when tabfm is
--                        installed (useful for A/B testing)
-- ----------------------------------------------------------------------------

-- ----- Classification: who needs follow-up testing? -----
DROP MODEL IF EXISTS PatientRiskScreener;
CREATE MODEL PatientRiskScreener
PREDICTING (needs_followup)
FROM TabFM.PatientScreening
USING {
    "pathtoclassifiers": "/opt/irisapp/demos/tabfm_foundation/iris_models/_staging/tabfm_classifier",
    "iscmodelsdisabled": 1,
    "userparams": {
        "n_estimators": 4,
        "max_num_rows": 1000
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
FROM TabFM.PatientScreening
WHERE split = 'test'
ORDER BY patient_id;


-- ----- Regression: how many kWh/day will the building consume? -----
DROP MODEL IF EXISTS BuildingEnergyForecaster;
CREATE MODEL BuildingEnergyForecaster
PREDICTING (kwh_day)
FROM TabFM.BuildingEnergy
USING {
    "pathtoregressors": "/opt/irisapp/demos/tabfm_foundation/iris_models/_staging/tabfm_regressor",
    "iscmodelsdisabled": 1,
    "userparams": {
        "n_estimators": 4,
        "max_num_rows": 1000
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
FROM TabFM.BuildingEnergy
WHERE split = 'test'
ORDER BY building_id;
