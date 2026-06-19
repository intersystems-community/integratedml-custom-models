-- Gaia Variable Star Detection — IntegratedML Model Setup
-- Uses the custom GaiaVariabilityDetector IRISModel

-- Train the variability classifier on ingested Gaia statistics
CREATE MODEL GaiaVariability PREDICTING (is_variable)
FROM GaiaObservationStats
USING {
    "pathtoclassifiers": "/opt/irisapp/demos/gaia_variable_stars/iris_models",
    "iscmodelsdisabled": 1
};

TRAIN MODEL GaiaVariability;

-- Validate model accuracy
VALIDATE MODEL GaiaVariability FROM GaiaObservationStats;
