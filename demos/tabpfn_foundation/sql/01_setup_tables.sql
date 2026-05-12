-- ----------------------------------------------------------------------------
-- TabPFN-3 foundation-model demo — table setup
--
-- Two small tabular datasets sized for TabPFN's sweet spot (≤10k rows):
--   * PatientScreening  → binary classification (needs_followup)
--   * BuildingEnergy    → continuous regression  (kwh_day)
--
-- Both tables include a held-out split column so we can train on one subset
-- and PREDICT() against another from the same table — useful for quick
-- experimentation without juggling separate train/test schemas.
-- ----------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS TabPFN;

-- ----- Patient screening (classification) -----
DROP TABLE IF EXISTS TabPFN.PatientScreening;
CREATE TABLE TabPFN.PatientScreening (
    patient_id        INT PRIMARY KEY,
    age               INT,
    sex               VARCHAR(8),
    bmi               DECIMAL(5,1),
    systolic_bp       INT,
    diastolic_bp      INT,
    fasting_glucose   INT,
    hdl               INT,
    ldl               INT,
    smoker            INT,
    family_history    INT,
    exercise_days     INT,
    needs_followup    INT,
    split             VARCHAR(10) -- 'train' or 'test'
);

-- ----- Building energy (regression) -----
DROP TABLE IF EXISTS TabPFN.BuildingEnergy;
CREATE TABLE TabPFN.BuildingEnergy (
    building_id        INT PRIMARY KEY,
    floor_area         INT,
    occupants          INT,
    hvac_type          VARCHAR(40),
    outdoor_temp_c     DECIMAL(5,1),
    insulation_rating  INT,
    is_weekend         INT,
    month              INT,
    appliance_density  DECIMAL(6,3),
    kwh_day            DECIMAL(8,1),
    split              VARCHAR(10)
);
