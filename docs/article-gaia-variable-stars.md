# Detecting Variable Stars with Gaia DR3 and InterSystems IRIS IntegratedML

An entry for the InterSystems Employee Programming Challenge #1

---

## The Challenge

The task: use publicly available Gaia observation data to find astronomical
objects whose brightness changed more than X% over time, produce a sortable
list, and run the whole pipeline automatically via a CI/CD script.

Gaia is ESA's space telescope that has measured the position and brightness of
over 1.5 billion stars. Its data is publicly available at the
[Gaia Science Archive](https://gea.esac.esa.int/archive/) and queryable with
standard SQL via ADQL.

My approach: bring the entire pipeline — download, ingest, model training, and
query — inside **InterSystems IRIS** using **IntegratedML Custom Models**, a
new capability
[announced in November 2025](https://community.intersystems.com/post/new-integratedml-custom-models-early-access-program-deploy-your-python-ml-models-sql)
that is targeting GA in IRIS 2026.1.

---

## Why IntegratedML for Astronomy?

The standard approach to this problem would be:

1. Download CSV from Gaia
2. Load into pandas
3. Compute statistics in Python
4. Write results to a file

The IntegratedML approach is different:

1. Download from Gaia (astroquery)
2. Ingest into IRIS (one `executemany`)
3. `TRAIN MODEL` — custom Python classifier runs inside IRIS
4. `SELECT … WHERE pct_change >= :threshold ORDER BY pct_change DESC`

The result set lives in the database. Downstream queries, joins to other
tables, and reporting tools can all reach it with plain SQL — no data
movement, no Python process keeping state.

---

## The Data: Gaia DR3

Gaia DR3 (Data Release 3) contains G-band photometry for ~1.5 billion sources.
The `gaia_source` table has per-source aggregate columns we need:

- `phot_g_mean_mag` — mean G-band magnitude (brighter = lower number)
- `phot_g_mean_flux` — mean flux in electrons/second
- `phot_g_mean_flux_error` — 1-sigma flux uncertainty
- `phot_g_n_obs` — number of CCD transits used

There is no per-transit light curve in `gaia_source` — that lives in the
`epoch_photometry` table, which is ~10 TB and requires a bulk download. For
this challenge I use an approximation that is standard in photometric surveys:

```text
σ_mag ≈ 1.086 / SNR
```

where `SNR = flux / flux_error`. This is Pogson's law linearised around the
mean — it gives the expected 1-sigma scatter in magnitudes for a stable source.
Sources with measured scatter significantly above this are variable candidates.

From σ_mag I derive a 3-sigma magnitude range:

```text
min_mag    = mean_mag - 3 × σ_mag
max_mag    = mean_mag + 3 × σ_mag
pct_change = (max_mag - min_mag) / mean_mag × 100
```

Sources with `pct_change >= threshold` are flagged as variable.

---

## The Pipeline

### Step 1 — Query Gaia Archive

```python
from astroquery.gaia import Gaia

query = """
SELECT TOP 5000
    source_id, ra, dec,
    phot_g_mean_mag,
    phot_g_mean_flux,
    phot_g_mean_flux_error,
    phot_g_n_obs
FROM gaiadr3.gaia_source
WHERE CONTAINS(
    POINT('ICRS', ra, dec),
    CIRCLE('ICRS', 56.75, 24.12, 1.0)
) = 1
AND phot_g_mean_mag IS NOT NULL
AND phot_g_n_obs > 10
"""

job = Gaia.launch_job_async(query)
df = job.get_results().to_pandas()
```

`astroquery.gaia` issues an asynchronous ADQL job against ESA's TAP service.
No authentication needed — the Gaia archive is fully public. The cone search
targets the Pleiades (ra=56.75, dec=24.12), a dense, well-studied field with
known variable members.

### Step 2 — Compute Variability Stats

```python
snr = df["phot_g_mean_flux"] / df["phot_g_mean_flux_error"]
df["std_mag"]    = (1.086 / snr.abs()).fillna(0.001)
df["min_mag"]    = df["phot_g_mean_mag"] - 3 * df["std_mag"]
df["max_mag"]    = df["phot_g_mean_mag"] + 3 * df["std_mag"]
df["mag_range"]  = df["max_mag"] - df["min_mag"]
df["pct_change"] = df["mag_range"] / df["phot_g_mean_mag"].abs() * 100
df["is_variable"] = (df["pct_change"] >= threshold).astype(int)
```

### Step 3 — Ingest into IRIS

```python
# Prefer in-process Embedded Python (no TCP) when running inside IRIS;
# fall back to the TCP driver for external connections.
def iris_connect():
    try:
        from iris_embedded_python import dbapi
        return dbapi.connect(namespace="USER", mode="embedded")
    except (ImportError, Exception):
        import intersystems_iris
        return intersystems_iris.dbapi.connect(
            hostname="localhost", port=1972,
            namespace="USER", username="demo", password="demo",
        )

conn = iris_connect()
cur = conn.cursor()
cur.execute("CREATE TABLE GaiaObservationStats (...)")
cur.executemany(
    "INSERT INTO GaiaObservationStats VALUES (?,?,?,?,?,?,?,?,?,?)", rows
)
conn.commit()
```

### Step 4 — Train an IntegratedML Custom Model

```sql
CREATE MODEL GaiaVariability PREDICTING (is_variable)
FROM GaiaObservationStats
USING {
    "pathtoclassifiers": "/opt/irisapp/demos/gaia_variable_stars/iris_models",
    "iscmodelsdisabled": 1
};

TRAIN MODEL GaiaVariability;
```

The `IRISModel` class (`gaia_variability_iris_model.py`) is a
`GradientBoostingClassifier` that adds three derived features before fitting:

- **CV** (coefficient of variation): `std_mag / mean_mag`
- **range-over-mean**: `mag_range / mean_mag`
- **log observation count**: `log(1 + n_obs)`

These features capture the signal-to-noise structure that distinguishes
genuinely variable sources from noisy faint ones.

### Step 5 — Query Results

```sql
SELECT
    source_id,
    ra,
    dec,
    min_mag  AS phot_g_mean_mag_min,
    max_mag  AS phot_g_mean_mag_max,
    ROUND((mag_range / mean_mag) * 100.0, 4) AS pct_change
FROM GaiaObservationStats
WHERE ROUND((mag_range / mean_mag) * 100.0, 4) >= 10
ORDER BY pct_change DESC;
```

Output (CSV to stdout, one record per line):

```csv
source_id,ra,dec,phot_g_mean_mag_min,phot_g_mean_mag_max,pct_change
66636515764786304,56.443201,24.331567,11.2341,13.0123,14.2100
...
```

---

## The IRISModel Contract

The key to this working inside IRIS is the `IRISModel` contract — a Python
class with `fit` / `predict` / `predict_proba` that IRIS's embedded Python
interpreter (`irispython`) loads at `TRAIN MODEL` time:

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class IRISModel:
    name = "gaia_variability_detector"

    def __init__(self, threshold_pct=10.0, **kwargs):
        self.threshold_pct = float(threshold_pct)
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
            )),
        ])

    def _features(self, X):
        X = np.asarray(X, dtype=float)
        mean_mag, std_mag, n_obs, mag_range = X[:,0], X[:,1], X[:,4], X[:,5]
        cv   = std_mag / (np.abs(mean_mag) + 1e-9)
        rom  = mag_range / (np.abs(mean_mag) + 1e-9)
        logn = np.log1p(n_obs)
        return np.column_stack([X, cv, rom, logn])

    def fit(self, X, y, **kwargs):
        self._pipeline.fit(self._features(X), y)
        return self

    def predict(self, X):
        return self._pipeline.predict(self._features(X))
```

The contract is intentionally minimal — just `fit`, `predict`, and a `name`
string. IRIS's
[Embedded Python](https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=GEPYTHON)
runtime (`irispython`) loads the file at `TRAIN MODEL` time, runs `fit`, then
serializes the trained model state for later `PREDICT()` calls in SQL.

A few gotchas that catch new users of Embedded Python and IntegratedML Custom
Models:

**Gotcha 1: packages install to `mgr/python`, not the system Python.**
Embedded Python has its own package directory — `<installdir>/mgr/python`.
Installing a package into your system Python or a venv does nothing for IRIS.
The correct install command targets that directory explicitly:

```bash
# Inside the container (Linux):
python3 -m pip install --target /usr/irissys/mgr/python scikit-learn tabpfn

# Or via docker exec from the host:
docker exec integratedml_iris python3 -m pip install \
    --target /usr/irissys/mgr/python scikit-learn tabpfn
```

Once a package is in `mgr/python`, any IRISModel file can import it freely.
Third-party packages — numpy, scikit-learn, pandas, anthropic, lightgbm,
tabpfn — all work exactly as they do in regular Python.

**Gotcha 2: your repo's internal packages are not on `sys.path`.**
IRISModel files are loaded from a staging directory that has no relationship
to your project root. `from shared.models import ClassificationModel` will
fail with `ModuleNotFoundError` — `shared/` is not visible. The fix is either
to inline what you need into the IRISModel file (usually the right answer for
small helpers), or to install your repo as a proper package into `mgr/python`:

```bash
python3 -m pip install --target /usr/irissys/mgr/python -e /opt/irisapp
```

**Gotcha 3: local development environment drift.**
When iterating on IRISModel files locally before deploying into the container,
your local Python environment may not match what IRIS sees. The
[`iris-embedded-python-wrapper`](https://github.com/grongierisc/iris-embedded-python-wrapper)
package (by Guillaume Rongier, available on PyPI) solves this: it lets you
bind a local virtual environment to Embedded Python so both sides see the same
packages:

```bash
pip install iris-embedded-python-wrapper
iris_embedded_python_wrapper bind   # point IRIS at your local venv
iris_embedded_python_wrapper unbind # restore IRIS's own mgr/python
```

This makes the develop-test-deploy loop much tighter.

---

## Going Further: AI Functions and TabPFN

The contest awards bonus points for using AI Hub. The same
[integratedml-custom-models repository](https://github.com/intersystems-community/integratedml-custom-models)
contains two additional demos that push the `IRISModel` pattern further:

### AI Functions

`demos/ai_functions/` implements seven SQL-native AI primitives —
`AI_COMPLETE`, `AI_SENTIMENT`, `AI_TRANSLATE`, `EMBED_TEXT`, `AI_SUMMARIZE`,
`AI_CLASSIFY`, `AI_EXTRACT` — as `IRISModel` files. The idea comes from
SingleStore's AI Functions announcement: what if you could call LLM operations
directly from SQL, like a built-in function?

```sql
SELECT
    ticket_id,
    PREDICT(AISentiment  USE message_text) AS sentiment,
    PREDICT(AIClassify   USE message_text) AS department,
    PREDICT(AISummarize  USE message_text) AS summary
FROM SupportTickets;
```

Each model runs offline by default (lexicon scorer, TF-IDF summarizer, regex
extractor). `AI_COMPLETE` upgrades to the real Claude API
(`claude-haiku-4-5`) when `ANTHROPIC_API_KEY` is set — no code change, just an
environment variable. This is the AI Hub integration: the Claude model is
called from inside an IntegratedML `PREDICT()` expression in SQL.

### TabPFN-3

`demos/tabpfn_foundation/` wraps
[Prior Labs' TabPFN-3](https://priorlabs.ai/technical-reports/tabpfn-3) — a
transformer foundation model for tabular data that performs in-context
learning without per-dataset training — as IntegratedML Custom Models.
Install `tabpfn` into the IRIS Python environment and you get a state-of-the-art
tabular model callable directly from SQL:

```sql
CREATE MODEL PatientRiskScreener PREDICTING (needs_followup)
FROM TabPFN.PatientScreening
USING {
    "pathtoclassifiers": "...",
    "iscmodelsdisabled": 1,
    "userparams": {"device": "cpu", "n_estimators": 8}
};
TRAIN MODEL PatientRiskScreener;
```

Both demos fall back to scikit-learn `GradientBoosting` when the optional
package is absent, so they work offline with no extra installs.

---

## Running It

```bash
# Default 10% threshold — prints CSV to stdout
python run_gaia_demo.py

# Custom threshold
python run_gaia_demo.py --threshold 5

# CI/CD entrypoint (no manual input)
./RunChallenge
THRESHOLD=5 ./RunChallenge
```

The `RunChallenge` script at the repo root starts IRIS if needed, installs
deps, and delegates to `run_gaia_demo.py`. GitHub Actions runs it automatically
on every push via `.github/workflows/run-challenge.yml`.

---

## Notes for New Users

**On the Gaia API:** `astroquery.gaia` is straightforward — ADQL is standard
SQL with geometry functions (`CONTAINS`, `CIRCLE`, `POINT`) bolted on, and the
ESA TAP service is public and well-documented. The one thing to establish
upfront: `gaia_source` has aggregate per-source statistics (all you need for
this challenge), while actual per-transit light curves live in
`epoch_photometry` (~10 TB, separate bulk download). Don't go looking for a
single table that has both.

**On Embedded Python and package installation:** New users often try
`pip install numpy` in their local terminal and then wonder why IRIS can't find
it. Embedded Python has its own package directory (`mgr/python`), separate from
your system Python and any venv. Everything has to be installed there
explicitly — see the gotchas above. Once you internalize that `irispython` is
just CPython with a specific `sys.path`, the rest follows naturally. The
[official Embedded Python docs](https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=GEPYTHON)
on docs.intersystems.com cover this thoroughly, including the Flexible Python
Runtime feature for choosing which CPython version IRIS uses.

**On using IntegratedML for a data engineering problem:** Survey astronomy
produces billions of rows of structured measurements — exactly the workload
databases were designed for. The value of running the classifier inside IRIS
rather than in a separate Python process is composability: once
`GaiaVariability` is trained, any SQL query can call `PREDICT()` on it,
including joins, subqueries, and aggregations over the result set. For a
pipeline like this (download → ingest → classify → query), keeping every step
close to the data eliminates an entire class of data-movement bugs.

---

## Source Code

All source code is on GitHub:
[intersystems-community/integratedml-custom-models](https://github.com/intersystems-community/integratedml-custom-models)

The Gaia demo lives under `demos/gaia_variable_stars/`. Related demos:
`demos/ai_functions/` (AI Hub bonus) and `demos/tabpfn_foundation/`
(foundation model demo).

| File                                                                   | Purpose                            |
| ---------------------------------------------------------------------- | ---------------------------------- |
| `run_gaia_demo.py`                                                     | End-to-end pipeline runner         |
| `RunChallenge`                                                         | CI/CD entrypoint (no manual input) |
| `demos/gaia_variable_stars/iris_models/gaia_variability_iris_model.py` | IntegratedML custom model          |
| `demos/gaia_variable_stars/sql/create_model.sql`                       | SQL to train the model             |
| `demos/gaia_variable_stars/sql/query_variable_stars.sql`               | SQL to query results               |
| `.github/workflows/run-challenge.yml`                                  | GitHub Actions CI                  |
| `demos/ai_functions/`                                                  | AI Functions demo (AI Hub bonus)   |
| `demos/tabpfn_foundation/`                                             | TabPFN-3 foundation model demo     |
