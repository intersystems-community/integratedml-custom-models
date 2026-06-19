# Gaia Variable Star Detection

Detects astronomical objects whose brightness changed over time by more than a
user-supplied threshold, using publicly available Gaia DR3 observations and
IntegratedML custom models running inside InterSystems IRIS.

## How it works

1. **Download** — `astroquery.gaia` issues an ADQL cone-search against the
   public Gaia Science Archive (no authentication required) and retrieves
   per-source photometric statistics including mean G-band magnitude, flux,
   flux error, and observation count.

2. **Ingest** — Results are loaded into an IRIS SQL table
   (`GaiaObservationStats`) via the Python DBAPI. Magnitude variability
   statistics (std, min, max, range, % change) are derived from the Pogson
   approximation: σ_mag ≈ 1.086 / SNR.

3. **Train** — An IntegratedML `CREATE MODEL … TRAIN MODEL` statement trains a
   `GradientBoostingClassifier` custom model (`GaiaVariabilityDetector`) on the
   ingested data, predicting `is_variable`.

4. **Query** — A plain SQL `SELECT` with a `WHERE pct_change >= :threshold`
   filter returns the sortable variable-star list.

## Installation

```bash
# Install dependencies
uv pip install astroquery intersystems-iris scikit-learn pandas numpy

# Or with pip
pip install astroquery intersystems-iris scikit-learn pandas numpy

# Start IRIS
docker compose up -d iris
```

## Usage

```bash
# Run with default 10% threshold (prints CSV to stdout)
python run_gaia_demo.py

# Custom threshold
python run_gaia_demo.py --threshold 5

# Custom sky region
python run_gaia_demo.py --threshold 10 --ra 83.82 --dec -5.39 --radius 1.0
```

## Output format

```csv
source_id,ra,dec,phot_g_mean_mag_min,phot_g_mean_mag_max,pct_change
4476732745067964544,56.750123,24.119876,12.1234,13.5678,11.2300
...
```

Results are sorted by `pct_change` descending.

## CI/CD

The `RunChallenge` script at the repo root runs the full pipeline without
manual input and prints the contest-format CSV:

```bash
./RunChallenge              # uses 10% default threshold
THRESHOLD=5 ./RunChallenge  # override threshold
```

## Notes on the Gaia API

- The `gaia_source` table covers ~1.5 billion sources; always use a cone
  search with `CONTAINS(POINT, CIRCLE)`.
- `phot_variable_flag = 'VARIABLE'` is Gaia's own classification but misses
  marginal variables. The SNR-derived σ_mag approach here catches more.
- For true light curves, query `gaiadr3.epoch_photometry` (much larger,
  requires chunked download).

## Notes on the IRIS Embedded Python environment

IRIS uses [Embedded Python](https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=GEPYTHON)
— a full CPython runtime that runs in-process with IRIS, accessible via the
`irispython` command. Install packages into it by targeting `mgr/python`:

```bash
# Inside the container
docker exec integratedml_iris python3 -m pip install \
    --target /usr/irissys/mgr/python astroquery scikit-learn
```

For local development, use
[`iris-embedded-python-wrapper`](https://github.com/grongierisc/iris-embedded-python-wrapper)
to bind your local venv to Embedded Python so your environment matches IRIS:

```bash
pip install iris-embedded-python-wrapper
iris_embedded_python_wrapper bind
```

IRISModel files in `iris_models/` are loaded from a staging directory inside
IRIS — they cannot import from this repo's own packages (`from shared.models
import ...` will fail). Any package installed into `mgr/python` is fine.
To expose your own repo modules, install the repo as a package:
`python3 -m pip install --target /usr/irissys/mgr/python -e .`
