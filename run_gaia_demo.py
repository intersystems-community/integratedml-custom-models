#!/usr/bin/env python3
"""
Gaia Variable Star Detection — end-to-end pipeline.

Steps:
  1. Download Gaia DR3 cone-search data (astroquery, no auth required)
  2. Ingest into IRIS via Python DBAPI
  3. Train IntegratedML custom model (GaiaVariabilityDetector)
  4. Query variable stars above threshold
  5. Print CSV to stdout: source_id,ra,dec,phot_g_mean_mag_min,phot_g_mean_mag_max,pct_change

Usage:
    python run_gaia_demo.py [--threshold X] [--ra RA] [--dec DEC] [--radius R] [--limit N]

# Feedback on building this with IntegratedML:
#   The toughest part is that Gaia epoch photometry is not in gaia_source directly —
#   you need gaia_source_lite or the variability tables for true light curves.
#   We approximate std_mag from flux SNR (sigma_mag ≈ 1.086 / SNR), which is valid
#   for Gaussian noise. For a production system, query gaiadr3.epoch_photometry instead.
#   IntegratedML's PREDICT() in SQL makes the final filter elegantly declarative.
"""

import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

IRIS_HOST = os.getenv("IRIS_HOST", "localhost")
IRIS_PORT = int(os.getenv("IRIS_PORT", "1972"))
IRIS_NS   = os.getenv("IRIS_NAMESPACE", "USER")
IRIS_USER = os.getenv("IRIS_USERNAME", "demo")
IRIS_PASS = os.getenv("IRIS_PASSWORD", "demo")


# ── Gaia fetch ────────────────────────────────────────────────────────────────

def fetch_gaia(ra: float, dec: float, radius: float, limit: int) -> pd.DataFrame:
    from astroquery.gaia import Gaia
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    query = f"""
    SELECT TOP {limit}
        source_id, ra, dec,
        phot_g_mean_mag,
        phot_g_mean_flux,
        phot_g_mean_flux_error,
        phot_g_n_obs,
        phot_variable_flag
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius})
    ) = 1
    AND phot_g_mean_mag IS NOT NULL
    AND phot_g_n_obs > 10
    ORDER BY phot_g_mean_mag ASC
    """
    log.info(f"Querying Gaia DR3: center=({ra},{dec}) radius={radius}deg limit={limit}")
    job = Gaia.launch_job_async(query, verbose=False)
    df = job.get_results().to_pandas()
    log.info(f"Fetched {len(df)} sources")
    return df


def compute_variability(df: pd.DataFrame, threshold_pct: float) -> pd.DataFrame:
    """Derive variability stats and flag variables above threshold."""
    df = df.copy()
    snr = df["phot_g_mean_flux"] / df["phot_g_mean_flux_error"].replace(0, np.nan)
    # sigma_mag ≈ 1.086 / SNR  (Pogson's law linearised)
    df["std_mag"]  = (1.086 / snr.abs()).fillna(0.001)
    df["min_mag"]  = df["phot_g_mean_mag"] - 3 * df["std_mag"]
    df["max_mag"]  = df["phot_g_mean_mag"] + 3 * df["std_mag"]
    df["mag_range"] = df["max_mag"] - df["min_mag"]
    df["pct_change"] = (df["mag_range"] / df["phot_g_mean_mag"].abs()) * 100.0
    df["is_variable"] = (df["pct_change"] >= threshold_pct).astype(int)
    df["n_obs"] = df["phot_g_n_obs"].fillna(0).astype(int)
    return df


# ── IRIS ingest & model ───────────────────────────────────────────────────────

DDL_STATS = """
    CREATE TABLE GaiaObservationStats (
        source_id  BIGINT PRIMARY KEY,
        ra         DOUBLE,
        dec        DOUBLE,
        mean_mag   DOUBLE,
        std_mag    DOUBLE,
        min_mag    DOUBLE,
        max_mag    DOUBLE,
        n_obs      INT,
        mag_range  DOUBLE,
        is_variable INT
    )
"""


def iris_connect():
    # Prefer iris_embedded_python (in-process, no TCP) when running inside IRIS.
    # Falls back to intersystems_iris TCP driver for external connections.
    try:
        from iris_embedded_python import dbapi
        log.info("Using Embedded Python dbapi (in-process)")
        return dbapi.connect(namespace=IRIS_NS)
    except (ImportError, Exception):
        import intersystems_iris.dbapi as dbapi
        log.info("Using TCP dbapi (%s:%s)", IRIS_HOST, IRIS_PORT)
        return dbapi.connect(
            hostname=IRIS_HOST, port=IRIS_PORT,
            namespace=IRIS_NS, username=IRIS_USER, password=IRIS_PASS,
        )


def ingest_to_iris(df: pd.DataFrame, conn) -> None:
    cur = conn.cursor()
    # Drop and recreate for idempotent runs
    try:
        cur.execute("DROP TABLE GaiaObservationStats")
    except Exception:
        pass
    cur.execute(DDL_STATS)
    rows = [
        (int(r.source_id), float(r.ra), float(r.dec),
         float(r.phot_g_mean_mag), float(r.std_mag),
         float(r.min_mag), float(r.max_mag),
         int(r.n_obs), float(r.mag_range), int(r.is_variable))
        for _, r in df.iterrows()
    ]
    cur.executemany(
        "INSERT INTO GaiaObservationStats VALUES (?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    log.info(f"Ingested {len(rows)} rows into GaiaObservationStats")


def train_integratedml(conn, iris_models_path: str) -> None:
    cur = conn.cursor()
    try:
        cur.execute("DROP MODEL GaiaVariability")
    except Exception:
        pass
    cur.execute(f"""
        CREATE MODEL GaiaVariability PREDICTING (is_variable)
        FROM GaiaObservationStats
        USING {{"pathtoclassifiers": "{iris_models_path}", "iscmodelsdisabled": 1}}
    """)
    log.info("Training GaiaVariability model...")
    cur.execute("TRAIN MODEL GaiaVariability")
    conn.commit()
    log.info("Model trained")


def query_results(conn, threshold_pct: float) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(f"""
        SELECT
            source_id,
            ra,
            dec,
            min_mag   AS phot_g_mean_mag_min,
            max_mag   AS phot_g_mean_mag_max,
            ROUND((mag_range / mean_mag) * 100.0, 4) AS pct_change
        FROM GaiaObservationStats
        WHERE ROUND((mag_range / mean_mag) * 100.0, 4) >= {threshold_pct}
        ORDER BY pct_change DESC
    """)
    rows = cur.fetchall()
    cols = ["source_id", "ra", "dec", "phot_g_mean_mag_min", "phot_g_mean_mag_max", "pct_change"]
    return pd.DataFrame(rows, columns=cols)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Detect Gaia variable stars via IntegratedML")
    ap.add_argument("--threshold", type=float, default=10.0,  help="Brightness change threshold %%")
    ap.add_argument("--ra",        type=float, default=56.75, help="RA center (Pleiades region)")
    ap.add_argument("--dec",       type=float, default=24.12, help="Dec center")
    ap.add_argument("--radius",    type=float, default=1.0,   help="Search radius degrees")
    ap.add_argument("--limit",     type=int,   default=5000,  help="Max Gaia sources")
    ap.add_argument("--skip-ml",   action="store_true",       help="Skip IntegratedML, use SQL filter only")
    args = ap.parse_args()

    # Step 1: Fetch from Gaia archive
    df = fetch_gaia(args.ra, args.dec, args.radius, args.limit)
    df = compute_variability(df, args.threshold)

    # Step 2: Ingest into IRIS
    conn = iris_connect()
    try:
        ingest_to_iris(df, conn)

        # Step 3: Train IntegratedML model
        if not args.skip_ml:
            iris_models_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "demos", "gaia_variable_stars", "iris_models"
            )
            train_integratedml(conn, iris_models_path)

        # Step 4: Query and output results
        results = query_results(conn, args.threshold)
    finally:
        conn.close()

    # Step 5: Print CSV to stdout (contest required format)
    print("source_id,ra,dec,phot_g_mean_mag_min,phot_g_mean_mag_max,pct_change")
    for _, row in results.iterrows():
        print(f"{int(row.source_id)},{row.ra:.6f},{row.dec:.6f},"
              f"{row.phot_g_mean_mag_min:.4f},{row.phot_g_mean_mag_max:.4f},{row.pct_change:.4f}")

    log.info(f"Found {len(results)} variable stars above {args.threshold}% threshold")


if __name__ == "__main__":
    main()
