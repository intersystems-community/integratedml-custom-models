"""
Gaia DR3 data ingestion into IRIS.

Downloads photometric time-series statistics for a sky region from the
Gaia Science Archive via ADQL (astroquery.gaia), then populates two tables:

  GaiaSources           — one row per source (ra, dec, mean mag, etc.)
  GaiaObservationStats  — aggregated stats per source (used for IntegratedML)

Usage:
    python ingest_gaia.py [--ra RA] [--dec DEC] [--radius DEG] [--limit N]

# Feedback on Gaia API:
#   The Gaia Science Archive ADQL endpoint is public and requires no auth.
#   astroquery.gaia wraps it cleanly. The gaia_source table has ~1.5B rows;
#   always filter by sky region. phot_g_mean_flux and its error give SNR.
#   phot_variable_flag = 'VARIABLE' is a Gaia-native classification but
#   misses many marginal variables — computing our own std/range is better.
"""

import os
import argparse
import logging
import pandas as pd
from astroquery.gaia import Gaia


def _iris_dbapi():
    """Return dbapi module: Embedded Python (in-process) preferred, TCP fallback."""
    try:
        from iris_embedded_python import dbapi
        return dbapi, {"mode": "embedded"}
    except (ImportError, Exception):
        import intersystems_iris
        dbapi = intersystems_iris.dbapi
        return dbapi, {
            "hostname": os.getenv("IRIS_HOST", "localhost"),
            "port": int(os.getenv("IRIS_PORT", "1972")),
            "username": os.getenv("IRIS_USERNAME", "demo"),
            "password": os.getenv("IRIS_PASSWORD", "demo"),
        }

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DDL_SOURCES = """
CREATE TABLE IF NOT EXISTS GaiaSources (
    source_id    BIGINT       PRIMARY KEY,
    ra           DOUBLE,
    "dec"        DOUBLE,
    mean_mag     DOUBLE,
    mag_err      DOUBLE,
    n_obs        INT,
    variability  VARCHAR(20)
)
"""

DDL_STATS = """
CREATE TABLE IF NOT EXISTS GaiaObservationStats (
    source_id    BIGINT       PRIMARY KEY,
    ra           DOUBLE,
    "dec"        DOUBLE,
    mean_mag     DOUBLE,
    std_mag      DOUBLE,
    min_mag      DOUBLE,
    max_mag      DOUBLE,
    n_obs        INT,
    mag_range    DOUBLE,
    is_variable  INT
)
"""


def fetch_gaia(ra, dec, radius, limit):
    """ADQL query: cone search on gaia_source, return per-source magnitude stats."""
    log.info(f"Querying Gaia DR3 — center ({ra}, {dec}), radius {radius} deg, limit {limit}")

    # Use ADQL aggregate over gaia_source which has epoch photometry stats inline
    query = f"""
    SELECT
        source_id,
        ra,
        dec,
        phot_g_mean_mag                                    AS mean_mag,
        phot_g_mean_flux_error / NULLIF(phot_g_mean_flux, 0) AS rel_err,
        phot_g_n_obs                                       AS n_obs,
        phot_variable_flag                                 AS variability,
        -- Approximate mag std from flux SNR: sigma_mag ≈ 1.086 / SNR
        1.086 / NULLIF(phot_g_mean_flux / NULLIF(phot_g_mean_flux_error, 0), 0) AS std_mag_approx
    FROM gaiadr3.gaia_source
    WHERE
        CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius})
        ) = 1
        AND phot_g_mean_mag IS NOT NULL
        AND phot_g_n_obs > 10
    ORDER BY phot_g_mean_mag ASC
    """
    if limit:
        query = query.rstrip() + f"\nLIMIT {limit}"

    job = Gaia.launch_job_async(query, verbose=False)
    tbl = job.get_results()
    df = tbl.to_pandas()
    log.info(f"Retrieved {len(df)} sources from Gaia")
    return df


def compute_stats(df, threshold_pct):
    """
    Build GaiaObservationStats rows.
    Mag range estimated from ±3*std; is_variable set by threshold.
    """
    df = df.copy()
    df["std_mag"] = df["std_mag_approx"].fillna(0.001).abs()
    df["min_mag"] = df["mean_mag"] - 3 * df["std_mag"]
    df["max_mag"] = df["mean_mag"] + 3 * df["std_mag"]
    df["mag_range"] = df["max_mag"] - df["min_mag"]

    # pct_change: magnitude change as % of mean (higher mag = fainter, so range over mean)
    # Astronomers: delta_mag → flux ratio. But contest asks simple % of phot_g_mean_mag.
    # We use: pct_change = mag_range / mean_mag * 100 (consistent with contest formula)
    df["pct_change"] = df["mag_range"] / df["mean_mag"].abs() * 100.0
    df["is_variable"] = (df["pct_change"] >= threshold_pct).astype(int)
    return df


def ingest(df, conn):
    cur = conn.cursor()
    cur.execute(DDL_SOURCES)
    cur.execute(DDL_STATS)
    conn.commit()

    src_rows = [(
        int(r.source_id), float(r.ra), float(r.dec),
        float(r.mean_mag), float(r.std_mag),
        int(r.n_obs), str(r.variability or "NOT_AVAILABLE")
    ) for _, r in df.iterrows()]

    stats_rows = [(
        int(r.source_id), float(r.ra), float(r.dec),
        float(r.mean_mag), float(r.std_mag),
        float(r.min_mag), float(r.max_mag),
        int(r.n_obs), float(r.mag_range), int(r.is_variable)
    ) for _, r in df.iterrows()]

    cur.executemany(
        "INSERT OR REPLACE INTO GaiaSources VALUES (?,?,?,?,?,?,?)", src_rows
    )
    cur.executemany(
        "INSERT OR REPLACE INTO GaiaObservationStats VALUES (?,?,?,?,?,?,?,?,?,?)", stats_rows
    )
    conn.commit()
    log.info(f"Ingested {len(src_rows)} sources into IRIS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra",        type=float, default=56.75,   help="RA center (degrees)")
    ap.add_argument("--dec",       type=float, default=24.12,   help="Dec center (degrees)")
    ap.add_argument("--radius",    type=float, default=0.5,     help="Search radius (degrees)")
    ap.add_argument("--limit",     type=int,   default=5000,    help="Max sources to fetch")
    ap.add_argument("--threshold", type=float, default=10.0,    help="Variability threshold %%")
    args = ap.parse_args()

    df = fetch_gaia(args.ra, args.dec, args.radius, args.limit)
    df = compute_stats(df, args.threshold)

    dbapi, extra = _iris_dbapi()
    conn = dbapi.connect(namespace=os.getenv("IRIS_NAMESPACE", "USER"), **extra)
    try:
        ingest(df, conn)
    finally:
        conn.close()

    log.info("Done.")


if __name__ == "__main__":
    main()
