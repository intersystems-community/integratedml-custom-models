"""Tests for the run_gaia_demo pipeline functions — no IRIS, no Gaia network."""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from run_gaia_demo import compute_variability


def _mock_gaia_df(n=20, seed=7):
    rng = np.random.default_rng(seed)
    source_ids = rng.integers(1_000_000_000, 9_999_999_999, n)
    ra  = rng.uniform(50, 60, n)
    dec = rng.uniform(20, 28, n)
    mean_mag = rng.uniform(10, 18, n)
    flux = 10 ** ((20 - mean_mag) / 2.5) * 1e6
    snr  = rng.uniform(5, 200, n)
    flux_err = flux / snr
    n_obs = rng.integers(20, 150, n)
    return pd.DataFrame({
        "source_id": source_ids,
        "ra": ra, "dec": dec,
        "phot_g_mean_mag": mean_mag,
        "phot_g_mean_flux": flux,
        "phot_g_mean_flux_error": flux_err,
        "phot_g_n_obs": n_obs,
        "phot_variable_flag": ["NOT_AVAILABLE"] * n,
    })


class TestComputeVariability:

    def test_columns_present(self):
        df = compute_variability(_mock_gaia_df(), threshold_pct=10.0)
        for col in ["std_mag", "min_mag", "max_mag", "mag_range", "pct_change", "is_variable"]:
            assert col in df.columns

    def test_min_less_than_max(self):
        df = compute_variability(_mock_gaia_df(), threshold_pct=10.0)
        assert (df["min_mag"] < df["max_mag"]).all()

    def test_is_variable_binary(self):
        df = compute_variability(_mock_gaia_df(50), threshold_pct=10.0)
        assert set(df["is_variable"].unique()).issubset({0, 1})

    def test_threshold_filter_consistency(self):
        df = compute_variability(_mock_gaia_df(100), threshold_pct=5.0)
        variable = df[df["is_variable"] == 1]
        assert (variable["pct_change"] >= 5.0).all()
        stable = df[df["is_variable"] == 0]
        assert (stable["pct_change"] < 5.0).all()

    def test_zero_flux_error_handled(self):
        df = _mock_gaia_df(10)
        df.loc[0, "phot_g_mean_flux_error"] = 0.0
        result = compute_variability(df, threshold_pct=10.0)
        assert result["std_mag"].notna().all()
        assert result["std_mag"].iloc[0] == pytest.approx(0.001)

    def test_pct_change_formula(self):
        df = _mock_gaia_df(5)
        result = compute_variability(df, threshold_pct=10.0)
        expected = result["mag_range"] / result["phot_g_mean_mag"].abs() * 100.0
        pd.testing.assert_series_equal(result["pct_change"], expected, check_names=False)
