"""Scoring rule checks: CRPS against the analytic Gaussian form first."""

import numpy as np
from scipy import stats

from nemforecastdemand.evaluation.calibration import (
    pit_gaussian,
    pit_histogram,
    pit_samples,
    reliability_table,
)
from nemforecastdemand.evaluation.diagnostics import ElboTrace, sampler_health, time_to_target_ess
from nemforecastdemand.evaluation.metrics import (
    crps_gaussian,
    crps_samples,
    energy_score,
    interval_coverage,
    log_score_gaussian,
    mase,
    paired_bootstrap_difference,
    pinball_loss,
)


def test_crps_gaussian_known_value():
    # At y = mean the closed form reduces to sd (sqrt(2/pi) - 1/sqrt(pi)).
    expected = 2.0 * (np.sqrt(2.0 / np.pi) - 1.0 / np.sqrt(np.pi))
    np.testing.assert_allclose(crps_gaussian(np.array([5.0]), 5.0, 2.0), expected, rtol=1e-12)


def test_crps_samples_matches_analytic_gaussian():
    rng = np.random.default_rng(0)
    mean = np.array([0.0, 2.0, -1.0, 10.0])
    sd = np.array([1.0, 0.5, 3.0, 2.0])
    y = np.array([0.3, 1.0, -4.0, 12.5])
    draws = rng.normal(mean, sd, size=(4000, 4))
    sample_based = crps_samples(y, draws, chunk=512)
    analytic = crps_gaussian(y, mean, sd)
    np.testing.assert_allclose(sample_based, analytic, rtol=0.05)


def test_crps_samples_chunk_invariance():
    rng = np.random.default_rng(1)
    y = np.array([1.0, -2.0])
    draws = rng.normal(size=(700, 2))
    full = crps_samples(y, draws, chunk=700)
    chunked = crps_samples(y, draws, chunk=128)
    np.testing.assert_allclose(full, chunked, rtol=1e-12)


def test_energy_score_reduces_to_crps_in_one_dimension():
    rng = np.random.default_rng(2)
    y = np.array([0.7])
    draws = rng.normal(size=(2000, 1))
    es = energy_score(y, draws, chunk=512)
    crps = crps_samples(y, draws, chunk=512)[0]
    np.testing.assert_allclose(es, crps, rtol=1e-10)


def test_pinball_median_is_half_mae():
    y = np.array([1.0, 2.0, 4.0])
    median = np.array([[2.0, 2.0, 2.0]])
    loss = pinball_loss(y, median, np.array([0.5]))
    np.testing.assert_allclose(loss, 0.5 * np.mean([1.0, 0.0, 2.0]))


def test_pinball_asymmetry():
    y = np.zeros(10)
    high = np.full((1, 10), 1.0)
    low_q = pinball_loss(y, high, np.array([0.1]))
    high_q = pinball_loss(y, high, np.array([0.9]))
    assert low_q > high_q  # over-forecasting hurts more at low quantiles


def test_log_score_gaussian_matches_scipy():
    y = np.array([1.5])
    np.testing.assert_allclose(log_score_gaussian(y, 1.0, 2.0), -stats.norm.logpdf(1.5, 1.0, 2.0))


def test_mase_scaling():
    y = np.array([10.0, 12.0])
    point = np.array([11.0, 11.0])
    assert mase(y, point, naive_abs_error=2.0) == 0.5


def test_interval_coverage():
    y = np.array([1.0, 5.0, 9.0, 20.0])
    assert interval_coverage(y, np.zeros(4), np.full(4, 10.0)) == 0.75


def test_paired_bootstrap_detects_difference():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 200)
    result = paired_bootstrap_difference(base + 1.0, base, n_boot=2000, seed=1)
    assert result["p_value"] < 0.01
    assert result["lower95"] > 0.5
    same = paired_bootstrap_difference(base, base.copy(), n_boot=2000, seed=1)
    assert same["difference"] == 0.0


def test_pit_uniform_under_calibration():
    rng = np.random.default_rng(4)
    y = rng.normal(2.0, 3.0, 5000)
    pit = pit_gaussian(y, 2.0, 3.0)
    assert abs(pit.mean() - 0.5) < 0.02
    assert abs(pit.std() - np.sqrt(1.0 / 12.0)) < 0.02
    density, _ = pit_histogram(pit, bins=10)
    assert np.all(np.abs(density - 1.0) < 0.15)


def test_pit_samples_near_uniform():
    rng = np.random.default_rng(5)
    y = rng.normal(size=2000)
    draws = rng.normal(size=(500, 2000))
    pit = pit_samples(y, draws)
    assert abs(pit.mean() - 0.5) < 0.03


def test_reliability_table():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    quantile_forecasts = np.tile(np.array([[2.5], [4.5]]), (1, 4))
    observed = reliability_table(y, quantile_forecasts, np.array([0.5, 0.95]))
    np.testing.assert_allclose(observed, [0.5, 1.0])


def test_elbo_trace_convergence_flag():
    steps = np.arange(100)
    flat = ElboTrace(steps, np.full(100, -50.0), np.zeros(100), np.zeros(100))
    rising = ElboTrace(steps, -50.0 + steps * 1.0, np.zeros(100), np.zeros(100))
    assert flat.converged(window=20)
    assert not rising.converged(window=20)


def test_sampler_health_counts():
    extra = {
        "diverging": np.array([[False, True, False], [False, False, False]]),
        "energy": np.array([[1.0, 2.0, 1.5], [1.0, 1.1, 0.9]]),
        "num_steps": np.array([[1023, 3, 7], [15, 15, 15]]),
    }
    health = sampler_health(extra, max_tree_depth=10)
    assert health.loc[0, "divergences"] == 1
    assert health.loc[0, "tree_depth_saturation"] == 1.0 / 3.0
    assert health.loc[1, "divergences"] == 0


def test_time_to_target_ess_scales():
    assert time_to_target_ess(10.0, 100.0, achieved_bulk_ess=200.0, target_bulk_ess=400.0) == 210.0
    assert time_to_target_ess(10.0, 100.0, achieved_bulk_ess=800.0, target_bulk_ess=400.0) == 60.0
