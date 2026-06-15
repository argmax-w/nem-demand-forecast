"""Divergence checks: closed forms first, then a sample-based cross-check."""

import numpy as np
from scipy import stats

from nemforecastdemand.evaluation.divergence import (
    gaussian_kl,
    gaussian_kl_1d,
    gaussian_kl_noise_floor,
    joint_gaussian_kl,
    knn_kl,
    marginal_kl,
    stack_unconstrained,
    to_unconstrained,
)


def test_gaussian_kl_1d_zero_for_identical():
    assert gaussian_kl_1d(1.5, 2.0, 1.5, 2.0) == 0.0


def test_gaussian_kl_1d_known_value():
    # KL(N(0,1) || N(0,2)) = log 2 + (1 + 0) / (2 * 4) - 1/2.
    expected = np.log(2.0) + 1.0 / 8.0 - 0.5
    np.testing.assert_allclose(gaussian_kl_1d(0.0, 1.0, 0.0, 2.0), expected, rtol=1e-12)


def test_gaussian_kl_1d_nonnegative_and_asymmetric():
    forward = gaussian_kl_1d(0.0, 1.0, 1.0, 3.0)
    reverse = gaussian_kl_1d(1.0, 3.0, 0.0, 1.0)
    assert forward > 0 and reverse > 0
    assert not np.isclose(forward, reverse)


def test_multivariate_reduces_to_scalar():
    got = gaussian_kl(np.array([0.0]), np.array([[1.0]]), np.array([1.0]), np.array([[4.0]]))
    expected = gaussian_kl_1d(0.0, 1.0, 1.0, 2.0)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_multivariate_zero_for_identical():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 5))
    cov = a @ a.T + np.eye(5)
    mean = rng.normal(size=5)
    np.testing.assert_allclose(gaussian_kl(mean, cov, mean, cov), 0.0, atol=1e-9)


def test_multivariate_matches_monte_carlo():
    rng = np.random.default_rng(1)
    mean_p, mean_q = np.array([0.0, 1.0]), np.array([0.5, -0.5])
    cov_p = np.array([[1.0, 0.4], [0.4, 1.2]])
    cov_q = np.array([[1.5, -0.3], [-0.3, 0.8]])
    closed = gaussian_kl(mean_p, cov_p, mean_q, cov_q)
    draws = rng.multivariate_normal(mean_p, cov_p, size=400_000)
    mc = float(
        np.mean(
            stats.multivariate_normal.logpdf(draws, mean_p, cov_p)
            - stats.multivariate_normal.logpdf(draws, mean_q, cov_q)
        )
    )
    np.testing.assert_allclose(closed, mc, rtol=0.03)


def test_marginal_kl_sums_to_joint_when_independent():
    rng = np.random.default_rng(2)
    p = rng.normal([0.0, 2.0], [1.0, 0.5], size=(60_000, 2))
    q = rng.normal([0.3, 1.5], [1.2, 0.7], size=(60_000, 2))
    # Diagonal covariance, so the joint divergence equals the marginal sum.
    np.testing.assert_allclose(marginal_kl(p, q).sum(), joint_gaussian_kl(p, q), rtol=0.02)


def test_knn_matches_gaussian_in_low_dimension():
    rng = np.random.default_rng(3)
    mean_p, mean_q = np.array([0.0, 0.0]), np.array([0.7, -0.4])
    cov_p = np.array([[1.0, 0.3], [0.3, 1.0]])
    cov_q = np.array([[1.4, -0.2], [-0.2, 0.9]])
    p = rng.multivariate_normal(mean_p, cov_p, size=8000)
    q = rng.multivariate_normal(mean_q, cov_q, size=8000)
    closed = gaussian_kl(mean_p, cov_p, mean_q, cov_q)
    np.testing.assert_allclose(knn_kl(p, q, k=5), closed, atol=0.05)


def test_noise_floor_shrinks_with_sample_size():
    rng = np.random.default_rng(4)
    samples = rng.multivariate_normal(np.zeros(4), np.eye(4), size=8000)
    small = gaussian_kl_noise_floor(samples, n_p=500, n_q=500, reps=10)
    large = gaussian_kl_noise_floor(samples, n_p=2000, n_q=2000, reps=10)
    assert small["mean"] > large["mean"] > 0.0


def test_to_unconstrained_transforms():
    np.testing.assert_allclose(to_unconstrained(np.array([2.0]), "real"), [2.0])
    np.testing.assert_allclose(to_unconstrained(np.array([np.e]), "positive"), [1.0])
    half = to_unconstrained(np.array([0.5]), "unit_interval")
    np.testing.assert_allclose(half, [0.0], atol=1e-12)
    # Midpoint of (-1, 1) maps to 0; symmetric points map to opposite signs.
    out = to_unconstrained(np.array([0.0, 0.5, -0.5]), ("interval", -1.0, 1.0))
    np.testing.assert_allclose(out[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(out[1], -out[2], rtol=1e-12)


def test_stack_unconstrained_orders_and_names():
    draws = {
        "phi1": np.full(3, 0.5),
        "phi2": np.zeros(3),
        "beta": np.ones((3, 2)),
    }
    supports = {"phi1": "unit_interval", "phi2": ("interval", -1.0, 1.0), "beta": "real"}
    matrix, names = stack_unconstrained(draws, supports)
    assert matrix.shape == (3, 4)
    assert names == ["phi1", "phi2", "beta[0]", "beta[1]"]
    np.testing.assert_allclose(matrix[:, :2], 0.0, atol=1e-12)
