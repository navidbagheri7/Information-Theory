"""Estimator math, exact finite statistics, optimization and portable model checks."""

import math

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import binom

from information_lab.bottleneck import multistart
from information_lab.estimators import gaussian_pairs, gaussian_plugin, gaussian_truth, ksg
from information_lab.quantization import decode, encode, gaussian_channel, lloyd_max
from information_lab.statistics import (
    binomial_log_probabilities,
    neyman_pearson,
    rare_event,
    wilson,
)


def test_ksg_matches_brute_chebyshev_neighbor_counts():
    x, y = gaussian_pairs(80, 2, 0.5, 71)
    joint = np.c_[x, y]
    distance = np.max(np.abs(joint[:, None] - joint[None, :]), axis=2)
    radius = np.sort(distance, axis=1)[:, 5]
    dx = np.max(np.abs(x[:, None] - x[None, :]), axis=2)
    dy = np.max(np.abs(y[:, None] - y[None, :]), axis=2)
    nx = (dx < radius[:, None]).sum(1)
    ny = (dy < radius[:, None]).sum(1)

    # For positive integers digamma(m)=sum_{j=1}^{m-1} 1/j - Euler gamma.
    def harmonic(n):
        return sum(1 / j for j in range(1, int(n)))

    expected = (
        harmonic(5)
        + harmonic(80)
        - np.mean([harmonic(a) + harmonic(b) for a, b in zip(nx, ny, strict=True)])
    ) / np.log(2)
    np.testing.assert_allclose(ksg(x, y, 5), expected, atol=1e-13)
    with pytest.raises(ValueError):
        ksg(np.zeros(10), np.zeros(10))


def test_gaussian_estimators_recover_known_information():
    x, y = gaussian_pairs(15000, 2, 0.7, 83)
    assert abs(gaussian_plugin(x, y) - gaussian_truth(2, 0.7)) < 0.04
    assert abs(ksg(x[:4000], y[:4000], 5) - gaussian_truth(2, 0.7)) < 0.12


def test_exact_statistics_and_rare_event_bounds():
    n, p, a = 50, 0.3, 0.7
    exact = sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(35, n + 1))
    np.testing.assert_allclose(
        np.exp(logsumexp(binomial_log_probabilities(n, p)[35:])), exact, rtol=2e-14
    )
    result = rare_event(n, p, a, 30000, 72)
    assert (
        result["type_lower_log_probability"]
        <= np.log(exact)
        <= result["type_union_upper_log_probability"]
    )
    assert (
        abs(np.exp(result["log_is_probability"] - np.log(exact)) - 1)
        < 5 * result["relative_standard_error"]
    )
    assert wilson(0, 100)[1] > 0.03
    assert wilson(100, 100)[0] < 0.97


def test_neyman_pearson_randomizes_to_the_exact_budget():
    result = neyman_pearson(35, 0.3, 0.7, 0.05)
    t, gamma = result["threshold"], result["boundary_randomization"]
    alpha = binom.sf(t, 35, 0.3) + gamma * binom.pmf(t, 35, 0.3)
    beta = binom.cdf(t - 1, 35, 0.7) + (1 - gamma) * binom.pmf(t, 35, 0.7)
    np.testing.assert_allclose(alpha, 0.05, atol=1e-13)
    np.testing.assert_allclose(np.exp(result["log_miss_probability"]), beta, rtol=2e-13)


def test_bottleneck_local_objectives_and_data_processing():
    pyx = np.array([0.05, 0.2, 0.8, 0.95])
    joint = np.stack([1 - pyx, pyx], axis=1) / 4
    result = multistart(joint, 2, 4, [1, 2, 3])
    for run in result["restarts"]:
        assert run["converged"]
        assert run["i_label_bits"] <= run["i_input_bits"] + 1e-12
        assert np.max(np.diff(run["trace"][:, 2])) < 1e-12
    assert result["selected"]["objective_bits"] == min(
        r["objective_bits"] for r in result["restarts"]
    )


def test_lloyd_training_descent_and_receiver_reconstruction():
    rng = np.random.default_rng(3)
    training, test = rng.normal(size=8000), rng.normal(size=4000)
    centers, fit = lloyd_max(training, 16)
    assert np.max(np.diff(fit["trace"][:, 0])) < 1e-12
    reconstructed = decode(encode(test, centers))
    independent = centers[np.argmin((test[:, None] - centers[None, :]) ** 2, axis=1)]
    np.testing.assert_array_equal(reconstructed, independent)


def test_gaussian_channel_power_mean_centering_and_analog_mse():
    settings = {"mean": 1, "variance": 3, "quantizer_bits": 4, "samples": 100000, "snr_db": 0}
    metric, arrays = gaussian_channel(settings, 3)
    assert abs(metric["analog_mse"] / metric["analog_theoretical_mse"] - 1) < 0.02
    assert abs(metric["analog_observed_power_per_use"] - 1) < 0.02
    assert metric["digital_mmse_mse"] < metric["digital_mse"]
    np.testing.assert_allclose(
        metric["digital_mse"], np.mean((arrays["digital"] - arrays["source"]) ** 2)
    )


def test_neural_bound_functions_and_safe_export(tmp_path):
    torch = pytest.importorskip("torch")
    from information_lab.neural import Critic, dv_value, export, infonce_value, load

    positive, negative = np.array([0.2, 0.7, -0.1]), np.array([1, 0.1, -0.5])
    oracle = positive.mean() - np.log(np.exp(negative).mean())
    np.testing.assert_allclose(dv_value(torch.tensor(positive), torch.tensor(negative)), oracle)
    matrix = np.array([[0.2, 0.1], [0.8, 1.2]])
    oracle = np.log(2) + np.mean(np.diag(matrix) - np.log(np.exp(matrix).sum(1)))
    np.testing.assert_allclose(infonce_value(torch.tensor(matrix)), oracle)
    critic = Critic(2, "mine_dv", 8)
    metadata = {"dimension": 2, "kind": "mine_dv", "width": 8}
    path = tmp_path / "model.npz"
    export(path, critic, metadata)
    restored, record = load(path)
    x, y = torch.randn(5, 2), torch.randn(5, 2)
    np.testing.assert_array_equal(
        critic.pair_score(x, y).detach(), restored.pair_score(x, y).detach()
    )
    assert record == metadata
