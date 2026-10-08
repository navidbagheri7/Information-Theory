"""Independent closed-form and generic constrained-optimizer oracles."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import rel_entr
from scipy.stats import entropy as scipy_entropy

from information_lab.channels import bec, bsc, capacity, jammer, water_filling, z_channel
from information_lab.rate_distortion import asymmetric_forbidden, bernoulli_hamming, solve


def test_channel_capacity_analytic_answers_and_course_channels():
    for p in [0, 0.1, 0.3, 0.5, 0.9, 1]:
        np.testing.assert_allclose(
            capacity(bsc(p))["lower_bits"], 1 - scipy_entropy([p, 1 - p], base=2), atol=1e-12
        )
        np.testing.assert_allclose(capacity(bec(p))["lower_bits"], 1 - p, atol=1e-12)
    z = capacity(z_channel(0.5))
    np.testing.assert_allclose(z["lower_bits"], np.log2(5 / 4), atol=1e-9)
    np.testing.assert_allclose(z["input"], [0.6, 0.4], atol=1e-8)
    first = np.array([[1, 0, 0], [0.5, 0.5, 0], [0, 0, 1]])
    second = first @ np.array([[1, 0, 0], [0, 0.5, 0.5], [0, 0.5, 0.5]])
    result = capacity(first)
    np.testing.assert_allclose(result["lower_bits"], np.log2(9 / 4), atol=1e-9)
    np.testing.assert_allclose(result["input"], [1 / 3, 2 / 9, 4 / 9], atol=1e-8)
    np.testing.assert_allclose(capacity(second)["lower_bits"], 1, atol=1e-9)


def test_capacity_certificate_against_slsqp():
    w = np.random.default_rng(83).dirichlet(np.ones(4), size=3)
    answer = capacity(w)
    conditional = np.array([scipy_entropy(row, base=2) for row in w])

    def objective(q):
        return -(scipy_entropy(q @ w, base=2) - q @ conditional)

    result = minimize(
        objective,
        np.ones(3) / 3,
        method="SLSQP",
        bounds=[(0, 1)] * 3,
        constraints={"type": "eq", "fun": lambda q: q.sum() - 1},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    assert result.success
    assert answer["converged"] and answer["gap_bits"] <= 1e-10
    assert answer["lower_bits"] - 1e-9 <= -result.fun <= answer["upper_bits"] + 1e-9
    assert np.min(np.diff(answer["trace"][:, 0])) > -1e-12


def test_jammer_side_information():
    q, p = 0.6, 0.1
    expected = q * (1 - scipy_entropy([p, 1 - p], base=2))
    np.testing.assert_allclose(capacity(jammer(q, p, True))["lower_bits"], expected)
    assert capacity(jammer(q, p, False))["upper_bits"] < expected


def test_water_filling_independent_optimizer_and_kkt():
    noise, budget = np.array([0.7, 2.0, 5.0, 11.0]), 8.0
    result = water_filling(noise, budget)
    oracle = minimize(
        lambda p: -0.5 * np.log2(1 + p / noise).sum(),
        np.ones(4) * budget / 4,
        method="SLSQP",
        bounds=[(0, budget)] * 4,
        constraints={"type": "eq", "fun": lambda p: p.sum() - budget},
        options={"ftol": 1e-12},
    )
    assert oracle.success
    np.testing.assert_allclose(result["power"].sum(), budget)
    np.testing.assert_allclose(result["capacity_bits_per_parallel_use"], -oracle.fun, atol=1e-10)
    active = result["power"] > 0
    np.testing.assert_allclose(noise[active] + result["power"][active], result["water_level"])
    assert np.all(noise[~active] >= result["water_level"])


def test_rate_distortion_closed_forms_and_forbidden_support():
    for probability in [0.2, 0.5]:
        for beta in [0.2, 1, 3, 8]:
            result = solve([1 - probability, probability], [[0, 1], [1, 0]], beta)
            assert result["converged"]
            expected = bernoulli_hamming(probability, result["distortion"])
            np.testing.assert_allclose(result["upper_rate_bits"], expected, atol=2e-9)
    for beta in [0, 0.3, 1, 4, 10]:
        result = solve([0.5, 0.5], [[0, np.inf], [1, 0]], beta)
        assert result["conditional"][0, 1] == 0
        assert result["converged"]
        np.testing.assert_allclose(
            result["upper_rate_bits"], asymmetric_forbidden(result["distortion"]), atol=2e-9
        )


def test_rd_primal_dual_certificate_with_generic_slsqp():
    p = np.array([0.2, 0.3, 0.5])
    cost = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]])
    result = solve(p, cost, 2.0)

    def objective(v):
        q = v.reshape(3, 3)
        joint = p[:, None] * q
        return rel_entr(joint, p[:, None] * joint.sum(0)[None, :]).sum() / np.log(2)

    oracle = minimize(
        objective,
        np.full(9, 1 / 3),
        method="SLSQP",
        bounds=[(1e-14, 1)] * 9,
        constraints=[
            {"type": "eq", "fun": lambda v: v.reshape(3, 3).sum(1) - 1},
            {
                "type": "ineq",
                "fun": lambda v: result["distortion"] - np.sum(p[:, None] * v.reshape(3, 3) * cost),
            },
        ],
        options={"ftol": 1e-11, "maxiter": 2000},
    )
    assert oracle.success
    assert result["lower_rate_bits"] - 2e-7 <= oracle.fun <= result["upper_rate_bits"] + 2e-7


def test_discretized_gaussian_convex_warm_start_retains_dual_certificate():
    from scipy.stats import norm

    from information_lab.rate_distortion import solve

    x = np.linspace(-3, 3, 17)
    p = np.diff(norm.cdf(np.r_[-np.inf, (x[:-1] + x[1:]) / 2, np.inf]))
    d = (x[:, None] - x[None, :]) ** 2
    result = solve(p, d, 0.7, tol=1e-6, accelerate=True)
    assert result["converged"] and result["acceleration_used"]
    assert 0 <= result["gap_bits"] <= 1e-6
    np.testing.assert_allclose(result["conditional"].sum(1), 1, atol=1e-12)
    assert np.all(result["conditional"] >= 0)
    # Independent constrained full conditional optimization is already checked
    # above. Here the dual is rebuilt directly in ordinary probabilities.
    r = result["reproduction"]
    kernel = np.exp(-0.7 * d)
    z = kernel @ r
    lower = (
        -p @ np.log(z) - np.log(((p / z) @ kernel).max()) - 0.7 * result["distortion"]
    ) / np.log(2)
    np.testing.assert_allclose(lower, result["lower_rate_bits"], atol=2e-7)
