"""Independent identities, exact course answers and support/assumption checks."""

import numpy as np
import pytest
from scipy.stats import entropy as scipy_entropy

from information_lab.information import (
    binary_entropy,
    conditional_entropy,
    conditional_mutual_information,
    cross_entropy,
    entropy,
    fano_bound,
    js,
    kl,
    mutual_information,
)
from information_lab.sources import (
    entropy_rate,
    finite_markov_law,
    parity_law,
    project_law,
    stationary,
    transformation_examples,
)


def test_exact_joint_example_and_chain_rules():
    p = np.array([[0.25, 0.25], [0, 0.5]])
    assert entropy(p.sum(1)) == 1
    assert entropy(p) == 1.5
    np.testing.assert_allclose(conditional_entropy(p), 0.6887218755408672, atol=1e-14)
    np.testing.assert_allclose(mutual_information(p), 0.31127812445913283, atol=1e-14)
    np.testing.assert_allclose(entropy(p), entropy(p.sum(0)) + conditional_entropy(p))


def test_support_and_entropy_units():
    assert entropy([1, 0]) == 0
    assert binary_entropy(0.5) == 1
    assert np.isinf(kl([1, 0], [0, 1]))
    assert js([1, 0], [0, 1]) == 1
    p, q = [0.3, 0.7], [0.2, 0.8]
    np.testing.assert_allclose(kl(p, q), scipy_entropy(p, q, base=2), atol=1e-14)
    np.testing.assert_allclose(cross_entropy(p, q), -np.array(p) @ np.log2(q))
    with pytest.raises(ValueError):
        entropy([1, 1])
    with pytest.raises(ValueError):
        entropy([-1, 2])


@pytest.mark.parametrize("n", [2, 3, 4, 10])
def test_parity_reveals_synergy(n):
    x, p = parity_law(n)
    pair = project_law(x[:, :2], p, [2, 2])
    np.testing.assert_allclose(mutual_information(pair), 1 if n == 2 else 0, atol=1e-14)
    if n >= 3:
        joint = project_law(x[:, :3], p, [2, 2, 2])
        np.testing.assert_allclose(
            conditional_mutual_information(joint), 1 if n == 3 else 0, atol=1e-14
        )
    previous_label = x[:, :-1] @ (2 ** np.arange(n - 1))
    joint = project_law(np.c_[previous_label, x[:, -1]], p, [2 ** (n - 1), 2])
    assert mutual_information(joint) == 1


def test_markov_entropy_and_nonmarkov_witnesses():
    w = np.array([[1 / 3, 2 / 3], [1 / 2, 1 / 2]])
    np.testing.assert_allclose(stationary(w), [3 / 7, 4 / 7])
    expected = 3 / 7 * scipy_entropy([1 / 3, 2 / 3], base=2) + 4 / 7
    np.testing.assert_allclose(entropy_rate(w), expected)
    _, p = finite_markov_law(w, 5)
    np.testing.assert_allclose(entropy(p), scipy_entropy([3 / 7, 4 / 7], base=2) + 4 * expected)
    result = transformation_examples()
    assert abs(result["ordinary_conditional_mi"]) < 1e-12
    assert result["swapped_conditional_mi"] > 1e-4
    assert result["summed_conditional_mi"] > 0.2
    assert result["source_rate"] < result["subsampled_rate"]
    with pytest.raises(ValueError):
        stationary(np.eye(2))


def test_fano_is_exact_for_uniform_symmetric_errors():
    for alphabet in [2, 3, 7]:
        for error in [0.01, 0.2, 0.6]:
            w = np.full((alphabet, alphabet), error / (alphabet - 1))
            np.fill_diagonal(w, 1 - error)
            h = scipy_entropy(w[0], base=2)
            np.testing.assert_allclose(fano_bound(error, alphabet), h, atol=1e-13)


def test_weak_typicality_does_not_imply_strong_typicality():
    # All-zero string under a fair source has surprisal H exactly; its type is wrong.
    weak_deviation = abs(-np.log2(0.5) - scipy_entropy([0.5, 0.5], base=2))
    strong_deviation = max(abs(np.array([1, 0]) - 0.5))
    assert weak_deviation == 0 and strong_deviation == 0.5
