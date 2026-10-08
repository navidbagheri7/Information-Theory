"""Stationary sources, sufficient-statistic typicality and exact finite examples."""

import itertools

import numpy as np

from .channels import channel
from .information import conditional_mutual_information, entropy, pmf


def stationary(matrix):
    w = channel(matrix)
    if w.shape[0] != w.shape[1]:
        raise ValueError("A Markov transition matrix must be square")
    equations = np.vstack([w.T - np.eye(len(w)), np.ones(len(w))])
    if np.linalg.matrix_rank(equations) != len(w):
        raise ValueError("Stationary law is not unique")
    rhs = np.r_[np.zeros(len(w)), 1.0]
    p = np.linalg.lstsq(equations, rhs, rcond=None)[0]
    if p.min() < -1e-10 or not np.allclose(p @ w, p, rtol=0, atol=1e-10):
        raise ValueError("Could not obtain a valid stationary law")
    p = np.maximum(p, 0)
    return p / p.sum()


def entropy_rate(matrix):
    w = channel(matrix)
    p = stationary(w)
    return float(sum(weight * entropy(row) for weight, row in zip(p, w, strict=True)))


def markov_draws(matrix, size, seed):
    w = channel(matrix)
    if size < 1:
        raise ValueError("Nonempty source required")
    rng = np.random.default_rng(seed)
    x = np.empty(size, dtype=np.int64)
    x[0] = rng.choice(len(w), p=stationary(w))
    uniforms = rng.random(size - 1)
    cumulative = w.cumsum(1)
    for i, u in enumerate(uniforms, start=1):
        x[i] = np.searchsorted(cumulative[x[i - 1]], u, side="right")
    return x


def finite_markov_law(matrix, size):
    w = channel(matrix)
    if size < 1 or len(w) ** size > 1000000:
        raise ValueError("Exact enumeration limited to one million sequences")
    sequences = np.array(list(itertools.product(range(len(w)), repeat=size)), dtype=np.int64)
    probability = stationary(w)[sequences[:, 0]]
    for i in range(1, size):
        probability *= w[sequences[:, i - 1], sequences[:, i]]
    return sequences, probability


def parity_law(size):
    if not 2 <= size <= 18:
        raise ValueError("Use 2–18 binary variables for exact parity enumeration")
    values = np.array(list(itertools.product([0, 1], repeat=size)))
    probabilities = (values.sum(1) % 2 == 0).astype(float) / 2 ** (size - 1)
    return values, probabilities


def project_law(values, probability, dimensions):
    """Aggregate labelled variables into a dense finite joint PMF."""
    probability = pmf(probability)
    values = np.asarray(values, dtype=np.int64)
    if values.shape != (len(probability), len(dimensions)):
        raise ValueError("Labels and declared dimensions disagree")
    index = np.ravel_multi_index(values.T, dimensions)
    return np.bincount(index, weights=probability, minlength=np.prod(dimensions)).reshape(
        dimensions
    )


def transformation_examples():
    w = np.array([[1 / 3, 2 / 3], [1 / 2, 1 / 2]])
    x3, p3 = finite_markov_law(w, 3)
    ordinary = project_law(x3[:, [0, 2, 1]], p3, [2, 2, 2])
    summed = np.stack([x3[:, 0], x3[:, 1] + x3[:, 2], x3[:, 0] + x3[:, 1]], axis=1)
    sum_law = project_law(summed, p3, [2, 3, 3])
    x4, p4 = finite_markov_law(w, 4)
    swapped = project_law(x4[:, [1, 3, 0]], p4, [2, 2, 2])
    return {
        "transition": w,
        "stationary": stationary(w),
        "source_rate": entropy_rate(w),
        "subsampled_transition": w @ w,
        "subsampled_rate": entropy_rate(w @ w),
        "ordinary_conditional_mi": conditional_mutual_information(ordinary),
        "swapped_conditional_mi": conditional_mutual_information(swapped),
        "summed_conditional_mi": conditional_mutual_information(sum_law),
        "swapped_rate": entropy_rate(w),
        "summed_rate_with_initial_anchor": entropy_rate(w),
        "scope": "Nonzero I(past;next|current) witnesses failure of first-order Markovity. Adjacent permutation and initially anchored sums are invertible finite-block transforms, preserving entropy rate.",
    }


def typicality_counts(source, blocklength, replicates, seed, weak_tolerance, strong_tolerance):
    p = pmf(source)
    if p.ndim != 1 or np.any(p <= 0) or min(blocklength, replicates) < 1:
        raise ValueError("Positive finite source and nonempty experiment required")
    counts = np.random.default_rng(seed).multinomial(blocklength, p, size=replicates)
    frequencies = counts / blocklength
    surprisal = -frequencies @ np.log2(p)
    weak = np.abs(surprisal - entropy(p)) <= weak_tolerance
    strong = np.max(np.abs(frequencies - p), axis=1) <= strong_tolerance / len(p)
    return {
        "counts": counts,
        "surprisal_bits": surprisal,
        "weak": weak,
        "strong": strong,
        "scope": "Multinomial symbol counts are the exact sufficient statistics of independent length-n IID source sequences. Strong tolerance follows the exercise's absolute epsilon/alphabet convention.",
    }
