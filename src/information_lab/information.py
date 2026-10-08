"""Finite-alphabet information quantities with explicit support semantics."""

import numpy as np
from scipy.special import xlogy

LOG2 = np.log(2.0)


def pmf(values):
    p = np.asarray(values, dtype=float)
    if not p.size or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("A PMF must be finite and nonnegative")
    if not np.isclose(p.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("Probabilities must sum to one; input is not silently normalized")
    return p


def entropy(p):
    p = pmf(p)
    return float(-xlogy(p, p).sum() / LOG2)


def binary_entropy(p):
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Binary probability must lie in [0, 1]")
    return entropy([p, 1 - p])


def kl(p, q):
    p, q = pmf(p), pmf(q)
    if p.shape != q.shape:
        raise ValueError("Aligned probability arrays must have the same shape")
    active = p > 0
    if np.any(q[active] == 0):
        return float("inf")
    return float(np.sum(p[active] * np.log(p[active] / q[active])) / LOG2)


def cross_entropy(p, q):
    return entropy(p) + kl(p, q)


def js(p, q):
    p, q = pmf(p), pmf(q)
    if p.shape != q.shape:
        raise ValueError("Aligned probability arrays must have the same shape")
    mixture = (p + q) / 2
    return (kl(p, mixture) + kl(q, mixture)) / 2


def mutual_information(joint):
    p = pmf(joint)
    if p.ndim != 2:
        raise ValueError("Mutual information takes a two-dimensional joint PMF")
    return kl(p, p.sum(1)[:, None] * p.sum(0)[None, :])


def conditional_entropy(joint):
    """H(X|Y), with joint[x,y]."""
    p = pmf(joint)
    if p.ndim != 2:
        raise ValueError("Conditional entropy takes a two-dimensional joint PMF")
    return entropy(p) - entropy(p.sum(0))


def conditional_mutual_information(joint):
    """I(X;Y|Z), with joint[x,y,z]."""
    p = pmf(joint)
    if p.ndim != 3:
        raise ValueError("Conditional MI requires three axes")
    result = 0.0
    for z, weight in enumerate(p.sum(axis=(0, 1))):
        if weight:
            result += weight * mutual_information(p[:, :, z] / weight)
    return float(result)


def empirical_mi(x, y):
    """Sparse discrete plug-in MI; empty cells never require a dense Cartesian table."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("Paired nonempty samples are required")
    x = x.reshape(len(x), -1)
    y = y.reshape(len(y), -1)
    _, cx = np.unique(x, axis=0, return_counts=True)
    _, cy = np.unique(y, axis=0, return_counts=True)
    _, cxy = np.unique(np.concatenate([x, y], axis=1), axis=0, return_counts=True)
    return entropy(cx / len(x)) + entropy(cy / len(x)) - entropy(cxy / len(x))


def fano_bound(error, alphabet):
    if alphabet < 2 or not 0 <= error <= 1:
        raise ValueError("Invalid error probability or alphabet size")
    return binary_entropy(error) + error * np.log2(alphabet - 1)
