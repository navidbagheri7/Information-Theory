"""Sample-based MI: Gaussian structure, sparse discretization and KSG neighbors."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

from .information import LOG2, empirical_mi


def pairs(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 3:
        raise ValueError("Paired nonempty sample matrices required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Samples must be finite")
    return x, y


def gaussian_pairs(size, dimension, correlation, seed):
    if size < 3 or dimension < 1 or not -1 < correlation < 1:
        raise ValueError("Invalid Gaussian pair experiment")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(size, dimension))
    y = correlation * x + np.sqrt(1 - correlation**2) * rng.normal(size=x.shape)
    return x, y


def gaussian_truth(dimension, correlation):
    return float(-dimension / 2 * np.log1p(-(correlation**2)) / LOG2)


def gaussian_plugin(x, y):
    x, y = pairs(x, y)
    covariance = np.cov(np.concatenate([x, y], axis=1), rowvar=False, ddof=1)
    dx = x.shape[1]
    determinants = [
        np.linalg.slogdet(c) for c in [covariance[:dx, :dx], covariance[dx:, dx:], covariance]
    ]
    if any(sign <= 0 for sign, _ in determinants):
        raise ValueError("Covariance must be positive definite; no hidden ridge added")
    return float((determinants[0][1] + determinants[1][1] - determinants[2][1]) / 2 / LOG2)


def binned_plugin(x, y, edges):
    x, y = pairs(x, y)
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("Increasing, fixed discretization edges required")
    # Exterior observations receive exterior bins; none are silently discarded.
    return empirical_mi(np.digitize(x, edges), np.digitize(y, edges))


def ksg(x, y, neighbors=5):
    """KSG estimator 1 with strict Chebyshev radii, returned in bits.

    Continuous samples only: exact duplicate joint samples raise instead of
    inventing noise or silently returning a discrete-data estimate.
    """
    x, y = pairs(x, y)
    if not 1 <= neighbors < len(x):
        raise ValueError("Neighbor count must be between one and sample size minus one")
    joint = np.concatenate([x, y], axis=1)
    distance = cKDTree(joint).query(joint, k=neighbors + 1, p=np.inf)[0][:, -1]
    if np.any(distance <= 0):
        raise ValueError("Joint ties require an explicitly chosen mixed/discrete estimator")
    radius = np.nextafter(distance, 0)
    nx = cKDTree(x).query_ball_point(x, radius, p=np.inf, return_length=True)
    ny = cKDTree(y).query_ball_point(y, radius, p=np.inf, return_length=True)
    estimate = digamma(neighbors) + digamma(len(x)) - np.mean(digamma(nx) + digamma(ny))
    return float(estimate / LOG2)
