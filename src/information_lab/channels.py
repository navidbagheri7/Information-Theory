"""Certified finite-channel capacity and Gaussian resource allocation."""

import numpy as np
from scipy.special import logsumexp, xlogy
from scipy.stats import norm

from .information import LOG2, binary_entropy, mutual_information, pmf


def channel(matrix):
    w = np.asarray(matrix, dtype=float)
    if w.ndim != 2 or min(w.shape) == 0 or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("A channel is a nonnegative finite input-by-output matrix")
    if not np.allclose(w.sum(1), 1, rtol=0, atol=1e-10):
        raise ValueError("Every channel row must sum to one")
    return w


def bsc(crossover):
    if not 0 <= crossover <= 1:
        raise ValueError("Crossover must lie in [0,1]")
    return np.array([[1 - crossover, crossover], [crossover, 1 - crossover]])


def bec(erasure):
    if not 0 <= erasure <= 1:
        raise ValueError("Erasure probability must lie in [0,1]")
    return np.array([[1 - erasure, 0, erasure], [0, 1 - erasure, erasure]])


def z_channel(crossover):
    if not 0 <= crossover <= 1:
        raise ValueError("Crossover must lie in [0,1]")
    return np.array([[1, 0], [crossover, 1 - crossover]])


def capacity(matrix, tol=1e-10, max_iter=100000):
    """Blahut–Arimoto, with I(q,W) <= C <= max_x KL(W_x||qW).

    A strictly positive initial input distribution covers every possible output.
    The returned tolerance certificate is in bits, not parameter-change units.
    """
    w = channel(matrix)
    if tol <= 0 or max_iter < 1:
        raise ValueError("Positive tolerance and iteration budget required")
    q = np.full(len(w), 1 / len(w))
    row_log = xlogy(w, w).sum(1)
    trace = []
    for _iteration in range(1, max_iter + 1):
        py = q @ w
        divergences = row_log - w @ np.log(np.maximum(py, np.finfo(float).tiny))
        lower, upper = float(q @ divergences / LOG2), float(divergences.max() / LOG2)
        trace.append([lower, upper])
        if upper - lower <= tol:
            break
        log_q = np.log(np.maximum(q, np.finfo(float).tiny)) + divergences
        q = np.exp(log_q - logsumexp(log_q))
    return {
        "input": q,
        "output": q @ w,
        "lower_bits": lower,
        "upper_bits": upper,
        "gap_bits": max(0.0, upper - lower),
        "iterations": _iteration,
        "converged": upper - lower <= tol,
        "trace": np.asarray(trace),
    }


def channel_information(q, matrix):
    q, w = pmf(q), channel(matrix)
    if q.ndim != 1 or len(q) != len(w):
        raise ValueError("Input PMF and channel input dimension disagree")
    return mutual_information(q[:, None] * w)


def jammer(free_probability, crossover, decoder_knows=False):
    q = float(free_probability)
    if not 0 <= q <= 1 or not 0 <= crossover <= 1:
        raise ValueError("Invalid jammer parameters")
    if not decoder_knows:
        return bsc(q * crossover + (1 - q) / 2)
    # Outputs: (free,0), (free,1), (blocked,0), (blocked,1).
    return np.concatenate([q * bsc(crossover), np.full((2, 2), (1 - q) / 2)], axis=1)


def water_filling(noise_variances, total_power):
    noise = np.asarray(noise_variances, dtype=float)
    if noise.ndim != 1 or not len(noise) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Strictly positive finite noise variances required")
    if not np.isfinite(total_power) or total_power < 0:
        raise ValueError("Power must be finite and nonnegative")
    ordered = np.sort(noise)
    for k in range(1, len(noise) + 1):
        level = (total_power + ordered[:k].sum()) / k
        if k == len(noise) or level <= ordered[k]:
            break
    power = np.maximum(level - noise, 0)
    return {
        "power": power,
        "water_level": float(level),
        "capacity_bits_per_parallel_use": float(0.5 * np.log2(1 + power / noise).sum()),
    }


def bsc_normal_approximation(crossover, blocklength, block_error):
    """Asymptotic normal approximation, not an exact finite-blocklength guarantee."""
    if not 0 < crossover < 0.5 or blocklength < 1 or not 0 < block_error < 0.5:
        raise ValueError("Interior BSC/error parameters and positive blocklength required")
    c = 1 - binary_entropy(crossover)
    v = crossover * (1 - crossover) * np.log2((1 - crossover) / crossover) ** 2
    rate = c - np.sqrt(v / blocklength) * norm.isf(block_error)
    return {"capacity": c, "dispersion_bits_squared": v, "second_order_rate": float(rate)}
