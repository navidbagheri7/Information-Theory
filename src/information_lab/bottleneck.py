"""Discrete information bottleneck: explicit local optimization and multiple starts."""

import numpy as np
from scipy.special import logsumexp, xlogy

from .information import mutual_information, pmf


def quantities(joint, conditional):
    p = pmf(joint)
    q = np.asarray(conditional, dtype=float)
    if p.ndim != 2 or q.ndim != 2 or q.shape[0] != p.shape[0]:
        raise ValueError("Input-label PMF and input-representation kernel must align")
    if np.any(q < 0) or not np.isfinite(q).all() or not np.allclose(q.sum(1), 1):
        raise ValueError("Representation kernel must be row-stochastic")
    xz = p.sum(1)[:, None] * q
    yz = p.T @ q
    return mutual_information(xz), mutual_information(yz)


def solve(joint, clusters, beta, seed, tol=1e-8, max_iter=10000):
    p = pmf(joint)
    if p.ndim != 2 or np.any(p.sum(1) <= 0) or np.any(p.sum(0) <= 0):
        raise ValueError("Remove unsupported input and label symbols first")
    if clusters < 1 or beta < 0 or tol <= 0 or max_iter < 1 or not np.isfinite([beta, tol]).all():
        raise ValueError("Invalid bottleneck parameters")
    px = p.sum(1)
    pyx = p / px[:, None]
    q = np.random.default_rng(seed).dirichlet(np.ones(clusters), size=len(p))
    trace = []
    for _iteration in range(1, max_iter + 1):
        pz = px @ q
        pyz = (p.T @ q) / pz[None, :]
        divergence = xlogy(pyx, pyx).sum(1)[:, None] - pyx @ np.log(pyz)
        log_q = np.log(pz)[None, :] - beta * divergence
        new_q = np.exp(log_q - logsumexp(log_q, axis=1)[:, None])
        new_q = np.maximum(new_q, np.finfo(float).tiny)
        new_q /= new_q.sum(1)[:, None]
        ixz, iyz = quantities(p, new_q)
        change = float(np.max(np.abs(new_q - q)))
        trace.append([ixz, iyz, ixz - beta * iyz, change])
        q = new_q
        if change <= tol:
            break
    return {
        "conditional": q,
        "i_input_bits": ixz,
        "i_label_bits": iyz,
        "objective_bits": ixz - beta * iyz,
        "fixed_point_change": change,
        "converged": change <= tol,
        "iterations": _iteration,
        "seed": seed,
        "trace": np.asarray(trace),
        "scope": "Self-consistent IB updates find a local stationary solution. A small residual is not a global-optimum certificate. Compare independently initialized restarts.",
    }


def multistart(joint, clusters, beta, seeds, **kwargs):
    results = [solve(joint, clusters, beta, seed, **kwargs) for seed in seeds]
    if not results:
        raise ValueError("At least one initialization required")
    best = min(range(len(results)), key=lambda i: results[i]["objective_bits"])
    return {"selected": results[best], "restarts": results, "selected_restart": best}
