"""Discrete rate–distortion optimization with a feasible dual certificate."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .information import LOG2, binary_entropy, mutual_information, pmf


def solve(source, distortion, beta, tol=1e-9, max_iter=100000, accelerate=False):
    """Minimize I_nats(X;Y) + beta * E[d(X,Y)].

    Infinite costs forbid transitions, including at beta=0. Source entries must
    be strictly positive: callers should remove zero-probability input symbols.
    Scale a_x=1/Z_x by max_y sum_x p_x a_x exp(-beta*d_xy) to get a feasible dual.
    Subtract beta*D to certify R(D) at the returned achieved distortion.
    """
    p = pmf(source)
    d = np.asarray(distortion, dtype=float)
    if p.ndim != 1 or np.any(p <= 0) or d.ndim != 2 or d.shape[0] != len(p):
        raise ValueError("Positive source PMF and aligned distortion matrix required")
    if np.isnan(d).any() or np.any(d < 0) or not np.isfinite(d).any(axis=1).all():
        raise ValueError("Every source symbol needs a finite nonnegative reproduction cost")
    if not np.isfinite(beta) or beta < 0 or tol <= 0 or max_iter < 1:
        raise ValueError("Invalid multiplier or convergence parameters")
    log_kernel = np.full(d.shape, -np.inf)
    finite = np.isfinite(d)
    log_kernel[finite] = -beta * d[finite]
    r = np.full(d.shape[1], 1 / d.shape[1])
    trace = []
    acceleration_used = False
    for _iteration in range(1, max_iter + 1):
        log_r = np.log(np.maximum(r, np.finfo(float).tiny))
        log_z = logsumexp(log_kernel + log_r[None, :], axis=1)
        q = np.exp(log_kernel + log_r[None, :] - log_z[:, None])
        new_r = p @ q
        joint = p[:, None] * q
        information = mutual_information(joint)
        achieved = float(np.sum(joint[finite] * d[finite]))
        # log s_y, including inactive reproduction symbols, enforces dual feasibility.
        log_s = logsumexp(np.log(p)[:, None] + log_kernel - log_z[:, None], axis=0)
        dual_lagrangian = float(-p @ log_z - log_s.max())
        lower = (dual_lagrangian - beta * achieved) / LOG2
        gap = information - lower
        trace.append([information, achieved, lower, gap])
        if gap <= tol:
            break
        r = new_r
        if accelerate and _iteration == 2000:
            # -sum p log(Kr) is convex in the reproduction marginal. This
            # constrained warm start helps ill-conditioned finite grids; the
            # independently feasible dual gap still decides convergence.
            kernel = np.exp(log_kernel)

            def objective(candidate, kernel=kernel):
                return -p @ np.log(kernel @ candidate)

            def gradient(candidate, kernel=kernel):
                return -(p / (kernel @ candidate)) @ kernel

            result = minimize(
                objective,
                r,
                jac=gradient,
                method="SLSQP",
                bounds=[(1e-15, 1)] * len(r),
                constraints=[
                    {
                        "type": "eq",
                        "fun": lambda candidate: candidate.sum() - 1,
                        "jac": lambda candidate: np.ones(len(candidate)),
                    }
                ],
                options={"ftol": 1e-14, "maxiter": 5000},
            )
            candidate = np.maximum(result.x, 1e-15)
            candidate /= candidate.sum()
            if objective(candidate) <= objective(r) + 1e-13:
                r = candidate
                acceleration_used = True
    return {
        "conditional": q,
        "reproduction": new_r,
        "distortion": achieved,
        "upper_rate_bits": information,
        "lower_rate_bits": lower,
        "gap_bits": max(0.0, gap),
        "beta_nats": float(beta),
        "iterations": _iteration,
        "converged": gap <= tol,
        "trace": np.asarray(trace),
        "acceleration_used": acceleration_used,
        "algorithm": "BA with optional convex marginal warm start"
        if accelerate
        else "Blahut-Arimoto",
    }


def bernoulli_hamming(probability, distortion):
    p = float(probability)
    if not 0 <= p <= 1 or not np.isfinite(distortion) or distortion < 0:
        raise ValueError("Invalid source probability or distortion")
    return max(0.0, binary_entropy(p) - binary_entropy(min(distortion, p, 1 - p)))


def asymmetric_forbidden(distortion):
    """Fair binary source, costs [[0,inf],[1,0]], under an upper distortion constraint."""
    if not np.isfinite(distortion) or distortion < 0:
        raise ValueError("Nonnegative finite distortion required")
    d = min(distortion, 0.5)
    return binary_entropy(0.5 - d) - 0.5 * binary_entropy(2 * d)


def gaussian_rate(variance, distortion):
    if variance <= 0 or distortion < 0 or not np.isfinite([variance, distortion]).all():
        raise ValueError("Invalid Gaussian variance or distortion")
    return float("inf") if distortion == 0 else max(0.0, 0.5 * np.log2(variance / distortion))
