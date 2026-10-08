"""Rare-event importance sampling and exact Neyman–Pearson/Stein experiments."""

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import binom, norm

from .information import LOG2, kl


def binomial_log_probabilities(size, probability):
    if size < 1 or not 0 < probability < 1:
        raise ValueError("Interior Bernoulli parameter and positive blocklength required")
    k = np.arange(size + 1)
    return (
        gammaln(size + 1)
        - gammaln(k + 1)
        - gammaln(size - k + 1)
        + k * np.log(probability)
        + (size - k) * np.log1p(-probability)
    )


def rare_event(size, probability, threshold, replicates, seed):
    if not 0 < probability < threshold < 1 or replicates < 2:
        raise ValueError("Need an upper-tail event and at least two trials")
    boundary = int(np.ceil(size * threshold))
    log_exact = float(logsumexp(binomial_log_probabilities(size, probability)[boundary:]))
    rng = np.random.default_rng(seed)
    naive = rng.binomial(size, probability, size=replicates)
    proposal = rng.binomial(size, threshold, size=replicates)
    accepted = proposal >= boundary
    log_weight = proposal * np.log(probability / threshold) + (size - proposal) * np.log(
        (1 - probability) / (1 - threshold)
    )
    # Rescaling retains logarithmic precision even when the probability underflows.
    maximum = float(log_weight[accepted].max()) if accepted.any() else 0.0
    weights = np.zeros(replicates)
    weights[accepted] = np.exp(log_weight[accepted] - maximum)
    mean = weights.mean()
    log_estimate = float(maximum + np.log(mean)) if mean > 0 else None
    relative_se = float(weights.std(ddof=1) / np.sqrt(replicates) / mean) if mean > 0 else None
    q = boundary / size
    exponent = kl([q, 1 - q], [probability, 1 - probability])
    return {
        "naive_counts": naive,
        "proposal_counts": proposal,
        "log_weights": log_weight,
        "naive_hits": int((naive >= boundary).sum()),
        "trials": replicates,
        "log_exact_probability": log_exact,
        "log_is_probability": log_estimate,
        "relative_standard_error": relative_se,
        "observed_exponent_bits": -log_exact / size / LOG2,
        "sanov_exponent_bits": exponent,
        "type_lower_log_probability": -size * exponent * LOG2 - np.log(size + 1),
        "type_union_upper_log_probability": min(0.0, -size * exponent * LOG2 + np.log(size + 1)),
        "scope": "IID binomial draws under a tilted Bernoulli proposal; likelihood weights restore the original law. Confidence uncertainty is Monte Carlo uncertainty, not a proof of a tail bound.",
    }


def neyman_pearson(size, null_probability, alternative_probability, alpha=0.05):
    if not 0 < null_probability < alternative_probability < 1 or not 0 < alpha < 1:
        raise ValueError("Ordered interior hypotheses and false-alarm budget required")
    log0 = binomial_log_probabilities(size, null_probability)
    log1 = binomial_log_probabilities(size, alternative_probability)
    threshold = int(binom.ppf(1 - alpha, size, null_probability))
    upper = float(np.exp(logsumexp(log0[threshold + 1 :])))
    randomization = float(np.clip((alpha - upper) / np.exp(log0[threshold]), 0, 1))
    terms = list(log1[:threshold])
    if randomization < 1:
        terms.append(float(log1[threshold] + np.log1p(-randomization)))
    log_miss = float(logsumexp(terms))
    attained = upper + randomization * np.exp(log0[threshold])
    return {
        "threshold": threshold,
        "boundary_randomization": randomization,
        "false_alarm": float(attained),
        "log_miss_probability": log_miss,
        "miss_exponent_bits": -log_miss / size / LOG2,
        "stein_limit_bits": kl(
            [null_probability, 1 - null_probability],
            [alternative_probability, 1 - alternative_probability],
        ),
    }


def wilson(errors, trials, confidence=0.95):
    """Binomial interval: use independent codewords for block-error uncertainty."""
    if trials < 1 or not 0 <= errors <= trials or not 0 < confidence < 1:
        raise ValueError("Invalid binomial observation")
    z = norm.ppf((1 + confidence) / 2)
    p = errors / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / denominator
    return [max(0.0, float(center - radius)), min(1.0, float(center + radius))]
