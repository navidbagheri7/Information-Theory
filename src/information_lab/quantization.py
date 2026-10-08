"""Scalar quantization and equal-channel-use Gaussian source/channel experiments."""

import struct

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from .coding import hamming_decode, hamming_encode
from .information import entropy
from .lossless import compress, decompress, huffman
from .rate_distortion import gaussian_rate


def uniform_centers(levels, mean=0.0, variance=1.0):
    if not 2 <= levels <= 256 or variance <= 0 or not np.isfinite([mean, variance]).all():
        raise ValueError("Use 2–256 centers and finite Gaussian parameters")
    edges = np.linspace(mean - 4 * np.sqrt(variance), mean + 4 * np.sqrt(variance), levels + 1)
    return (edges[:-1] + edges[1:]) / 2


def indices(values, centers):
    centers, values = np.asarray(centers), np.asarray(values)
    if (
        centers.ndim != 1
        or len(centers) < 2
        or not np.all(np.diff(centers) > 0)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Ordered centers and finite observations required")
    return np.searchsorted((centers[:-1] + centers[1:]) / 2, values).astype(np.uint8)


def lloyd_max(training, levels, mean=0.0, variance=1.0, tol=1e-8, max_iter=1000):
    training = np.asarray(training)
    if training.ndim != 1 or not len(training) or not np.isfinite(training).all():
        raise ValueError("Finite nonempty training sample required")
    centers = mean + np.sqrt(variance) * norm.ppf((np.arange(levels) + 0.5) / levels)
    trace = []
    for _iteration in range(1, max_iter + 1):
        selected = indices(training, centers)
        counts = np.bincount(selected, minlength=levels)
        sums = np.bincount(selected, weights=training, minlength=levels)
        new = centers.copy()
        np.divide(sums, counts, out=new, where=counts > 0)
        change = float(np.max(np.abs(new - centers)))
        centers = new
        distortion = float(np.mean((training - centers[indices(training, centers)]) ** 2))
        trace.append([distortion, change])
        if change <= tol:
            break
    return centers, {"iterations": _iteration, "converged": change <= tol, "trace": np.array(trace)}


def encode(values, centers):
    centers = np.asarray(centers, dtype=float)
    if not 2 <= len(centers) <= 256:
        raise ValueError("Portable format supports 2–256 reconstruction centers")
    labels = indices(values, centers)
    return (
        b"ITQ1"
        + struct.pack(">H", len(centers))
        + centers.astype(">f8").tobytes()
        + compress(labels.tobytes(), "huffman")
    )


def decode(encoded):
    if encoded[:4] != b"ITQ1" or len(encoded) < 6:
        raise ValueError("Invalid quantizer container")
    count = struct.unpack_from(">H", encoded, 4)[0]
    if not 2 <= count <= 256 or len(encoded) < 6 + 8 * count:
        raise ValueError("Missing reconstruction centers")
    centers = np.frombuffer(encoded[6 : 6 + 8 * count], dtype=">f8")
    labels = np.frombuffer(decompress(encoded[6 + 8 * count :], "huffman"), dtype=np.uint8)
    if np.any(labels >= count) or not np.isfinite(centers).all():
        raise ValueError("Invalid center or encoded index")
    return centers[labels].astype(float)


def experiment(settings, seed):
    mean, variance, levels = settings["mean"], settings["variance"], settings["levels"]
    rng = np.random.default_rng(seed + 10000)
    training = rng.normal(mean, np.sqrt(variance), settings["train_size"])
    test = np.random.default_rng(seed + 20000).normal(
        mean, np.sqrt(variance), settings["test_size"]
    )
    fitting = None
    if settings["method"] == "uniform":
        centers = uniform_centers(levels, mean, variance)
    elif settings["method"] == "lloyd_max":
        centers, fitting = lloyd_max(training, levels, mean, variance)
    else:
        raise ValueError("Unknown scalar quantizer")
    labels = indices(test, centers)
    restored = centers[labels]
    encoded = encode(test, centers)
    if not np.array_equal(decode(encoded), restored):
        raise RuntimeError("Portable scalar reconstruction differs")
    distortion = float(np.mean((test - restored) ** 2))
    counts = np.bincount(labels, minlength=levels)
    codes = huffman(counts)
    payload = sum(int(counts[i]) * len(code) for i, code in codes.items())
    metrics = {
        "distortion": distortion,
        "entropy_bits": entropy(counts / len(test)),
        "payload_bits_per_sample": payload / len(test),
        "container_bits_per_sample": len(encoded) * 8 / len(test),
        "gaussian_lower_rate_bits": gaussian_rate(variance, distortion),
        "encoded_bytes": len(encoded),
        "fit_iterations": fitting["iterations"] if fitting else None,
        "fit_converged": fitting["converged"] if fitting else None,
        "scope": "Uniform prior-based or training-only Lloyd–Max fixed-level scalar quantizer. Independent test draws; Huffman index frequencies and all float64 reconstruction centers are transmitted. An asymptotic Gaussian R(D) curve is a limit, not the measured rate of this finite scalar codec.",
    }
    arrays = {
        "training": training,
        "test": test,
        "centers": centers,
        "indices": labels,
        "reconstruction": restored,
        "fit_trace": fitting["trace"] if fitting else np.empty((0, 2)),
    }
    return metrics, arrays, encoded


def gaussian_channel(settings, seed):
    """Compare k-use digital and k-repeat analog at the same expected power/use."""
    mean, variance = settings["mean"], settings["variance"]
    resolution = settings["quantizer_bits"]
    if resolution not in [4, 8]:
        raise ValueError("This matched-use experiment supports 4/8 bits with Hamming (7,4)")
    size = settings["samples"]
    snr = 10 ** (settings["snr_db"] / 10)
    noise_variance = 1 / snr
    uses = resolution * 7 // 4
    rng = np.random.default_rng(seed)
    source = rng.normal(mean, np.sqrt(variance), size)
    centers = uniform_centers(2**resolution, mean, variance)
    label = indices(source, centers)
    gray = label ^ (label >> 1)
    message = ((gray[:, None] >> np.arange(resolution)[None, :]) & 1).astype(np.uint8)
    word = hamming_encode(message)
    received = 1 - 2 * word.astype(float) + rng.normal(0, np.sqrt(noise_variance), word.shape)
    restored_bits = hamming_decode(2 * received / noise_variance, soft=True)
    restored_gray = restored_bits @ (2 ** np.arange(resolution))
    restored_label = restored_gray.copy()
    shift = 1
    while shift < resolution:
        restored_label ^= restored_label >> shift
        shift *= 2
    digital = centers[restored_label]
    possible_label = np.arange(2**resolution, dtype=np.uint16)
    possible_gray = possible_label ^ (possible_label >> 1)
    possible_bits = ((possible_gray[:, None] >> np.arange(resolution)[None, :]) & 1).astype(
        np.uint8
    )
    possible_words = hamming_encode(possible_bits)
    boundaries = np.r_[-np.inf, (centers[:-1] + centers[1:]) / 2, np.inf]
    standardized = (boundaries - mean) / np.sqrt(variance)
    prior = np.diff(norm.cdf(standardized))
    conditional_means = (
        mean
        + np.sqrt(variance) * (norm.pdf(standardized[:-1]) - norm.pdf(standardized[1:])) / prior
    )
    posterior_log = (
        received @ (1 - 2 * possible_words.astype(float)).T / noise_variance
        + np.log(prior)[None, :]
    )
    posterior = np.exp(posterior_log - logsumexp(posterior_log, axis=1)[:, None])
    digital_map = centers[np.argmax(posterior_log, axis=1)]
    digital_mmse = posterior @ conditional_means
    # The nonzero mean is removed before transmission, satisfying expected P=1.
    signal = (source - mean) / np.sqrt(variance)
    averaged = signal + rng.normal(0, np.sqrt(noise_variance / uses), size)
    analog = mean + np.sqrt(variance) / (1 + noise_variance / uses) * averaged
    digital_error = (digital - source) ** 2
    analog_error = (analog - source) ** 2
    return {
        "channel_uses_per_source": uses,
        "source_symbols_per_use": 1 / uses,
        "digital_mse": float(digital_error.mean()),
        "analog_mse": float(analog_error.mean()),
        "digital_map_mse": float(np.mean((digital_map - source) ** 2)),
        "digital_mmse_mse": float(np.mean((digital_mmse - source) ** 2)),
        "analog_theoretical_mse": variance / (1 + uses * snr),
        "ideal_separation_distortion": variance / (1 + snr) ** uses,
        "quantization_only_mse": float(np.mean((centers[label] - source) ** 2)),
        "source_clipping_fraction": float(np.mean(np.abs(source - mean) > 4 * np.sqrt(variance))),
        "digital_power_per_use": 1.0,
        "analog_observed_power_per_use": float(np.mean(signal**2)),
        "scope": "Same channel uses/source and expected power/use, AWGN variance=1/SNR. Compare soft-ML Hamming, source-prior MAP, exact source-prior posterior mean, and repeated analog MMSE. Gaussian bin probabilities/conditional means are known-prior quantities, not fitted to test data. Ideal separation is an asymptotic Gaussian-input bound, not an achieved finite-code rate.",
    }, {
        "source": source,
        "received": received,
        "source_index": label,
        "centers": centers,
        "analog_observation": averaged,
        "digital": digital,
        "digital_map": digital_map,
        "digital_mmse": digital_mmse,
        "analog": analog,
        "digital_squared_error": digital_error,
        "analog_squared_error": analog_error,
    }
