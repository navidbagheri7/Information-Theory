"""Configured, resumable scientific experiments with portable numerical evidence."""

import hashlib
import json
import time
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.stats import norm

from . import (
    bottleneck,
    channels,
    coding,
    estimators,
    information,
    lossless,
    quantization,
    rate_distortion,
    sources,
    statistics,
)
from .artifacts import plain, provenance, sha256, write_json


def foundations(settings):
    task = settings["task"]
    if task == "joint_entropy":
        p = np.array([[0.25, 0.25], [0, 0.5]])
        return {
            "h_x": information.entropy(p.sum(1)),
            "h_y": information.entropy(p.sum(0)),
            "h_xy": information.entropy(p),
            "h_x_given_y": information.conditional_entropy(p),
            "mi_bits": information.mutual_information(p),
        }, {"joint": p}
    if task == "parity":
        n = settings["variables"]
        x, p = sources.parity_law(n)
        pair = sources.project_law(x[:, :2], p, [2, 2])
        label = x[:, :-1] @ (2 ** np.arange(n - 1))
        full = sources.project_law(np.c_[label, x[:, -1]], p, [2 ** (n - 1), 2])
        conditional = None
        triple = np.empty((0,))
        if n >= 3:
            triple = sources.project_law(x[:, :3], p, [2, 2, 2])
            conditional = information.conditional_mutual_information(triple)
        return {
            "pair_mi": information.mutual_information(pair),
            "conditional_mi": conditional,
            "joint_to_last_mi": information.mutual_information(full),
            "scope": "Pairwise independence can hide a perfect parity constraint. Conditioned MI can increase or decrease; it is not monotone under conditioning.",
        }, {"states": x, "probability": p, "pair_joint": pair, "triple_joint": triple}
    if task == "mixing":
        p = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
        q = (1 - settings["fraction"]) * p + settings["fraction"] / len(p)
        return {
            "original_entropy": information.entropy(p),
            "mixed_entropy": information.entropy(q),
            "scope": "Mixing with the uniform distribution increases entropy; this is a finite-alphabet majorization example.",
        }, {"original": p, "mixed": q}
    if task == "entropy_extrema":
        m, first, last = settings["alphabet"], settings["first"], settings["last"]
        remaining = 1 - first - last
        maximum = np.r_[first, np.full(m - 2, remaining / (m - 2)), last]
        minimum = np.r_[first, remaining, np.zeros(m - 3), last]
        return {
            "minimum_entropy": information.entropy(minimum),
            "maximum_entropy": information.entropy(maximum),
        }, {"minimizer": minimum, "maximizer": maximum}
    if task == "data_processing":
        first, second = settings["first_crossover"], settings["second_crossover"]
        w1, w2 = channels.bsc(first), channels.bsc(second)
        p = np.ones(2) / 2
        joint = p[:, None, None] * w1[:, :, None] * w2[None, :, :]
        xy, xz = joint.sum(2), joint.sum(1)
        return {
            "i_x_y": information.mutual_information(xy),
            "i_x_z": information.mutual_information(xz),
            "h_x_given_z": information.conditional_entropy(xz),
        }, {"joint": joint, "xy": xy, "xz": xz}
    if task == "fano":
        m, error = settings["alphabet"], settings["error"]
        w = np.full((m, m), error / (m - 1))
        np.fill_diagonal(w, 1 - error)
        joint = w / m
        return {
            "conditional_entropy": information.conditional_entropy(joint),
            "fano_upper_bits": information.fano_bound(error, m),
            "mi_bits": information.mutual_information(joint),
        }, {"joint": joint}
    if task == "weighted_huffman":
        p = np.array([1 / 3, 1 / 4, 1 / 6, 1 / 12, 1 / 16, 1 / 16, 1 / 24])
        cost = np.array(settings["costs"])
        weights = p * cost
        codes = lossless.huffman(weights)
        lengths = np.array([len(codes[i]) for i in range(len(p))])
        relaxed = -np.log2(weights / weights.sum())
        return {
            "source_entropy": information.entropy(p),
            "expected_length": float(p @ lengths),
            "expected_cost": float(weights @ lengths),
            "relaxed_lower_cost": float(weights @ relaxed),
            "additive_cost_bound": float(weights.sum()),
            "codes": codes,
        }, {"probability": p, "costs": cost, "lengths": lengths, "relaxed_lengths": relaxed}
    if task == "strictness_counterexample":
        w1, w2 = np.array([[0.2, 0.8], [0.2, 0.8]]), np.array([[0.7, 0.3], [0.7, 0.3]])
        values = [channels.channel_information([0.4, 0.6], w) for w in [w1, w2, (w1 + w2) / 2]]
        return {
            "first_mi": values[0],
            "second_mi": values[1],
            "mixture_mi": values[2],
            "strictness_slack": (values[0] + values[1]) / 2 - values[2],
            "scope": "Distinct input-independent channels all have I=0 and C=0. Convexity holds with equality, disproving the exercise's unqualified strict-convexity assertion. An input-independent channel similarly disproves strict concavity in input PMFs.",
        }, {"first_channel": w1, "second_channel": w2, "mixture_channel": (w1 + w2) / 2}
    raise ValueError("Unknown foundation experiment")


def channel_experiment(settings):
    task = settings["task"]
    if task == "finite_length":
        result = channels.bsc_normal_approximation(
            settings["noise"], settings["blocklength"], settings["block_error"]
        )
        result["scope"] = (
            "Second-order asymptotic approximation, not an exact finite-length guarantee. Negative rates are preserved to expose breakdown at small n."
        )
        return result, {"channel": channels.bsc(settings["noise"])}
    if task == "water_filling":
        w = np.array(settings["noise_variances"])
        answer = channels.water_filling(w, settings["total_power"])
        return {k: v for k, v in answer.items() if k != "power"}, {
            "noise": w,
            "power": answer["power"],
        }
    if task == "bsc":
        w = channels.bsc(settings["noise"])
    elif task == "bec":
        w = channels.bec(settings["noise"])
    elif task == "z_channel":
        w = channels.z_channel(settings["noise"])
    elif task == "course_channel1":
        w = np.array([[1, 0, 0], [0.5, 0.5, 0], [0, 0, 1]])
    elif task == "course_channel2":
        w = np.array([[1, 0, 0], [0.5, 0.25, 0.25], [0, 0.5, 0.5]])
    elif task == "weakly_symmetric":
        w = np.array([[0.4, 0.1, 0.2, 0.3], [0.1, 0.4, 0.3, 0.2]])
    elif task == "jammer":
        w = channels.jammer(
            settings["free_probability"], settings["noise"], settings["decoder_knows"]
        )
    else:
        raise ValueError("Unknown channel")
    answer = channels.capacity(w, tol=settings.get("tolerance", 1e-10))
    return {k: v for k, v in answer.items() if k not in ["trace", "input", "output"]}, {
        "channel": w,
        "input": answer["input"],
        "output": answer["output"],
        "trace": answer["trace"],
    }


def distortion_experiment(settings):
    task = settings["task"]
    if task == "bernoulli_hamming":
        p, cost = np.array([1 - settings["p_one"], settings["p_one"]]), np.array([[0, 1], [1, 0]])
        x, y = np.arange(2), np.arange(2)
    elif task == "asymmetric_forbidden":
        p, cost = np.array([0.5, 0.5]), np.array([[0, np.inf], [1, 0]])
        x, y = np.arange(2), np.arange(2)
    elif task == "four_pam":
        x, y = np.array([-3, -1, 1, 3]), np.arange(-3, 4)
        p, cost = np.ones(4) / 4, (x[:, None] - y[None, :]) ** 2
    elif task == "discretized_gaussian":
        x = y = np.linspace(-3, 3, 17)
        edges = np.r_[-np.inf, (x[:-1] + x[1:]) / 2, np.inf]
        p, cost = np.diff(norm.cdf(edges)), (x[:, None] - y[None, :]) ** 2
    else:
        raise ValueError("Unknown rate–distortion source")
    answer = rate_distortion.solve(
        p,
        cost,
        settings["beta"],
        tol=settings.get("tolerance", 1e-8),
        accelerate=settings.get("accelerate", False),
    )
    metric = {k: v for k, v in answer.items() if k not in ["conditional", "reproduction", "trace"]}
    if task == "bernoulli_hamming":
        metric["analytic_rate_bits"] = rate_distortion.bernoulli_hamming(
            settings["p_one"], answer["distortion"]
        )
    elif task == "asymmetric_forbidden":
        metric["analytic_rate_bits"] = rate_distortion.asymmetric_forbidden(answer["distortion"])
    else:
        metric["analytic_rate_bits"] = None
    return metric, {
        "source": p,
        "source_values": x,
        "reproduction_values": y,
        "distortion_matrix": cost,
        "conditional": answer["conditional"],
        "reproduction": answer["reproduction"],
        "trace": answer["trace"],
    }


def coding_experiment(settings, seed):
    method, k, trials = settings["method"], settings["message_bits"], settings["trials"]
    rng = np.random.default_rng(seed)
    messages = rng.integers(0, 2, (trials, k), dtype=np.uint8)
    h, g = coding.ldpc_matrices(k, settings.get("graph_seed", 991))
    if method == "uncoded":

        def encode(x):
            return x.copy()

        def decode(llr):
            return ((llr < 0).astype(np.uint8), {})
    elif method == "repetition3":

        def encode(x):
            return np.repeat(x, 3, axis=1)

        def decode(llr):
            return ((llr.reshape(len(llr), k, 3).sum(2) < 0).astype(np.uint8), {})
    elif method in ["hamming_hard", "hamming_soft"]:
        encode = coding.hamming_encode

        def decode(llr):
            return (coding.hamming_decode(llr, soft=method == "hamming_soft"), {})
    elif method == "convolutional":
        encode = coding.convolutional_encode

        def decode(llr):
            return (coding.viterbi_decode(llr), {})
    elif method == "polar_sc":

        def encode(x):
            return coding.polar_encode(x, 2 * k, settings.get("design_erasure", 0.5))

        def decode(llr):
            return (coding.polar_decode(llr, k, settings.get("design_erasure", 0.5)), {})
    elif method in ["ldpc_bp", "ldpc_min_sum"]:

        def encode(x):
            return (x @ g) % 2

        algorithm = "sum_product" if method == "ldpc_bp" else "normalized_min_sum"

        def decode(llr):
            return coding.ldpc_decode(llr, h, method=algorithm, max_iter=settings["max_iterations"])
    else:
        raise ValueError("Unknown code family")
    n = encode(messages[:1]).shape[1]
    rate = k / n
    ebn0 = 10 ** (settings["ebn0_db"] / 10)
    variance = 1 / (2 * rate * ebn0)
    outputs, iteration_counts, syndrome = [], [], []
    seconds = [0.0, 0.0]
    examples = {}
    for start in range(0, trials, settings.get("batch_size", 256)):
        original = messages[start : start + settings.get("batch_size", 256)]
        began = time.perf_counter()
        word = encode(original)
        seconds[0] += time.perf_counter() - began
        received = 1 - 2 * word.astype(float) + rng.normal(0, np.sqrt(variance), word.shape)
        llr = 2 * received / variance
        began = time.perf_counter()
        restored, diagnostics = decode(llr)
        seconds[1] += time.perf_counter() - began
        outputs.append(restored)
        if diagnostics:
            # LDPC returns full codewords; its first k coordinates are systematic.
            outputs[-1] = restored[:, :k]
            iteration_counts.extend(diagnostics["iterations"].tolist())
            syndrome.extend(diagnostics["syndrome_satisfied"].tolist())
        if start == 0:
            examples = {
                "example_encoded": word[:32],
                "example_received": received[:32],
                "example_llr": llr[:32],
            }
    restored = np.concatenate(outputs)
    error_counts = np.sum(restored != messages, axis=1)
    errors = int((error_counts > 0).sum())
    metrics = {
        "bit_error_rate": float(error_counts.sum() / (trials * k)),
        "block_error_rate": errors / trials,
        "block_error_interval": statistics.wilson(errors, trials),
        "block_errors": errors,
        "bit_errors": int(error_counts.sum()),
        "trials": trials,
        "message_bits": k,
        "coded_bits": n,
        "actual_rate": rate,
        "noise_variance": variance,
        "encode_seconds": seconds[0],
        "decode_seconds": seconds[1],
        "gaussian_capacity_bits_per_real_use": float(0.5 * np.log2(1 + 1 / variance)),
        "bpsk_information_bits_per_real_use": coding.bpsk_information(variance),
        "mean_decoder_iterations": float(np.mean(iteration_counts)) if iteration_counts else None,
        "syndrome_satisfaction_fraction": float(np.mean(syndrome)) if syndrome else None,
        "scope": "Independent uniform 64-bit messages; all termination bits count in actual rate. BPSK energy/message-bit normalization gives noise variance=1/(2*actual_rate*Eb/N0). Wilson intervals use independent messages, not correlated decoded bits. Short owned LDPC and BEC-designed Polar SC are demonstrations, not standard 5G profiles or MAP guarantees.",
    }
    return metrics, {
        "message": messages,
        "decoded": restored,
        "bit_error_counts": error_counts,
        "decoder_iterations": np.array(iteration_counts),
        "syndrome_satisfied": np.array(syndrome),
        **examples,
        "parity_check": h if method.startswith("ldpc") else np.empty((0, 0)),
    }


def lossless_experiment(settings, seed):
    kind, size = settings["task"], settings["max_bytes"]
    rng = np.random.default_rng(seed)
    if kind == "iid_biased":
        raw = rng.choice(np.array([0, 1], dtype=np.uint8), size=size, p=[0.9, 0.1]).tobytes()
        theoretical = information.binary_entropy(0.1)
    elif kind == "markov_binary":
        matrix = np.array([[0.97, 0.03], [0.03, 0.97]])
        raw = sources.markov_draws(matrix, size, seed).astype(np.uint8).tobytes()
        theoretical = sources.entropy_rate(matrix)
    elif kind == "uniform_bytes":
        raw = rng.integers(0, 256, size, dtype=np.uint8).tobytes()
        theoretical = 8.0
    else:
        raw = (files("information_lab") / "default_configs/corpora" / (kind + ".txt")).read_bytes()[
            :size
        ]
        theoretical = None
    began = time.perf_counter()
    encoded = lossless.compress(raw, settings["method"], settings.get("window", 4096))
    encode_time = time.perf_counter() - began
    began = time.perf_counter()
    restored = lossless.decompress(encoded, settings["method"])
    decode_time = time.perf_counter() - began
    if restored != raw:
        raise RuntimeError("Lossless round-trip failed")
    values = np.frombuffer(raw, dtype=np.uint8)
    count = np.bincount(values, minlength=256)
    p = count / len(raw)
    method = settings["method"]
    header_bytes = None
    if method in lossless.METHODS:
        header_bytes = lossless.HEADER.size
        if method in ["huffman", "shannon"]:
            header_bytes += 1024
        elif method == "lz77":
            header_bytes += 2
    metric = {
        "source_bytes": len(raw),
        "encoded_bytes": len(encoded),
        "bits_per_byte": len(encoded) * 8 / len(raw),
        "owned_header_bytes": header_bytes,
        "padded_payload_bits_per_byte": (len(encoded) - header_bytes) * 8 / len(raw)
        if header_bytes is not None
        else None,
        "byte_entropy_bits": information.entropy(p),
        "known_entropy_rate_bits": theoretical,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
        "round_trip_exact": True,
        "encode_seconds": encode_time,
        "decode_seconds": decode_time,
        "scope": "Complete encoded file sizes, including model tables, lengths, padding and checksum. Huffman/Shannon are two-pass block codes with transmitted frequencies; arithmetic learns online with shared pseudocounts; LZ77 uses its documented gamma/offset format. Real-corpus entropy is empirical byte entropy, not a claim about its entropy rate.",
    }
    return (
        metric,
        {"original": values, "decoded": np.frombuffer(restored, dtype=np.uint8), "counts": count},
        encoded,
    )


def evaluate(settings, seed, output, data_root, device):
    family = settings["family"]
    if family == "foundations":
        return (*foundations(settings), None)
    if family == "channels":
        return (*channel_experiment(settings), None)
    if family == "rate_distortion":
        return (*distortion_experiment(settings), None)
    if family == "coding":
        return (*coding_experiment(settings, seed), None)
    if family == "lossless":
        return lossless_experiment(settings, seed)
    if family == "quantization":
        return quantization.experiment(settings, seed)
    if family == "source_channel":
        return (*quantization.gaussian_channel(settings, seed), None)
    if family == "estimation":
        x, y = estimators.gaussian_pairs(
            settings["samples"], settings["dimension"], settings["correlation"], seed + 300000
        )
        method = settings["method"]
        began = time.perf_counter()
        if method == "ksg":
            estimate = estimators.ksg(x, y, settings["neighbors"])
        elif method == "gaussian_plugin":
            estimate = estimators.gaussian_plugin(x, y)
        elif method == "binned_plugin":
            estimate = estimators.binned_plugin(x, y, np.linspace(-3, 3, settings["bin_edges"]))
        else:
            raise ValueError("Unknown estimator")
        return (
            {
                "estimate_bits": estimate,
                "true_bits": estimators.gaussian_truth(
                    settings["dimension"], settings["correlation"]
                ),
                "seconds": time.perf_counter() - began,
                "scope": "Same independent Gaussian sample pairs per seed across estimators. Empirical discretization has sparse-cell bias; Gaussian plug-in assumes Gaussianity; KSG is continuous and dimension/sample-size sensitive. Finite estimates may be negative or exceed truth and are not clipped.",
            },
            {"x": x, "y": y},
            None,
        )
    if family == "statistics":
        if settings["task"] == "rare_events":
            result = statistics.rare_event(
                settings["blocklength"],
                settings["probability"],
                settings["threshold"],
                settings["trials"],
                seed,
            )
            arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
            return {k: v for k, v in result.items() if k not in arrays}, arrays, None
        result = statistics.neyman_pearson(
            settings["blocklength"], settings["null"], settings["alternative"], settings["alpha"]
        )
        return (
            result,
            {
                "null_logpmf": statistics.binomial_log_probabilities(
                    settings["blocklength"], settings["null"]
                ),
                "alternative_logpmf": statistics.binomial_log_probabilities(
                    settings["blocklength"], settings["alternative"]
                ),
            },
            None,
        )
    if family == "sources":
        if settings["task"] == "markov_transformations":
            result = sources.transformation_examples()
            arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
            return {k: v for k, v in result.items() if k not in arrays}, arrays, None
        result = sources.typicality_counts(
            settings["probability"],
            settings["blocklength"],
            settings["trials"],
            seed,
            settings["weak_tolerance"],
            settings["strong_tolerance"],
        )
        return (
            {
                "weak_fraction": float(result["weak"].mean()),
                "strong_fraction": float(result["strong"].mean()),
                "true_entropy_bits": information.entropy(settings["probability"]),
                "scope": result["scope"],
            },
            {k: v for k, v in result.items() if isinstance(v, np.ndarray)},
            None,
        )
    if family == "bottleneck":
        pyx = np.array([0.05, 0.15, 0.2, 0.35, 0.65, 0.8, 0.85, 0.95])
        joint = np.c_[1 - pyx, pyx] / len(pyx)
        fit = bottleneck.multistart(
            joint,
            settings["clusters"],
            settings["beta"],
            [seed + j * 100 for j in range(settings["restarts"])],
        )
        selected = fit["selected"]
        metric = {k: v for k, v in selected.items() if k not in ["conditional", "trace"]}
        metric.update(
            selected_restart=fit["selected_restart"],
            source_label_mi=information.mutual_information(joint),
        )
        arrays = {
            "joint": joint,
            "conditional": selected["conditional"],
            "trace": selected["trace"],
            "restart_objectives": np.array([r["objective_bits"] for r in fit["restarts"]]),
            "restart_changes": np.array([r["fixed_point_change"] for r in fit["restarts"]]),
        }
        return metric, arrays, None
    if family == "neural":
        from .neural import train

        return (*train(settings, seed, output / "model.npz", device), None)
    if family == "images":
        from .images import experiment

        return (*experiment(settings, data_root, output, device), None)
    raise ValueError("Unknown experiment family")


def run_case(settings, seed, destination, data_root, device="cpu"):
    output = Path(destination) / settings["family"] / f"{settings['id']}_s{seed}"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run.json"
    source = provenance()
    if path.exists():
        existing = json.loads(path.read_text())
        if (
            existing["config"] != settings
            or existing["source_sha256"] != source["source_sha256"]
            or existing.get("resource_sha256") != source["resource_sha256"]
        ):
            raise RuntimeError(
                "Existing evidence is stale; choose a separate destination: " + str(path)
            )
        for name, digest in existing["artifacts"].items():
            if sha256(output / name) != digest:
                raise RuntimeError("Existing numerical artifact changed: " + name)
        return existing, True
    metric, arrays, encoded = evaluate(settings, seed, output, data_root, device)
    temporary = output / "data.npz.tmp"
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(output / "data.npz")
    if encoded is not None:
        (output / "stream.bin").write_bytes(encoded)
    artifacts = {
        p.name: sha256(p)
        for p in sorted(output.iterdir())
        if p.is_file() and p.name not in ["run.json", "run.json.tmp"]
    }
    record = plain(
        {
            "schema": 1,
            "config": settings,
            "seed": seed,
            "metrics": metric,
            "artifacts": artifacts,
            **source,
        }
    )
    write_json(path, record)
    return record, False
