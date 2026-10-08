"""Audit all configured evidence, with independent metric and decoding oracles."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import digamma, logsumexp
from scipy.stats import binom, norm

from information_lab.artifacts import provenance, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def close(actual, expected, atol=1e-9):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=1e-9)


def entropy(p):
    p = np.asarray(p)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def mi(joint):
    p = np.asarray(joint)
    denominator = p.sum(1)[:, None] * p.sum(0)[None, :]
    supported = p > 0
    return float(np.sum(p[supported] * np.log2(p[supported] / denominator[supported])))


def ksg_oracle(x, y, neighbors):
    """Brute-force distances, independent of the package's tree implementation."""
    terms = []
    for start in range(0, len(x), 64):
        dx = cdist(x[start : start + 64], x, metric="chebyshev")
        dy = cdist(y[start : start + 64], y, metric="chebyshev")
        joint = np.maximum(dx, dy)
        joint[np.arange(len(joint)), np.arange(start, start + len(joint))] = np.inf
        radius = np.nextafter(np.partition(joint, neighbors - 1, axis=1)[:, neighbors - 1], 0)
        # Package comparisons use <= nextafter(kth distance, 0).
        nx = (dx <= radius[:, None]).sum(1)
        ny = (dy <= radius[:, None]).sum(1)
        terms.extend(digamma(nx) + digamma(ny))
    return float((digamma(neighbors) + digamma(len(x)) - np.mean(terms)) / np.log(2))


def audit(path, case, data_root, sources_hash):
    record = json.loads(path.read_text())
    assert record["config"] == case
    assert record["source_sha256"] == sources_hash
    assert record["resource_sha256"] == provenance()["resource_sha256"]
    for name, digest in record["artifacts"].items():
        assert Path(name).name == name and sha256(path.parent / name) == digest
    with np.load(path.parent / "data.npz", allow_pickle=False) as archive:
        a = {k: archive[k] for k in archive.files}
    m, family = record["metrics"], case["family"]
    if family == "foundations":
        task = case["task"]
        if task == "joint_entropy":
            p = a["joint"]
            close(m["h_xy"], entropy(p))
            close(m["h_x"], entropy(p.sum(1)))
            close(m["h_x_given_y"], entropy(p) - entropy(p.sum(0)))
            close(m["mi_bits"], mi(p))
        elif task == "parity":
            states, p = a["states"], a["probability"]
            assert np.all(states[p > 0].sum(1) % 2 == 0)
            close(p.sum(), 1)
            close(m["pair_mi"], mi(a["pair_joint"]))
            close(m["joint_to_last_mi"], 1)
            if case["variables"] >= 3:
                close(m["conditional_mi"], 1 if case["variables"] == 3 else 0)
        elif task == "mixing":
            close(m["original_entropy"], entropy(a["original"]))
            close(m["mixed_entropy"], entropy(a["mixed"]))
            assert m["mixed_entropy"] + 1e-12 >= m["original_entropy"]
        elif task == "entropy_extrema":
            close(m["minimum_entropy"], entropy(a["minimizer"]))
            close(m["maximum_entropy"], entropy(a["maximizer"]))
        elif task == "data_processing":
            close(m["i_x_y"], mi(a["xy"]))
            close(m["i_x_z"], mi(a["xz"]))
            assert m["i_x_y"] + 1e-12 >= m["i_x_z"]
        elif task == "fano":
            close(m["conditional_entropy"], entropy(a["joint"]) - entropy(a["joint"].sum(0)))
            close(m["conditional_entropy"], m["fano_upper_bits"])
        elif task == "weighted_huffman":
            close(m["expected_length"], a["probability"] @ a["lengths"])
            close(m["expected_cost"], (a["probability"] * a["costs"]) @ a["lengths"])
            assert m["relaxed_lower_cost"] <= m["expected_cost"] + 1e-12
        else:
            close([m["first_mi"], m["second_mi"], m["mixture_mi"]], 0)
    elif family == "channels":
        if case["task"] == "finite_length":
            p = case["noise"]
            capacity = 1 - entropy([p, 1 - p])
            v = p * (1 - p) * math.log2((1 - p) / p) ** 2
            close(
                m["second_order_rate"],
                capacity - math.sqrt(v / case["blocklength"]) * norm.isf(case["block_error"]),
            )
        elif case["task"] == "water_filling":
            close(a["power"].sum(), case["total_power"])
            assert (a["power"] >= 0).all()
            close(
                m["capacity_bits_per_parallel_use"],
                0.5 * np.log2(1 + a["power"] / a["noise"]).sum(),
            )
            active = a["power"] > 1e-10
            close((a["noise"] + a["power"])[active], m["water_level"])
            assert ((a["noise"] + a["power"])[~active] >= m["water_level"] - 1e-10).all()
        else:
            w, q = a["channel"], a["input"]
            py = q @ w
            div = []
            for row in w:
                support = row > 0
                div.append(np.sum(row[support] * np.log2(row[support] / py[support])))
            close(m["lower_bits"], q @ div)
            close(m["upper_bits"], max(div))
            assert m["converged"] and m["gap_bits"] <= case.get("tolerance", 1e-10) + 1e-12
            expected = None
            if case["task"] == "bsc":
                expected = 1 - entropy([case["noise"], 1 - case["noise"]])
            if case["task"] == "bec":
                expected = 1 - case["noise"]
            if case["task"] == "course_channel1":
                expected = math.log2(9 / 4)
            if case["task"] == "course_channel2":
                expected = 1
            if case["task"] == "weakly_symmetric":
                expected = 2 - entropy(w[0])
            if case["task"] == "jammer":
                qfree, p = case["free_probability"], case["noise"]
                expected = (
                    qfree * (1 - entropy([p, 1 - p]))
                    if case["decoder_knows"]
                    else 1 - entropy([qfree * p + (1 - qfree) / 2, 1 - qfree * p - (1 - qfree) / 2])
                )
            if expected is not None:
                assert m["lower_bits"] - 1e-9 <= expected <= m["upper_bits"] + 1e-9
    elif family == "rate_distortion":
        p, q, d = a["source"], a["conditional"], a["distortion_matrix"]
        j = p[:, None] * q
        close(q.sum(1), 1)
        assert (q >= 0).all()
        assert np.all(q[~np.isfinite(d)] == 0)
        finite = np.isfinite(d)
        distortion = (j[finite] * d[finite]).sum()
        close(m["distortion"], distortion)
        close(m["upper_rate_bits"], mi(j))
        kernel = np.zeros(d.shape)
        kernel[finite] = np.exp(-case["beta"] * d[finite])
        z = kernel @ a["reproduction"]
        lower = (
            -p @ np.log(z) - np.log(((p / z) @ kernel).max()) - case["beta"] * distortion
        ) / np.log(2)
        assert lower <= m["upper_rate_bits"] + 1e-9
        close(m["lower_rate_bits"], lower, atol=3 * case["tolerance"])
        assert m["converged"] and m["gap_bits"] <= case["tolerance"] + 1e-12
        if m["analytic_rate_bits"] is not None:
            assert (
                m["lower_rate_bits"] - 1e-8
                <= m["analytic_rate_bits"]
                <= m["upper_rate_bits"] + 1e-8
            )
    elif family == "coding":
        error = (a["message"] != a["decoded"]).sum(1)
        np.testing.assert_array_equal(error, a["bit_error_counts"])
        close(m["bit_error_rate"], error.sum() / a["message"].size)
        close(m["block_error_rate"], np.mean(error > 0))
        assert m["block_errors"] == np.count_nonzero(error) and m["bit_errors"] == error.sum()
        close(m["actual_rate"], case["message_bits"] / a["example_encoded"].shape[1])
        variance = 1 / (2 * m["actual_rate"] * 10 ** (case["ebn0_db"] / 10))
        close(m["noise_variance"], variance)
        close(a["example_llr"], 2 * a["example_received"] / variance)
        z = norm.isf(0.025)
        phat = np.mean(error > 0)
        n = len(error)
        denom = 1 + z * z / n
        center = (phat + z * z / (2 * n)) / denom
        radius = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
        close(m["block_error_interval"], [max(0, center - radius), min(1, center + radius)])
        if case["method"].startswith("ldpc"):
            close(m["mean_decoder_iterations"], a["decoder_iterations"].mean())
            close(m["syndrome_satisfaction_fraction"], a["syndrome_satisfied"].mean())
    elif family == "lossless":
        from information_lab.lossless import decompress

        raw = a["original"].tobytes()
        decoded = decompress((path.parent / "stream.bin").read_bytes(), case["method"])
        assert decoded == raw == a["decoded"].tobytes()
        assert sha256(path.parent / "stream.bin") == m["encoded_sha256"]
        close(m["bits_per_byte"], (path.parent / "stream.bin").stat().st_size * 8 / len(raw))
        close(m["byte_entropy_bits"], entropy(a["counts"] / len(raw)))
    elif family == "quantization":
        from information_lab.quantization import decode

        decoded = decode((path.parent / "stream.bin").read_bytes())
        np.testing.assert_array_equal(decoded, a["reconstruction"])
        close(m["distortion"], np.mean((a["test"] - decoded) ** 2))
        close(
            m["container_bits_per_sample"],
            (path.parent / "stream.bin").stat().st_size * 8 / len(decoded),
        )
        assert (np.diff(a["centers"]) > 0).all()
        if len(a["fit_trace"]):
            assert (np.diff(a["fit_trace"][:, 0]) <= 1e-10).all()
    elif family == "source_channel":
        source = a["source"]
        mean = case["mean"]
        var = case["variance"]
        noise = 10 ** (-case["snr_db"] / 10)
        uses = case["quantizer_bits"] * 7 // 4
        for name in ["digital", "digital_map", "digital_mmse", "analog"]:
            close(m[name + "_mse"], np.mean((source - a[name]) ** 2))
        close(a["analog"], mean + np.sqrt(var) / (1 + noise / uses) * a["analog_observation"])
        close(m["analog_theoretical_mse"], var / (1 + uses / noise))
        close(m["ideal_separation_distortion"], var / (1 + 1 / noise) ** uses)
        close(m["analog_observed_power_per_use"], np.mean((source - mean) ** 2) / var)
        centers = a["centers"]
        edges = np.r_[-np.inf, (centers[:-1] + centers[1:]) / 2, np.inf]
        z = (edges - mean) / np.sqrt(var)
        prior = np.diff(norm.cdf(z))
        conditional_mean = mean + np.sqrt(var) * (norm.pdf(z[:-1]) - norm.pdf(z[1:])) / prior
        from information_lab.coding import hamming_encode

        labels = np.arange(len(centers))
        gray = labels ^ (labels >> 1)
        bits = ((gray[:, None] >> np.arange(case["quantizer_bits"])) & 1).astype(np.uint8)
        words = hamming_encode(bits)
        scores = a["received"] @ (1 - 2 * words.astype(float)).T / noise + np.log(prior)[None, :]
        posterior = np.exp(scores - logsumexp(scores, axis=1)[:, None])
        close(a["digital_mmse"], posterior @ conditional_mean)
        close(a["digital_map"], centers[np.argmax(scores, axis=1)])
    elif family == "estimation":
        x, y = a["x"], a["y"]
        close(m["true_bits"], -case["dimension"] / 2 * np.log2(1 - case["correlation"] ** 2))
        if case["method"] == "ksg":
            expected = ksg_oracle(x, y, case["neighbors"])
        elif case["method"] == "gaussian_plugin":
            joint = np.cov(np.c_[x, y], rowvar=False)
            d = x.shape[1]
            expected = (
                np.linalg.slogdet(joint[:d, :d])[1]
                + np.linalg.slogdet(joint[d:, d:])[1]
                - np.linalg.slogdet(joint)[1]
            ) / (2 * np.log(2))
        else:
            edges = np.linspace(-3, 3, case["bin_edges"])
            xl = np.searchsorted(edges, x)
            yl = np.searchsorted(edges, y)
            _, xi = np.unique(xl, axis=0, return_inverse=True)
            _, yi = np.unique(yl, axis=0, return_inverse=True)
            from sklearn.metrics import mutual_info_score

            expected = mutual_info_score(xi, yi) / np.log(2)
        close(m["estimate_bits"], expected, atol=1e-10)
    elif family == "neural":
        import torch

        from information_lab.neural import evaluate, load

        torch.set_num_threads(1)
        model, metadata = load(path.parent / "model.npz")
        assert metadata["selected_step"] == m["selected_step"]
        result = evaluate(
            model,
            torch.tensor(a["test_x"], dtype=torch.float32),
            torch.tensor(a["test_y"], dtype=torch.float32),
            case["batch_size"],
        )
        close(m["estimate_bits"], result["estimate_bits"], atol=3e-5)
        close(m["validation_bits"], a["history"][:, 2].max(), atol=1e-7)
        close(m["true_bits"], -case["dimension"] / 2 * np.log2(1 - case["correlation"] ** 2))
        if case["method"] == "mine_dv":
            close(
                m["estimate_bits"],
                (
                    a["joint_scores"].mean()
                    - logsumexp(a["product_scores"])
                    + np.log(len(a["product_scores"]))
                )
                / np.log(2),
                atol=3e-5,
            )
        else:
            close(m["estimate_bits"], a["batch_estimates_bits"].mean())
            assert m["estimate_bits"] <= np.log2(case["batch_size"]) + 1e-7
    elif family == "statistics":
        n = case["blocklength"]
        if case["task"] == "rare_events":
            k = int(np.ceil(n * case["threshold"]))
            logpmf = binom.logpmf(np.arange(n + 1), n, case["probability"])
            close(m["log_exact_probability"], logsumexp(logpmf[k:]))
            assert m["naive_hits"] == np.count_nonzero(a["naive_counts"] >= k)
            accepted = a["proposal_counts"] >= k
            expected = logsumexp(a["log_weights"][accepted]) - np.log(case["trials"])
            close(m["log_is_probability"], expected)
            assert (
                m["type_lower_log_probability"]
                <= m["log_exact_probability"]
                <= m["type_union_upper_log_probability"]
            )
        else:
            t = m["threshold"]
            random = m["boundary_randomization"]
            close(
                m["false_alarm"],
                binom.sf(t, n, case["null"]) + random * binom.pmf(t, n, case["null"]),
            )
            close(m["false_alarm"], case["alpha"])
            terms = binom.logpmf(np.arange(t + 1), n, case["alternative"])
            terms[-1] += np.log1p(-random)
            close(m["log_miss_probability"], logsumexp(terms))
    elif family == "bottleneck":
        p = a["joint"]
        q = a["conditional"]
        close(q.sum(1), 1)
        ixz = mi(p.sum(1)[:, None] * q)
        iyz = mi(p.T @ q)
        close(m["i_input_bits"], ixz)
        close(m["i_label_bits"], iyz)
        close(m["objective_bits"], ixz - case["beta"] * iyz)
        close(m["objective_bits"], a["restart_objectives"].min())
        assert iyz <= mi(p) + 1e-9
        assert (np.diff(a["trace"][:, 2]) <= 1e-8).all()
    elif family == "sources":
        if case["task"] == "typicality":
            f = a["counts"] / case["blocklength"]
            p = np.array(case["probability"])
            surprisal = -f @ np.log2(p)
            close(a["surprisal_bits"], surprisal)
            close(
                m["weak_fraction"], np.mean(abs(surprisal - entropy(p)) <= case["weak_tolerance"])
            )
            close(
                m["strong_fraction"],
                np.mean(np.max(abs(f - p), axis=1) <= case["strong_tolerance"] / len(p)),
            )
        else:
            close(a["stationary"] @ a["transition"], a["stationary"])
            close(a["subsampled_transition"], a["transition"] @ a["transition"])
            close(
                m["source_rate"],
                sum(
                    v * entropy(row)
                    for v, row in zip(a["stationary"], a["transition"], strict=True)
                ),
            )
            close(m["ordinary_conditional_mi"], 0)
            assert m["swapped_conditional_mi"] > 0 and m["summed_conditional_mi"] > 0
    elif family == "images":
        from PIL import Image

        from information_lab.images import decode, metrics

        original = np.array(
            Image.open(ROOT / "assets/images/originals" / f"{case['image']}.png").convert("RGB")
        )
        np.testing.assert_array_equal(original, a["original"])
        encoded = (path.parent / "image.bin").read_bytes()
        decoded = decode(encoded, case["method"], data_root)
        np.testing.assert_array_equal(decoded, a["decoded"])
        np.testing.assert_array_equal(decoded, np.array(Image.open(path.parent / "decoded.png")))
        close(m["container_bpp"], len(encoded) * 8 / np.prod(original.shape[:2]))
        expected = metrics(original, decoded)
        for name in ["mse_rgb8", "psnr_db", "ssim_rgb", "ms_ssim_rgb"]:
            close(m[name], expected[name])
        close(m["encode_seconds"], np.median(a["timings"][:, 0]))
        close(m["decode_seconds"], np.median(a["timings"][:, 1]))
        assert sha256(path.parent / "image.bin") == m["encoded_sha256"]
    else:
        raise AssertionError("Missing family oracle: " + family)
    return {
        "path": str(path.relative_to(ROOT)),
        "record_sha256": sha256(path),
        "family": family,
        "artifacts": len(record["artifacts"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "results/reference")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "configs/reference.json").read_text())
    source = provenance()["source_sha256"]
    audited = []
    for case in protocol["cases"]:
        for seed in protocol["seeds"] if case.get("stochastic", True) else [protocol["seeds"][0]]:
            path = args.reference / case["family"] / f"{case['id']}_s{seed}" / "run.json"
            audited.append(audit(path, case, args.data_root, source))
            print("Verified", case["family"], case["id"], seed, flush=True)
    discovered = list(args.reference.glob("*/*/run.json"))
    assert len(discovered) == len(audited)
    write_json(
        ROOT / "docs/reference_validation.json",
        {
            "schema": 1,
            "status": "passed",
            "runs": len(audited),
            "source_sha256": source,
            "validator_sha256": sha256(__file__),
            "scope": "Artifact hashes, protocol coverage, independent scalar/matrix oracles, brute-force KSG neighbors, safe CPU critic inference, actual saved image/lossless/quantizer decoding. Statistical outcomes and local optimizer convergence remain experiment results, not guaranteed success.",
            "records": audited,
        },
    )
    print("PASS", len(audited), "reference records", flush=True)


if __name__ == "__main__":
    main()
