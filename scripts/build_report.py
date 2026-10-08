"""Render source-bound recorded evidence and field-specific comparison tables."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from information_lab.artifacts import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "foundations": "Entropy and dependence",
    "sources": "Sources and typicality",
    "channels": "Channel capacity",
    "coding": "Noisy communication",
    "lossless": "Lossless compression",
    "rate_distortion": "Rate and distortion",
    "quantization": "Scalar quantization",
    "source_channel": "Source–channel transmission",
    "statistics": "Rare events and hypothesis tests",
    "estimation": "Information estimation",
    "neural": "Learned information estimators",
    "bottleneck": "Information bottleneck",
    "images": "Real image compression",
}
METHOD_LABELS = {
    "ldpc_bp": "LDPC: sum-product belief propagation",
    "ldpc_min_sum": "LDPC: normalized min-sum",
    "hamming_soft": "Hamming (7,4): soft maximum likelihood",
    "hamming_hard": "Hamming (7,4): hard syndrome decoding",
    "convolutional": "Convolutional K7: soft Viterbi",
    "polar_sc": "Polar (128,64): successive cancellation",
    "repetition3": "Repetition (3,1): soft combining",
    "uncoded": "Uncoded BPSK",
    "mine_dv": "MINE-style DV critic",
    "infonce": "InfoNCE embedding critic",
    "bmshj2018-factorized": "Factorized prior (Ballé)",
    "bmshj2018-hyperprior": "Scale hyperprior (Ballé)",
    "mbt2018-mean": "Mean-scale hyperprior (Minnen)",
    "cheng2020-anchor": "Context/residual codec (Cheng)",
    "jpeg": "JPEG (4:4:4)",
    "webp": "WebP",
    "avif": "AVIF (4:4:4)",
    "jpeg2000": "JPEG2000 (irreversible)",
    "arithmetic0": "Adaptive arithmetic: order zero",
    "arithmetic1": "Adaptive arithmetic: order one",
    "huffman": "Huffman",
    "shannon": "Shannon",
    "deflate": "DEFLATE",
    "zstd": "Zstandard",
    "lz77": "Window LZ77",
    "gaussian_plugin": "Gaussian plug-in",
    "binned_plugin": "Sparse histogram plug-in",
    "ksg": "KSG nearest neighbors",
    "lloyd_max": "Lloyd–Max",
    "uniform": "Uniform quantizer",
}


def descriptors(c):
    family = c["family"]
    method = c.get("method", "Exact finite calculation")
    if family == "images":
        return c["image"].title(), method, f"point {c['point']}"
    if family in ["estimation", "neural"]:
        scenario = f"Gaussian dimension {c['dimension']}, correlation {c['correlation']}"
        setting = (
            f"n={c['samples']}"
            if family == "estimation"
            else f"{c['steps']} steps, batch {c['batch_size']}"
        )
        return scenario, method, setting
    if family == "coding":
        return "BPSK over AWGN, 64-bit messages", method, f"Eb/N0 = {c['ebn0_db']} dB"
    if family == "lossless":
        return c["task"].replace("_", " "), method, f"up to {c['max_bytes']} source bytes"
    if family == "quantization":
        return "Gaussian mean 1, variance 3", method, f"{c['levels']} reconstruction levels"
    if family == "source_channel":
        return (
            f"{c['quantizer_bits']} quantization bits, {c['quantizer_bits'] * 7 // 4} channel uses",
            "Digital / analog receivers",
            f"SNR = {c['snr_db']} dB",
        )
    if family == "bottleneck":
        return (
            f"8-symbol relevance task, {c['clusters']} representation states",
            "5-start self-consistent solver",
            f"beta = {c['beta']}",
        )
    if family == "rate_distortion":
        scenario = c["task"].replace("_", " ")
        if "p_one" in c:
            scenario += f", p(1)={c['p_one']}"
        return scenario, "Certified optimization", f"beta = {c['beta']} nats/distortion"
    if family == "sources":
        if c["task"] == "markov_transformations":
            return (
                "Markov transformations",
                "Exact finite enumeration",
                "Original / subsampled / reordered / sums",
            )
        return (
            "IID source " + str(c["probability"]),
            "Exact multinomial draws",
            f"blocklength {c['blocklength']}",
        )
    if family == "statistics":
        return (
            c["task"].replace("_", " "),
            "Tilted importance sampling"
            if c["task"] == "rare_events"
            else "Exact randomized Neyman–Pearson",
            f"blocklength {c['blocklength']}",
        )
    if family == "channels":
        task = c["task"].replace("_", " ")
        if c["task"] == "finite_length":
            return (
                task,
                "Second-order approximation",
                f"n={c['blocklength']}, error={c['block_error']}",
            )
        if c["task"] == "water_filling":
            return task, "Exact Gaussian allocation", f"total power {c['total_power']}"
        if c["task"] == "jammer":
            return (
                task,
                "Decoder state " + ("known" if c["decoder_knows"] else "unknown"),
                f"free probability {c['free_probability']}",
            )
        return (
            task,
            "Certified Blahut–Arimoto",
            f"noise {c['noise']}" if "noise" in c else "Supplied course channel",
        )
    return (
        c["task"].replace("_", " "),
        "Exact finite calculation",
        ", ".join(
            f"{k}={v}" for k, v in c.items() if k not in ["family", "id", "task", "stochastic"]
        )
        or "Exact case",
    )


def visual(record, data, destination):
    c, m = record["config"], record["metrics"]
    f = c["family"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    left, right = axes
    fig.suptitle(f"{LABELS[f]} · {c['id']} · seed {record['seed']}", fontsize=10)

    def heat(ax, a, title, xlabels=None, ylabels=None):
        image = ax.imshow(a, aspect="auto", cmap="Blues", vmin=0)
        fig.colorbar(image, ax=ax, shrink=0.75)
        ax.set_title(title, fontsize=10)
        columns = np.arange(0, a.shape[1], max(1, a.shape[1] // 8))
        rows = np.arange(0, a.shape[0], max(1, a.shape[0] // 8))
        ax.set_xticks(
            columns, labels=[f"{xlabels[i]:g}" for i in columns] if xlabels is not None else columns
        )
        ax.set_yticks(
            rows, labels=[f"{ylabels[i]:g}" for i in rows] if ylabels is not None else rows
        )

    if f == "foundations":
        if "joint" in data and data["joint"].ndim == 2:
            p = data["joint"]
            heat(left, p, "Joint probability P(X,Y)")
            left.set_xlabel("Y")
            left.set_ylabel("X")
            right.bar(
                ["H(X)", "H(Y)", "H(X,Y)", "I(X;Y)"],
                [entropy(p.sum(1)), entropy(p.sum(0)), entropy(p), mutual_information(p)],
            )
            right.set_ylabel("bits")
        elif "states" in data:
            supported = data["states"][data["probability"] > 0]
            heat(left, supported[:64], "First supported parity states (0/1)")
            left.set_xlabel("Variable")
            left.set_ylabel("Supported state")
            right.bar(["Pair MI", "Joint-to-last MI"], [m["pair_mi"], m["joint_to_last_mi"]])
            right.set_ylabel("bits")
            right.set_ylim(0, 1.15)
        elif "original" in data:
            left.bar(
                np.arange(len(data["original"])) - 0.18,
                data["original"],
                width=0.36,
                label="Original",
            )
            left.bar(np.arange(len(data["mixed"])) + 0.18, data["mixed"], width=0.36, label="Mixed")
            left.legend()
            left.set_ylabel("Probability")
            right.bar(["Original", "Mixed"], [m["original_entropy"], m["mixed_entropy"]])
            right.set_ylabel("Entropy (bits)")
        elif "minimizer" in data:
            left.bar(range(len(data["minimizer"])), data["minimizer"])
            left.set_title("Minimum-entropy feasible PMF")
            right.bar(range(len(data["maximizer"])), data["maximizer"])
            right.set_title("Maximum-entropy feasible PMF")
        elif "xy" in data:
            heat(left, data["xy"], "P(X,Y), first channel")
            heat(right, data["xz"], "P(X,Z), degraded channel")
        elif "lengths" in data:
            left.bar(range(len(data["lengths"])), data["lengths"], label="Integer")
            left.plot(data["relaxed_lengths"], "o", label="Relaxed weighted optimum")
            left.legend()
            left.set_ylabel("Codeword length (bits)")
            right.bar(range(len(data["probability"])), data["probability"] * data["costs"])
            right.set_ylabel("Probability × letter cost")
        else:
            heat(left, data["first_channel"], "First input-independent channel")
            heat(right, data["second_channel"], "Different input-independent channel")
    elif f == "channels":
        if "power" in data:
            left.bar(range(len(data["noise"])), data["noise"], label="Noise")
            left.bar(
                range(len(data["power"])),
                data["power"],
                bottom=data["noise"],
                label="Allocated power",
            )
            left.axhline(m["water_level"], ls="--", color="black", label="Water level")
            left.legend()
            left.set_ylabel("Power")
            right.bar(range(len(data["power"])), 0.5 * np.log2(1 + data["power"] / data["noise"]))
            right.set_ylabel("Capacity contribution (bits/real use)")
        else:
            heat(left, data["channel"], "Channel transition probabilities")
            left.set_xlabel("Output")
            left.set_ylabel("Input")
            if "trace" in data:
                trace = data["trace"]
                marker = "o" if len(trace) <= 30 else None
                right.plot(
                    np.arange(len(trace)) + 1, trace[:, 0], marker=marker, label="Lower bound"
                )
                right.plot(
                    np.arange(len(trace)) + 1,
                    trace[:, 1],
                    marker=marker,
                    ls="--",
                    label="Upper bound",
                )
                right.set_xlabel("BA iteration")
                right.set_ylabel("Capacity bounds (bits/use)")
                right.legend()
            else:
                right.bar(
                    ["Capacity", "Second-order rate"], [m["capacity"], m["second_order_rate"]]
                )
                right.set_ylabel("bits/use; approximation may break down")
    elif f == "rate_distortion":
        heat(
            left,
            data["conditional"],
            "Reconstruction kernel P(Y|X)",
            data["reproduction_values"],
            data["source_values"],
        )
        left.set_xlabel("Reconstruction")
        left.set_ylabel("Source symbol")
        trace = data["trace"]
        marker = "o" if len(trace) <= 30 else None
        right.plot(trace[:, 0], marker=marker, label="Primal rate at current D")
        right.plot(trace[:, 2], marker=marker, ls="--", label="Dual bound at current D")
        right.set_title("Bounds at each iteration's attained distortion", fontsize=9)
        right.legend()
        right.set_xlabel("Iteration (warm start declared if used)")
        right.set_ylabel("bits/source symbol")
    elif f == "coding":
        rows = 12
        heat(left, data["message"][:rows], "Original message bits")
        heat(
            right,
            (data["message"] != data["decoded"])[:rows],
            f"Decoded bit errors · BER={m['bit_error_rate']:.4g}",
        )
        for ax in axes:
            ax.set_xlabel("Message bit")
            ax.set_ylabel("Independent message")
    elif f == "lossless":
        left.bar(np.arange(256), data["counts"] / data["counts"].sum())
        left.set_xlim(-1, 256)
        left.set_xlabel("Byte value")
        left.set_ylabel("Empirical probability")
        values = [8, m["bits_per_byte"], m["byte_entropy_bits"]]
        labels = ["Raw bytes", "Complete stream", "Marginal entropy"]
        if m["known_entropy_rate_bits"] is not None:
            values.append(m["known_entropy_rate_bits"])
            labels.append("Known source rate")
        right.bar(labels, values)
        right.tick_params(axis="x", labelrotation=20)
        right.set_ylabel("bits/source byte")
        right.set_title("Marginal entropy differs from entropy rate")
    elif f == "quantization":
        left.hist(data["test"], bins=50, density=True, alpha=0.5, label="Held-out source")
        left.plot(data["centers"], np.zeros(len(data["centers"])), "|", ms=12, label="Centers")
        left.legend()
        left.set_xlabel("Source value")
        n = min(512, len(data["test"]))
        right.scatter(data["test"][:n], data["reconstruction"][:n], s=7, alpha=0.5)
        right.plot([-5, 7], [-5, 7], ls="--", color="black")
        right.set_xlabel("Held-out original")
        right.set_ylabel("Decoded reconstruction")
    elif f == "source_channel":
        for name in ["digital", "digital_mmse", "analog"]:
            left.scatter(
                data["source"][:256],
                data[name][:256],
                s=8,
                alpha=0.45,
                label=name.replace("_", " "),
            )
        left.plot([-5, 7], [-5, 7], ls="--", color="black")
        left.legend(fontsize=8)
        left.set_xlabel("Original source")
        left.set_ylabel("Receiver reconstruction")
        right.bar(
            ["Digital ML", "Digital MAP", "Digital MMSE", "Analog"],
            [m["digital_mse"], m["digital_map_mse"], m["digital_mmse_mse"], m["analog_mse"]],
        )
        right.axhline(m["analog_theoretical_mse"], ls="--", label="Analog theory")
        right.legend()
        right.set_ylabel("Held-out MSE, equal channel uses")
    elif f == "estimation":
        left.scatter(data["x"][:512, 0], data["y"][:512, 0], s=8, alpha=0.4)
        left.set_xlabel("X, first coordinate")
        left.set_ylabel("Y, first coordinate")
        left.set_title("Projection only; MI concerns all coordinates")
        right.bar(["Estimate", "Known MI"], [m["estimate_bits"], m["true_bits"]])
        right.set_ylabel("bits")
        right.set_title("Finite estimate, not a certified bound")
    elif f == "neural":
        history = data["history"]
        left.plot(history[:, 0], history[:, 1], label="Training objective")
        left.plot(history[:, 0], history[:, 2], label="Validation objective")
        left.axvline(m["selected_step"], ls=":", color="black", label="Selected checkpoint")
        left.set_xlabel("Training step")
        left.set_ylabel("Estimated bits")
        left.legend(fontsize=8)
        right.bar(["Held-out estimate", "Known Gaussian MI"], [m["estimate_bits"], m["true_bits"]])
        right.set_ylabel("bits")
        if m["batch_ceiling_bits"] is not None:
            right.axhline(
                m["batch_ceiling_bits"], ls="--", color="red", label="InfoNCE batch ceiling"
            )
            right.legend(fontsize=8)
    elif f == "bottleneck":
        heat(left, data["conditional"], "Representation kernel P(Z|X)")
        left.set_xlabel("Representation Z")
        left.set_ylabel("Input X")
        trace = data["trace"]
        right.plot(trace[:, 0], label="I(X;Z)")
        right.plot(trace[:, 1], label="I(Y;Z)")
        right.set_xlabel("Self-consistent iteration")
        right.set_ylabel("bits")
        right.legend()
        right.set_title("Selected local restart; no global certificate")
    elif f == "sources":
        if "surprisal_bits" in data:
            left.hist(data["surprisal_bits"], bins=30, density=True)
            left.axvline(m["true_entropy_bits"], color="black", ls="--", label="Source entropy")
            left.legend()
            left.set_xlabel("Surprisal (bits/source symbol)")
            right.bar(
                ["Weak typical", "Strong typical"], [m["weak_fraction"], m["strong_fraction"]]
            )
            right.set_ylim(0, 1.05)
            right.set_ylabel("Observed fraction of IID blocks")
        else:
            heat(left, data["transition"], "Original transition P")
            heat(right, data["subsampled_transition"], "Subsampled transition P²")
    elif f == "statistics":
        if c["task"] == "rare_events":
            left.hist(data["naive_counts"], bins=25, alpha=0.6, label="Naive source")
            left.hist(data["proposal_counts"], bins=25, alpha=0.6, label="Tilted proposal")
            left.axvline(
                np.ceil(c["blocklength"] * c["threshold"]),
                color="black",
                ls="--",
                label="Tail threshold",
            )
            left.legend(fontsize=8)
            left.set_xlabel("Number of ones")
            right.bar(
                ["Exact", "Importance sampling"],
                [m["log_exact_probability"] / np.log(10), m["log_is_probability"] / np.log(10)],
            )
            right.set_ylabel("log10 tail probability")
            right.set_title(f"Naive hits: {m['naive_hits']} / {m['trials']}")
        else:
            x = np.arange(len(data["null_logpmf"]))
            left.plot(x, np.exp(data["null_logpmf"]), label="Null")
            left.plot(x, np.exp(data["alternative_logpmf"]), label="Alternative")
            left.axvline(m["threshold"], ls="--", color="black", label="Randomized threshold")
            left.legend(fontsize=8)
            left.set_xlabel("Number of ones")
            right.bar(
                ["Finite exponent", "Stein limit"], [m["miss_exponent_bits"], m["stein_limit_bits"]]
            )
            right.set_ylabel("bits/source symbol")
    else:
        raise ValueError(f)
    for ax in axes:
        ax.grid(axis="y", alpha=0.18)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=100)
    plt.close(fig)


def entropy(p):
    p = np.asarray(p)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def mutual_information(p):
    support = p > 0
    denominator = p.sum(1)[:, None] * p.sum(0)[None, :]
    return np.sum(p[support] * np.log2(p[support] / denominator[support]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render-family",
        action="append",
        choices=list(LABELS),
        help="Re-render selected figure families and reuse existing others; default renders all",
    )
    args = parser.parse_args()
    protocol = json.loads((ROOT / "configs/reference.json").read_text())
    expected = sum(
        len(protocol["seeds"]) if c.get("stochastic", True) else 1 for c in protocol["cases"]
    )
    paths = sorted((ROOT / "results/reference").glob("*/*/run.json"))
    assert len(paths) == expected
    rows = []
    for path in paths:
        record = json.loads(path.read_text())
        c = record["config"]
        scenario, method, setting = descriptors(c)
        key = path.parent.name
        family = c["family"]
        if family == "images":
            asset = path.parent / "decoded.png"
        else:
            asset = ROOT / "assets/experiments" / family / (key + ".png")
            if not args.render_family or family in args.render_family:
                with np.load(path.parent / "data.npz", allow_pickle=False) as archive:
                    visual(record, {k: archive[k] for k in archive.files}, asset)
            if not asset.exists():
                raise FileNotFoundError(
                    "Run the full figure builder before reusing families: " + str(asset)
                )
        row = {
            "key": family + "/" + key,
            "family": family,
            "label": LABELS[family],
            "scenario": scenario,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "setting": setting,
            "seed": record["seed"],
            "config": c,
            "metrics": record["metrics"],
            "record": "../" + str(path.relative_to(ROOT)),
            "record_sha256": sha256(path),
            "asset": "../" + str(asset.relative_to(ROOT)),
            "asset_sha256": sha256(asset),
            "data": "../" + str((path.parent / "data.npz").relative_to(ROOT)),
        }
        if family == "images":
            row["original"] = "../assets/images/originals/" + c["image"] + ".png"
            row["original_sha256"] = sha256(ROOT / "assets/images/originals" / f"{c['image']}.png")
        rows.append(row)
        print("Indexed", row["key"], flush=True)
    write_json(
        ROOT / "docs/results.json",
        {"schema": 1, "protocol_sha256": sha256(ROOT / "configs/reference.json"), "rows": rows},
    )
    metrics = sorted({k for row in rows for k in row["metrics"]})
    with (ROOT / "docs/comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=["family", "scenario", "method", "setting", "seed", *metrics]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{k: row[k] for k in ["family", "scenario", "method", "setting", "seed"]},
                    **{
                        k: json.dumps(v) if isinstance(v, (dict, list)) else v
                        for k, v in row["metrics"].items()
                    },
                }
            )
    families = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
    text = [
        "# Measured reference results",
        "",
        f"The configured protocol contains {len(protocol['cases'])} cases and {len(rows)} measured runs. All tables link to recorded evidence. See [measurement contracts](mathematical_contracts.md) for units, assumptions, uncertainty and limits, and [the explorer](explorer.html) for field-specific comparisons.",
        "",
        "| Field | Recorded runs |",
        "|---|---:|",
    ]
    text += [f"| {LABELS[f]} | {len(rs)} |" for f, rs in families.items()]
    text += [
        "",
        "## Selected findings",
        "",
        "These numerical examples show practical behavior; they are not universal method rankings.",
        "",
    ]
    selected = next(
        r
        for r in rows
        if r["family"] == "source_channel"
        and r["config"]["quantizer_bits"] == 4
        and r["config"]["snr_db"] == 0
        and r["seed"] == 41
    )
    m = selected["metrics"]
    text += [
        f"- At 0 dB and seven channel uses per Gaussian source symbol, analog MSE is {m['analog_mse']:.5f}; digital Hamming soft-ML MSE is {m['digital_mse']:.5f}, and source-prior posterior-mean MSE is {m['digital_mmse_mse']:.5f}. The matched-use power budget makes this a meaningful finite-code comparison."
    ]
    selected = [
        r
        for r in rows
        if r["family"] == "neural"
        and r["config"]["dimension"] == 16
        and r["config"]["correlation"] == 0.95
        and r["config"]["method"] == "infonce"
    ]
    values = [r["metrics"]["estimate_bits"] for r in selected]
    text += [
        f"- For 16 dimensions and correlation 0.95, Gaussian MI is {selected[0]['metrics']['true_bits']:.5f} bits. InfoNCE estimates range from {min(values):.5f} to {max(values):.5f} bits across three seeds, approaching the 8-bit batch ceiling rather than the truth. MINE-style estimates and strong Gaussian/KSG baselines are available alongside this result."
    ]
    selected = [
        r
        for r in rows
        if r["family"] == "statistics"
        and r["config"]["task"] == "rare_events"
        and r["config"]["blocklength"] == 500
    ]
    m = selected[0]["metrics"]
    text += [
        f"- At binomial blocklength 500, naive simulation observes {m['naive_hits']} tail hits in {m['trials']} trials while the exact natural-log probability is {m['log_exact_probability']:.5f}. Importance sampling estimates the same rare probability with reported relative uncertainty.",
        "",
        "## Image codec examples",
        "",
        "Four attributed photographs, complete actual containers, RGB8 decoded-pixel metrics, CPU timing medians. Pretrained model weights are shared external inputs and not amortized per image. This small resized set supports a demonstration, not a benchmark or SOTA claim.",
        "",
        "| Coffee codec | Point | Complete bpp | PSNR dB | MS-SSIM | Decode seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        families["images"], key=lambda r: (r["config"]["method"], r["metrics"]["container_bpp"])
    ):
        if row["config"]["image"] != "coffee":
            continue
        c, m = row["config"], row["metrics"]
        text.append(
            f"| {c['method']} | {c['point']} | {m['container_bpp']:.5f} | {m['psnr_db']:.3f} | {m['ms_ssim_rgb']:.5f} | {m['decode_seconds']:.5f} |"
        )
    text += [
        "",
        "All scalar metrics are in [comparison.csv](comparison.csv), full row/asset associations in [results.json](results.json), and independent validation in [reference_validation.json](reference_validation.json). Exact numeric quality settings differ between codec families; compare rate and decoded quality.",
    ]
    results_text = "\n".join(text)
    for name in [
        "mathematical_contracts.md",
        "explorer.html",
        "comparison.csv",
        "results.json",
        "reference_validation.json",
    ]:
        results_text = results_text.replace("(" + name + ")", "(../docs/" + name + ")")
    (ROOT / "results/README.md").write_text(results_text + "\n")
    (ROOT / "docs/results.md").write_text(
        "\n".join(text).replace("(mathematical_contracts.md)", "(mathematical_contracts.md)") + "\n"
    )
    write_json(
        ROOT / "docs/report_validation.json",
        {
            "schema": 1,
            "status": "generated",
            "runs": len(rows),
            "builder_sha256": sha256(__file__),
            "results_sha256": sha256(ROOT / "docs/results.json"),
            "assets": [
                {
                    "key": r["key"],
                    "path": r["asset"],
                    "sha256": r["asset_sha256"],
                    "record_sha256": r["record_sha256"],
                }
                for r in rows
            ],
        },
    )
    print("COMPLETE", len(rows), "source-bound records")


if __name__ == "__main__":
    main()
