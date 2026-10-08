"""Cross-method figures with matching source laws, rates and uncertainty units."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from information_lab.artifacts import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = sorted((ROOT / "results/reference").glob("*/*/run.json"))
    records = [json.loads(p.read_text()) for p in paths]
    mappings = []
    out = ROOT / "assets/previews"
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name, selected, scope):
        path = out / (name + ".png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        mappings.append(
            {
                "file": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "scope": scope,
                "records": [
                    {
                        "key": r["config"]["family"]
                        + "/"
                        + r["config"]["id"]
                        + "_s"
                        + str(r["seed"]),
                        "run_sha256": sha256(
                            ROOT
                            / "results/reference"
                            / r["config"]["family"]
                            / (r["config"]["id"] + "_s" + str(r["seed"]))
                            / "run.json"
                        ),
                    }
                    for r in selected
                ],
            }
        )

    rs = [r for r in records if r["config"]["family"] == "coding"]
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for method in sorted({r["config"]["method"] for r in rs}):
        xs, ys, lo, hi = [], [], [], []
        for ebn0 in sorted({r["config"]["ebn0_db"] for r in rs}):
            batch = [
                r for r in rs if r["config"]["method"] == method and r["config"]["ebn0_db"] == ebn0
            ]
            n = sum(r["metrics"]["trials"] for r in batch)
            errors = sum(r["metrics"]["block_errors"] for r in batch)
            p = errors / n
            z = norm.isf(0.025)
            denom = 1 + z * z / n
            center = (p + z * z / (2 * n)) / denom
            radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
            xs.append(ebn0)
            ys.append(p if p > 0 else np.nan)
            lo.append(max(center - radius, 1e-12))
            hi.append(center + radius)
        line = ax.plot(xs, ys, "o-", label=method)[0]
        ax.fill_between(xs, lo, hi, color=line.get_color(), alpha=0.10)
        zero = np.isnan(ys)
        if zero.any():
            ax.scatter(
                np.array(xs)[zero], np.array(hi)[zero], marker="v", color=line.get_color(), s=45
            )
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.1)
    ax.set_xlabel("Eb/N0 (dB), common energy per message bit")
    ax.set_ylabel("Message error probability")
    ax.set_title(
        "Short-code comparison · 6,144 independent 64-bit messages per point\nWilson 95% intervals; downward markers denote zero-error upper bounds"
    )
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=9)
    save(
        fig,
        "coding_comparison",
        rs,
        "Pool independent messages across seeds; 95% Wilson word-error intervals. Zero observed word errors use the nonzero confidence upper bound, not a fabricated positive error count.",
    )
    rs = [
        r
        for r in records
        if r["config"]["family"] in ["estimation", "neural"]
        and r["config"]["correlation"] == 0.95
        and (r["config"]["family"] == "neural" or r["config"]["samples"] == 4096)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    methods = ["gaussian_plugin", "binned_plugin", "ksg", "mine_dv", "infonce"]
    for dimension, ax in zip([1, 4, 16], axes, strict=True):
        means, stds = [], []
        for method in methods:
            values = [
                r["metrics"]["estimate_bits"]
                for r in rs
                if r["config"]["dimension"] == dimension and r["config"]["method"] == method
            ]
            means.append(np.mean(values))
            stds.append(np.std(values, ddof=1))
        ax.bar(np.arange(5), means, yerr=stds, capsize=4)
        ax.set_xticks(
            np.arange(5),
            ["Gaussian", "Histogram", "KSG", "MINE-style", "InfoNCE"],
            rotation=35,
            ha="right",
        )
        truth = -dimension / 2 * np.log2(1 - 0.95**2)
        ax.axhline(truth, color="black", ls="--", label=f"Known MI {truth:.2f} bits")
        ax.axhline(8, color="red", ls=":", label="InfoNCE ceiling")
        ax.set_title(f"{dimension} dimensions, rho=0.95")
        ax.set_ylabel("Estimated bits")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Same 4,096 held-out Gaussian pairs · means ± one standard deviation over 3 seeds")
    save(
        fig,
        "mi_comparison",
        rs,
        "Same Gaussian sample laws and held-out draws; means and seed standard deviations, not confidence bounds. Gaussian plug-in assumes the correct parametric family; contrastive ceiling and dimensional bias remain visible.",
    )
    rs = [
        r
        for r in records
        if r["config"]["family"] == "lossless" and r["config"]["max_bytes"] == 65536
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    tasks = ["iid_biased", "markov_binary", "owned_code", "ecoli_excerpt"]
    methods = ["huffman", "shannon", "arithmetic0", "arithmetic1", "lz77", "deflate", "zstd"]
    for ax, subset in zip(axes, [tasks[:2], tasks[2:]], strict=True):
        for j, task in enumerate(subset):
            values = [
                np.mean(
                    [
                        r["metrics"]["bits_per_byte"]
                        for r in rs
                        if r["config"]["task"] == task and r["config"]["method"] == method
                    ]
                )
                for method in methods
            ]
            ax.bar(
                np.arange(7) + (j - 0.5) * 0.36, values, width=0.36, label=task.replace("_", " ")
            )
        ax.axhline(8, color="black", ls="--", label="Original byte storage")
        ax.set_xticks(range(7), methods, rotation=35, ha="right")
        ax.set_ylabel("Complete stream bits/source byte")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Real lossless streams · headers, models, padding and checksums count")
    save(
        fig,
        "lossless_comparison",
        rs,
        "Same maximum input lengths, actual source lengths saved per record. Each binary symbol occupies one raw byte. Rates are complete streams; dependent Markov structure can beat its marginal entropy. Real corpora have one deterministic input, not artificial replicated seeds.",
    )
    rs = [r for r in records if r["config"]["family"] == "images"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for image, ax in zip(["astronaut", "coffee", "chelsea", "rocket"], axes.flat, strict=True):
        for method in sorted({r["config"]["method"] for r in rs}):
            batch = sorted(
                [
                    r
                    for r in rs
                    if r["config"]["image"] == image and r["config"]["method"] == method
                ],
                key=lambda r: r["metrics"]["container_bpp"],
            )
            ax.plot(
                [r["metrics"]["container_bpp"] for r in batch],
                [r["metrics"]["psnr_db"] for r in batch],
                "o-",
                label=method,
                ms=4,
            )
        ax.set_xlabel("Actual complete container bits/pixel")
        ax.set_ylabel("PSNR on decoded RGB8 (dB)")
        ax.set_title(image.title())
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "Actual image streams · four credited resized photographs, no benchmark/SOTA claim"
    )
    save(
        fig,
        "image_comparison",
        rs,
        "Same input photograph per panel, actual measured bytes and decoded RGB8 pixels. Lines connect three recorded points within a codec and do not certify intermediate rate points. External shared weights are not amortized per image.",
    )
    rs = [r for r in records if r["config"]["family"] == "source_channel"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for bits, ax in zip([4, 8], axes, strict=True):
        xs = sorted({r["config"]["snr_db"] for r in rs})
        for metric in ["digital_mse", "digital_map_mse", "digital_mmse_mse", "analog_mse"]:
            means = [
                np.mean(
                    [
                        r["metrics"][metric]
                        for r in rs
                        if r["config"]["quantizer_bits"] == bits and r["config"]["snr_db"] == snr
                    ]
                )
                for snr in xs
            ]
            ax.plot(xs, means, "o-", label=metric.replace("_", " "))
        theory = [
            next(
                r["metrics"]["analog_theoretical_mse"]
                for r in rs
                if r["config"]["quantizer_bits"] == bits and r["config"]["snr_db"] == snr
            )
            for snr in xs
        ]
        ax.plot(xs, theory, ls="--", color="black", label="Analog theory")
        ax.set_yscale("log")
        ax.set_xlabel("SNR (dB), same power/channel use")
        ax.set_ylabel("Held-out mean squared error")
        ax.set_title(f"{bits} quantization bits · {bits * 7 // 4} uses/source")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.suptitle("Finite digital and analog receivers · Gaussian mean 1, variance 3 · three seeds")
    save(
        fig,
        "source_channel_comparison",
        rs,
        "Same expected power/use and channel uses/source within each panel. Digital ML, prior MAP and posterior mean are distinct receivers. Means over three independent seeds; the mean-centered analog theoretical MSE is a separate exact calculation.",
    )
    rs = [r for r in records if r["config"]["family"] == "bottleneck"]
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for clusters in [2, 3, 4]:
        selected = [r for r in rs if r["config"]["clusters"] == clusters]
        ax.scatter(
            [r["metrics"]["i_input_bits"] for r in selected],
            [r["metrics"]["i_label_bits"] for r in selected],
            label=f"{clusters} representation states",
            s=25,
            alpha=0.7,
        )
    ax.set_xlabel("I(X;Z), retained input information (bits)")
    ax.set_ylabel("I(Y;Z), retained relevant information (bits)")
    ax.set_title("Discrete information bottleneck · local solutions across beta and three seeds")
    ax.legend()
    ax.grid(alpha=0.2)
    save(
        fig,
        "bottleneck_comparison",
        rs,
        "Selected best objective from five local restarts per beta and seed. This is a measured relevance/compression tradeoff, not a certified global frontier.",
    )
    write_json(
        ROOT / "docs/comparison_figures.json",
        {"schema": 1, "builder_sha256": sha256(__file__), "figures": mappings},
    )
    print("Generated", len(mappings), "source-bound cross-method comparison figures")


if __name__ == "__main__":
    main()
