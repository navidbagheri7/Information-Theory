"""Generate explicit scientific protocols; no hidden notebook settings."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cases = []


def add(family, name, **settings):
    cases.append({"family": family, "id": name, **settings})


add("foundations", "joint_entropy", task="joint_entropy", stochastic=False)
for n in [2, 3, 10]:
    add("foundations", f"parity_{n}", task="parity", variables=n, stochastic=False)
for fraction in [0, 0.25, 0.5, 0.75, 1]:
    add(
        "foundations",
        f"mix_{int(fraction * 100)}",
        task="mixing",
        fraction=fraction,
        stochastic=False,
    )
add(
    "foundations",
    "fixed_mass_extrema",
    task="entropy_extrema",
    alphabet=7,
    first=0.2,
    last=0.1,
    stochastic=False,
)
for eps in [0, 0.1, 0.3, 0.5]:
    add(
        "foundations",
        f"data_processing_{int(100 * eps)}",
        task="data_processing",
        first_crossover=0.1,
        second_crossover=eps,
        stochastic=False,
    )
for error in [0, 0.1, 0.3, 0.75]:
    add(
        "foundations",
        f"fano_{int(100 * error)}",
        task="fano",
        alphabet=4,
        error=error,
        stochastic=False,
    )
for name, costs in [("ordinary", [1] * 7), ("weighted", [1, 2, 1, 3, 1, 2, 4])]:
    add("foundations", name + "_huffman", task="weighted_huffman", costs=costs, stochastic=False)
add("foundations", "strictness_counterexample", task="strictness_counterexample", stochastic=False)
add("sources", "markov_transformations", task="markov_transformations", stochastic=False)
for source_name, probability in [("uniform", [0.25] * 4), ("skewed", [0.6, 0.25, 0.1, 0.05])]:
    for n in [16, 64, 256, 1024]:
        add(
            "sources",
            f"{source_name}_{n}",
            task="typicality",
            probability=probability,
            blocklength=n,
            trials=10000,
            weak_tolerance=0.1,
            strong_tolerance=0.2,
        )
for task in ["bsc", "bec", "z_channel"]:
    for eps in [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 1]:
        add("channels", f"{task}_{int(100 * eps)}", task=task, noise=eps, stochastic=False)
for task in ["course_channel1", "course_channel2"]:
    add("channels", task, task=task, stochastic=False)
add("channels", "weakly_symmetric", task="weakly_symmetric", stochastic=False)
for n in [32, 128, 512, 2048]:
    for epsilon in [0.001, 0.01, 0.1]:
        add(
            "channels",
            f"finite_length_{n}_{int(1000 * epsilon)}",
            task="finite_length",
            noise=0.1,
            blocklength=n,
            block_error=epsilon,
            stochastic=False,
        )
for side in [False, True]:
    for q in [0.25, 0.5, 0.75, 1]:
        add(
            "channels",
            f"jammer_{int(100 * q)}_{'known' if side else 'unknown'}",
            task="jammer",
            free_probability=q,
            noise=0.1,
            decoder_knows=side,
            stochastic=False,
        )
for power in [0.25, 1, 4, 16]:
    add(
        "channels",
        f"water_filling_{str(power).replace('.', 'p')}",
        task="water_filling",
        noise_variances=[0.5, 1, 2, 4],
        total_power=power,
        stochastic=False,
    )
betas = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.2, 1.6, 2, 3, 4, 6, 8, 12, 16, 24]
for task, params in [
    ("bernoulli_hamming", {"p_one": 0.5}),
    ("biased_hamming", {"p_one": 0.2}),
    ("asymmetric_forbidden", {}),
    ("four_pam", {}),
    ("discretized_gaussian", {}),
]:
    for i, beta in enumerate(betas):
        add(
            "rate_distortion",
            f"{task}_{i:02}",
            task="bernoulli_hamming" if task == "biased_hamming" else task,
            beta=beta,
            tolerance=1e-6 if task == "discretized_gaussian" else 1e-8,
            accelerate=task == "discretized_gaussian",
            stochastic=False,
            **params,
        )
for method in [
    "uncoded",
    "repetition3",
    "hamming_hard",
    "hamming_soft",
    "convolutional",
    "polar_sc",
    "ldpc_bp",
    "ldpc_min_sum",
]:
    for ebn0 in [-1, 1, 3, 5, 7]:
        add(
            "coding",
            f"{method}_{'m1' if ebn0 < 0 else ebn0}",
            method=method,
            message_bits=64,
            trials=2048,
            ebn0_db=ebn0,
            batch_size=256,
            max_iterations=50,
            graph_seed=991,
            design_erasure=0.5,
        )
for task in ["iid_biased", "markov_binary", "uniform_bytes", "owned_code", "ecoli_excerpt"]:
    for size in [4096, 65536]:
        for method in [
            "huffman",
            "shannon",
            "arithmetic0",
            "arithmetic1",
            "lz77",
            "deflate",
            "zstd",
        ]:
            add(
                "lossless",
                f"{task}_{size}_{method}",
                task=task,
                max_bytes=size,
                method=method,
                window=4096,
                stochastic=task not in ["owned_code", "ecoli_excerpt"],
            )
for method in ["uniform", "lloyd_max"]:
    for levels in [2, 4, 8, 16, 32, 64, 128, 256]:
        add(
            "quantization",
            f"{method}_{levels}",
            method=method,
            levels=levels,
            mean=1,
            variance=3,
            train_size=8192,
            test_size=8192,
        )
for bits in [4, 8]:
    for snr in [-5, 0, 5, 10, 15]:
        add(
            "source_channel",
            f"hamming_{bits}_{'m5' if snr < 0 else snr}",
            quantizer_bits=bits,
            mean=1,
            variance=3,
            samples=4096,
            snr_db=snr,
        )
for dimension in [1, 4, 16]:
    for rho in [0, 0.3, 0.7, 0.95]:
        for n in [256, 1024, 4096]:
            for method in ["gaussian_plugin", "binned_plugin", "ksg"]:
                add(
                    "estimation",
                    f"{method}_d{dimension}_r{int(100 * rho)}_n{n}",
                    method=method,
                    dimension=dimension,
                    correlation=rho,
                    samples=n,
                    neighbors=5,
                    bin_edges=9,
                )
for n in [20, 50, 100, 200, 500]:
    add(
        "statistics",
        f"rare_event_{n}",
        task="rare_events",
        blocklength=n,
        probability=0.1,
        threshold=0.3,
        trials=10000,
    )
for n in [16, 32, 64, 128, 256, 512, 1024, 2048]:
    add(
        "statistics",
        f"stein_{n}",
        task="hypothesis_test",
        blocklength=n,
        null=0.1,
        alternative=0.3,
        alpha=0.05,
        stochastic=False,
    )
for clusters in [2, 3, 4]:
    for i, beta in enumerate([0.1, 0.5, 0.9, 1.5, 2, 4, 8, 16, 32]):
        add("bottleneck", f"clusters{clusters}_beta{i}", clusters=clusters, beta=beta, restarts=5)
for method in ["mine_dv", "infonce"]:
    for d in [1, 4, 16]:
        for rho in [0, 0.7, 0.95]:
            add(
                "neural",
                f"{method}_d{d}_r{int(100 * rho)}",
                method=method,
                dimension=d,
                correlation=rho,
                train_size=8192,
                validation_size=4096,
                test_size=4096,
                batch_size=256,
                steps=1500,
                learning_rate=0.001,
                validate_every=100,
                width=128,
            )
for name in ["astronaut", "coffee", "chelsea", "rocket"]:
    for method, points in [
        ("jpeg", [35, 60, 85]),
        ("webp", [35, 60, 85]),
        ("avif", [35, 60, 85]),
        ("jpeg2000", [20, 10, 5]),
        ("bmshj2018-factorized", [1, 3, 5]),
        ("bmshj2018-hyperprior", [1, 3, 5]),
        ("mbt2018-mean", [1, 3, 5]),
        ("cheng2020-anchor", [1, 3, 5]),
    ]:
        for point in points:
            add(
                "images",
                f"{name}_{method.replace('-', '_')}_{point}",
                image=name,
                method=method,
                point=point,
                timing_repeats=3,
                stochastic=False,
            )

protocol = {"schema": 1, "name": "reference", "seeds": [41, 42, 43], "cases": cases}
quick = []
for family in dict.fromkeys(c["family"] for c in cases):
    if family in ["images", "neural"]:
        continue  # Optional extras have separate explicit smoke tests.
    case = next(c.copy() for c in cases if c["family"] == family)
    for key, value in [
        ("trials", 128),
        ("train_size", 512),
        ("test_size", 512),
        ("samples", 256),
        ("max_bytes", 512),
        ("restarts", 2),
    ]:
        if key in case:
            case[key] = value
    quick.append(case)
for destination in [ROOT / "configs", ROOT / "src/information_lab/default_configs"]:
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in [
        ("reference", protocol),
        ("quick", {"schema": 1, "name": "quick", "seeds": [41], "cases": quick}),
    ]:
        (destination / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
print(
    f"{len(cases)} configured cases; {sum(3 if c.get('stochastic', True) else 1 for c in cases)} seeded runs"
)
