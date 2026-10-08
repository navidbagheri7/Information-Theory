"""Portable command-line entry point for configured evidence generation."""

import argparse
import json
import os
import re
from importlib.resources import files
from pathlib import Path

from .artifacts import write_json
from .experiments import run_case


def read_protocol(path):
    protocol = json.loads(Path(path).read_text())
    cases = protocol["cases"]
    identifiers = []
    for case in cases:
        if not re.fullmatch(r"[a-z0-9_]+", case["id"]):
            raise ValueError("Case identifiers must be plain lowercase names")
        if not re.fullmatch(r"[a-z0-9_]+", case["family"]):
            raise ValueError("Family names must be plain lowercase names")
        identifiers.append((case["family"], case["id"]))
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Case identifiers must be unique within each family")
    if not protocol["seeds"] or len(set(protocol["seeds"])) != len(protocol["seeds"]):
        raise ValueError("At least one unique experiment seed required")
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="Copy packaged quick/full configurations")
    initialize.add_argument("destination", type=Path)
    prepare = commands.add_parser(
        "prepare", help="Prepare credited images, DNA and optional weights"
    )
    prepare.add_argument("--models", action="store_true")
    prepare.add_argument("--project-root", type=Path, required=True)
    prepare.add_argument("--data-root", type=Path, required=True)
    run = commands.add_parser("run", help="Run or resume a configured protocol")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get("INFORMATION_LAB_DATA", "~/.cache/information-lab")
        ).expanduser(),
    )
    run.add_argument("--family", action="append", help="Repeat to select families")
    run.add_argument("--seed", type=int, action="append", help="Override protocol seeds")
    run.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.command == "init":
        args.destination.mkdir(parents=True, exist_ok=True)
        for name in ["quick.json", "reference.json"]:
            path = args.destination / name
            if path.exists():
                raise FileExistsError("Refusing to overwrite: " + str(path))
            path.write_bytes((files("information_lab") / "default_configs" / name).read_bytes())
        return
    if args.command == "prepare":
        from .prepare import prepare as prepare_data

        prepare_data(args.data_root, args.project_root, args.models)
        return
    protocol = read_protocol(args.config)
    selected = [c for c in protocol["cases"] if not args.family or c["family"] in args.family]
    if not selected:
        raise ValueError("No matching experiment cases")
    entries = []
    seeds = args.seed if args.seed is not None else protocol["seeds"]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds")
    for case in selected:
        for seed in seeds if case.get("stochastic", True) else [protocol["seeds"][0]]:
            record, resumed = run_case(case, seed, args.output, args.data_root, args.device)
            entry = {"family": case["family"], "id": case["id"], "seed": seed, "resumed": resumed}
            entries.append(entry)
            print(json.dumps({**entry, "metrics": record["metrics"]}), flush=True)
    # Family workers write separate indexes and never race over a shared manifest.
    suffix = "_".join(sorted(args.family or ["all"])) + "_" + "_".join(map(str, seeds))
    write_json(
        args.output / ("index_" + suffix + ".json"),
        {"schema": 1, "config": protocol, "cases": entries},
    )


if __name__ == "__main__":
    main()
