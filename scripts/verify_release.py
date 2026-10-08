"""Build distributions and verify their contents and behavior outside the checkout."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, cwd, env, log):
    print("Checking:", " ".join(map(str, command)), flush=True)
    with log.open("w") as handle:
        subprocess.run(
            command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    for name in ("reference_validation", "notebook_validation", "browser_validation"):
        evidence = json.loads((ROOT / "docs" / (name + ".json")).read_text())
        assert evidence["status"] == "passed", f"Missing passing {name}"
        if name == "browser_validation":
            assert evidence["explorer_sha256"] == digest(ROOT / "docs/explorer.html")
            assert evidence["results_sha256"] == digest(ROOT / "docs/results.json")
        elif name == "notebook_validation":
            for record in evidence["records"]:
                assert record["sha256"] == digest(ROOT / record["file"])
    scratch = ROOT / ".local/release_verify"
    scratch.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="check-", dir=scratch))
    distribution = ROOT / ".local/release_dist"
    distribution.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONNOUSERSITE="1"
    )
    env.pop("PYTHONPATH", None)
    run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(distribution),
        ],
        ROOT,
        env,
        work / "build.log",
    )
    archive = distribution / "information_theory_lab-1.0.0.tar.gz"
    wheel = distribution / "information_theory_lab-1.0.0-py3-none-any.whl"
    public = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    public = sorted({p for p in public if p and p != "docs/release_artifacts.json"})
    extracted = work / "source"
    extracted.mkdir()
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            destination = (extracted / member.name).resolve()
            assert destination.is_relative_to(extracted.resolve())
            assert not member.issym() and not member.islnk()
            parts = Path(member.name).parts
            assert not any(part in {".local", ".git"} for part in parts), member.name
            assert parts[1:3] != ("legacy", "originals"), member.name
            assert not member.name.endswith((".pdf", ".pth", ".pt"))
        stream.extractall(extracted)
    source = next(extracted.iterdir())
    missing = [name for name in public if not (source / name).is_file()]
    assert not missing, f"Missing public archive files: {missing}"
    inventory = []
    for name in public:
        original, copy = ROOT / name, source / name
        assert original.is_file() and copy.is_file(), f"Missing public archive file: {name}"
        assert digest(copy) == digest(original), f"Archive mismatch: {name}"
        inventory.append(
            {"file": name, "sha256": digest(original), "bytes": original.stat().st_size}
        )
    (work / "public_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    source_env = {**env, "PYTHONPATH": str(source / "src"), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    run(
        [
            sys.executable,
            "-c",
            "import information_lab; from pathlib import Path; "
            f"assert Path(information_lab.__file__).resolve().is_relative_to(Path({str(source)!r}))",
        ],
        source,
        source_env,
        work / "source_identity.log",
    )
    run([sys.executable, "-m", "pytest", "-q"], source, source_env, work / "source_tests.log")
    run(
        [
            sys.executable,
            "-m",
            "information_lab.cli",
            "run",
            "--config",
            "configs/quick.json",
            "--output",
            str(work / "source_quick"),
        ],
        source,
        source_env,
        work / "source_quick.log",
    )
    run(
        [
            sys.executable,
            "scripts/execute_notebooks.py",
            "--output-receipt",
            str(work / "source_notebooks.json"),
        ],
        source,
        source_env,
        work / "source_notebooks.log",
    )
    target = work / "wheel_target"
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--target",
            str(target),
            str(wheel),
        ],
        work,
        env,
        work / "wheel_install.log",
    )
    with zipfile.ZipFile(wheel) as stream:
        assert not any(
            n.startswith(("results/", "assets/", "legacy/", "docs/")) for n in stream.namelist()
        )
    runtime_files = []
    for original in sorted((ROOT / "src/information_lab").rglob("*")):
        if original.is_file() and "__pycache__" not in original.parts:
            relative = original.relative_to(ROOT / "src")
            assert digest(original) == digest(target / relative)
            runtime_files.append(str(relative))
    wheel_run = work / "wheel_run"
    wheel_run.mkdir()
    critic_source = ROOT / "results/reference/neural/mine_dv_d4_r70_s41"
    for name in ("model.npz", "data.npz", "run.json"):
        shutil.copy2(critic_source / name, wheel_run / ("critic_" + name))
    image_source = ROOT / "results/reference/images/coffee_cheng2020_anchor_3_s41"
    for name in ("image.bin", "decoded.png"):
        shutil.copy2(image_source / name, wheel_run / name)
    probe = wheel_run / "probe.py"
    probe.write_text(
        """import json
from pathlib import Path
import importlib.metadata
import numpy as np
import torch
from PIL import Image
import information_lab
from information_lab.neural import load, evaluate
from information_lab.images import decode
import sys
assert Path(information_lab.__file__).resolve().is_relative_to(Path(sys.argv[1]))
assert importlib.metadata.version('information-theory-lab') == '1.0.0'
critic, metadata = load('critic_model.npz', 'cpu')
record = json.loads(Path('critic_run.json').read_text())
with np.load('critic_data.npz', allow_pickle=False) as arrays:
    value = evaluate(critic, torch.tensor(arrays['test_x'], dtype=torch.float32),
                     torch.tensor(arrays['test_y'], dtype=torch.float32),
                     record['config']['batch_size'])['estimate_bits']
assert np.isclose(value, record['metrics']['estimate_bits'], rtol=0, atol=3e-5)
actual = decode(Path('image.bin').read_bytes(), 'cheng2020-anchor', sys.argv[2], 'cpu')
np.testing.assert_array_equal(actual, np.asarray(Image.open('decoded.png').convert('RGB')))
print(json.dumps({'import': str(Path(information_lab.__file__).resolve()),
                  'critic_estimate_bits': float(value), 'image_byte_exact': True}))
"""
    )
    wheel_env = {**env, "PYTHONPATH": str(target), "PYTHONNOUSERSITE": "1"}
    console = target / "bin/information-lab"
    assert console.is_file(), "Installed wheel is missing its console entry point"
    run(
        [str(console), "init", "configs"],
        wheel_run,
        wheel_env,
        work / "wheel_init.log",
    )
    run(
        [
            str(console),
            "run",
            "--config",
            "configs/quick.json",
            "--output",
            "results",
        ],
        wheel_run,
        wheel_env,
        work / "wheel_quick.log",
    )
    run(
        [sys.executable, str(probe), str(target.resolve()), str(args.data_root.resolve())],
        wheel_run,
        wheel_env,
        work / "wheel_inference.log",
    )
    notebook_result = json.loads((work / "source_notebooks.json").read_text())
    assert notebook_result["status"] == "passed" and notebook_result["notebooks"] == 12
    tests = (work / "source_tests.log").read_text()
    assert "48 passed" in tests
    receipt = {
        "schema": 1,
        "status": "passed",
        "validator_sha256": digest(__file__),
        "source_archive": {
            "file": archive.name,
            "sha256": digest(archive),
            "bytes": archive.stat().st_size,
        },
        "wheel": {"file": wheel.name, "sha256": digest(wheel), "bytes": wheel.stat().st_size},
        "public_files_identical_to_checkout": len(inventory),
        "public_inventory_sha256": digest(work / "public_inventory.json"),
        "installed_runtime_resources_verified": len(runtime_files),
        "source_archive_tests_passed": 48,
        "source_archive_notebooks": notebook_result["notebooks"],
        "source_archive_code_cells": notebook_result["code_cells"],
        "source_and_wheel_quick_protocol": "passed",
        "installed_wheel_neural_cpu_and_actual_image_decode": "passed",
        "execution_environment": "Project interpreter with scientific dependencies already installed; offline wheel --no-deps install into an isolated target; asserted source/archive import identities. Not a fresh dependency-resolution test.",
        "receipt_excluded_from_archive": "Avoids an archive/self-hash cycle; receipt is published in the checkout.",
        "private_logs_directory": str(work.relative_to(ROOT)),
    }
    (ROOT / "docs/release_artifacts.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        "PASS: source archive, tests, notebooks, wheel quick protocol and portable inference",
        flush=True,
    )


if __name__ == "__main__":
    main()
