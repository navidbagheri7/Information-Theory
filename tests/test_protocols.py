"""Integration contracts for saved experiments and scientifically matched settings."""

import json
from importlib.resources import files

import numpy as np
import pytest

from information_lab.artifacts import plain, write_json
from information_lab.cli import main, read_protocol
from information_lab.experiments import coding_experiment, run_case


def test_quick_cli_all_core_families_and_resume(tmp_path, capsys):
    main(["init", str(tmp_path / "configs")])
    config = tmp_path / "configs/quick.json"
    protocol = read_protocol(config)
    for case in protocol["cases"]:
        first, resumed = run_case(case, 41, tmp_path / "runs", tmp_path / "data")
        assert not resumed
        second, resumed = run_case(case, 41, tmp_path / "runs", tmp_path / "data")
        assert resumed and first == second
    main(["run", "--config", str(config), "--output", str(tmp_path / "runs")])
    for line in capsys.readouterr().out.splitlines():
        assert json.loads(line)["resumed"]
    with pytest.raises(FileExistsError):
        main(["init", str(tmp_path / "configs")])


def test_resume_rejects_changed_numerical_artifact(tmp_path):
    case = {"id": "joint", "family": "foundations", "task": "joint_entropy"}
    run_case(case, 41, tmp_path, tmp_path)
    path = tmp_path / "foundations/joint_s41/data.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="artifact changed"):
        run_case(case, 41, tmp_path, tmp_path)


def test_no_silent_nonfinite_evidence(tmp_path):
    with pytest.raises(ValueError, match="Nonfinite"):
        write_json(tmp_path / "bad.json", {"result": np.inf})
    assert plain(np.bool_(True)) is True


@pytest.mark.parametrize(
    "method",
    [
        "uncoded",
        "repetition3",
        "hamming_hard",
        "hamming_soft",
        "convolutional",
        "polar_sc",
        "ldpc_bp",
        "ldpc_min_sum",
    ],
)
def test_configured_decoders_high_snr_and_actual_energy(method):
    config = json.loads((files("information_lab") / "default_configs/reference.json").read_text())
    case = next(
        c.copy() for c in config["cases"] if c["family"] == "coding" and c["method"] == method
    )
    case.update(trials=32, ebn0_db=30)
    metric, arrays = coding_experiment(case, 41)
    np.testing.assert_array_equal(arrays["message"], arrays["decoded"])
    np.testing.assert_allclose(metric["noise_variance"], 1 / (2 * metric["actual_rate"] * 1000))
    assert metric["coded_bits"] == arrays["example_encoded"].shape[1]


def test_resume_rejects_changed_packaged_corpus_fingerprint(tmp_path, monkeypatch):
    from information_lab import experiments

    case = {"id": "joint", "family": "foundations", "task": "joint_entropy"}
    record, _ = run_case(case, 41, tmp_path, tmp_path)
    assert "default_configs/corpora/owned_code.txt" in record["resource_sha256"]
    current = experiments.provenance()
    current["resource_sha256"] = {
        **current["resource_sha256"],
        "default_configs/corpora/owned_code.txt": "changed",
    }
    monkeypatch.setattr(experiments, "provenance", lambda: current)
    with pytest.raises(RuntimeError, match="stale"):
        run_case(case, 41, tmp_path, tmp_path)
