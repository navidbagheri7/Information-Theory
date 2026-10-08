# Reproduce the experiments

The repository includes measured reference records, portable arrays/streams, source hashes, configurations and rendered evidence. Opening the explorer only reads those recorded results. It does not train a network in the browser or require a GPU.

## Install and try the core

Use Python 3.10 or newer in your own environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
information-lab run --config configs/quick.json --output results/smoke
pytest
```

The core quick protocol requires no downloaded model, dataset or GPU. Two attributed compression fixtures are bundled. Installing the wheel also provides `information-lab init my-configs` to export both configurations outside a source checkout.

For every optional example:

```bash
python -m pip install -e '.[dev,neural,images,notebooks,browser]'
export INFORMATION_LAB_DATA="$HOME/.cache/information-lab"
information-lab prepare --project-root . --data-root "$INFORMATION_LAB_DATA" --models
```

Preparation fetches the versioned NCBI reference with its official checksum, makes the four attributed images, and downloads the twelve explicitly configured official model checkpoints. All large downloads live under the chosen data root. The owner's reference cache is `/data/navid.bagheri-shouraki/ML_data/information`; it is a local choice, never required on another computer. Original course PDFs are excluded from Git and source distributions.

## Full reference protocol

Choose a fresh output directory when source or input fingerprints differ:

```bash
information-lab run --config configs/reference.json --output results/my-run \
  --data-root "$INFORMATION_LAB_DATA" --family coding
information-lab run --config configs/reference.json --output results/my-run \
  --data-root "$INFORMATION_LAB_DATA" --family neural --device cuda:0
information-lab run --config configs/reference.json --output results/my-run \
  --data-root "$INFORMATION_LAB_DATA" --family images --device cpu
```

Omit `--family` to run every family. Repeat it to select several families. Omit `--device` for CPU; all neural examples also support CPU. The reference uses seeds 41/42/43 for stochastic cases and one deterministic record where repeated seeds would be meaningless. The saved JSON specifies every grid point and training budget. Three separate neural workers used one seed per L40S; they did not change the protocol or data splits.

Use `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` for controlled CPU timing. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before GPU startup. PyTorch deterministic algorithms and disabled TF32 are used by neural training. Exact floating-point identity across GPU/CPU, CUDA versions or library builds is not promised; explicit CPU inference tolerances are audited. Codec timing is local wall time and not a hardware-independent ranking.

Each case stores `run.json`, `data.npz`, and relevant real streams/model arrays/decoded images. Records include full source and dependency provenance and SHA256 hashes of artifacts. Completed records are resumed only when their configuration and fingerprints match and all saved artifact digests verify. A changed or partial run is not silently declared complete. Family workers write separate index files to avoid manifest races.

Resuming preserves the original device and dependency provenance. Changing `--device` or installed library versions does not force a completed experiment to run again: use a fresh output directory for a hardware or dependency comparison. `requirements-reference.txt` records the actual tested versions; its CUDA wheels require the corresponding official PyTorch wheel index. The ordinary installation instructions use the supported version ranges instead.

## Inspect and audit

```bash
python scripts/validate_reference.py --data-root "$INFORMATION_LAB_DATA"
python scripts/build_report.py
python scripts/build_explorer.py
python scripts/execute_notebooks.py
python scripts/audit_explorer.py
```

The independent reference audit reconstructs every configured metric or invariant, brute-forces KSG distances independently of its tree implementation, checks exact/dual analytic results, loads portable critics on CPU, and decodes the stored bytes using the external cached models. Original/decoded pixel arrays must match the displayed images. The browser audit exercises every enabled finite selector combination, slider endpoints/midpoints, applicable comparisons, links and images; it separately checks live numerical formulas and mobile rendering.

Reference arrays intentionally retain the detailed evidence behind the figures. The wheel contains the small runtime and fixtures; the source distribution also contains measured reference evidence and notebooks. Generated release checks document the exact archive contents and import identity.

Run `python scripts/verify_release.py --data-root "$INFORMATION_LAB_DATA"` to build and check both distributions, execute the source-archive tests and notebooks, and verify installed-wheel inference. This check needs the optional development/notebook/image dependencies and already prepared model cache. `docs/release_artifacts.json` is written afterward and deliberately excluded from the source archive: an archive cannot include a receipt containing its own final hash. It remains available in the repository checkout.
