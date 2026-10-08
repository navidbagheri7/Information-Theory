"""Portable evidence serialization, hashing and experiment provenance."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Nonfinite numeric evidence must be explicitly represented, not hidden")
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def provenance():
    root = Path(__file__).parent
    versions = {}
    for name in [
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "Pillow",
        "zstandard",
        "torch",
        "torchvision",
        "compressai",
        "scikit-image",
        "pytorch-msssim",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "source_sha256": {p.name: sha256(p) for p in sorted(root.glob("*.py"))},
        "resource_sha256": {
            str(p.relative_to(root)): sha256(p)
            for p in sorted((root / "default_configs").rglob("*"))
            if p.is_file() and p.suffix in [".json", ".txt"]
        },
        "information_unit": "bits; natural-log optimization multipliers explicitly labelled",
    }
