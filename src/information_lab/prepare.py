"""Prepare attributed datasets and external models in an explicitly chosen cache."""

import gzip
import hashlib
import shutil
from importlib.resources import files
from pathlib import Path
from urllib.request import urlopen

from .artifacts import sha256, write_json

NCBI_BASE = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/"
NCBI_FILE = "GCF_000005845.2_ASM584v2_genomic.fna.gz"


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        with urlopen(url, timeout=30) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    return destination


def prepare(data_root, project_root, with_models=False):
    """Preserve origin/hash metadata; never modify the shared Conda environment."""
    data_root, project_root = Path(data_root).resolve(), Path(project_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    public = project_root / "assets/corpora"
    public.mkdir(parents=True, exist_ok=True)
    packaged = project_root / "src/information_lab/default_configs/corpora"
    originals = project_root.parent / "MCMC_project/src/sampling_lab"
    if originals.exists():
        selected = ["targets.py", "samplers.py", "ebm.py", "diagnostics.py", "posterior.py"]
        content, components = [], []
        for name in selected:
            path = originals / name
            content.append(
                f"\n# Original MIT source: MCMC_project/src/sampling_lab/{name}\n"
                + path.read_text()
            )
            components.append({"file": name, "sha256": sha256(path)})
        corpus = "# Navid Bagheri-Shouraki — MIT scientific source-code corpus\n" + "".join(content)
        (public / "owned_code.txt").write_text(corpus)
        code_record = {
            "name": "owned_code",
            "source": "Owner's MIT Sampling & Energy Lab",
            "components": components,
            "license": "MIT, see LICENSE",
        }
    else:
        resource = files("information_lab") / "default_configs/corpora/owned_code.txt"
        (public / "owned_code.txt").write_bytes(resource.read_bytes())
        code_record = {
            "name": "owned_code",
            "source": "Bundled owned MIT source-code corpus",
            "license": "MIT, see LICENSE",
        }
    sequence_file = download(NCBI_BASE + NCBI_FILE, data_root / "genomics" / NCBI_FILE)
    with urlopen(NCBI_BASE + "md5checksums.txt", timeout=30) as response:
        checksums = response.read().decode()
    expected = next(
        line.split()[0] for line in checksums.splitlines() if line.split()[-1].endswith(NCBI_FILE)
    )
    if hashlib.md5(sequence_file.read_bytes()).hexdigest() != expected:
        raise ValueError("NCBI reference sequence does not match its upstream checksum")
    with gzip.open(sequence_file, "rt") as file:
        lines = file.read().splitlines()
    headers = [line for line in lines if line.startswith(">")]
    if len(headers) != 1 or "NC_000913.3" not in headers[0]:
        raise ValueError("Unexpected E. coli assembly structure")
    sequence = "".join(line.strip() for line in lines if not line.startswith(">")).upper().encode()
    if set(sequence) - set(b"ACGT"):
        raise ValueError("Unexpected non-ACGT symbols in this versioned reference")
    (public / "ecoli_excerpt.txt").write_bytes(sequence[:200000])
    genome_record = {
        "name": "ecoli_excerpt",
        "accession": "NC_000913.3",
        "assembly": "GCF_000005845.2",
        "source": NCBI_BASE + NCBI_FILE,
        "upstream_md5": expected,
        "download_sha256": sha256(sequence_file),
        "full_sequence_length": len(sequence),
        "full_sequence_sha256": hashlib.sha256(sequence).hexdigest(),
        "selection": "First 200,000 chromosome bases, no FASTA header or newline",
        "license": "Public NCBI reference sequence; factual biological sequence, source credited",
    }
    records = [code_record, genome_record]
    for record in records:
        path = public / (record["name"] + ".txt")
        record.update(file=path.name, bytes=path.stat().st_size, sha256=sha256(path))
    write_json(public / "manifest.json", records)
    if packaged.parent.exists():
        packaged.mkdir(parents=True, exist_ok=True)
        for path in public.iterdir():
            shutil.copy2(path, packaged / path.name)
    from .images import LEARNED, load_model, prepare_images

    image_records = prepare_images(data_root)
    image_public = project_root / "assets/images/originals"
    image_public.mkdir(parents=True, exist_ok=True)
    for record in image_records:
        shutil.copy2(data_root / "images" / record["file"], image_public / record["file"])
    write_json(project_root / "assets/images/manifest.json", image_records)
    models = []
    if with_models:
        for architecture in LEARNED:
            for quality in [1, 3, 5]:
                print("Preparing", architecture, quality, flush=True)
                _, record = load_model(architecture, quality, str(data_root), "cpu", True)
                models.append(record)
        write_json(project_root / "docs/model_provenance.json", models)
    write_json(
        data_root / "manifest.json", {"corpora": records, "images": image_records, "models": models}
    )
    return {"corpora": records, "images": image_records, "models": models}
