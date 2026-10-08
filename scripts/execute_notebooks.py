"""Execute every notebook with this interpreter and preserve an auditable receipt."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from information_lab.artifacts import provenance, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-receipt", type=Path, default=ROOT / "docs/notebook_validation.json"
    )
    args = parser.parse_args()
    kernel_root = ROOT / ".local/jupyter"
    kernel = kernel_root / "kernels/information-lab"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Information Theory Lab validation",
                "language": "python",
                "env": {
                    "PYTHONPATH": str(ROOT / "src"),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "MPLBACKEND": "Agg",
                },
            }
        )
    )
    os.environ["JUPYTER_PATH"] = str(kernel_root) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    completed = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        started = time.perf_counter()
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="information-lab",
            resources={"metadata": {"path": str(ROOT / "notebooks")}},
        )
        client.execute()
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is not None
                assert not any(out.output_type == "error" for out in cell.outputs)
        nbformat.write(notebook, path)
        completed.append(
            {
                "file": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
                "seconds": time.perf_counter() - started,
            }
        )
        print("Executed", path.name, flush=True)
    assert len(completed) == 12
    write_json(
        args.output_receipt,
        {
            "schema": 1,
            "status": "passed",
            "notebooks": len(completed),
            "code_cells": sum(r["code_cells"] for r in completed),
            "executor_sha256": sha256(__file__),
            "runtime": provenance(),
            "records": completed,
        },
    )
    print("PASS", len(completed), "executed notebooks", flush=True)


if __name__ == "__main__":
    main()
