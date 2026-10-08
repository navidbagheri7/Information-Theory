"""Build an offline standalone explorer from immutable recorded row associations."""

import json
from pathlib import Path

from information_lab.artifacts import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "docs/results.json").read_text())
    studies = json.loads((ROOT / "docs/comparison_figures.json").read_text())["figures"]
    data["studies"] = [
        {
            "file": s["file"],
            "sha256": s["sha256"],
            "scope": s["scope"],
            "title": Path(s["file"]).stem.replace("_", " "),
        }
        for s in studies
    ]
    template = (ROOT / "scripts/explorer_template.html").read_text()
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    html = template.replace("__RUN_COUNT__", str(len(data["rows"]))).replace(
        "__RECORD_DATA__", encoded
    )
    (ROOT / "docs/explorer.html").write_text(html)
    (ROOT / "index.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="refresh" content="0; url=docs/explorer.html"><title>Information Theory Lab</title></head><body><p><a href="docs/explorer.html">Open the interactive Information Theory Lab</a></p></body></html>\n'
    )
    write_json(
        ROOT / "docs/explorer_manifest.json",
        {
            "schema": 1,
            "runs": len(data["rows"]),
            "explorer_sha256": sha256(ROOT / "docs/explorer.html"),
            "results_sha256": sha256(ROOT / "docs/results.json"),
            "builder_sha256": sha256(__file__),
            "template_sha256": sha256(ROOT / "scripts/explorer_template.html"),
            "scope": "Offline recorded evidence. Every finite selection is bound to an actual run, decoded image or numerical plot. Live binary formulas and photo reveal are separate from recorded experiments.",
        },
    )
    print("Built", len(data["rows"]), "recorded interactive selections")


if __name__ == "__main__":
    main()
