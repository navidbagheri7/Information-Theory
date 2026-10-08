"""Readable static HTML guides for offline and GitHub Pages previews."""

import html
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]


def main():
    for path in sorted((ROOT / "docs").glob("*.md")):
        title = next(
            line.lstrip("# ") for line in path.read_text().splitlines() if line.startswith("# ")
        )
        body = markdown.markdown(path.read_text(), extensions=["tables", "fenced_code", "toc"])
        body = re.sub(r'(href="[^"]+)\.md(?=["#])', r"\1.html", body)
        page = (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>"
            "body{max-width:1100px;margin:32px auto;padding:0 20px;font:16px/1.65 system-ui;"
            "color:#132d48;background:#f8fafc}a{color:#007c88}h1,h2,h3{line-height:1.25}"
            "table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}"
            "th,td{border:1px solid #d7e4ec;padding:10px;vertical-align:top}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf3f7;padding:16px;"
            "border-radius:10px}code{font-size:.9em}img{max-width:100%;height:auto}"
            '</style></head><body><nav><a href="explorer.html">← Interactive lab</a> · '
            f'<a href="{path.name}">Markdown source</a></nav>{body}</body></html>\n'
        )
        path.with_suffix(".html").write_text(page)
        print("Rendered guide", path.name)


if __name__ == "__main__":
    main()
