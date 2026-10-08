"""Exercise every recorded UI selection, numerical display and local asset over HTTP."""

import argparse
import hashlib
import html
import json
import math
import re
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

from playwright.sync_api import sync_playwright

from information_lab.artifacts import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            # Rapid user selections can cancel a previous image request.
            # All required responses are checked separately against file hashes.
            pass


def compare_expected(rows, current):
    c = current["config"]
    family = c["family"]
    result = []
    for row in rows:
        other = row["config"]
        f = other["family"]
        matched = False
        if family == "neural":
            matched = (
                f in ["neural", "estimation"]
                and other["dimension"] == c["dimension"]
                and other["correlation"] == c["correlation"]
                and (f == "neural" or other["samples"] == c["test_size"])
            )
        elif f != family:
            continue
        elif family == "images":
            matched = other["image"] == c["image"]
        elif family == "coding":
            matched = other["ebn0_db"] == c["ebn0_db"]
        elif family == "lossless":
            matched = other["task"] == c["task"] and other["max_bytes"] == c["max_bytes"]
        elif family == "quantization":
            matched = (
                other["levels"] == c["levels"]
                and other["mean"] == c["mean"]
                and other["variance"] == c["variance"]
            )
        elif family == "estimation":
            matched = all(other[k] == c[k] for k in ["dimension", "correlation", "samples"])
        elif family == "bottleneck":
            matched = other["clusters"] == c["clusters"]
        elif family == "rate_distortion":
            matched = other["task"] == c["task"] and other.get("p_one") == c.get("p_one")
        elif family == "source_channel":
            matched = other["quantizer_bits"] == c["quantizer_bits"]
        elif family == "sources":
            matched = other["task"] == c["task"] and other.get("probability") == c.get(
                "probability"
            )
        else:
            matched = other["task"] == c["task"]
        if matched:
            result.append(row["key"])
    return set(result)


def assert_visible_number(text, value):
    if value is None:
        assert text == "Not applicable"
        return
    if isinstance(value, bool):
        assert text == ("Yes" if value else "No")
        return
    if isinstance(value, list):
        numbers = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)]
        assert len(numbers) == len(value)
        for a, b in zip(numbers, value, strict=True):
            assert math.isclose(a, b, rel_tol=5e-6, abs_tol=1e-12)
        return
    assert math.isclose(float(text), value, rel_tol=5e-6, abs_tol=1e-12), (text, value)


def select(page, row):
    for control, value in [
        ("field", row["family"]),
        ("scenario", row["scenario"]),
        ("method", row["method"]),
        ("setting", row["setting"]),
        ("seed", row["seed"]),
    ]:
        locator = page.locator("#" + control)
        if locator.input_value() != str(value):
            assert not locator.is_disabled()
            locator.select_option(str(value))
    page.wait_for_function("key => window.selectionEvidence?.key === key", arg=row["key"])
    active = "#decodedImage" if row["family"] == "images" else "#experimentFigure"
    page.locator(active).wait_for(state="visible")
    page.wait_for_function(
        "selector => {const i=document.querySelector(selector);return i.complete&&i.naturalWidth>0}",
        arg=active,
    )
    if row["family"] == "images":
        page.wait_for_function(
            'document.querySelector("#originalImage").complete && document.querySelector("#originalImage").naturalWidth > 0'
        )


def displayed(page, row, rows):
    shown = page.evaluate(
        """() => ({key:document.getElementById('selectionStatus').dataset.key, title:document.getElementById('resultTitle').textContent, setting:document.getElementById('selectionStatus').textContent, config:JSON.parse(document.getElementById('config').textContent),scope:document.getElementById('scope').textContent,cards:[...document.querySelectorAll('.metric')].map(n=>({key:n.dataset.metric,value:JSON.parse(n.dataset.value),text:n.querySelector('.metric-value').textContent})), table:[...document.querySelectorAll('#comparison tbody tr')].map(n=>({key:n.dataset.key,cells:[...n.querySelectorAll('td[data-metric]')].map(t=>({key:t.dataset.metric,value:JSON.parse(t.dataset.value),text:t.textContent}))})),evidence:window.selectionEvidence,asset:document.querySelector(window.currentRecord.family==='images'?'#decodedImage':'#experimentFigure').getAttribute('src'),links:[...document.querySelectorAll('a[href]')].map(a=>a.href),images:[...document.images].filter(i=>i.offsetParent!==null).map(i=>i.src)})"""
    )
    assert shown["key"] == row["key"] and shown["config"] == row["config"]
    assert row["scenario"] in shown["title"] and row["method_label"] in shown["title"]
    assert row["setting"] in shown["setting"] and row["key"] in shown["setting"]
    assert (
        shown["evidence"]["asset_sha256"] == row["asset_sha256"]
        and shown["evidence"]["record_sha256"] == row["record_sha256"]
    )
    assert shown["asset"] == row["asset"]
    assert shown["cards"] and shown["table"]
    lookup = {r["key"]: r for r in rows}
    for card in shown["cards"]:
        assert card["value"] == row["metrics"][card["key"]]
        assert_visible_number(card["text"], card["value"])
    assert {r["key"] for r in shown["table"]} == compare_expected(rows, row)
    for tr in shown["table"]:
        source = lookup[tr["key"]]
        assert tr["cells"]
        for cell in tr["cells"]:
            assert cell["value"] == source["metrics"].get(cell["key"])
            assert_visible_number(cell["text"], cell["value"])
    if row["metrics"].get("scope"):
        assert shown["scope"] == row["metrics"]["scope"]
    return shown


def binary(page, value):
    locator = page.locator("#crossover")
    if value == 0:
        locator.focus()
        locator.press("Home")
    elif value == 0.5:
        locator.focus()
        locator.press("End")
    else:
        locator.evaluate(
            '(node,value)=>{node.value=value;node.dispatchEvent(new Event("input",{bubbles:true}))}',
            value,
        )
    values = page.evaluate(
        """() => ({p:Number(document.getElementById('crossover').value),entropy:Number(document.getElementById('liveEntropy').dataset.value),capacity:Number(document.getElementById('liveCapacity').dataset.value),entropyText:document.getElementById('liveEntropy').textContent,capacityText:document.getElementById('liveCapacity').textContent})"""
    )
    expected = (
        0 if value in [0, 1] else -value * math.log2(value) - (1 - value) * math.log2(1 - value)
    )
    assert (
        values["p"] == value
        and math.isclose(values["entropy"], expected, abs_tol=1e-12)
        and math.isclose(values["capacity"], 1 - expected, abs_tol=1e-12)
    )
    assert_visible_number(values["entropyText"], expected)
    assert_visible_number(values["capacityText"], 1 - expected)


def reveal(page, value):
    locator = page.locator("#reveal")
    if value in [0, 100]:
        locator.focus()
        locator.press("Home" if value == 0 else "End")
    else:
        locator.evaluate(
            '(node,value)=>{node.value=value;node.dispatchEvent(new Event("input",{bubbles:true}))}',
            value,
        )
    values = page.evaluate(
        """() => ({value:Number(document.getElementById('reveal').value),clip:document.getElementById('decodedImage').style.clipPath,label:document.getElementById('revealValue').textContent})"""
    )
    assert values["value"] == value and values["clip"] == f"inset(0px 0px 0px {value}%)"
    assert values["label"] == f"{value}% original area; {100 - value}% decoded area."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/browser_audit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "docs/results.json").read_text())
    rows = data["rows"]
    assert len(rows) == 1117
    for row in rows:
        record_path = (ROOT / "docs" / row["record"]).resolve()
        record = json.loads(record_path.read_text())
        assert record["metrics"] == row["metrics"] and record["config"] == row["config"]
        assert sha256(record_path) == row["record_sha256"]
        assert sha256((ROOT / "docs" / row["asset"]).resolve()) == row["asset_sha256"]
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    errors = []
    links = set()
    pictures = set()
    count = 0
    sliders = 0
    details = 0
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
            context = browser.new_context(
                viewport={"width": 1440, "height": 1000}, device_scale_factor=1
            )
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(base + "/")
            page.wait_for_url("**/docs/explorer.html")
            page.wait_for_function("window.explorerReady===true")
            assert (
                page.evaluate("window.currentRecord.key") == "images/coffee_cheng2020_anchor_3_s41"
            )
            page.screenshot(path=str(args.output / "default_desktop.png"))
            for row in rows:
                select(page, row)
                shown = displayed(page, row, rows)
                links.update(shown["links"])
                pictures.update(shown["images"])
                for value in [0, 0.25, 0.5]:
                    binary(page, value)
                    sliders += 1
                if row["family"] == "images":
                    for value in [0, 50, 100]:
                        reveal(page, value)
                        sliders += 1
                page.locator("#configurationDetails summary").click()
                assert page.locator("#config").is_visible()
                page.locator("#configurationDetails summary").click()
                assert not page.locator("#config").is_visible()
                details += 2
                count += 1
                if count % 50 == 0:
                    print("Exercised", count, "recorded selections", flush=True)
            requests = []
            queue = sorted(links | pictures)
            visited = set()
            while queue:
                url = queue.pop(0)
                parsed = urlparse(url)
                if not url.startswith(base) or parsed.fragment:
                    continue
                if url in visited:
                    continue
                visited.add(url)
                response = context.request.get(url)
                assert response.status == 200, (url, response.status)
                payload = response.body()
                relative = unquote(parsed.path).lstrip("/")
                path = (ROOT / relative).resolve()
                assert path.is_relative_to(ROOT) and path.is_file()
                digest = hashlib.sha256(payload).hexdigest()
                assert digest == sha256(path)
                requests.append({"file": relative, "sha256": digest, "bytes": len(payload)})
                if path.suffix == ".html":
                    for href in re.findall(r'href="([^"]+)"', payload.decode()):
                        queue.append(urljoin(url, html.unescape(href)))
            chosen = {
                "images": ("coffee_cheng2020_anchor_3", 41),
                "coding": ("ldpc_bp_1", 41),
                "foundations": ("parity_3", 41),
                "sources": ("uniform_16", 41),
                "channels": ("course_channel1", 41),
                "rate_distortion": ("asymmetric_forbidden_10", 41),
                "quantization": ("lloyd_max_16", 41),
                "source_channel": ("hamming_4_0", 41),
                "statistics": ("rare_event_500", 41),
                "estimation": ("ksg_d4_r70_n4096", 41),
                "neural": ("infonce_d16_r95", 41),
                "bottleneck": ("clusters3_beta5", 41),
                "lossless": ("markov_binary_65536_arithmetic1", 41),
            }
            for family, (identifier, seed) in chosen.items():
                row = next(
                    r
                    for r in rows
                    if r["family"] == family
                    and r["config"]["id"] == identifier
                    and r["seed"] == seed
                )
                select(page, row)
                reveal(page, 50) if family == "images" else None
                page.locator("section.panel").nth(1).screenshot(
                    path=str(args.output / (family + "_evidence.png"))
                )
            mobile = context.new_page()
            mobile.set_viewport_size({"width": 390, "height": 844})
            mobile.on("pageerror", lambda error: errors.append(str(error)))
            mobile.goto(base + "/docs/explorer.html")
            mobile.wait_for_function("window.explorerReady===true")
            for family, (identifier, seed) in chosen.items():
                row = next(
                    r
                    for r in rows
                    if r["family"] == family
                    and r["config"]["id"] == identifier
                    and r["seed"] == seed
                )
                select(mobile, row)
                displayed(mobile, row, rows)
                assert mobile.evaluate("document.documentElement.scrollWidth<=window.innerWidth+1")
                mobile.locator("section.panel").nth(1).screenshot(
                    path=str(args.output / (family + "_mobile.png"))
                )
            offline = context.new_page()
            offline.on("pageerror", lambda error: errors.append(str(error)))
            offline.goto((ROOT / "docs/explorer.html").as_uri())
            offline.wait_for_function("window.explorerReady===true")
            for family in chosen:
                row = next(r for r in rows if r["family"] == family)
                select(offline, row)
                displayed(offline, row, rows)
            assert not errors, errors
            result = {
                "schema": 1,
                "status": "passed",
                "recorded_selections": count,
                "slider_checks": sliders,
                "details_open_close_checks": details,
                "mobile_fields": len(chosen),
                "offline_fields": len(chosen),
                "local_http_files": len(requests),
                "javascript_errors": errors,
                "browser": browser.version,
                "audit_sha256": sha256(__file__),
                "explorer_sha256": sha256(ROOT / "docs/explorer.html"),
                "results_sha256": sha256(ROOT / "docs/results.json"),
                "scope": "Every recorded finite selection, independently matched scenario comparisons and rendered numeric cells, live binary formulas, photo reveal endpoints, expanded configuration, all visible local links/images and their served hashes. Desktop and 390px mobile screenshots; offline file navigation across all 13 fields. External primary-source URLs are attribution links, not included in the local HTTP audit.",
                "http_files": requests,
            }
            write_json(ROOT / "docs/browser_validation.json", result)
            write_json(args.output / "receipt.json", result)
            print(
                "PASS",
                count,
                "recorded selections",
                sliders,
                "slider checks",
                len(requests),
                "served files",
                flush=True,
            )
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
