
"""
PHASE 3B - STEP 3B.9
Frozen Manifest Image Access Test

- Verifies the frozen manifest SHA256
- Looks up one selected tile in AT2 and GT450
- Verifies scanner, stainer, tile ID and acquisition
- Inspects image dimensions and reference format
- Does not download image bytes or modify the manifest
"""

from pathlib import Path
from urllib.parse import urlparse
import hashlib
import json

import pandas as pd
import requests


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "multi_region_research"

MANIFEST = (
    BASE / "outputs/validated/"
    "phase3b_frozen_tile_manifest.csv"
)

REPORT = (
    BASE / "outputs/reports/"
    "09_paired_image_access_test.json"
)

EXPECTED_HASH = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)

API = "https://datasets-server.huggingface.co/filter"
DATASET = "owkin/plism-dataset-tiles"

SCANNERS = {
    "AT2": "GIVH_AT2_to_GMH_S60",
    "GT450": "GIVH_GT450_to_GMH_S60",
}

MAX_RESPONSE_BYTES = 100_000


def fetch_tile(session, tile_id, scanner):
    where = (
        f'"tile_id"=\'{tile_id}\' '
        f'AND "stainer"=\'GIVH\' '
        f'AND "scanner"=\'{scanner}\''
    )

    params = {
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "where": where,
        "offset": 0,
        "length": 2,
    }

    with session.get(
        API,
        params=params,
        stream=True,
        timeout=(10, 25),
    ) as response:

        response.raise_for_status()

        raw = bytearray()

        for chunk in response.iter_content(16384):
            raw.extend(chunk)

            if len(raw) > MAX_RESPONSE_BYTES:
                raise RuntimeError(
                    "Metadata response exceeded safety limit"
                )

        data = json.loads(raw)

    rows = data.get("rows", [])

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one {scanner} record, "
            f"received {len(rows)}"
        )

    if int(data.get("num_rows_total", -1)) != 1:
        raise RuntimeError(
            f"Unexpected match count for {scanner}"
        )

    record = rows[0]["row"]

    if record.get("tile_id") != tile_id:
        raise RuntimeError("Tile ID mismatch")

    if record.get("scanner") != scanner:
        raise RuntimeError("Scanner mismatch")

    if record.get("stainer") != "GIVH":
        raise RuntimeError("Stainer mismatch")

    if record.get("slide_id") != SCANNERS[scanner]:
        raise RuntimeError("Acquisition ID mismatch")

    png = record.get("png")

    if not isinstance(png, dict):
        raise RuntimeError("Unexpected PNG reference format")

    src = png.get("src")
    height = png.get("height")
    width = png.get("width")

    if not isinstance(src, str) or not src:
        raise RuntimeError("Missing PNG reference")

    if (
        not isinstance(height, int)
        or not isinstance(width, int)
        or height <= 0
        or width <= 0
    ):
        raise RuntimeError("Invalid image dimensions")

    parsed = urlparse(src)

    return {
        "scanner": scanner,
        "tile_id": tile_id,
        "slide_id": record["slide_id"],
        "image_width": width,
        "image_height": height,
        "reference_scheme": parsed.scheme or "relative",
        "reference_host": parsed.hostname or "",
        "reference_present": True,
    }


print("=" * 70)
print("PHASE 3B - PAIRED IMAGE ACCESS TEST")
print("=" * 70)

print("\n[1] Verifying frozen manifest...")

if not MANIFEST.is_file():
    raise SystemExit(f"Missing manifest: {MANIFEST}")

actual_hash = hashlib.sha256(
    MANIFEST.read_bytes()
).hexdigest()

if actual_hash != EXPECTED_HASH:
    raise RuntimeError(
        "Frozen manifest SHA256 mismatch. "
        "Do not continue with image retrieval."
    )

df = pd.read_csv(MANIFEST)

if df.empty:
    raise RuntimeError("Frozen manifest is empty")

sample = df.iloc[0]
tile_id = str(sample["tile_id"])

print("Manifest fingerprint: VERIFIED")
print("Region:", sample["region_id"])
print("Selected tile:", tile_id)

print("\n[2] Checking exact scanner-paired metadata...")

results = []

with requests.Session() as session:
    for scanner in SCANNERS:
        result = fetch_tile(session, tile_id, scanner)
        results.append(result)

        print(f"\n{scanner}")
        print("  Tile ID:", result["tile_id"])
        print("  Acquisition:", result["slide_id"])
        print(
            "  Dimensions:",
            f"{result['image_width']} x "
            f"{result['image_height']}"
        )
        print(
            "  Reference scheme:",
            result["reference_scheme"]
        )
        print(
            "  Reference host:",
            result["reference_host"] or "(none)"
        )

print("\n[3] Validating pair dimensions...")

if (
    results[0]["image_width"] != results[1]["image_width"]
    or results[0]["image_height"] != results[1]["image_height"]
):
    raise RuntimeError("Scanner image dimensions differ")

print("Pair dimensions: MATCHED")

REPORT.parent.mkdir(parents=True, exist_ok=True)

report = {
    "phase": "3B",
    "step": "3B.9",
    "manifest_sha256": actual_hash,
    "sample_region": str(sample["region_id"]),
    "sample_tile_id": tile_id,
    "scanner_results": results,
    "metadata_access_verified": True,
    "image_bytes_downloaded": False,
    "pixel_content_verified": False,
    "all_720_images_verified": False,
}

REPORT.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("PAIRED IMAGE ACCESS TEST COMPLETED")
print("=" * 70)

print("Metadata and image references: VERIFIED")
print("Pixel content: NOT YET VERIFIED")
print("Images downloaded: NONE")
print("Report:", REPORT)
print("\nSTEP 3B.9 COMPLETED")
