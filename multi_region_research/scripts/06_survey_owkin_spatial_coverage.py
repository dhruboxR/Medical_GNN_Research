
"""
PHASE 3B - STEP 3B.6
Owkin Spatial Coverage Survey

Purpose:
- Survey multiple API offsets across GIVH AT2 and GT450.
- Measure cross-scanner tile-ID overlap in each sampled band.
- Inspect the spatial extent of sampled tile coordinates.
- Compare sampled positions with the existing shortlist.
- Decide whether broader metadata collection is worthwhile.

No image downloads.
No Parquet reads.
No changes to existing region selections.

Sampling is systematic by API offset, not random.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import re
import time

import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"

BASE = ROOT / "multi_region_research"

TABLES = BASE / "outputs" / "tables"
REPORTS = BASE / "outputs" / "reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

API_URL = "https://datasets-server.huggingface.co/filter"

DATASET = "owkin/plism-dataset-tiles"

STAINER = "GIVH"
SCANNERS = ("AT2", "GT450")

OFFSETS = [0, 3200, 6400, 9600, 12800, 16000]
PAGE_SIZE = 100

MAX_RESPONSE_BYTES = 400_000

TIMEOUT = (10, 25)
MAX_ATTEMPTS = 3

PATTERN = re.compile(
    r"^tile_(\d+)_(\d+)_(\d+)$"
)


# ============================================================
# API FUNCTION
# ============================================================

def fetch_metadata_page(session, scanner, offset):

    params = {
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "where": (
            f'"stainer"=\'{STAINER}\' '
            f'AND "scanner"=\'{scanner}\''
        ),
        "offset": offset,
        "length": PAGE_SIZE,
    }

    for attempt in range(1, MAX_ATTEMPTS + 1):

        try:
            with session.get(
                API_URL,
                params=params,
                stream=True,
                timeout=TIMEOUT,
            ) as response:

                response.raise_for_status()

                raw = bytearray()

                for chunk in response.iter_content(16384):
                    raw.extend(chunk)

                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise RuntimeError(
                            "Response size limit exceeded."
                        )

                data = json.loads(raw)

                if not isinstance(data.get("rows"), list):
                    raise RuntimeError(
                        "Invalid Dataset Viewer response."
                    )

                return data

        except (
            requests.RequestException,
            ValueError,
        ) as exc:

            print(
                f"  Attempt {attempt} failed: {exc}",
                flush=True,
            )

            if attempt == MAX_ATTEMPTS:
                raise

            time.sleep(attempt)

    raise RuntimeError("API request failed.")


# ============================================================
# STEP 1: INITIALIZE
# ============================================================

print("=" * 70, flush=True)
print("PHASE 3B - OWKIN SPATIAL COVERAGE SURVEY", flush=True)
print("=" * 70, flush=True)

print("\n[1] Preparing systematic offset survey...", flush=True)

print(f"Offsets: {OFFSETS}")
print(f"Scanners: {SCANNERS}")
print(f"Records per scanner per offset: {PAGE_SIZE}")

records = []
band_results = []
reported_totals = {}


# ============================================================
# STEP 2: QUERY SCANNER METADATA
# ============================================================

print("\n[2] Retrieving distributed metadata samples...", flush=True)

with requests.Session() as session:

    for offset in OFFSETS:

        print(f"\nOffset: {offset}", flush=True)

        scanner_data = {}

        for scanner in SCANNERS:

            data = fetch_metadata_page(
                session,
                scanner,
                offset,
            )

            if data.get("num_rows_total") is not None:
                reported_totals[scanner] = int(
                    data["num_rows_total"]
                )

            parsed = {}

            for item in data["rows"]:

                row = item.get("row", {})

                if (
                    row.get("stainer") != STAINER
                    or row.get("scanner") != scanner
                ):
                    raise ValueError(
                        "Unexpected acquisition metadata."
                    )

                tile_id = str(row.get("tile_id", ""))

                match = PATTERN.match(tile_id)

                if match is None:
                    raise ValueError(
                        f"Invalid tile ID: {tile_id}"
                    )

                level, x, y = map(int, match.groups())

                if tile_id in parsed:
                    raise ValueError(
                        f"Duplicate tile ID: {tile_id}"
                    )

                parsed[tile_id] = {
                    "offset": offset,
                    "scanner": scanner,
                    "tile_id": tile_id,
                    "level": level,
                    "x": x,
                    "y": y,
                    "slide_id": str(row.get("slide_id", "")),
                }

            scanner_data[scanner] = parsed

            records.extend(parsed.values())

            print(
                f"  {scanner}: {len(parsed)} tiles",
                flush=True,
            )

        ids_a = set(scanner_data["AT2"])
        ids_b = set(scanner_data["GT450"])

        shared = ids_a & ids_b

        # Report coordinates from paired positions only.
        paired_positions = [
            scanner_data["AT2"][tile_id]
            for tile_id in shared
        ]

        result = {
            "offset": offset,
            "at2_records": len(ids_a),
            "gt450_records": len(ids_b),
            "shared_tile_ids": len(shared),
            "at2_only": len(ids_a - ids_b),
            "gt450_only": len(ids_b - ids_a),
            "paired_x_min": (
                min(r["x"] for r in paired_positions)
                if paired_positions else None
            ),
            "paired_x_max": (
                max(r["x"] for r in paired_positions)
                if paired_positions else None
            ),
            "paired_y_min": (
                min(r["y"] for r in paired_positions)
                if paired_positions else None
            ),
            "paired_y_max": (
                max(r["y"] for r in paired_positions)
                if paired_positions else None
            ),
            "paired_levels": ";".join(
                map(
                    str,
                    sorted({
                        r["level"] for r in paired_positions
                    })
                )
            ),
        }

        band_results.append(result)

        print(f"  Shared IDs: {len(shared)}")

        if paired_positions:
            print(
                f"  X range: "
                f"{result['paired_x_min']} - "
                f"{result['paired_x_max']}"
            )

            print(
                f"  Y range: "
                f"{result['paired_y_min']} - "
                f"{result['paired_y_max']}"
            )

        time.sleep(0.2)


# ============================================================
# STEP 3: VALIDATE RESULTS
# ============================================================

print("\n[3] Validating survey data...", flush=True)

records_df = pd.DataFrame(records)
bands_df = pd.DataFrame(band_results)

if records_df.empty:
    raise RuntimeError("No metadata records retrieved.")

duplicates = int(
    records_df.duplicated(
        ["scanner", "tile_id"]
    ).sum()
)

if duplicates:
    raise RuntimeError(
        f"Repeated scanner/tile IDs: {duplicates}"
    )

total_matched = int(
    bands_df["shared_tile_ids"].sum()
)

print(f"Metadata records collected: {len(records_df)}")
print(f"Matched IDs across sampled bands: {total_matched}")
print(f"Duplicate records: {duplicates}")


# ============================================================
# STEP 4: SAVE RESULTS
# ============================================================

print("\n[4] Saving spatial survey outputs...", flush=True)

records_path = (
    TABLES / "plism_spatial_coverage_samples.csv"
)

bands_path = (
    TABLES / "plism_spatial_coverage_by_offset.csv"
)

report_path = (
    REPORTS / "06_owkin_spatial_coverage_survey.json"
)

records_df.to_csv(records_path, index=False)
bands_df.to_csv(bands_path, index=False)

report = {
    "phase": "3B",
    "step": "3B.6",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset": DATASET,
    "stainer": STAINER,
    "scanners": list(SCANNERS),
    "sample_offsets": OFFSETS,
    "page_size": PAGE_SIZE,
    "reported_scanner_totals": reported_totals,
    "metadata_records_collected": len(records_df),
    "paired_positions_across_sampled_bands":
        total_matched,
    "bands": band_results,
    "images_downloaded": 0,
    "full_dataset_inventory": False,
    "biological_independence_verified": False,
    "interpretation": (
        "A systematic metadata coverage survey. "
        "Results cannot establish representative sampling, "
        "tissue identity, or specimen independence."
    ),
}

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3B.6 - SPATIAL COVERAGE SURVEY COMPLETED")
print("=" * 70)

print("\nSurvey summary:")

print(
    bands_df[
        [
            "offset",
            "shared_tile_ids",
            "paired_x_min",
            "paired_x_max",
            "paired_y_min",
            "paired_y_max",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(records_path)
print(bands_path)
print(report_path)

print("\nImage downloads: NONE")
print("Biological independence: UNVERIFIED")
print("Full inventory: NOT YET COLLECTED")

print("\nSTEP 3B.6 COMPLETED")
