
"""
PHASE 3B.4 - API-Based Owkin Region Discovery

Uses the Hugging Face Dataset Viewer API instead of
opening remote Parquet shards.

- Up to 2,000 metadata rows per scanner
- 100 rows per API request
- Explicit network timeouts and response-size limits
- No image downloads
- Saves only metadata and provisional region selections

This is a bounded discovery sample, not a complete
dataset-wide inventory.
"""

from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import json
import re
import time

import pandas as pd
import requests


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"

BASE = ROOT / "multi_region_research"
TABLES = BASE / "outputs/tables"
REPORTS = BASE / "outputs/reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

API = "https://datasets-server.huggingface.co/filter"

DATASET = "owkin/plism-dataset-tiles"
SCANNERS = ("AT2", "GT450")
STAINER = "GIVH"

PAGE_SIZE = 100
MAX_PAGES_PER_SCANNER = 20

MAX_RESPONSE_BYTES = 400_000
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 20

WINDOW = 6
MIN_MATCHED = 24
MAX_REGIONS = 12

PILOT_ROWS = set(range(100, 110))
PILOT_COLS = set(range(126, 136))

PATTERN = re.compile(
    r"^tile_(\d+)_(\d+)_(\d+)$"
)


# --------------------------------------------------
# API HELPER
# --------------------------------------------------

def fetch_page(session, scanner, offset):

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

    with session.get(
        API,
        params=params,
        stream=True,
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    ) as response:

        response.raise_for_status()

        content = bytearray()

        for chunk in response.iter_content(16384):
            content.extend(chunk)

            if len(content) > MAX_RESPONSE_BYTES:
                raise RuntimeError(
                    "API response exceeded safety limit."
                )

        return json.loads(content)


# --------------------------------------------------
# STEP 1: RETRIEVE BOUNDED SCANNER METADATA
# --------------------------------------------------

print("=" * 70, flush=True)
print("PHASE 3B.4 - OWKIN API REGION DISCOVERY", flush=True)
print("=" * 70, flush=True)

print("\n[1] Retrieving scanner metadata...", flush=True)

inventory = defaultdict(dict)
scanner_totals = {}
request_counts = {}

with requests.Session() as session:

    for scanner in SCANNERS:

        print(f"\nScanner: {scanner}", flush=True)

        scanner_inventory = {}
        reported_total = None
        completed_requests = 0

        for page in range(MAX_PAGES_PER_SCANNER):

            offset = page * PAGE_SIZE

            # Retry transient connection issues twice.
            data = None

            for attempt in range(3):

                try:
                    data = fetch_page(
                        session, scanner, offset
                    )
                    break

                except (
                    requests.RequestException,
                    ValueError,
                ) as exc:

                    print(
                        f"  Request issue: {exc}",
                        flush=True
                    )

                    if attempt == 2:
                        raise

                    time.sleep(attempt + 1)

            if data is None:
                raise RuntimeError("No API response.")

            rows = data.get("rows", [])

            if not isinstance(rows, list):
                raise RuntimeError(
                    "Unexpected API response format."
                )

            completed_requests += 1

            total = data.get("num_rows_total")

            if total is not None:
                reported_total = int(total)

            for item in rows:

                record = item.get("row", {})

                if (
                    record.get("scanner") != scanner
                    or record.get("stainer") != STAINER
                ):
                    raise RuntimeError(
                        "Unexpected scanner/stainer record."
                    )

                tile_id = str(
                    record.get("tile_id", "")
                )

                match = PATTERN.match(tile_id)

                if not match:
                    continue

                level, x, y = map(
                    int, match.groups()
                )

                if tile_id in scanner_inventory:
                    raise RuntimeError(
                        f"Duplicate tile ID: {tile_id}"
                    )

                scanner_inventory[tile_id] = {
                    "tile_id": tile_id,
                    "level": level,
                    "x": x,
                    "y": y,
                    "scanner": scanner,
                    "slide_id": str(
                        record.get("slide_id", "")
                    ),
                }

            print(
                f"  Page {page + 1:02d}: "
                f"{len(rows)} rows, "
                f"{len(scanner_inventory)} valid tile IDs",
                flush=True,
            )

            if len(rows) < PAGE_SIZE:
                break

            if (
                reported_total is not None
                and offset + len(rows) >= reported_total
            ):
                break

            # Small delay to avoid rapid API calls.
            time.sleep(0.15)

        inventory[scanner] = scanner_inventory
        scanner_totals[scanner] = reported_total
        request_counts[scanner] = completed_requests

        print(
            f"  Total discovered: "
            f"{len(scanner_inventory)}",
            flush=True
        )

        print(
            f"  Dataset matching rows: "
            f"{reported_total}",
            flush=True
        )


# --------------------------------------------------
# STEP 2: FIND MATCHED TILE IDENTIFIERS
# --------------------------------------------------

print("\n[2] Matching scanner tile IDs...", flush=True)

at2_ids = set(inventory["AT2"])
gt450_ids = set(inventory["GT450"])

shared_ids = at2_ids & gt450_ids

print(f"AT2 observed IDs: {len(at2_ids)}")
print(f"GT450 observed IDs: {len(gt450_ids)}")
print(f"Shared IDs: {len(shared_ids)}")

print(
    "Note: The scanner inventories are truncated "
    "to the requested API pages."
)

matched_records = []

for tile_id in sorted(shared_ids):

    at2 = inventory["AT2"][tile_id]
    gt450 = inventory["GT450"][tile_id]

    matched_records.append({
        "tile_id": tile_id,
        "level": at2["level"],
        "x": at2["x"],
        "y": at2["y"],
        "at2_slide_id": at2["slide_id"],
        "gt450_slide_id": gt450["slide_id"],
    })

matched_df = pd.DataFrame(
    matched_records,
    columns=[
        "tile_id", "level", "x", "y",
        "at2_slide_id", "gt450_slide_id"
    ]
)


# --------------------------------------------------
# STEP 3: DISCOVER SPATIAL WINDOWS
# --------------------------------------------------

print("\n[3] Finding candidate spatial windows...", flush=True)

positions_by_level = defaultdict(set)

for row in matched_df.itertuples(index=False):
    positions_by_level[int(row.level)].add(
        (int(row.x), int(row.y))
    )

candidates = []

for level, positions in sorted(
    positions_by_level.items()
):

    for x0, y0 in sorted(positions):

        footprint = {
            (x, y)
            for x in range(x0, x0 + WINDOW)
            for y in range(y0, y0 + WINDOW)
        }

        # Exclude any overlap with the Phase 1 pilot.
        if level == 16 and any(
            x in PILOT_ROWS and y in PILOT_COLS
            for x, y in footprint
        ):
            continue

        matched = footprint & positions

        if len(matched) < MIN_MATCHED:
            continue

        candidates.append({
            "level": level,
            "x_start": x0,
            "y_start": y0,
            "x_end": x0 + WINDOW - 1,
            "y_end": y0 + WINDOW - 1,
            "matched_positions": len(matched),
            "coverage_fraction": (
                len(matched) / (WINDOW * WINDOW)
            ),
            "_footprint": footprint,
        })

print(f"Candidate windows: {len(candidates)}")


# --------------------------------------------------
# STEP 4: SELECT NON-OVERLAPPING REGIONS
# --------------------------------------------------

print("\n[4] Selecting provisional regions...", flush=True)

candidates.sort(
    key=lambda r: (
        -r["matched_positions"],
        r["level"],
        r["x_start"],
        r["y_start"],
    )
)

selected = []
occupied = set()

for candidate in candidates:

    if len(selected) >= MAX_REGIONS:
        break

    footprint = {
        (candidate["level"], x, y)
        for x, y in candidate["_footprint"]
    }

    if footprint & occupied:
        continue

    selected.append(candidate)
    occupied.update(footprint)


# --------------------------------------------------
# STEP 5: SAVE METADATA
# --------------------------------------------------

print("\n[5] Saving results...", flush=True)

matched_path = (
    TABLES / "plism_api_matched_tile_inventory.csv"
)

region_path = (
    TABLES / "plism_api_region_shortlist.csv"
)

report_path = (
    REPORTS / "04b_owkin_api_region_discovery.json"
)

matched_df.to_csv(matched_path, index=False)

region_records = []

for index, region in enumerate(selected, 1):

    region_records.append({
        "region_id": f"OWKIN_REGION_{index:03d}",
        "level": region["level"],
        "x_start": region["x_start"],
        "y_start": region["y_start"],
        "x_end": region["x_end"],
        "y_end": region["y_end"],
        "matched_positions": region["matched_positions"],
        "coverage_fraction": region["coverage_fraction"],
        "tissue_type": "UNVERIFIED",
        "specimen_id": "UNVERIFIED",
    })

regions_df = pd.DataFrame(
    region_records,
    columns=[
        "region_id", "level", "x_start",
        "y_start", "x_end", "y_end",
        "matched_positions", "coverage_fraction",
        "tissue_type", "specimen_id",
    ]
)

regions_df.to_csv(region_path, index=False)

report = {
    "phase": "3B",
    "step": "3B.4 API",
    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset": DATASET,
    "stainer": STAINER,
    "scanners": list(SCANNERS),
    "max_pages_per_scanner": MAX_PAGES_PER_SCANNER,
    "page_size": PAGE_SIZE,
    "requests_completed": request_counts,
    "reported_scanner_totals": scanner_totals,
    "observed_tile_counts": {
        scanner: len(inventory[scanner])
        for scanner in SCANNERS
    },
    "matched_tile_ids": len(shared_ids),
    "candidate_windows": len(candidates),
    "selected_regions": len(selected),
    "full_dataset_scanned": False,
    "original_wsi_coordinate_mapping_verified": False,
    "biological_independence_verified": False,
    "images_downloaded": 0,
    "limitations": [
        "Only initial API pages were examined",
        "Matched ID coverage may be incomplete",
        "No tissue labels have been verified",
        "No biological independence has been established",
        "No source WSI coordinate transform was inferred",
    ],
}

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)


# --------------------------------------------------
# FINAL OUTPUT
# --------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 3B.4 API DISCOVERY COMPLETED")
print("=" * 70)

print(f"AT2 observed: {len(at2_ids)}")
print(f"GT450 observed: {len(gt450_ids)}")
print(f"Matched tile IDs: {len(shared_ids)}")
print(f"Candidate windows: {len(candidates)}")
print(f"Selected regions: {len(selected)}")

if not regions_df.empty:
    print("\nProvisional region shortlist:")
    print(regions_df.to_string(index=False))

print("\nSaved:")
print(matched_path)
print(region_path)
print(report_path)

print("\nImage downloads: NONE")
print("Biological independence: UNVERIFIED")
print("Original WSI mapping: UNVERIFIED")

