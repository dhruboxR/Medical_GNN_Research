
"""
PHASE 3B - STEP 3B.7
Distributed Owkin-Native Region Sampling

Inputs:
    plism_api_matched_tile_inventory.csv

New API sampling:
    5 distributed offsets
    700 records per scanner per offset

Region selection:
    6 x 6 fully populated windows
    Minimum inter-region grid separation: 6
    Maximum 2 regions per sampling band
    Maximum 12 regions total

Outputs:
    Distributed matched tile inventory
    Distributed region shortlist
    Exact tile-to-region manifest
    Pairwise region distances
    Sampling report

No images downloaded.
No biological independence inferred.
"""

from pathlib import Path
from collections import defaultdict
from itertools import combinations
from datetime import datetime, timezone
import json
import math
import re
import time

import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "multi_region_research"

TABLES = BASE / "outputs/tables"
REPORTS = BASE / "outputs/reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

PREVIOUS_INVENTORY = (
    TABLES / "plism_api_matched_tile_inventory.csv"
)

API = "https://datasets-server.huggingface.co/filter"
DATASET = "owkin/plism-dataset-tiles"

STAINER = "GIVH"
SCANNERS = ("AT2", "GT450")

# Preserve the existing initial inventory as Band 0.
NEW_BANDS = {
    "BAND_1": 3000,
    "BAND_2": 6200,
    "BAND_3": 9400,
    "BAND_4": 12400,
    "BAND_5": 15400,
}

RECORDS_PER_BAND = 700
PAGE_SIZE = 100

WINDOW = 6
MIN_SEPARATION = 6.0
MAX_PER_BAND = 2
MAX_REGIONS = 12

MAX_RESPONSE_BYTES = 400_000
TIMEOUT = (10, 25)
MAX_ATTEMPTS = 3

EXPECTED_SLIDES = {
    "AT2": "GIVH_AT2_to_GMH_S60",
    "GT450": "GIVH_GT450_to_GMH_S60",
}

TILE_PATTERN = re.compile(
    r"^tile_(\d+)_(\d+)_(\d+)$"
)


# ============================================================
# HELPERS
# ============================================================

def parse_tile_id(tile_id):
    match = TILE_PATTERN.fullmatch(str(tile_id))

    if match is None:
        raise ValueError(f"Invalid tile ID: {tile_id}")

    return tuple(map(int, match.groups()))


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

    for attempt in range(1, MAX_ATTEMPTS + 1):

        try:
            with session.get(
                API,
                params=params,
                timeout=TIMEOUT,
                stream=True,
            ) as response:

                response.raise_for_status()

                raw = bytearray()

                for chunk in response.iter_content(16384):

                    raw.extend(chunk)

                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise RuntimeError(
                            "Response size limit exceeded"
                        )

                data = json.loads(raw)

                if not isinstance(data.get("rows"), list):
                    raise ValueError(
                        "Unexpected response structure"
                    )

                return data

        except (
            requests.RequestException,
            ValueError,
        ) as exc:

            print(
                f"    Attempt {attempt}: {exc}",
                flush=True
            )

            if attempt == MAX_ATTEMPTS:
                raise

            time.sleep(attempt)

    raise RuntimeError("API request failed")


def rectangle_distance(a, b):

    dx = max(
        0,
        a["x_start"] - b["x_end"],
        b["x_start"] - a["x_end"],
    )

    dy = max(
        0,
        a["y_start"] - b["y_end"],
        b["y_start"] - a["y_end"],
    )

    return math.hypot(dx, dy)


def region_is_separated(candidate, selected):

    for previous in selected:

        # Coordinate levels cannot be compared directly.
        if candidate["level"] != previous["level"]:
            continue

        if rectangle_distance(
            candidate, previous
        ) < MIN_SEPARATION:
            return False

    return True


# ============================================================
# STEP 1: LOAD EXISTING MATCHED INVENTORY
# ============================================================

print("=" * 70, flush=True)
print("PHASE 3B - DISTRIBUTED REGION SAMPLING", flush=True)
print("=" * 70, flush=True)

print("\n[1] Loading existing metadata...", flush=True)

if not PREVIOUS_INVENTORY.is_file():
    raise SystemExit(
        f"Missing: {PREVIOUS_INVENTORY}"
    )

previous = pd.read_csv(
    PREVIOUS_INVENTORY,
    dtype=str
)

required = {
    "tile_id", "level", "x", "y",
    "at2_slide_id", "gt450_slide_id"
}

if required - set(previous.columns):
    raise ValueError(
        "Previous inventory has missing columns"
    )

matched_inventory = {}

for row in previous.itertuples(index=False):

    tile_id = str(row.tile_id)

    level, x, y = parse_tile_id(tile_id)

    if (
        level != int(row.level)
        or x != int(row.x)
        or y != int(row.y)
    ):
        raise ValueError(
            f"Coordinate mismatch: {tile_id}"
        )

    if (
        row.at2_slide_id != EXPECTED_SLIDES["AT2"]
        or row.gt450_slide_id != EXPECTED_SLIDES["GT450"]
    ):
        raise ValueError(
            "Unexpected scanner acquisition"
        )

    if tile_id in matched_inventory:
        raise ValueError("Duplicate previous tile ID")

    matched_inventory[tile_id] = {
        "tile_id": tile_id,
        "level": level,
        "x": x,
        "y": y,
        "band": "BAND_0",
        "at2_slide_id": row.at2_slide_id,
        "gt450_slide_id": row.gt450_slide_id,
    }

print(f"Previously matched tiles: {len(matched_inventory)}")


# ============================================================
# STEP 2: RETRIEVE DISTRIBUTED API BANDS
# ============================================================

print("\n[2] Retrieving distributed metadata bands...", flush=True)

band_summaries = []

with requests.Session() as session:

    for band_name, start_offset in NEW_BANDS.items():

        print(
            f"\n{band_name} | start offset={start_offset}",
            flush=True
        )

        scanner_data = {}

        for scanner in SCANNERS:

            records = {}
            reported_total = None

            for local_offset in range(
                0, RECORDS_PER_BAND, PAGE_SIZE
            ):

                offset = start_offset + local_offset

                data = fetch_page(
                    session,
                    scanner,
                    offset
                )

                if data.get("num_rows_total") is not None:
                    reported_total = int(
                        data["num_rows_total"]
                    )

                for item in data["rows"]:

                    row = item.get("row", {})

                    if (
                        row.get("stainer") != STAINER
                        or row.get("scanner") != scanner
                    ):
                        raise ValueError(
                            "Scanner/stainer mismatch"
                        )

                    tile_id = str(
                        row.get("tile_id", "")
                    )

                    level, x, y = parse_tile_id(
                        tile_id
                    )

                    slide_id = str(
                        row.get("slide_id", "")
                    )

                    if slide_id != EXPECTED_SLIDES[scanner]:
                        raise ValueError(
                            f"Unexpected slide ID: {slide_id}"
                        )

                    if tile_id in records:
                        raise ValueError(
                            f"Duplicate tile ID in {band_name}"
                        )

                    records[tile_id] = {
                        "level": level,
                        "x": x,
                        "y": y,
                        "slide_id": slide_id,
                    }

                print(
                    f"  {scanner}: "
                    f"{len(records)} records",
                    flush=True
                )

                if len(data["rows"]) < PAGE_SIZE:
                    break

                time.sleep(0.1)

            scanner_data[scanner] = records

        ids_a = set(scanner_data["AT2"])
        ids_b = set(scanner_data["GT450"])

        shared = ids_a & ids_b

        print(
            f"  Matched tile IDs: {len(shared)}",
            flush=True
        )

        added = 0

        for tile_id in sorted(shared):

            at2 = scanner_data["AT2"][tile_id]
            gt450 = scanner_data["GT450"][tile_id]

            if (
                at2["level"], at2["x"], at2["y"]
            ) != (
                gt450["level"], gt450["x"], gt450["y"]
            ):
                raise ValueError(
                    "Scanner coordinate mismatch"
                )

            if tile_id in matched_inventory:
                continue

            matched_inventory[tile_id] = {
                "tile_id": tile_id,
                "level": at2["level"],
                "x": at2["x"],
                "y": at2["y"],
                "band": band_name,
                "at2_slide_id": at2["slide_id"],
                "gt450_slide_id": gt450["slide_id"],
            }

            added += 1

        band_summaries.append({
            "band": band_name,
            "start_offset": start_offset,
            "at2_records": len(ids_a),
            "gt450_records": len(ids_b),
            "matched_ids": len(shared),
            "new_unique_matched_ids": added,
            "at2_only_ids": len(ids_a - ids_b),
            "gt450_only_ids": len(ids_b - ids_a),
        })

        print(
            f"  New unique positions: {added}",
            flush=True
        )


# ============================================================
# STEP 3: BUILD CANDIDATE REGIONS PER BAND
# ============================================================

print("\n[3] Generating 6x6 candidate windows...", flush=True)

positions_by_band = defaultdict(set)

for record in matched_inventory.values():

    key = (
        record["band"],
        record["level"]
    )

    positions_by_band[key].add(
        (record["x"], record["y"])
    )

candidates = []

for (band, level), positions in sorted(
    positions_by_band.items()
):

    for x0, y0 in sorted(positions):

        x1 = x0 + WINDOW - 1
        y1 = y0 + WINDOW - 1

        # Prevent overlap with original Phase 1 pilot.
        if level == 16 and (
            x0 <= 109 and x1 >= 100
            and y0 <= 135 and y1 >= 126
        ):
            continue

        footprint = {
            (x, y)
            for x in range(x0, x1 + 1)
            for y in range(y0, y1 + 1)
        }

        if not footprint.issubset(positions):
            continue

        candidates.append({
            "band": band,
            "level": level,
            "x_start": x0,
            "y_start": y0,
            "x_end": x1,
            "y_end": y1,
            "matched_positions": WINDOW * WINDOW,
        })

print(f"Fully populated candidate windows: {len(candidates)}")


# ============================================================
# STEP 4: SELECT SPATIALLY DISTRIBUTED REGIONS
# ============================================================

print("\n[4] Selecting distributed regions...", flush=True)

selected = []
band_counts = defaultdict(int)

# Round-robin selection gives different bands a
# chance to contribute before selecting a second
# region from any single band.
bands = ["BAND_0", *NEW_BANDS.keys()]

for selection_round in range(MAX_PER_BAND):

    for band in bands:

        if len(selected) >= MAX_REGIONS:
            break

        pool = [
            c for c in candidates
            if c["band"] == band
            and c not in selected
            and region_is_separated(c, selected)
        ]

        if not pool:
            continue

        # Prefer the most distant candidate from
        # already selected regions.
        if selected:

            def distance_score(candidate):
                distances = [
                    rectangle_distance(candidate, other)
                    for other in selected
                    if candidate["level"] == other["level"]
                ]
                return min(distances) if distances else float("inf")

            chosen = max(
                pool,
                key=lambda c: (
                    distance_score(c),
                    -c["x_start"],
                    -c["y_start"]
                )
            )

        else:
            chosen = min(
                pool,
                key=lambda c: (
                    c["x_start"],
                    c["y_start"]
                )
            )

        selected.append(chosen)
        band_counts[band] += 1

print(f"Selected regions: {len(selected)}")


# ============================================================
# STEP 5: CREATE EXACT TILE MANIFEST
# ============================================================

print("\n[5] Creating region tile manifest...", flush=True)

region_records = []
manifest_records = []

for index, region in enumerate(selected, 1):

    region_id = f"OWKIN_DISTRIBUTED_{index:03d}"

    region_records.append({
        "region_id": region_id,
        **region,
        "tissue_type": "UNVERIFIED",
        "specimen_id": "UNVERIFIED",
    })

    for x in range(
        region["x_start"],
        region["x_end"] + 1
    ):

        for y in range(
            region["y_start"],
            region["y_end"] + 1
        ):

            tile_id = (
                f"tile_{region['level']}_{x}_{y}"
            )

            if tile_id not in matched_inventory:
                raise RuntimeError(
                    f"Missing matched tile: {tile_id}"
                )

            original = matched_inventory[tile_id]

            # Every region is drawn from one
            # verified paired acquisition.
            manifest_records.append({
                "region_id": region_id,
                "band": region["band"],
                "tile_id": tile_id,
                "level": region["level"],
                "x": x,
                "y": y,
                "at2_slide_id":
                    original["at2_slide_id"],
                "gt450_slide_id":
                    original["gt450_slide_id"],
            })

regions_df = pd.DataFrame(region_records)
manifest_df = pd.DataFrame(manifest_records)

if (
    not manifest_df.empty
    and manifest_df["tile_id"].duplicated().any()
):
    raise RuntimeError(
        "Duplicate tile across selected regions"
    )


# ============================================================
# STEP 6: VERIFY PAIRWISE DISTANCES
# ============================================================

print("\n[6] Checking separation...", flush=True)

distance_records = []

for a, b in combinations(
    range(len(selected)), 2
):

    region_a = selected[a]
    region_b = selected[b]

    if region_a["level"] != region_b["level"]:
        distance = None
    else:
        distance = rectangle_distance(
            region_a, region_b
        )

    if distance is not None and distance < MIN_SEPARATION:
        raise RuntimeError(
            "Region separation validation failed"
        )

    distance_records.append({
        "region_a": f"OWKIN_DISTRIBUTED_{a+1:03d}",
        "region_b": f"OWKIN_DISTRIBUTED_{b+1:03d}",
        "distance_grid_units": distance,
    })

distance_df = pd.DataFrame(
    distance_records,
    columns=[
        "region_a",
        "region_b",
        "distance_grid_units",
    ]
)

minimum_distance = (
    float(distance_df["distance_grid_units"].min())
    if not distance_df.empty
    else None
)

print(f"Region pairs checked: {len(distance_df)}")
print(f"Minimum grid separation: {minimum_distance}")


# ============================================================
# STEP 7: SAVE RESULTS
# ============================================================

print("\n[7] Saving sampling outputs...", flush=True)

inventory_path = (
    TABLES / "plism_distributed_matched_tiles.csv"
)

regions_path = (
    TABLES / "plism_distributed_region_shortlist.csv"
)

manifest_path = (
    TABLES / "plism_distributed_tile_manifest.csv"
)

distance_path = (
    TABLES / "plism_distributed_region_distances.csv"
)

report_path = (
    REPORTS / "07_distributed_region_sampling.json"
)

pd.DataFrame(
    matched_inventory.values()
).to_csv(
    inventory_path,
    index=False
)

regions_df.to_csv(regions_path, index=False)
manifest_df.to_csv(manifest_path, index=False)
distance_df.to_csv(distance_path, index=False)

report = {
    "phase": "3B",
    "step": "3B.7",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset": DATASET,
    "stainer": STAINER,
    "initial_matched_tiles": len(previous),
    "new_sample_bands": band_summaries,
    "total_unique_matched_tiles":
        len(matched_inventory),
    "fully_populated_candidate_windows":
        len(candidates),
    "selected_regions": len(selected),
    "regions_per_band": dict(band_counts),
    "selected_matched_tile_positions":
        len(manifest_df),
    "potential_paired_images":
        2 * len(manifest_df),
    "minimum_grid_separation":
        minimum_distance,
    "biological_independence_verified": False,
    "original_wsi_mapping_verified": False,
    "full_dataset_inventory": False,
    "images_downloaded": 0,
    "limitations": [
        "Metadata collected from selected API bands",
        "Not a random or exhaustive dataset sample",
        "A sampling band does not identify a biological specimen",
        "No tissue labels transferred from original PLISM",
        "No specimen-level independence established",
        "Grid separation is not verified physical separation"
    ]
}

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3B.7 - DISTRIBUTED SAMPLING COMPLETED")
print("=" * 70)

print(f"Unique matched tiles: {len(matched_inventory)}")
print(f"Full 6x6 windows: {len(candidates)}")
print(f"Selected regions: {len(selected)}")
print(f"Matched selected positions: {len(manifest_df)}")
print(f"Potential images: {2 * len(manifest_df)}")
print(f"Minimum grid separation: {minimum_distance}")

print("\nRegions per sampling band:")
for band in bands:
    print(f"  {band}: {band_counts[band]}")

if not regions_df.empty:
    print("\nFinal provisional region shortlist:")
    print(regions_df.to_string(index=False))

print("\nSaved:")
print(inventory_path)
print(regions_path)
print(manifest_path)
print(distance_path)
print(report_path)

print("\nImage downloads: NONE")
print("Biological independence: UNVERIFIED")
print("Dataset-wide inventory: INCOMPLETE")

print("\nSTEP 3B.7 COMPLETED")
