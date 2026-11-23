
"""
PHASE 3B - STEP 3B.1
PLISM Multi-Region Sampling Feasibility

Objective:
Identify non-overlapping candidate tissue regions
using the original PLISM WSI annotation metadata.

IMPORTANT:
- Original WSI coordinates are NOT Owkin tile IDs.
- Tissue labels are NOT verified specimen IDs.
- No image registration or biological independence is
  established by this experiment.
- No images or model weights are downloaded.

Outputs:
1. Coordinate validation summary
2. Candidate region inventory
3. Non-overlapping provisional region shortlist
4. JSON feasibility report
"""

from pathlib import Path
from urllib.request import Request, urlopen
from collections import defaultdict
from datetime import datetime, timezone
import json

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"

WORKSPACE = ROOT / "multi_region_research"

TABLE_DIR = WORKSPACE / "outputs" / "tables"
REPORT_DIR = WORKSPACE / "outputs" / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ARTICLE_ID = 23614422
FILENAME = "PLISM_wsi_en.csv"

STAIN = "GIVH"
SCANNERS = ("AT2", "GT450")

GRID_STEP = 1024

WINDOW_SIZE = 6
MIN_ANNOTATED_POSITIONS = 24

MAX_REGIONS = 12
MAX_REGIONS_PER_TISSUE = 2

MAX_SOURCE_BYTES = 25 * 1024 * 1024
CHUNK_SIZE = 10000


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def fetch_annotation_url():

    url = (
        f"https://api.figshare.com/v2/articles/"
        f"{ARTICLE_ID}"
    )

    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "PLISM-Phase3B/1.0"
        }
    )

    with urlopen(request, timeout=60) as response:
        article = json.load(response)

    matches = [
        item
        for item in article.get("files", [])
        if item.get("name") == FILENAME
    ]

    if not matches:
        raise RuntimeError(
            "Official WSI annotation file was not found."
        )

    file_info = matches[0]

    file_size = int(file_info.get("size") or 0)

    if not (0 < file_size <= MAX_SOURCE_BYTES):
        raise RuntimeError(
            f"Unexpected file size: {file_size}"
        )

    download_url = file_info.get("download_url")

    if not download_url:
        raise RuntimeError("Missing annotation URL.")

    return download_url, file_size


def parse_grid_coordinate(value):
    """
    Parse original PLISM WSI coordinate strings
    such as 100352_61440.

    Returns integer grid coordinates only when
    the values match the observed 1024-unit grid.
    """

    parts = str(value).strip().split("_")

    if len(parts) != 2:
        return None

    try:
        x, y = map(int, parts)
    except ValueError:
        return None

    if x < 0 or y < 0:
        return None

    if x % GRID_STEP or y % GRID_STEP:
        return None

    return (x // GRID_STEP, y // GRID_STEP)


def generate_windows(tissue, positions):
    """
    Generate provisional square regions consisting
    of annotations with the same tissue label.
    """

    if not positions:
        return []

    xs = sorted({x for x, y in positions})
    ys = sorted({y for x, y in positions})

    if not xs or not ys:
        return []

    candidates = []

    # Restrict anchors to existing coordinates.
    for x0 in xs:
        for y0 in ys:

            selected = {
                (x, y)
                for x, y in positions
                if x0 <= x < x0 + WINDOW_SIZE
                and y0 <= y < y0 + WINDOW_SIZE
            }

            count = len(selected)

            if count < MIN_ANNOTATED_POSITIONS:
                continue

            coverage = count / (WINDOW_SIZE ** 2)

            candidates.append({
                "tissue_type": tissue,
                "grid_x_start": x0,
                "grid_y_start": y0,
                "grid_x_end": x0 + WINDOW_SIZE - 1,
                "grid_y_end": y0 + WINDOW_SIZE - 1,
                "annotated_positions": count,
                "window_positions": WINDOW_SIZE ** 2,
                "coverage_fraction": coverage,
                "_positions": selected
            })

    return candidates


# ============================================================
# STEP 1: GET ANNOTATION SOURCE
# ============================================================

print("=" * 70, flush=True)
print("PHASE 3B - MULTI-REGION SAMPLING FEASIBILITY", flush=True)
print("=" * 70, flush=True)

print("\n[1] Retrieving official annotation metadata...", flush=True)

download_url, source_size = fetch_annotation_url()

print(f"Annotation file: {FILENAME}")
print(f"Size: {source_size / 1024**2:.2f} MiB")
print("Image downloads: none")


# ============================================================
# STEP 2: STREAM METADATA
# ============================================================

print("\n[2] Reading scanner annotation records...", flush=True)

# Coordinate -> scanner -> tissue label set
annotations = defaultdict(
    lambda: defaultdict(set)
)

# Keep counts to identify duplicate annotations.
row_counts = defaultdict(int)

processed = 0
invalid_coordinates = 0
selected_rows = 0

request = Request(
    download_url,
    headers={
        "Accept-Encoding": "identity",
        "User-Agent": "PLISM-Phase3B/1.0"
    }
)

with urlopen(request, timeout=120) as response:

    content_length = response.headers.get("Content-Length")

    if (
        content_length is not None
        and int(content_length) > MAX_SOURCE_BYTES
    ):
        raise RuntimeError(
            "Response exceeds the configured size limit."
        )

    reader = pd.read_csv(
        response,
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        usecols=[
            "stain",
            "device",
            "tissue_type",
            "coordinate"
        ]
    )

    for chunk_number, chunk in enumerate(reader, 1):

        processed += len(chunk)

        chunk = chunk[
            (chunk["stain"].str.strip() == STAIN)
            &
            (
                chunk["device"].str.strip()
                .isin(SCANNERS)
            )
        ]

        for record in chunk.itertuples(index=False):

            scanner = record.device.strip()
            tissue = record.tissue_type.strip()

            grid = parse_grid_coordinate(
                record.coordinate
            )

            selected_rows += 1

            if grid is None or not tissue:
                invalid_coordinates += 1
                continue

            annotations[grid][scanner].add(tissue)

            row_counts[(grid, scanner)] += 1

        print(
            f"Processed {processed:,} annotation rows",
            flush=True
        )


# ============================================================
# STEP 3: VERIFY SCANNER LABEL PAIRING
# ============================================================

print("\n[3] Validating cross-scanner tissue labels...", flush=True)

tissue_positions = defaultdict(set)

paired_coordinates = 0
conflicting_coordinates = 0
unmatched_coordinates = 0
duplicate_coordinate_groups = sum(
    count > 1 for count in row_counts.values()
)

for grid, scanner_labels in annotations.items():

    if not all(
        scanner in scanner_labels
        for scanner in SCANNERS
    ):
        unmatched_coordinates += 1
        continue

    label_a = scanner_labels["AT2"]
    label_b = scanner_labels["GT450"]

    if (
        len(label_a) != 1
        or len(label_b) != 1
        or label_a != label_b
    ):
        conflicting_coordinates += 1
        continue

    label = next(iter(label_a))

    tissue_positions[label].add(grid)
    paired_coordinates += 1

print(f"Valid paired coordinates: {paired_coordinates}")
print(f"Tissue labels: {len(tissue_positions)}")
print(f"Conflicting coordinates: {conflicting_coordinates}")
print(f"Unmatched coordinates: {unmatched_coordinates}")
print(f"Invalid rows: {invalid_coordinates}")
print(f"Duplicate coordinate groups: {duplicate_coordinate_groups}")

if (
    invalid_coordinates
    or conflicting_coordinates
    or unmatched_coordinates
    or duplicate_coordinate_groups
):
    raise RuntimeError(
        "Coordinate validation was not clean. "
        "Review these issues before sampling."
    )


# ============================================================
# STEP 4: GENERATE REGION CANDIDATES
# ============================================================

print("\n[4] Finding candidate spatial windows...", flush=True)

candidates = []

for tissue in sorted(tissue_positions):

    positions = tissue_positions[tissue]

    windows = generate_windows(
        tissue,
        positions
    )

    candidates.extend(windows)

print(f"Candidate windows: {len(candidates)}")

if not candidates:
    print(
        "No windows met the screening threshold. "
        "Do not lower the threshold without reviewing "
        "the tissue coordinate distribution."
    )


# ============================================================
# STEP 5: SELECT NON-OVERLAPPING REGIONS
# ============================================================

print("\n[5] Selecting provisional candidate regions...", flush=True)

# Prioritize:
# 1. More annotated positions
# 2. Higher coverage
# 3. Deterministic tissue/coordinate order

candidates.sort(
    key=lambda item: (
        -item["annotated_positions"],
        -item["coverage_fraction"],
        item["tissue_type"],
        item["grid_x_start"],
        item["grid_y_start"]
    )
)

selected = []
used_positions = set()
used_tissues = defaultdict(int)

# First pass: seek different tissue labels.
# Second pass: fill remaining slots without overlap.

for diversity_pass in (True, False):

    for candidate in candidates:

        if len(selected) >= MAX_REGIONS:
            break

        tissue = candidate["tissue_type"]
        positions = candidate["_positions"]

        if used_tissues[tissue] >= MAX_REGIONS_PER_TISSUE:
            continue

        if diversity_pass and used_tissues[tissue] > 0:
            continue

        # Prevent overlapping full window footprints,
        # including positions that are not annotated.
        footprint = {
            (x, y)
            for x in range(
                candidate["grid_x_start"],
                candidate["grid_x_end"] + 1
            )
            for y in range(
                candidate["grid_y_start"],
                candidate["grid_y_end"] + 1
            )
        }

        if footprint & used_positions:
            continue

        used_positions.update(footprint)
        used_tissues[tissue] += 1

        selected.append(candidate)

    if len(selected) >= MAX_REGIONS:
        break


# ============================================================
# STEP 6: SAVE REGION INVENTORIES
# ============================================================

print("\n[6] Saving candidate inventories...", flush=True)

public_fields = [
    "tissue_type",
    "grid_x_start",
    "grid_y_start",
    "grid_x_end",
    "grid_y_end",
    "annotated_positions",
    "window_positions",
    "coverage_fraction"
]

all_candidates_df = pd.DataFrame(
    [
        {key: item[key] for key in public_fields}
        for item in candidates
    ],
    columns=public_fields
)

selected_records = []

for index, item in enumerate(selected, 1):

    record = {
        "region_id": f"REGION_{index:03d}",
        **{
            key: item[key]
            for key in public_fields
        },
        "source_coordinate_system": "Original PLISM WSI",
        "owkin_tile_mapping_verified": False,
        "independent_specimen_verified": False,
    }

    selected_records.append(record)

selected_df = pd.DataFrame(
    selected_records,
    columns=[
        "region_id",
        *public_fields,
        "source_coordinate_system",
        "owkin_tile_mapping_verified",
        "independent_specimen_verified"
    ]
)

all_path = (
    TABLE_DIR / "plism_region_candidates.csv"
)

selected_path = (
    TABLE_DIR / "plism_provisional_region_shortlist.csv"
)

all_candidates_df.to_csv(all_path, index=False)
selected_df.to_csv(selected_path, index=False)


# ============================================================
# STEP 7: FEASIBILITY REPORT
# ============================================================

print("\n[7] Generating feasibility report...", flush=True)

report = {
    "phase": "3B",
    "step": "3B.1",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "source": FILENAME,
    "stain": STAIN,
    "scanners": list(SCANNERS),
    "source_records_processed": processed,
    "target_scanner_annotation_rows": selected_rows,
    "valid_paired_coordinates": paired_coordinates,
    "tissue_labels": len(tissue_positions),
    "conflicting_coordinates": conflicting_coordinates,
    "unmatched_coordinates": unmatched_coordinates,
    "invalid_annotation_rows": invalid_coordinates,
    "duplicate_coordinate_groups": duplicate_coordinate_groups,
    "grid_step": GRID_STEP,
    "window_size": WINDOW_SIZE,
    "minimum_annotated_positions": MIN_ANNOTATED_POSITIONS,
    "candidate_windows": len(candidates),
    "selected_provisional_regions": len(selected),
    "selected_tissue_labels": len({
        item["tissue_type"] for item in selected
    }),
    "images_downloaded": 0,
    "owkin_tile_coordinate_mapping_verified": False,
    "biological_specimen_independence_verified": False,
    "ready_to_extract_new_owkin_tiles": False,
    "limitations": [
        "Original WSI coordinates are not mapped to Owkin tiles",
        "Tissue labels are not verified patient or specimen IDs",
        "Non-overlapping windows do not prove independent specimens",
        "The 1024-unit grid interpretation is provisional",
        "Candidate screening does not validate tissue-core boundaries",
        "Candidate feasibility does not establish image registration"
    ]
}

report_path = (
    REPORT_DIR / "01_multi_region_sampling_feasibility.json"
)

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3B.1 - SAMPLING FEASIBILITY COMPLETED")
print("=" * 70)

print(f"Paired coordinates: {paired_coordinates}")
print(f"Available tissue labels: {len(tissue_positions)}")
print(f"Candidate windows: {len(candidates)}")
print(f"Provisional regions selected: {len(selected)}")
print(f"Distinct selected tissue labels: {report['selected_tissue_labels']}")

if not selected_df.empty:
    print("\nProvisional region shortlist:")

    print(
        selected_df[
            [
                "region_id",
                "tissue_type",
                "annotated_positions",
                "coverage_fraction",
                "grid_x_start",
                "grid_y_start"
            ]
        ].to_string(index=False)
    )

print("\nSaved:")
print(all_path)
print(selected_path)
print(report_path)

print("\nOwkin coordinate mapping: UNVERIFIED")
print("Independent biological specimens: UNVERIFIED")
print("Image downloads: NONE")
print("\nSTEP 3B.1 COMPLETED")

