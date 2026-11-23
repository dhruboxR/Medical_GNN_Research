
"""
Phase 3A - Step 3.8
PLISM WSI Coordinate-Level Tissue Annotation Consistency

Checks:
1. Coordinate overlap between GIVH AT2 and GT450
2. Tissue label consistency at corresponding coordinates
3. Duplicate and conflicting coordinate annotations
4. Whether tissue labels can serve as provisional
   grouping information

Does NOT map coordinates to Owkin tiles.
Does NOT download images or save the full annotation CSV.
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

ROOT = Path.cwd()

OUTPUT = ROOT / "experimental_research" / "outputs"
TABLE_DIR = OUTPUT / "tables"
REPORT_DIR = OUTPUT / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ARTICLE_ID = 23614422
FILENAME = "PLISM_wsi_en.csv"

STAIN = "GIVH"
SCANNERS = ("AT2", "GT450")

CHUNK_SIZE = 10000
MAX_FILE_BYTES = 25 * 1024 * 1024

print("=" * 70, flush=True)
print("PHASE 3A - WSI COORDINATE LABEL CONSISTENCY")
print("=" * 70, flush=True)


# ============================================================
# STEP 1: GET OFFICIAL ANNOTATION FILE INFORMATION
# ============================================================

print("\n[1] Retrieving official WSI file information...",
      flush=True)

api_url = (
    f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
)

request = Request(
    api_url,
    headers={
        "Accept": "application/json",
        "User-Agent": "PLISM-Coordinate-Audit/1.0"
    }
)

with urlopen(request, timeout=60) as response:
    article = json.load(response)

files = [
    item for item in article.get("files", [])
    if item.get("name") == FILENAME
]

if not files:
    raise SystemExit("WSI annotation file not found.")

file_info = files[0]
file_size = int(file_info.get("size") or 0)
download_url = file_info.get("download_url")

if not (0 < file_size <= MAX_FILE_BYTES):
    raise SystemExit("Annotation file exceeds allowed size.")

if not download_url:
    raise SystemExit("Missing official download URL.")

print(f"Source: {FILENAME}")
print(f"Annotation size: {file_size / 1024**2:.2f} MiB")


# ============================================================
# STEP 2: INITIALIZE COORDINATE MAPS
# ============================================================

print("\n[2] Preparing scanner coordinate maps...")

# Each coordinate maps to a set of tissue labels.
coordinate_labels = {
    scanner: defaultdict(set)
    for scanner in SCANNERS
}

coordinate_counts = {
    scanner: defaultdict(int)
    for scanner in SCANNERS
}

scanner_rows = {scanner: 0 for scanner in SCANNERS}

total_rows = 0


# ============================================================
# STEP 3: READ OFFICIAL WSI ANNOTATIONS
# ============================================================

print("\n[3] Processing original WSI annotations...",
      flush=True)

csv_request = Request(
    download_url,
    headers={
        "Accept-Encoding": "identity",
        "User-Agent": "PLISM-Coordinate-Audit/1.0"
    }
)

with urlopen(csv_request, timeout=120) as response:

    content_length = response.headers.get("Content-Length")

    if (
        content_length is not None
        and int(content_length) > MAX_FILE_BYTES
    ):
        raise SystemExit("Response exceeds allowed size.")

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

        total_rows += len(chunk)

        filtered = chunk[
            (chunk["stain"].str.strip() == STAIN)
            &
            (chunk["device"].str.strip().isin(SCANNERS))
        ]

        for row in filtered.itertuples(index=False):

            scanner = row.device.strip()
            coord = row.coordinate.strip()
            label = row.tissue_type.strip()

            if not coord:
                continue

            scanner_rows[scanner] += 1
            coordinate_labels[scanner][coord].add(label)
            coordinate_counts[scanner][coord] += 1

        print(
            f"Chunk {chunk_number}: "
            f"{total_rows:,} rows processed",
            flush=True
        )


# ============================================================
# STEP 4: IDENTIFY SHARED AND UNMATCHED COORDINATES
# ============================================================

print("\n[4] Comparing coordinate inventories...")

a_coords = set(coordinate_labels["AT2"])
b_coords = set(coordinate_labels["GT450"])

shared = a_coords & b_coords
only_a = a_coords - b_coords
only_b = b_coords - a_coords

print(f"AT2 coordinates: {len(a_coords)}")
print(f"GT450 coordinates: {len(b_coords)}")
print(f"Shared coordinates: {len(shared)}")
print(f"AT2-only coordinates: {len(only_a)}")
print(f"GT450-only coordinates: {len(only_b)}")


# ============================================================
# STEP 5: VERIFY LABEL AGREEMENT
# ============================================================

print("\n[5] Checking coordinate-level tissue labels...")

records = []

for coord in sorted(a_coords | b_coords):

    labels_a = coordinate_labels["AT2"].get(coord, set())
    labels_b = coordinate_labels["GT450"].get(coord, set())

    present_both = coord in shared

    exact_agreement = (
        present_both
        and labels_a == labels_b
        and len(labels_a) == 1
        and len(labels_b) == 1
        and "" not in labels_a
        and "" not in labels_b
    )

    records.append({
        "coordinate": coord,
        "at2_present": coord in a_coords,
        "gt450_present": coord in b_coords,
        "at2_labels": ";".join(sorted(labels_a)),
        "gt450_labels": ";".join(sorted(labels_b)),
        "at2_annotation_count":
            coordinate_counts["AT2"].get(coord, 0),
        "gt450_annotation_count":
            coordinate_counts["GT450"].get(coord, 0),
        "exact_single_label_agreement": exact_agreement,
        "annotation_conflict": (
            present_both and not exact_agreement
        ),
        "coordinate_system": "ORIGINAL_PLISM_WSI",
        "owkin_tile_mapping_verified": False
    })

results = pd.DataFrame(records)

agreement_count = int(
    results["exact_single_label_agreement"].sum()
)

conflict_count = int(
    results["annotation_conflict"].sum()
)

duplicates_a = sum(
    count > 1
    for count in coordinate_counts["AT2"].values()
)

duplicates_b = sum(
    count > 1
    for count in coordinate_counts["GT450"].values()
)

print(f"Single-label agreements: {agreement_count}")
print(f"Conflicting shared coordinates: {conflict_count}")
print(f"AT2 duplicate coordinate groups: {duplicates_a}")
print(f"GT450 duplicate coordinate groups: {duplicates_b}")

agreement_rate = (
    agreement_count / len(shared)
    if shared else 0.0
)

print(
    f"Shared-coordinate label agreement: "
    f"{agreement_rate:.4%}"
)


# ============================================================
# STEP 6: SAVE RESULTS
# ============================================================

print("\n[6] Saving consistency assessment...")

table_path = (
    TABLE_DIR / "plism_givh_coordinate_label_consistency.csv"
)

results.to_csv(table_path, index=False)

report = {
    "phase": "3A",
    "step": "3.8",
    "generated_utc":
        datetime.now(timezone.utc).isoformat(),
    "annotation_source": FILENAME,
    "stain": STAIN,
    "at2_annotation_rows": scanner_rows["AT2"],
    "gt450_annotation_rows": scanner_rows["GT450"],
    "at2_unique_coordinates": len(a_coords),
    "gt450_unique_coordinates": len(b_coords),
    "shared_coordinates": len(shared),
    "at2_only_coordinates": len(only_a),
    "gt450_only_coordinates": len(only_b),
    "single_label_agreements": agreement_count,
    "conflicting_shared_coordinates": conflict_count,
    "shared_coordinate_label_agreement": agreement_rate,
    "at2_duplicate_coordinate_groups": duplicates_a,
    "gt450_duplicate_coordinate_groups": duplicates_b,
    "original_wsi_coordinates_consistent": (
        len(shared) > 0
        and len(only_a) == 0
        and len(only_b) == 0
        and conflict_count == 0
    ),
    "owkin_tile_mapping_verified": False,
    "independent_biological_specimens_verified": False
}

report_path = (
    REPORT_DIR / "08_wsi_coordinate_label_consistency.json"
)

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("WSI COORDINATE LABEL AUDIT COMPLETED")
print("=" * 70)

print(f"Shared coordinates: {len(shared)}")
print(f"Exact label agreements: {agreement_count}")
print(f"Label conflicts: {conflict_count}")
print(f"Agreement rate: {agreement_rate:.4%}")

print("Owkin coordinate mapping: UNVERIFIED")
print("Biological independence: UNVERIFIED")

print("\nSaved outputs:")
print(table_path)
print(report_path)

print("\nSTEP 3.8 COMPLETED")
