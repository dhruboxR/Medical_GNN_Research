
"""
PHASE 4.1.3 - PLISM EXPANSION PLANNING

Build a candidate expansion plan from the verified
91-acquisition inventory.

Design:
A. All seven scanners for reference condition GIVH
B. AT2 and GT450 across all 13 conditions

Union = 31 unique acquisitions.

This is a provisional acquisition selection.
Tile overlap, biological independence and final data
splits must be validated separately.

No images downloaded. No historical data modified.
"""

from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json

ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "phase4_main_research"

TABLES = BASE / "outputs/tables"
REPORTS = BASE / "outputs/reports"
MANIFESTS = BASE / "datasets/manifests"

INPUT = TABLES / "02_plism_acquisitions.csv"
PREVIOUS = REPORTS / "02_plism_acquisition_audit.json"

MATRIX = TABLES / "03_plism_acquisition_matrix.csv"
SELECTED = MANIFESTS / "03_candidate_acquisitions.csv"
REPORT = REPORTS / "03_plism_expansion_plan.json"

REFERENCE_CONDITION = "GIVH"
REFERENCE_SCANNERS = {"AT2", "GT450"}

EXPECTED_SCANNERS = {
    "AT2", "GT450", "P", "S210",
    "S360", "S60", "SQ"
}

EXPECTED_CONDITIONS = {
    "GIVH", "GIV", "GMH", "GM", "GVH", "GV",
    "HRH", "HR", "KRH", "KR", "LMH", "LM", "MY"
}


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)

    return h.hexdigest()


def write_csv_exclusive(path, rows, columns):
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


print("=" * 70)
print("PHASE 4.1.3 - PLISM DATASET EXPANSION PLAN")
print("=" * 70)

print("\n[1] Validating previous acquisition inventory...")

for path in (INPUT, PREVIOUS):
    if not path.is_file():
        raise RuntimeError(f"Missing previous input: {path}")

for path in (MATRIX, SELECTED, REPORT):
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite: {path}")

with PREVIOUS.open(encoding="utf-8") as f:
    previous = json.load(f)

if previous.get("acquisition_files_found") != 91:
    raise RuntimeError("Previous audit did not find 91 acquisitions.")

with INPUT.open(encoding="utf-8", newline="") as f:
    records = list(csv.DictReader(f))

if len(records) != 91:
    raise RuntimeError(f"Expected 91 records, found {len(records)}")

if any(r["parse_status"] != "heuristic" for r in records):
    raise RuntimeError("Unparsed slide identifiers exist.")

keys = [
    (r["source_condition"], r["source_scanner_token"])
    for r in records
]

if len(set(keys)) != 91:
    raise RuntimeError("Duplicate scanner-condition combinations.")

conditions = {condition for condition, _ in keys}
scanners = {scanner for _, scanner in keys}

if conditions != EXPECTED_CONDITIONS:
    raise RuntimeError("Unexpected condition identifiers.")

if scanners != EXPECTED_SCANNERS:
    raise RuntimeError("Unexpected scanner identifiers.")

lookup = {
    (r["source_condition"], r["source_scanner_token"]): r
    for r in records
}

for condition in sorted(conditions):
    for scanner in sorted(scanners):
        if (condition, scanner) not in lookup:
            raise RuntimeError(
                f"Missing combination: {condition}/{scanner}"
            )

print("Complete acquisition matrix: VERIFIED")
print("Conditions:", len(conditions))
print("Scanners:", len(scanners))
print("Acquisitions:", len(records))


print("\n[2] Building the scanner-condition matrix...")

matrix_rows = []

for condition in sorted(conditions):
    row = {"condition": condition}

    for scanner in sorted(scanners):
        row[scanner] = lookup[
            (condition, scanner)
        ]["slide_id"]

    matrix_rows.append(row)


print("\n[3] Selecting candidate acquisitions...")

selected_rows = []

for condition in sorted(conditions):
    for scanner in sorted(scanners):

        cross_scanner = (
            condition == REFERENCE_CONDITION
        )

        cross_condition = (
            scanner in REFERENCE_SCANNERS
        )

        if not (cross_scanner or cross_condition):
            continue

        source = lookup[(condition, scanner)]

        selected_rows.append({
            "slide_id": source["slide_id"],
            "source_condition": condition,
            "scanner_token": scanner,
            "source_file": source["source_file"],
            "cross_scanner_arm": int(cross_scanner),
            "cross_condition_arm": int(cross_condition),
            "selection_status": "provisional",
            "tile_coordinate_overlap_verified": False,
            "slide_metadata_verified": False,
            "independent_specimen_verified": False,
        })

if len(selected_rows) != 31:
    raise RuntimeError(
        f"Expected 31 candidates, found {len(selected_rows)}"
    )

if len({r["slide_id"] for r in selected_rows}) != 31:
    raise RuntimeError("Duplicate acquisition selection.")

print("Candidate acquisition count:", len(selected_rows))
print("Cross-scanner arm: 7 acquisitions")
print("Cross-condition arm: 26 acquisitions")
print("Unique acquisitions: 31")

print("\nSelected scanner coverage:")
for scanner in sorted(scanners):
    count = sum(
        r["scanner_token"] == scanner
        for r in selected_rows
    )
    print(f"  {scanner}: {count}")

print("\nSelected condition coverage:")
for condition in sorted(conditions):
    count = sum(
        r["source_condition"] == condition
        for r in selected_rows
    )
    print(f"  {condition}: {count}")


print("\n[4] Saving provisional research plan...")

for directory in (TABLES, REPORTS, MANIFESTS):
    directory.mkdir(parents=True, exist_ok=True)

write_csv_exclusive(
    MATRIX,
    matrix_rows,
    ["condition"] + sorted(scanners),
)

write_csv_exclusive(
    SELECTED,
    selected_rows,
    [
        "slide_id",
        "source_condition",
        "scanner_token",
        "source_file",
        "cross_scanner_arm",
        "cross_condition_arm",
        "selection_status",
        "tile_coordinate_overlap_verified",
        "slide_metadata_verified",
        "independent_specimen_verified",
    ],
)

report = {
    "phase": "4",
    "step": "4.1.3",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_inventory_sha256": sha256(INPUT),
    "reference_condition": REFERENCE_CONDITION,
    "reference_scanners": sorted(REFERENCE_SCANNERS),
    "total_acquisitions": len(records),
    "conditions": sorted(conditions),
    "scanners": sorted(scanners),
    "candidate_acquisitions": len(selected_rows),
    "cross_scanner_arm": 7,
    "cross_condition_arm": 26,
    "overlap_between_arms": 2,
    "selection_is_provisional": True,
    "tile_metadata_verified": False,
    "coordinate_overlap_verified": False,
    "biological_independence_verified": False,
    "final_data_split_frozen": False,
    "image_downloads_performed": 0,
    "next_step": (
        "Validate the selected slide IDs and paired tile "
        "coordinates against the original PLISM Parquet data."
    ),
    "limitations": [
        "Scanner/condition categories derived from filenames",
        "No tile-level overlap verified",
        "No independent biological samples established",
        "No train/validation/test assignments made",
        "Selected acquisition count is not an image budget",
    ],
}

with REPORT.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 70)
print("PHASE 4.1.3 - EXPANSION PLAN COMPLETED")
print("=" * 70)

print("Original acquisitions:", len(records))
print("Provisional acquisitions:", len(selected_rows))
print("Tile coordinate verification: PENDING")
print("Biological independence: UNVERIFIED")
print("Image downloads: NONE")

print("\nSaved:")
print(MATRIX)
print(SELECTED)
print(REPORT)

print("\nSTEP 4.1.3 COMPLETED")
