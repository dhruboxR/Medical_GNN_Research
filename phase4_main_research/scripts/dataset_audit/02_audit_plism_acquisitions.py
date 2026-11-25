
"""
PHASE 4.1.2 - PLISM ACQUISITION INVENTORY

Discover available PLISM acquisition identifiers from
the official Owkin companion repository.

This is a metadata-only inventory, not a tile download.

Output:
- CSV of acquisition filenames and parsed identifiers
- CSV of acquisition-condition groups
- JSON provenance and inventory report

Important:
Filename parsing is heuristic.
Acquisition identity is not biological independence.
"""

from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
from urllib.request import Request, urlopen
import csv
import hashlib
import json
import re

ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "phase4_main_research"

META = BASE / "datasets/metadata"
TABLES = BASE / "outputs/tables"
REPORTS = BASE / "outputs/reports"

PREVIOUS = REPORTS / "01_phase4_dataset_inventory.json"

ACQUISITIONS = TABLES / "02_plism_acquisitions.csv"
GROUPS = TABLES / "02_plism_acquisition_groups.csv"
REPORT = REPORTS / "02_plism_acquisition_audit.json"
RAW_META = META / "02_companion_repo_metadata.json"

DATASET_ID = "owkin/plism-dataset"

URL = (
    "https://huggingface.co/api/datasets/"
    + DATASET_ID
    + "?blobs=true"
)

EXPECTED_TILE_COUNT_PER_ACQUISITION = 16278

OUTPUTS = (ACQUISITIONS, GROUPS, REPORT, RAW_META)

for path in OUTPUTS:
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite previous output: {path}"
        )

if not PREVIOUS.is_file():
    raise RuntimeError(
        "Step 4.1.1 must be completed first."
    )

with PREVIOUS.open(encoding="utf-8") as f:
    earlier = json.load(f)

if not earlier.get("remote_metadata_retrieved"):
    raise RuntimeError(
        "Step 4.1.1 remote metadata was not retrieved."
    )

for folder in (META, TABLES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)


def fetch_json(url):
    request = Request(
        url,
        headers={
            "User-Agent": "MedicalGNNResearch/Phase4-Audit",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=90) as response:
        return json.load(response)


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def parse_slide_id(slide_id):
    """
    Example:
    GIVH_AT2_to_GMH_S60

    Splits first at '_to_', then interprets the
    last underscore-separated tokens as device names.

    These are tentative filename-derived fields.
    """
    if "_to_" not in slide_id:
        return {
            "source_condition": "",
            "source_scanner_token": "",
            "reference_condition": "",
            "reference_scanner_token": "",
            "parse_status": "unparsed",
        }

    source, target = slide_id.split("_to_", 1)

    source_parts = source.rsplit("_", 1)
    target_parts = target.rsplit("_", 1)

    if len(source_parts) != 2 or len(target_parts) != 2:
        return {
            "source_condition": "",
            "source_scanner_token": "",
            "reference_condition": "",
            "reference_scanner_token": "",
            "parse_status": "unparsed",
        }

    return {
        "source_condition": source_parts[0],
        "source_scanner_token": source_parts[1],
        "reference_condition": target_parts[0],
        "reference_scanner_token": target_parts[1],
        "parse_status": "heuristic",
    }


def write_csv(path, rows, columns):
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


print("=" * 70, flush=True)
print("PHASE 4.1.2 - PLISM ACQUISITION AUDIT", flush=True)
print("=" * 70, flush=True)

print("\n[1] Fetching companion repository metadata...", flush=True)

remote = fetch_json(URL)

siblings = remote.get("siblings") or []

print("Repository:", remote.get("id"))
print("Revision:", remote.get("sha"))
print("Repository entries:", len(siblings))

# A companion HDF5 file normally has a name such as
# GIVH_AT2_to_GMH_S60.tif.h5.
# We do not download or open the HDF5 payload.

h5_files = sorted(
    {
        item["rfilename"]
        for item in siblings
        if str(item.get("rfilename", "")).lower().endswith(".h5")
    }
)

if not h5_files:
    raise RuntimeError(
        "No HDF5 acquisitions found in repository metadata. "
        "Inspect the repository tree before proceeding."
    )

print("HDF5 acquisition files:", len(h5_files))

records = []

for filename in h5_files:
    base_name = Path(filename).name

    slide_id = re.sub(
        r"(?i)(?:\.tif|\.tiff)?\.h5$",
        "",
        base_name,
    )

    parsed = parse_slide_id(slide_id)

    records.append({
        "slide_id": slide_id,
        "source_file": filename,
        **parsed,
        "published_tile_count": (
            EXPECTED_TILE_COUNT_PER_ACQUISITION
        ),
        "biologically_independent": "unverified",
    })

if len(set(r["slide_id"] for r in records)) != len(records):
    raise RuntimeError(
        "Duplicate slide identifiers after filename parsing."
    )

print("\n[2] Summarizing acquisition conditions...", flush=True)

source_scanners = Counter(
    r["source_scanner_token"]
    for r in records
    if r["parse_status"] == "heuristic"
)

source_conditions = Counter(
    r["source_condition"]
    for r in records
    if r["parse_status"] == "heuristic"
)

grouped = defaultdict(list)

for record in records:
    if record["parse_status"] != "heuristic":
        continue

    key = (
        record["source_condition"],
        record["reference_condition"],
        record["reference_scanner_token"],
    )

    grouped[key].append(record)

group_rows = []

for key, members in sorted(grouped.items()):
    scanners = sorted({
        r["source_scanner_token"]
        for r in members
    })

    group_rows.append({
        "source_condition": key[0],
        "reference_condition": key[1],
        "reference_scanner_token": key[2],
        "acquisition_count": len(members),
        "scanner_tokens": ";".join(scanners),
        "slide_ids": ";".join(
            sorted(r["slide_id"] for r in members)
        ),
        "contains_at2": "AT2" in scanners,
        "contains_gt450": "GT450" in scanners,
    })

print("\nSource scanner tokens:")
for name, count in source_scanners.most_common():
    print(f"  {name}: {count}")

print("\nSource condition tokens:")
for name, count in source_conditions.most_common():
    print(f"  {name}: {count}")

print("\nLargest acquisition groups:")
for group in sorted(
    group_rows,
    key=lambda r: r["acquisition_count"],
    reverse=True,
)[:15]:
    print(
        " ",
        group["source_condition"],
        "->",
        group["reference_condition"],
        group["reference_scanner_token"],
        "| acquisitions:",
        group["acquisition_count"],
        "| scanners:",
        group["scanner_tokens"],
    )

print("\n[3] Saving evidence...", flush=True)

# Save source response exactly for provenance.
with RAW_META.open("x", encoding="utf-8") as f:
    json.dump(remote, f, indent=2)

write_csv(
    ACQUISITIONS,
    records,
    [
        "slide_id",
        "source_file",
        "source_condition",
        "source_scanner_token",
        "reference_condition",
        "reference_scanner_token",
        "parse_status",
        "published_tile_count",
        "biologically_independent",
    ],
)

write_csv(
    GROUPS,
    group_rows,
    [
        "source_condition",
        "reference_condition",
        "reference_scanner_token",
        "acquisition_count",
        "scanner_tokens",
        "slide_ids",
        "contains_at2",
        "contains_gt450",
    ],
)

report = {
    "phase": "4",
    "step": "4.1.2",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "companion_dataset": DATASET_ID,
    "companion_revision": remote.get("sha"),
    "previous_inventory_sha256": checksum(PREVIOUS),
    "acquisition_files_found": len(h5_files),
    "parsed_acquisitions": sum(
        r["parse_status"] == "heuristic"
        for r in records
    ),
    "unique_source_scanner_tokens": dict(source_scanners),
    "unique_source_condition_tokens": dict(source_conditions),
    "acquisition_groups": len(group_rows),
    "published_tiles_per_acquisition": (
        EXPECTED_TILE_COUNT_PER_ACQUISITION
    ),
    "metadata_only": True,
    "original_pngs_downloaded": 0,
    "scanner_fields_verified_from_tile_rows": False,
    "coordinate_overlap_verified": False,
    "biological_independence_verified": False,
    "next_step": (
        "Confirm candidate slide identifiers and scanner metadata "
        "against the original tile repository, then plan "
        "matched coordinate selection and controlled download."
    ),
}

with REPORT.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 70)
print("PHASE 4.1.2 - ACQUISITION AUDIT COMPLETED")
print("=" * 70)

print("Acquisitions:", len(records))
print("Parsed acquisitions:", report["parsed_acquisitions"])
print("Acquisition groups:", len(group_rows))
print("Unique scanner tokens:", len(source_scanners))
print("Unique condition tokens:", len(source_conditions))

print("\nSaved:")
print(ACQUISITIONS)
print(GROUPS)
print(REPORT)
print(RAW_META)

print("\nImages downloaded: NONE")
print("Original Phase 3B cohort: UNCHANGED")
print("Independent specimens: UNVERIFIED")
print("\nSTEP 4.1.2 COMPLETED")
