
"""
PHASE 4.1.1 - DATASET INVENTORY

Goals:
1. Verify the frozen Phase 3B cohort.
2. Audit local research storage.
3. Retrieve public Owkin PLISM dataset metadata.
4. Inventory published dataset repository files.
5. Save evidence for planning dataset expansion.

This script does NOT:
- Download image tiles
- Modify Phase 3B artifacts
- Train models
- Assume independent biological samples

Requires Python 3.10+.
Uses only the Python standard library.
"""

from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import csv
import hashlib
import json
import shutil
import sys


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"

PHASE3 = ROOT / "multi_region_research"
PHASE4 = ROOT / "phase4_main_research"

DATA_ROOT = ROOT / "datasets" / "PLISM"

MANIFEST = (
    PHASE3
    / "outputs/validated"
    / "phase3b_frozen_tile_manifest.csv"
)

FEATURE_ARCHIVE = (
    PHASE3
    / "outputs/features"
    / "plism_phase3b_dinov2_features.npz"
)

OLD_IMAGES = (
    DATA_ROOT
    / "phase3b_original_png"
)

REPORT_DIR = PHASE4 / "outputs/reports"
TABLE_DIR = PHASE4 / "outputs/tables"
METADATA_DIR = PHASE4 / "datasets/metadata"

REPORT_PATH = (
    REPORT_DIR
    / "01_phase4_dataset_inventory.json"
)

SHARD_PATH = (
    TABLE_DIR
    / "01_plism_published_file_inventory.csv"
)

REMOTE_METADATA_PATH = (
    METADATA_DIR
    / "owkin_plism_tiles_dataset_info.json"
)

EXPECTED_MANIFEST_SHA256 = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)

DATASET_ID = "owkin/plism-dataset-tiles"

API_URL = (
    "https://huggingface.co/api/datasets/"
    + DATASET_ID
    + "?blobs=true"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def gib(num_bytes):
    return round(
        num_bytes / (1024 ** 3),
        3,
    )


def fetch_json(url):
    request = Request(
        url,
        headers={
            "User-Agent": (
                "MedicalGNNResearch-Phase4-DatasetAudit/1.0"
            ),
            "Accept": "application/json",
        },
    )

    with urlopen(
        request,
        timeout=60,
    ) as response:
        return json.load(response)


def write_json_exclusive(path, payload):
    with path.open(
        "x",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )


def read_csv_rows(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def image_inventory(folder):
    """
    Audit existing PNG files only.
    Does not recalculate expensive image hashes.
    """

    result = {}

    for scanner in ("AT2", "GT450"):
        scanner_dir = folder / scanner

        files = (
            list(scanner_dir.glob("*.png"))
            if scanner_dir.exists()
            else []
        )

        result[scanner] = {
            "directory": str(scanner_dir),
            "exists": scanner_dir.is_dir(),
            "png_count": len(files),
            "total_gib": gib(
                sum(
                    path.stat().st_size
                    for path in files
                )
            ),
        }

    return result


def extract_file_size(item):
    """
    Hugging Face repository metadata may expose
    file size either directly or through LFS info.
    """

    size = item.get("size")

    if isinstance(size, (int, float)):
        return int(size)

    lfs = item.get("lfs") or {}

    size = lfs.get("size")

    if isinstance(size, (int, float)):
        return int(size)

    return None


# ============================================================
# STEP 1: SAFETY CHECKS
# ============================================================

print("=" * 70, flush=True)
print(
    "PHASE 4.1.1 - PLISM DATASET INVENTORY",
    flush=True,
)
print("=" * 70, flush=True)

print(
    "\n[1] Checking research workspace...",
    flush=True,
)

for path in (
    MANIFEST,
    FEATURE_ARCHIVE,
):
    if not path.is_file():
        raise RuntimeError(
            f"Required historical input missing: {path}"
        )

for folder in (
    REPORT_DIR,
    TABLE_DIR,
    METADATA_DIR,
):
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

for path in (
    REPORT_PATH,
    SHARD_PATH,
    REMOTE_METADATA_PATH,
):
    if path.exists():
        raise RuntimeError(
            "Refusing to overwrite a prior audit:\n"
            f"{path}"
        )

manifest_hash = sha256_file(MANIFEST)

if manifest_hash != EXPECTED_MANIFEST_SHA256:
    raise RuntimeError(
        "Frozen Phase 3B manifest hash mismatch."
    )

print(
    "Phase 3B frozen manifest: VERIFIED",
    flush=True,
)

print(
    "Phase 3B feature archive: FOUND",
    flush=True,
)


# ============================================================
# STEP 2: LOCAL DATA INVENTORY
# ============================================================

print(
    "\n[2] Auditing existing research data...",
    flush=True,
)

rows = read_csv_rows(MANIFEST)

if len(rows) != 360:
    raise RuntimeError(
        "Expected 360 frozen tile positions."
    )

region_counts = Counter(
    row["region_id"]
    for row in rows
)

if (
    len(region_counts) != 10
    or any(n != 36 for n in region_counts.values())
):
    raise RuntimeError(
        "Unexpected frozen region distribution."
    )

unique_tile_ids = {
    row["tile_id"]
    for row in rows
}

if len(unique_tile_ids) != 360:
    raise RuntimeError(
        "Duplicate tile IDs in frozen manifest."
    )

images = image_inventory(OLD_IMAGES)

print(
    "Frozen regions:",
    len(region_counts),
    flush=True,
)

print(
    "Matched tile positions:",
    len(rows),
    flush=True,
)

for scanner in ("AT2", "GT450"):
    item = images[scanner]

    print(
        f"Existing {scanner} PNG files:",
        item["png_count"],
        flush=True,
    )

disk = shutil.disk_usage(ROOT)

disk_info = {
    "total_gib": gib(disk.total),
    "used_gib": gib(disk.used),
    "free_gib": gib(disk.free),
}

print(
    "Free disk space (GiB):",
    disk_info["free_gib"],
    flush=True,
)


# ============================================================
# STEP 3: FETCH OFFICIAL DATASET METADATA
# ============================================================

print(
    "\n[3] Fetching official PLISM metadata...",
    flush=True,
)

remote = None
remote_error = None

try:
    remote = fetch_json(API_URL)

except (
    HTTPError,
    URLError,
    TimeoutError,
    ValueError,
    OSError,
) as exc:

    remote_error = (
        type(exc).__name__
        + ": "
        + str(exc)
    )

    print(
        "Remote metadata unavailable:",
        remote_error,
        flush=True,
    )

if remote is not None:

    print(
        "Remote dataset:",
        remote.get("id"),
        flush=True,
    )

    print(
        "Repository revision:",
        remote.get("sha"),
        flush=True,
    )

    print(
        "Repository metadata: RETRIEVED",
        flush=True,
    )


# ============================================================
# STEP 4: INVENTORY REPOSITORY FILES
# ============================================================

print(
    "\n[4] Inventorying published files...",
    flush=True,
)

file_records = []

if remote is not None:

    siblings = remote.get("siblings") or []

    for item in siblings:

        filename = item.get("rfilename", "")

        if not filename:
            continue

        suffix = Path(filename).suffix.lower()

        file_records.append({
            "repository_file": filename,
            "extension": suffix,
            "file_size_bytes": extract_file_size(item),
            "is_parquet": suffix == ".parquet",
        })

parquet_files = [
    row
    for row in file_records
    if row["is_parquet"]
]

known_remote_sizes = [
    row["file_size_bytes"]
    for row in file_records
    if row["file_size_bytes"] is not None
]

print(
    "Published repository entries:",
    len(file_records),
    flush=True,
)

print(
    "Parquet files identified:",
    len(parquet_files),
    flush=True,
)

print(
    "Entries with known sizes:",
    len(known_remote_sizes),
    flush=True,
)

if known_remote_sizes:
    print(
        "Total known published file size (GiB):",
        gib(sum(known_remote_sizes)),
        flush=True,
    )

# Important: repository-file inventory is not
# an exhaustive tile-level slide inventory.
# Unique specimen counts cannot be established here.


# ============================================================
# STEP 5: SAVE AUDIT
# ============================================================

print(
    "\n[5] Saving Phase 4 inventory...",
    flush=True,
)

audit = {
    "phase": "4",
    "step": "4.1.1",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "python_version": sys.version,
    "dataset_id": DATASET_ID,
    "dataset_metadata_url": API_URL,
    "remote_metadata_retrieved": remote is not None,
    "remote_error": remote_error,
    "remote_revision": (
        remote.get("sha")
        if remote is not None
        else None
    ),
    "remote_repository_entries": len(file_records),
    "remote_parquet_file_count": len(parquet_files),
    "known_remote_file_bytes": (
        sum(known_remote_sizes)
        if known_remote_sizes
        else None
    ),
    "phase3b_manifest_sha256": manifest_hash,
    "phase3b_feature_sha256": sha256_file(
        FEATURE_ARCHIVE
    ),
    "phase3b_regions": len(region_counts),
    "phase3b_tile_positions": len(rows),
    "phase3b_local_pngs": images,
    "local_disk": disk_info,
    "independent_specimens_confirmed": False,
    "tile_level_slide_inventory_completed": False,
    "next_step": (
        "Inspect remote Parquet metadata and calculate "
        "scanner/slide/coordinate coverage without "
        "downloading image payloads."
    ),
    "data_modified": False,
    "image_downloads_performed": False,
}

if remote is not None:
    write_json_exclusive(
        REMOTE_METADATA_PATH,
        remote,
    )

with SHARD_PATH.open(
    "x",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "repository_file",
            "extension",
            "file_size_bytes",
            "is_parquet",
        ],
    )

    writer.writeheader()
    writer.writerows(file_records)

write_json_exclusive(
    REPORT_PATH,
    audit,
)


# ============================================================
# FINAL REPORT
# ============================================================

print(
    "\n" + "=" * 70,
    flush=True,
)

print(
    "PHASE 4.1.1 - DATASET INVENTORY COMPLETED",
    flush=True,
)

print(
    "=" * 70,
    flush=True,
)

print(
    "Frozen Phase 3B positions:",
    len(rows),
    flush=True,
)

print(
    "Local original PNG files:",
    sum(
        item["png_count"]
        for item in images.values()
    ),
    flush=True,
)

print(
    "Remote metadata retrieved:",
    remote is not None,
    flush=True,
)

print(
    "Remote Parquet files:",
    len(parquet_files),
    flush=True,
)

print(
    "Free disk GiB:",
    disk_info["free_gib"],
    flush=True,
)

print("\nSaved:", flush=True)
print(REPORT_PATH, flush=True)
print(SHARD_PATH, flush=True)

if remote is not None:
    print(
        REMOTE_METADATA_PATH,
        flush=True,
    )

print(
    "\nPhase 3B data: UNCHANGED",
    flush=True,
)

print(
    "Image downloads: NONE",
    flush=True,
)

print(
    "Independent specimens: NOT YET VERIFIED",
    flush=True,
)

print(
    "\nSTEP 4.1.1 COMPLETED",
    flush=True,
)
