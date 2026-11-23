
"""
Phase 3A - Step 3.7
Full PLISM WSI Annotation Metadata Analysis

Reads the official Figshare WSI annotation CSV
in chunks without saving the source file.

No WSI images or PNG tiles are downloaded.
"""

from pathlib import Path
from urllib.request import Request, urlopen
from datetime import datetime, timezone
from collections import defaultdict
import json

import pandas as pd


# ======================================================
# CONFIGURATION
# ======================================================

ROOT = Path.cwd()

OUTPUT = ROOT / "experimental_research" / "outputs"

TABLE_DIR = OUTPUT / "tables"
REPORT_DIR = OUTPUT / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ARTICLE_ID = 23614422
FILENAME = "PLISM_wsi_en.csv"

CHUNK_SIZE = 10000

# Safety limit for the remote annotation file
MAX_FILE_BYTES = 25 * 1024 * 1024

print("=" * 70, flush=True)
print("PHASE 3A - FULL PLISM WSI METADATA ANALYSIS", flush=True)
print("=" * 70, flush=True)


# ======================================================
# STEP 1: GET OFFICIAL FILE METADATA
# ======================================================

print("\n[1] Retrieving Figshare file metadata...", flush=True)

api_url = (
    f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
)

request = Request(
    api_url,
    headers={
        "User-Agent": "PLISM-Metadata-Research/1.0",
        "Accept": "application/json"
    }
)

with urlopen(request, timeout=60) as response:
    article = json.load(response)

matching_files = [
    item
    for item in article.get("files", [])
    if item.get("name") == FILENAME
]

if not matching_files:
    raise SystemExit("Official WSI annotation file not found.")

file_info = matching_files[0]

size_bytes = int(file_info.get("size") or 0)

if size_bytes <= 0 or size_bytes > MAX_FILE_BYTES:
    raise SystemExit(
        f"Unexpected annotation file size: {size_bytes}"
    )

download_url = file_info.get("download_url")

if not download_url:
    raise SystemExit("No official file URL available.")

print(f"File: {FILENAME}")
print(f"Size: {size_bytes / 1024**2:.2f} MiB")
print("Mode: streamed CSV analysis")


# ======================================================
# STEP 2: INITIALIZE DATA STRUCTURES
# ======================================================

print("\n[2] Preparing metadata analysis...", flush=True)

total_rows = 0
invalid_coordinate_rows = 0

group_counts = defaultdict(int)
group_tissues = defaultdict(set)
group_coordinates = defaultdict(set)

# Record original annotation coordinate behavior.
coordinate_modulo_counts = defaultdict(int)

required_columns = {
    "stain",
    "device",
    "tissue_type",
    "coordinate",
    "path"
}


# ======================================================
# STEP 3: STREAM THE WSI ANNOTATION CSV
# ======================================================

print("\n[3] Reading annotation records...", flush=True)

csv_request = Request(
    download_url,
    headers={
        "User-Agent": "PLISM-Metadata-Research/1.0",
        "Accept-Encoding": "identity"
    }
)

with urlopen(csv_request, timeout=120) as response:

    content_length = response.headers.get(
        "Content-Length"
    )

    if (
        content_length is not None
        and int(content_length) > MAX_FILE_BYTES
    ):
        raise SystemExit("Response exceeds safety size limit.")

    chunks = pd.read_csv(
        response,
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )

    for chunk_number, chunk in enumerate(chunks, 1):

        missing = required_columns - set(chunk.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        total_rows += len(chunk)

        for row in chunk.itertuples(index=False):

            stain = str(row.stain).strip()
            device = str(row.device).strip()
            tissue = str(row.tissue_type).strip()
            coordinate = str(row.coordinate).strip()

            key = (stain, device)

            group_counts[key] += 1

            if tissue:
                group_tissues[key].add(tissue)

            if coordinate:
                group_coordinates[key].add(coordinate)

                parts = coordinate.split("_")

                if (
                    len(parts) == 2
                    and all(part.isdigit() for part in parts)
                ):
                    x, y = map(int, parts)

                    if x % 1024 == 0 and y % 1024 == 0:
                        coordinate_modulo_counts[
                            "both_divisible_by_1024"
                        ] += 1
                    else:
                        coordinate_modulo_counts[
                            "not_both_divisible_by_1024"
                        ] += 1

                else:
                    invalid_coordinate_rows += 1

        print(
            f"Processed chunk {chunk_number}: "
            f"{total_rows:,} rows",
            flush=True
        )


# ======================================================
# STEP 4: BUILD SCANNER GROUP INVENTORY
# ======================================================

print("\n[4] Building acquisition group inventory...", flush=True)

group_records = []

for key in sorted(group_counts):

    stain, device = key

    group_records.append({
        "stain": stain,
        "device": device,
        "annotation_rows": group_counts[key],
        "unique_tissue_types": len(group_tissues[key]),
        "unique_coordinates": len(group_coordinates[key])
    })

groups_df = pd.DataFrame(group_records)

groups_path = (
    TABLE_DIR / "plism_full_wsi_acquisition_groups.csv"
)

groups_df.to_csv(groups_path, index=False)

print(f"Total annotation rows: {total_rows:,}")
print(f"Staining conditions: {groups_df['stain'].nunique()}")
print(f"Devices: {groups_df['device'].nunique()}")
print(f"Acquisition groups: {len(groups_df)}")


# ======================================================
# STEP 5: FOCUS ON GIVH AT2 AND GT450
# ======================================================

print("\n[5] Comparing GIVH AT2 and GT450...", flush=True)

key_a = ("GIVH", "AT2")
key_b = ("GIVH", "GT450")

tissues_a = group_tissues[key_a]
tissues_b = group_tissues[key_b]

coords_a = group_coordinates[key_a]
coords_b = group_coordinates[key_b]

shared_tissues = tissues_a & tissues_b
shared_coordinates = coords_a & coords_b

comparison = {
    "at2_annotation_rows": group_counts[key_a],
    "gt450_annotation_rows": group_counts[key_b],
    "at2_tissue_types": len(tissues_a),
    "gt450_tissue_types": len(tissues_b),
    "shared_tissue_types": len(shared_tissues),
    "at2_unique_coordinates": len(coords_a),
    "gt450_unique_coordinates": len(coords_b),
    "identical_coordinate_strings": len(shared_coordinates),
    "scanner_pair_present": (
        group_counts[key_a] > 0
        and group_counts[key_b] > 0
    )
}

for key, value in comparison.items():
    print(f"{key}: {value}")

print(
    "\nImportant: Matching coordinate strings do not "
    "establish spatial registration."
)


# ======================================================
# STEP 6: SAVE TISSUE-TYPE OVERLAP
# ======================================================

print("\n[6] Saving tissue-type comparison...", flush=True)

all_tissues = sorted(tissues_a | tissues_b)

tissue_records = [
    {
        "tissue_type": tissue,
        "present_at2": tissue in tissues_a,
        "present_gt450": tissue in tissues_b,
        "shared": (
            tissue in tissues_a and tissue in tissues_b
        )
    }
    for tissue in all_tissues
]

tissue_df = pd.DataFrame(
    tissue_records,
    columns=[
        "tissue_type",
        "present_at2",
        "present_gt450",
        "shared"
    ]
)

tissue_path = (
    TABLE_DIR / "plism_givh_tissue_type_overlap.csv"
)

tissue_df.to_csv(tissue_path, index=False)


# ======================================================
# STEP 7: SAVE FINAL REPORT
# ======================================================

print("\n[7] Generating research report...", flush=True)

report = {
    "phase": "3A",
    "step": "3.7",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "source_file": FILENAME,
    "total_annotation_rows": total_rows,
    "staining_conditions": int(
        groups_df["stain"].nunique()
    ),
    "devices": int(groups_df["device"].nunique()),
    "acquisition_groups": len(groups_df),
    "coordinate_format_invalid_rows":
        invalid_coordinate_rows,
    "coordinate_grid_checks":
        dict(coordinate_modulo_counts),
    "givh_scanner_comparison": comparison,
    "shared_tissue_type_labels":
        sorted(shared_tissues),
    "tissue_core_mapping_verified": False,
    "independent_specimen_count": None,
    "registration_transform_verified": False
}

report_path = (
    REPORT_DIR / "07_full_wsi_metadata_analysis.json"
)

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("FULL WSI ANNOTATION ANALYSIS COMPLETED")
print("=" * 70)

print(f"Total records: {total_rows:,}")
print(f"Acquisition groups: {len(groups_df)}")
print(f"Shared GIVH tissue types: {len(shared_tissues)}")
print(f"Identical coordinate strings: {len(shared_coordinates)}")
print("Independent specimen identity: UNVERIFIED")
print("Owkin coordinate mapping: UNVERIFIED")

print("\nSaved outputs:")
print(groups_path)
print(tissue_path)
print(report_path)

print("\nSTEP 3.7 COMPLETED")
