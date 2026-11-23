
"""
PHASE 3B - STEP 3B.15
Frozen Tile Shard and Row-Group Discovery

Objective:
    Locate candidate Parquet row groups for the 720
    frozen scanner-tile records using footer statistics.

Method:
    - Verify frozen manifest SHA256
    - Inspect the first 12 Parquet shard footers
    - Use scanner, stainer, and tile_id min/max statistics
    - Estimate candidate PNG column transfer cost
    - Save candidate locations and discovery report

IMPORTANT:
    Min/max filtering produces candidate locations,
    not verified exact row matches.

No image bytes downloaded.
No full Parquet shards downloaded.
Frozen manifest unchanged.
"""

from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import struct

import pandas as pd
import pyarrow.parquet as pq
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

MANIFEST_PATH = (
    BASE / "outputs/validated/"
    "phase3b_frozen_tile_manifest.csv"
)

EXPECTED_HASH = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)

DATASET = "owkin/plism-dataset-tiles"

BASE_URL = (
    "https://huggingface.co/datasets/"
    f"{DATASET}/resolve/main/data/"
)

SHARD_INDICES = list(range(12))
TOTAL_SHARDS = 293

MAX_FOOTER_BYTES = 16 * 1024 * 1024
TIMEOUT = (10, 30)

EXPECTED_SLIDES = {
    "AT2": "GIVH_AT2_to_GMH_S60",
    "GT450": "GIVH_GT450_to_GMH_S60",
}


# ============================================================
# HELPERS
# ============================================================

def sha256(data):
    return hashlib.sha256(data).hexdigest()


def get_shard_name(index):
    return (
        f"train-{index:05d}-of-"
        f"{TOTAL_SHARDS:05d}.parquet"
    )


def fetch_range(session, url, start, end):

    expected = end - start + 1

    if expected <= 0 or expected > MAX_FOOTER_BYTES:
        raise RuntimeError("Invalid metadata range size")

    with session.get(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
        },
        stream=True,
        timeout=TIMEOUT,
        allow_redirects=True,
    ) as response:

        if response.status_code != 206:
            raise RuntimeError(
                f"Expected HTTP 206, got {response.status_code}"
            )

        content_range = response.headers.get(
            "Content-Range", ""
        )

        if not content_range.startswith(
            f"bytes {start}-{end}/"
        ):
            raise RuntimeError(
                f"Unexpected Content-Range: {content_range}"
            )

        data = bytearray()

        for chunk in response.iter_content(65536):
            data.extend(chunk)

            if len(data) > expected:
                raise RuntimeError("Oversized range response")

    if len(data) != expected:
        raise RuntimeError("Incomplete metadata response")

    return bytes(data)


def read_footer(session, url):

    response = session.head(
        url,
        allow_redirects=True,
        timeout=TIMEOUT,
    )
    response.raise_for_status()

    size_text = response.headers.get("Content-Length")

    if not size_text or not size_text.isdigit():
        raise RuntimeError("Missing file length")

    file_size = int(size_text)

    footer = fetch_range(
        session, url, file_size - 8, file_size - 1
    )

    metadata_size = struct.unpack("<I", footer[:4])[0]

    if footer[4:] != b"PAR1":
        raise RuntimeError("Invalid Parquet magic")

    if not 0 < metadata_size <= MAX_FOOTER_BYTES:
        raise RuntimeError("Unsafe metadata size")

    metadata_start = file_size - 8 - metadata_size

    if metadata_start < 4:
        raise RuntimeError("Invalid footer location")

    metadata_bytes = fetch_range(
        session,
        url,
        metadata_start,
        file_size - 9,
    )

    synthetic = b"PAR1" + metadata_bytes + footer
    metadata = pq.read_metadata(io.BytesIO(synthetic))

    return metadata, metadata_size


def decode_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None:
        return None
    return str(value)


def can_contain(column, target):
    """
    Returns True when footer statistics cannot
    rule out a target value.

    Missing statistics are treated as unknown.
    """
    if column is None:
        return True

    stats = column.statistics

    if stats is None or not stats.has_min_max:
        return True

    minimum = decode_value(stats.min)
    maximum = decode_value(stats.max)

    if minimum is None or maximum is None:
        return True

    return minimum <= target <= maximum


# ============================================================
# STEP 1: VALIDATE FROZEN MANIFEST
# ============================================================

print("=" * 70, flush=True)
print("PHASE 3B - PARQUET TILE LOCATION DISCOVERY", flush=True)
print("=" * 70, flush=True)

print("\n[1] Verifying frozen manifest...", flush=True)

if not MANIFEST_PATH.is_file():
    raise RuntimeError("Frozen manifest missing")

actual_hash = sha256(MANIFEST_PATH.read_bytes())

if actual_hash != EXPECTED_HASH:
    raise RuntimeError("Frozen manifest SHA256 mismatch")

manifest = pd.read_csv(MANIFEST_PATH)

if len(manifest) != 360:
    raise RuntimeError("Unexpected frozen manifest size")

targets = []

for row in manifest.itertuples(index=False):

    for scanner in EXPECTED_SLIDES:
        targets.append({
            "region_id": str(row.region_id),
            "tile_id": str(row.tile_id),
            "scanner": scanner,
            "stainer": "GIVH",
            "slide_id": EXPECTED_SLIDES[scanner],
        })

print("Frozen manifest: VERIFIED")
print("Matched positions:", len(manifest))
print("Scanner-tile targets:", len(targets))


# ============================================================
# STEP 2: INSPECT ROW-GROUP STATISTICS
# ============================================================

print("\n[2] Scanning Parquet footer metadata...", flush=True)

candidate_records = []
shard_records = []
candidate_groups = {}

with requests.Session() as session:

    for shard_index in SHARD_INDICES:

        shard = get_shard_name(shard_index)
        url = BASE_URL + shard

        print(
            f"\nShard {shard_index:03d}: {shard}",
            flush=True,
        )

        metadata, footer_bytes = read_footer(
            session, url
        )

        shard_candidate_count = 0
        shard_candidate_groups = 0

        for rg_index in range(
            metadata.num_row_groups
        ):

            group = metadata.row_group(rg_index)

            columns = {
                group.column(i).path_in_schema:
                group.column(i)
                for i in range(group.num_columns)
            }

            relevant_columns = {
                key: columns.get(key)
                for key in (
                    "tile_id",
                    "scanner",
                    "stainer",
                    "slide_id",
                )
            }

            possible = []

            for target in targets:

                if all(
                    can_contain(
                        relevant_columns[key],
                        target[key]
                    )
                    for key in relevant_columns
                ):
                    possible.append(target)

            if not possible:
                continue

            shard_candidate_groups += 1
            shard_candidate_count += len(possible)

            png_columns = [
                column
                for path, column in columns.items()
                if (
                    path == "png"
                    or path.startswith("png.")
                )
            ]

            # Estimate the compressed bytes required
            # for the entire nested PNG column.
            png_bytes = sum(
                col.total_compressed_size
                for col in png_columns
            )

            group_id = (shard, rg_index)

            candidate_groups[group_id] = {
                "shard": shard,
                "row_group": rg_index,
                "row_count": group.num_rows,
                "estimated_png_column_bytes": png_bytes,
            }

            for target in possible:
                candidate_records.append({
                    **target,
                    "shard": shard,
                    "row_group": rg_index,
                    "row_group_rows": group.num_rows,
                    "estimated_png_column_bytes": png_bytes,
                    "location_verified": False,
                })

        shard_records.append({
            "shard": shard,
            "rows": metadata.num_rows,
            "row_groups": metadata.num_row_groups,
            "footer_bytes": footer_bytes,
            "candidate_row_groups": shard_candidate_groups,
            "candidate_target_locations": shard_candidate_count,
        })

        print(
            "  Row groups:",
            metadata.num_row_groups,
            flush=True,
        )

        print(
            "  Candidate row groups:",
            shard_candidate_groups,
            flush=True,
        )

        print(
            "  Candidate target locations:",
            shard_candidate_count,
            flush=True,
        )


# ============================================================
# STEP 3: SUMMARIZE COVERAGE
# ============================================================

print("\n[3] Summarizing candidate coverage...", flush=True)

candidates_df = pd.DataFrame(
    candidate_records,
    columns=[
        "region_id", "tile_id", "scanner",
        "stainer", "slide_id", "shard",
        "row_group", "row_group_rows",
        "estimated_png_column_bytes",
        "location_verified",
    ]
)

shards_df = pd.DataFrame(shard_records)

candidate_groups_df = pd.DataFrame(
    candidate_groups.values(),
    columns=[
        "shard", "row_group", "row_count",
        "estimated_png_column_bytes",
    ]
)

target_keys = {
    (item["tile_id"], item["scanner"])
    for item in targets
}

candidate_keys = {
    (str(row.tile_id), str(row.scanner))
    for row in candidates_df.itertuples(index=False)
}

unlocated_keys = target_keys - candidate_keys

covered = len(candidate_keys)
unlocated = len(unlocated_keys)

print(f"Total target records: {len(target_keys)}")
print(f"Targets with candidate locations: {covered}")
print(f"No candidate in inspected shards: {unlocated}")

estimated_bytes = int(
    candidate_groups_df[
        "estimated_png_column_bytes"
    ].sum()
) if not candidate_groups_df.empty else 0

print(
    "Candidate PNG-column transfer estimate:",
    f"{estimated_bytes / 1_000_000:.2f} MB"
)

print(
    "Note: Candidate group estimates may overstate "
    "actual retrieval work until row matches are verified."
)


# ============================================================
# STEP 4: SAVE OUTPUTS
# ============================================================

print("\n[4] Saving discovery tables...", flush=True)

candidates_path = (
    TABLES / "plism_frozen_tile_candidate_locations.csv"
)

shards_path = (
    TABLES / "plism_frozen_shard_location_summary.csv"
)

groups_path = (
    TABLES / "plism_candidate_image_row_groups.csv"
)

report_path = (
    REPORTS / "15_frozen_tile_location_discovery.json"
)

candidates_df.to_csv(
    candidates_path, index=False
)

shards_df.to_csv(
    shards_path, index=False
)

candidate_groups_df.to_csv(
    groups_path, index=False
)

report = {
    "phase": "3B",
    "step": "3B.15",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "frozen_manifest_sha256": actual_hash,
    "shard_count_inspected": len(SHARD_INDICES),
    "total_shards": TOTAL_SHARDS,
    "total_target_records": len(target_keys),
    "targets_with_candidate_locations": covered,
    "targets_without_candidate_locations": unlocated,
    "candidate_row_groups": len(candidate_groups_df),
    "estimated_png_chunk_bytes": estimated_bytes,
    "exact_tile_locations_verified": False,
    "images_downloaded": 0,
    "full_shards_downloaded": 0,
    "frozen_manifest_modified": False,
    "notes": [
        "Only the first 12 shards were inspected",
        "Parquet statistics use lexicographic string ranges",
        "Candidate row groups must be checked against actual rows",
        "Metadata column chunks may require additional HTTP requests",
        "PNG chunk estimates are approximate transfer costs",
        "No biological independence established",
    ],
}

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3B.15 - LOCATION DISCOVERY COMPLETED")
print("=" * 70)

print(f"Shards inspected: {len(SHARD_INDICES)}")
print(f"Total target images: {len(target_keys)}")
print(f"Targets with candidate locations: {covered}")
print(f"Targets not yet located: {unlocated}")
print(f"Candidate row groups: {len(candidate_groups_df)}")

print("\nCandidate locations by scanner:")

for scanner in EXPECTED_SLIDES:
    count = sum(
        1 for tile_id, sc in candidate_keys
        if sc == scanner
    )
    print(f"  {scanner}: {count}")

print("\nSaved:")
print(candidates_path)
print(shards_path)
print(groups_path)
print(report_path)

print("\nImage data downloaded: NONE")
print("Actual row locations: NOT YET VERIFIED")
print("Frozen manifest: UNCHANGED")
print("\nSTEP 3B.15 COMPLETED")
