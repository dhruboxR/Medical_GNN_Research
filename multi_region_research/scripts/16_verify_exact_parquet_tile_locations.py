
"""
PHASE 3B - STEP 3B.16
Exact PLISM Parquet Tile-Location Verification

Inputs:
  - Frozen 360-position tile manifest
  - Phase 3B.15 candidate locations

Procedure:
  1. Verify frozen manifest SHA256.
  2. Load candidate shard/row-group locations.
  3. Open relevant remote Parquet files using HTTP ranges.
  4. Read ONLY small metadata columns in candidate groups.
  5. Match exact tile ID, scanner, stainer and slide ID.
  6. Record exact shard, row group and row index.
  7. Audit missing, duplicated and unexpected matches.

No PNG/image columns are read.
No complete Parquet shards are downloaded.
The frozen manifest remains unchanged.
"""

from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json

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

CANDIDATE_PATH = (
    TABLES / "plism_frozen_tile_candidate_locations.csv"
)

EXACT_PATH = (
    TABLES / "plism_exact_parquet_tile_locations.csv"
)

AUDIT_PATH = (
    TABLES / "plism_exact_location_target_audit.csv"
)

GROUP_PATH = (
    TABLES / "plism_verified_image_row_groups.csv"
)

REPORT_PATH = (
    REPORTS / "16_exact_parquet_tile_locations.json"
)

EXPECTED_MANIFEST_HASH = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)

BASE_URL = (
    "https://huggingface.co/datasets/"
    "owkin/plism-dataset-tiles/resolve/main/data/"
)

EXPECTED_SLIDES = {
    "AT2": "GIVH_AT2_to_GMH_S60",
    "GT450": "GIVH_GT450_to_GMH_S60",
}

METADATA_COLUMNS = [
    "tile_id",
    "scanner",
    "stainer",
    "slide_id",
]

TIMEOUT = (10, 40)

MAX_SINGLE_READ = 2 * 1024 * 1024
MAX_SHARD_TRANSFER = 35 * 1024 * 1024


# ============================================================
# REMOTE RANGE READER
# ============================================================

class HTTPRangeFile(io.RawIOBase):

    def __init__(self, url, session):
        super().__init__()

        self.url = url
        self.session = session

        self.position = 0
        self.bytes_received = 0
        self.requests_made = 0

        response = session.head(
            url,
            allow_redirects=True,
            timeout=TIMEOUT,
        )

        response.raise_for_status()

        content_length = response.headers.get(
            "Content-Length"
        )

        if not content_length or not content_length.isdigit():
            raise RuntimeError(
                "Remote file length unavailable."
            )

        self.file_size = int(content_length)

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):

        if whence == io.SEEK_SET:
            new_position = offset

        elif whence == io.SEEK_CUR:
            new_position = self.position + offset

        elif whence == io.SEEK_END:
            new_position = self.file_size + offset

        else:
            raise ValueError("Invalid seek mode.")

        if not 0 <= new_position <= self.file_size:
            raise ValueError(
                "Seek outside remote file."
            )

        self.position = int(new_position)

        return self.position

    def read(self, size=-1):

        if size is None or size < 0:
            raise RuntimeError(
                "Unbounded reads are prohibited."
            )

        if size == 0 or self.position >= self.file_size:
            return b""

        size = min(
            size,
            self.file_size - self.position
        )

        if size > MAX_SINGLE_READ:
            raise RuntimeError(
                f"Remote read too large: {size} bytes."
            )

        if (
            self.bytes_received + size
            > MAX_SHARD_TRANSFER
        ):
            raise RuntimeError(
                "Per-shard transfer limit exceeded."
            )

        start = self.position
        end = start + size - 1

        with self.session.get(
            self.url,
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
                    f"Expected HTTP 206, "
                    f"received {response.status_code}."
                )

            expected_range = f"bytes {start}-{end}/"

            actual_range = response.headers.get(
                "Content-Range", ""
            )

            if not actual_range.startswith(expected_range):
                raise RuntimeError(
                    f"Unexpected Content-Range: {actual_range}"
                )

            data = bytearray()

            for chunk in response.iter_content(65536):
                data.extend(chunk)

                if len(data) > size:
                    raise RuntimeError(
                        "Response exceeded requested range."
                    )

        if len(data) != size:
            raise RuntimeError(
                f"Incomplete response: {len(data)}/{size}"
            )

        self.position += size
        self.bytes_received += size
        self.requests_made += 1

        return bytes(data)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


# ============================================================
# STEP 1: VERIFY INPUTS
# ============================================================

print("=" * 70, flush=True)
print("PHASE 3B - EXACT PARQUET LOCATION VALIDATION", flush=True)
print("=" * 70, flush=True)

print("\n[1] Verifying input files...", flush=True)

for path in (MANIFEST_PATH, CANDIDATE_PATH):
    if not path.is_file():
        raise RuntimeError(f"Missing input file: {path}")

actual_hash = hashlib.sha256(
    MANIFEST_PATH.read_bytes()
).hexdigest()

if actual_hash != EXPECTED_MANIFEST_HASH:
    raise RuntimeError(
        "Frozen manifest SHA256 mismatch."
    )

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"tile_id": str, "region_id": str},
)

candidates = pd.read_csv(
    CANDIDATE_PATH,
    dtype={
        "tile_id": str,
        "scanner": str,
        "shard": str,
        "region_id": str,
    },
)

required_candidate_columns = {
    "tile_id", "scanner", "shard", "row_group",
    "region_id",
}

if required_candidate_columns - set(candidates.columns):
    raise RuntimeError(
        "Candidate table has missing columns."
    )

if len(manifest) != 360:
    raise RuntimeError(
        "Unexpected frozen manifest size."
    )

if manifest["tile_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate tile IDs in frozen manifest."
    )

targets = {}

for row in manifest.itertuples(index=False):

    for scanner, slide in EXPECTED_SLIDES.items():

        key = (str(row.tile_id), scanner)

        targets[key] = {
            "region_id": str(row.region_id),
            "tile_id": str(row.tile_id),
            "scanner": scanner,
            "stainer": "GIVH",
            "slide_id": slide,
        }

if len(targets) != 720:
    raise RuntimeError(
        "Expected exactly 720 scanner-tile targets."
    )

print("Frozen manifest: VERIFIED")
print("Target scanner-tile records:", len(targets))
print("Candidate location rows:", len(candidates))


# ============================================================
# STEP 2: ORGANIZE CANDIDATE ROW GROUPS
# ============================================================

print("\n[2] Organizing candidate groups...", flush=True)

group_targets = defaultdict(set)

for row in candidates.itertuples(index=False):

    key = (str(row.tile_id), str(row.scanner))

    if key not in targets:
        raise RuntimeError(
            f"Unexpected candidate target: {key}"
        )

    group_key = (
        str(row.shard),
        int(row.row_group),
    )

    group_targets[group_key].add(key)

shard_groups = defaultdict(list)

for shard, row_group in group_targets:
    shard_groups[shard].append(row_group)

print("Candidate groups:", len(group_targets))
print("Relevant shards:", len(shard_groups))

# ============================================================
# STEP 3: VERIFY EXACT ROW MATCHES
# ============================================================

print("\n[3] Reading metadata columns...", flush=True)

exact_records = []
group_stats = []

total_requests = 0
total_bytes = 0

with requests.Session() as session:

    for shard in sorted(shard_groups):

        print(
            f"\nShard: {shard}",
            flush=True
        )

        reader = HTTPRangeFile(
            BASE_URL + shard,
            session
        )

        parquet = pq.ParquetFile(
            reader,
            pre_buffer=False,
        )

        for rg_index in sorted(shard_groups[shard]):

            if not (
                0 <= rg_index < parquet.metadata.num_row_groups
            ):
                raise RuntimeError(
                    f"Invalid row group: {shard} / {rg_index}"
                )

            table = parquet.read_row_group(
                rg_index,
                columns=METADATA_COLUMNS,
                use_threads=False,
            )

            rows = table.to_pylist()

            possible_keys = group_targets[
                (shard, rg_index)
            ]

            local_matches = 0

            for row_index, record in enumerate(rows):

                tile_id = str(
                    record.get("tile_id", "")
                )

                scanner = str(
                    record.get("scanner", "")
                )

                key = (tile_id, scanner)

                if key not in possible_keys:
                    continue

                expected = targets[key]

                if (
                    record.get("stainer")
                    != expected["stainer"]
                    or record.get("slide_id")
                    != expected["slide_id"]
                ):
                    continue

                exact_records.append({
                    **expected,
                    "shard": shard,
                    "row_group": rg_index,
                    "row_index_in_group": row_index,
                    "row_group_rows": len(rows),
                    "exact_metadata_verified": True,
                })

                local_matches += 1

            group_stats.append({
                "shard": shard,
                "row_group": rg_index,
                "rows_read": len(rows),
                "candidate_targets": len(possible_keys),
                "exact_matches": local_matches,
            })

            print(
                f"  RG {rg_index:02d}: "
                f"{len(rows)} rows | "
                f"{local_matches} exact matches",
                flush=True,
            )

        total_requests += reader.requests_made
        total_bytes += reader.bytes_received

        print(
            f"  Network: {reader.requests_made} "
            f"requests, {reader.bytes_received:,} bytes",
            flush=True,
        )


# ============================================================
# STEP 4: AUDIT MATCH COMPLETENESS
# ============================================================

print("\n[4] Auditing exact locations...", flush=True)

counts = defaultdict(int)

for record in exact_records:
    key = (
        record["tile_id"],
        record["scanner"],
    )
    counts[key] += 1

missing_keys = [
    key for key in targets
    if counts[key] == 0
]

duplicate_keys = [
    key for key in targets
    if counts[key] > 1
]

audit_records = []

for key, target in sorted(targets.items()):

    count = counts[key]

    audit_records.append({
        **target,
        "exact_location_count": count,
        "status": (
            "VERIFIED"
            if count == 1
            else "MISSING"
            if count == 0
            else "DUPLICATE"
        ),
    })

exact_df = pd.DataFrame(
    exact_records,
    columns=[
        "region_id", "tile_id", "scanner",
        "stainer", "slide_id", "shard",
        "row_group", "row_index_in_group",
        "row_group_rows", "exact_metadata_verified",
    ],
)

audit_df = pd.DataFrame(audit_records)
group_df = pd.DataFrame(group_stats)

if not exact_df.empty:
    exact_df = exact_df.sort_values(
        ["shard", "row_group", "row_index_in_group"]
    ).reset_index(drop=True)

print("Exact records found:", len(exact_df))
print("Unique targets verified:", sum(
    counts[key] == 1 for key in targets
))
print("Missing targets:", len(missing_keys))
print("Duplicate targets:", len(duplicate_keys))
print("Metadata bytes transferred:", total_bytes)


# ============================================================
# STEP 5: SAVE RESULTS
# ============================================================

print("\n[5] Saving exact location audit...", flush=True)

exact_df.to_csv(EXACT_PATH, index=False)
audit_df.to_csv(AUDIT_PATH, index=False)
group_df.to_csv(GROUP_PATH, index=False)

complete = (
    len(missing_keys) == 0
    and len(duplicate_keys) == 0
    and len(exact_df) == 720
)

report = {
    "phase": "3B",
    "step": "3B.16",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "frozen_manifest_sha256": actual_hash,
    "target_count": len(targets),
    "candidate_row_groups_checked": len(group_df),
    "shards_checked": len(shard_groups),
    "exact_records_found": len(exact_df),
    "unique_targets_verified": sum(
        counts[key] == 1 for key in targets
    ),
    "missing_targets": len(missing_keys),
    "duplicate_targets": len(duplicate_keys),
    "missing_target_examples": [
        {"tile_id": key[0], "scanner": key[1]}
        for key in missing_keys[:20]
    ],
    "duplicate_target_examples": [
        {"tile_id": key[0], "scanner": key[1]}
        for key in duplicate_keys[:20]
    ],
    "http_range_requests": total_requests,
    "metadata_bytes_transferred": total_bytes,
    "all_exact_locations_verified": complete,
    "png_columns_read": False,
    "images_downloaded": 0,
    "full_parquet_shards_downloaded": 0,
    "frozen_manifest_modified": False,
    "biological_independence_verified": False,
}

REPORT_PATH.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3B.16 - EXACT LOCATION AUDIT COMPLETED")
print("=" * 70)

print(f"Target records: {len(targets)}")
print(f"Candidate row groups checked: {len(group_df)}")
print(f"Exact records found: {len(exact_df)}")
print(f"Missing targets: {len(missing_keys)}")
print(f"Duplicate targets: {len(duplicate_keys)}")

print(f"HTTP requests: {total_requests}")
print(f"Metadata transferred: {total_bytes / 1_000_000:.3f} MB")

print("\nSaved:")
print(EXACT_PATH)
print(AUDIT_PATH)
print(GROUP_PATH)
print(REPORT_PATH)

print("\nAll exact locations verified:", complete)
print("Image data downloaded: NONE")
print("Frozen manifest: UNCHANGED")

if not complete:
    print(
        "\nVALIDATION INCOMPLETE: "
        "Resolve missing/duplicate locations "
        "before retrieving image chunks."
    )
else:
    print(
        "\nVALIDATION PASSED: "
        "All 720 scanner-tile records have "
        "one verified source location."
    )

print("\nSTEP 3B.16 COMPLETED")
