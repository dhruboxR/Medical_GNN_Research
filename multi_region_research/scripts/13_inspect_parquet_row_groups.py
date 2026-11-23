
"""
PHASE 3B - STEP 3B.13
Remote PLISM Parquet Footer and Row-Group Inspection

Purpose:
    Determine whether original PNG image retrieval can use
    bounded HTTP byte-range requests.

Method:
    1. Read the final 8 bytes of each remote Parquet file.
    2. Decode the footer metadata length.
    3. Retrieve only the footer metadata bytes.
    4. Parse metadata using PyArrow.
    5. Inspect row groups, tile_id and png column chunks.
    6. Report available tile_id statistics.

No image data is downloaded.
No complete Parquet shards are downloaded.
No changes to the frozen sampling manifest.
"""

from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
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

DATASET = "owkin/plism-dataset-tiles"

SHARD_INDICES = [0, 1, 2]
TOTAL_SHARDS = 293

BASE_URL = (
    "https://huggingface.co/datasets/"
    f"{DATASET}/resolve/main/data/"
)

MAX_FOOTER_BYTES = 16 * 1024 * 1024
TIMEOUT = (10, 30)


# ============================================================
# HELPERS
# ============================================================

def shard_name(index):
    return (
        f"train-{index:05d}-of-"
        f"{TOTAL_SHARDS:05d}.parquet"
    )


def fetch_exact_range(session, url, start, end):
    """
    Fetch an inclusive byte range.

    Reject full-file responses and unexpected ranges.
    """

    expected_length = end - start + 1

    headers = {
        "Range": f"bytes={start}-{end}",
        "Accept-Encoding": "identity",
    }

    with session.get(
        url,
        headers=headers,
        stream=True,
        timeout=TIMEOUT,
        allow_redirects=True,
    ) as response:

        if response.status_code != 206:
            raise RuntimeError(
                "Server did not return partial content: "
                f"HTTP {response.status_code}"
            )

        content_range = response.headers.get(
            "Content-Range", ""
        )

        if not content_range.startswith(
            f"bytes {start}-{end}/"
        ):
            raise RuntimeError(
                "Unexpected Content-Range: "
                f"{content_range}"
            )

        content = bytearray()

        for chunk in response.iter_content(16384):
            content.extend(chunk)

            if len(content) > expected_length:
                raise RuntimeError(
                    "Range response exceeded expected size"
                )

    if len(content) != expected_length:
        raise RuntimeError(
            "Incomplete byte-range response: "
            f"{len(content)} != {expected_length}"
        )

    return bytes(content)


def get_remote_file_size(session, url):
    response = session.head(
        url,
        allow_redirects=True,
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    value = response.headers.get(
        "Content-Length"
    )

    if value is None or not value.isdigit():
        raise RuntimeError(
            "Missing remote Content-Length"
        )

    return int(value)


def decode_statistic(value):
    """
    Convert Parquet statistics into JSON-safe text.
    """

    if value is None:
        return None

    if isinstance(value, bytes):
        return value.decode(
            "utf-8", errors="replace"
        )

    return str(value)


def read_remote_parquet_metadata(session, url):
    file_size = get_remote_file_size(
        session, url
    )

    if file_size < 12:
        raise RuntimeError(
            "File too small to contain Parquet metadata"
        )

    # Final 8 bytes:
    #   4-byte little-endian metadata length
    #   4-byte PAR1 magic
    footer = fetch_exact_range(
        session,
        url,
        file_size - 8,
        file_size - 1,
    )

    metadata_length = struct.unpack(
        "<I", footer[:4]
    )[0]

    if footer[4:] != b"PAR1":
        raise RuntimeError(
            "Invalid Parquet footer signature"
        )

    if not (
        0 < metadata_length <= MAX_FOOTER_BYTES
    ):
        raise RuntimeError(
            "Parquet metadata exceeds configured "
            f"safety limit: {metadata_length}"
        )

    metadata_start = (
        file_size - 8 - metadata_length
    )

    if metadata_start < 4:
        raise RuntimeError(
            "Invalid metadata offset"
        )

    metadata_bytes = fetch_exact_range(
        session,
        url,
        metadata_start,
        file_size - 9,
    )

    # Construct a small in-memory Parquet container
    # containing a header and the original footer.
    #
    # This is used ONLY for metadata parsing.
    # Column data offsets still refer to the remote file.
    synthetic_file = (
        b"PAR1"
        + metadata_bytes
        + footer
    )

    metadata = pq.read_metadata(
        io.BytesIO(synthetic_file)
    )

    return {
        "file_size": file_size,
        "footer_bytes": metadata_length,
        "footer_start": metadata_start,
        "metadata": metadata,
    }


# ============================================================
# START
# ============================================================

print("=" * 70, flush=True)
print("PHASE 3B - REMOTE PARQUET METADATA INSPECTION", flush=True)
print("=" * 70, flush=True)

shard_records = []
row_group_records = []
column_records = []

with requests.Session() as session:

    for shard_index in SHARD_INDICES:

        name = shard_name(shard_index)
        url = BASE_URL + name

        print(
            f"\n[1] Inspecting {name}...",
            flush=True,
        )

        remote = read_remote_parquet_metadata(
            session, url
        )

        metadata = remote["metadata"]

        print(
            "  Remote file size:",
            remote["file_size"],
            "bytes",
            flush=True,
        )

        print(
            "  Footer metadata:",
            remote["footer_bytes"],
            "bytes",
            flush=True,
        )

        print(
            "  Total rows:",
            metadata.num_rows,
            flush=True,
        )

        print(
            "  Row groups:",
            metadata.num_row_groups,
            flush=True,
        )

        columns = metadata.schema.names

        print(
            "  Schema columns:",
            columns,
            flush=True,
        )

        shard_records.append({
            "shard": name,
            "file_size_bytes": remote["file_size"],
            "footer_metadata_bytes": remote["footer_bytes"],
            "num_rows": metadata.num_rows,
            "num_row_groups": metadata.num_row_groups,
            "schema_columns": ";".join(columns),
        })

        # ====================================================
        # INSPECT ROW GROUPS
        # ====================================================

        print(
            "\n[2] Inspecting row-group column chunks...",
            flush=True,
        )

        for rg_index in range(
            metadata.num_row_groups
        ):

            group = metadata.row_group(
                rg_index
            )

            row_group_records.append({
                "shard": name,
                "row_group": rg_index,
                "num_rows": group.num_rows,
                "total_byte_size": group.total_byte_size,
                "num_columns": group.num_columns,
            })

            for col_index in range(
                group.num_columns
            ):

                col = group.column(col_index)

                column_path = col.path_in_schema

                stats = col.statistics

                has_min_max = bool(
                    stats is not None
                    and stats.has_min_max
                )

                minimum = (
                    decode_statistic(stats.min)
                    if has_min_max
                    else None
                )

                maximum = (
                    decode_statistic(stats.max)
                    if has_min_max
                    else None
                )

                column_records.append({
                    "shard": name,
                    "row_group": rg_index,
                    "column": column_path,
                    "compression": col.compression,
                    "compressed_size": col.total_compressed_size,
                    "uncompressed_size": col.total_uncompressed_size,
                    "data_page_offset": col.data_page_offset,
                    "dictionary_page_offset":
                        col.dictionary_page_offset,
                    "has_min_max": has_min_max,
                    "minimum": minimum,
                    "maximum": maximum,
                })

        # ====================================================
        # DISPLAY TARGET COLUMN SUMMARY
        # ====================================================

        shard_cols = [
            row for row in column_records
            if row["shard"] == name
        ]

        for target in ("tile_id", "png"):

            matches = [
                row for row in shard_cols
                if row["column"] == target
                or row["column"].startswith(
                    target + "."
                )
            ]

            if not matches:
                print(
                    f"  {target}: not found",
                    flush=True,
                )
                continue

            compressed_sizes = [
                row["compressed_size"]
                for row in matches
            ]

            print(
                f"\n  Column: {target}",
                flush=True,
            )

            print(
                "    Column chunks:",
                len(matches)
            )

            print(
                "    Total compressed bytes:",
                sum(compressed_sizes)
            )

            print(
                "    Largest compressed chunk:",
                max(compressed_sizes)
            )

            print(
                "    Chunks with min/max:",
                sum(
                    row["has_min_max"]
                    for row in matches
                )
            )

            if target == "tile_id":
                for row in matches[:5]:
                    print(
                        f"    RG {row['row_group']}: "
                        f"min={row['minimum']} | "
                        f"max={row['maximum']}"
                    )


# ============================================================
# SAVE TABLES
# ============================================================

print("\n[3] Saving metadata inventory...", flush=True)

shards_df = pd.DataFrame(shard_records)
groups_df = pd.DataFrame(row_group_records)
columns_df = pd.DataFrame(column_records)

shards_path = (
    TABLES /
    "plism_remote_parquet_shard_metadata.csv"
)

groups_path = (
    TABLES /
    "plism_remote_parquet_row_groups.csv"
)

columns_path = (
    TABLES /
    "plism_remote_parquet_column_chunks.csv"
)

report_path = (
    REPORTS /
    "13_remote_parquet_metadata_inspection.json"
)

shards_df.to_csv(
    shards_path, index=False
)

groups_df.to_csv(
    groups_path, index=False
)

columns_df.to_csv(
    columns_path, index=False
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

tile_columns = columns_df[
    columns_df["column"] == "tile_id"
]

png_columns = columns_df[
    columns_df["column"].eq("png")
    | columns_df["column"].str.startswith("png.")
]

tile_statistics_available = bool(
    not tile_columns.empty
    and tile_columns["has_min_max"].any()
)

report = {
    "phase": "3B",
    "step": "3B.13",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset": DATASET,
    "shards_inspected": len(shards_df),
    "total_rows_in_tested_shards": int(
        shards_df["num_rows"].sum()
    ),
    "total_row_groups": int(
        shards_df["num_row_groups"].sum()
    ),
    "tile_id_min_max_available":
        tile_statistics_available,
    "total_png_compressed_bytes": int(
        png_columns["compressed_size"].sum()
    ) if not png_columns.empty else 0,
    "metadata_only": True,
    "image_bytes_downloaded": 0,
    "full_parquet_files_downloaded": 0,
    "individual_tile_retrieval_feasible":
        "NOT_YET_ESTABLISHED",
    "frozen_manifest_modified": False,
}

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3B.13 - METADATA INSPECTION COMPLETED")
print("=" * 70)

print(
    "Shards inspected:",
    len(shards_df)
)

print(
    "Total rows inspected via metadata:",
    int(shards_df["num_rows"].sum())
)

print(
    "Total row groups:",
    int(shards_df["num_row_groups"].sum())
)

print(
    "Tile-ID min/max statistics available:",
    tile_statistics_available
)

if not groups_df.empty:
    print(
        "Smallest row group:",
        int(groups_df["num_rows"].min()),
        "rows"
    )

    print(
        "Largest row group:",
        int(groups_df["num_rows"].max()),
        "rows"
    )

if not png_columns.empty:
    print(
        "Largest PNG column chunk:",
        int(png_columns["compressed_size"].max()),
        "bytes"
    )

print("\nSaved:")
print(shards_path)
print(groups_path)
print(columns_path)
print(report_path)

print("\nOriginal images downloaded: NONE")
print("Frozen manifest: UNCHANGED")
print("Individual tile retrieval: NOT YET VERIFIED")
print("\nSTEP 3B.13 COMPLETED")
