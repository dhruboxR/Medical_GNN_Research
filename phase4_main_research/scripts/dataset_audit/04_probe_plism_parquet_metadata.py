
"""
PHASE 4.1.4 - PLISM REMOTE PARQUET METADATA PROBE

Verify the structure of original Owkin PLISM Parquet
shards using remote HTTP byte-range requests.

Probe three representative shards:
    - First
    - Middle
    - Last

Read Parquet schema, row-group metadata, and column
statistics. Do not read PNG column values.

This step does not establish exact tile intersections.
Those require a subsequent metadata-column audit.

Outputs:
    - Parquet shard metadata CSV
    - Row-group metadata CSV
    - JSON audit report

No original research data is modified.
"""

from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import quote

import csv
import hashlib
import json

import fsspec
import pyarrow.parquet as pq


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "phase4_main_research"

INPUT = (
    BASE / "outputs/tables"
    / "01_plism_published_file_inventory.csv"
)

INVENTORY = (
    BASE / "outputs/reports"
    / "01_phase4_dataset_inventory.json"
)

OUTPUT_TABLE = (
    BASE / "outputs/tables"
    / "04_parquet_metadata_probe.csv"
)

OUTPUT_GROUPS = (
    BASE / "outputs/tables"
    / "04_parquet_rowgroup_probe.csv"
)

OUTPUT_REPORT = (
    BASE / "outputs/reports"
    / "04_parquet_metadata_probe.json"
)

DATASET_ID = "owkin/plism-dataset-tiles"

REQUIRED_COLUMNS = {
    "slide_id",
    "tile_id",
    "scanner",
    "stainer",
    "png",
}

METADATA_COLUMNS = (
    "slide_id",
    "tile_id",
    "scanner",
    "stainer",
)

OUTPUTS = (
    OUTPUT_TABLE,
    OUTPUT_GROUPS,
    OUTPUT_REPORT,
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def decode_stat(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None:
        return None
    return str(value)


def csv_write(path, rows, columns):
    with path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(rows)


print("=" * 70, flush=True)
print("PHASE 4.1.4 - PARQUET METADATA PROBE", flush=True)
print("=" * 70, flush=True)

print("\n[1] Checking previous inventory...", flush=True)

for path in (INPUT, INVENTORY):
    if not path.is_file():
        raise RuntimeError(f"Missing input: {path}")

for path in OUTPUTS:
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite previous output: {path}"
        )

with INVENTORY.open(encoding="utf-8") as f:
    previous = json.load(f)

revision = previous.get("remote_revision")

if not revision:
    raise RuntimeError(
        "Missing pinned dataset revision."
    )

with INPUT.open(
    encoding="utf-8",
    newline="",
) as f:
    files = list(csv.DictReader(f))

parquet_files = sorted(
    [
        row["repository_file"]
        for row in files
        if row["repository_file"].lower().endswith(
            ".parquet"
        )
    ]
)

if len(parquet_files) != 293:
    raise RuntimeError(
        f"Expected 293 Parquet shards, "
        f"found {len(parquet_files)}"
    )

positions = sorted(
    set([
        0,
        len(parquet_files) // 2,
        len(parquet_files) - 1,
    ])
)

selected = [
    parquet_files[index]
    for index in positions
]

print("Pinned revision:", revision)
print("Published Parquet shards:", len(parquet_files))
print("Probe shards:", len(selected))

# Remote HTTPFileSystem supplies seekable byte-range
# reads. PyArrow reads Parquet footer metadata.

filesystem = fsspec.filesystem(
    "http",
    block_size=256 * 1024,
    timeout=90,
)

table_rows = []
group_rows = []

print("\n[2] Inspecting remote Parquet footers...", flush=True)

for position, filename in zip(positions, selected):

    url = (
        f"https://huggingface.co/datasets/"
        f"{DATASET_ID}/resolve/{revision}/"
        f"{quote(filename, safe='/')}"
    )

    print(
        f"\n  Shard index {position}: {filename}",
        flush=True,
    )

    try:
        with filesystem.open(
            url,
            mode="rb",
            block_size=256 * 1024,
        ) as stream:

            parquet = pq.ParquetFile(stream)

            metadata = parquet.metadata
            schema = parquet.schema_arrow

            column_names = set(schema.names)

            missing = (
                REQUIRED_COLUMNS - column_names
            )

            if missing:
                raise RuntimeError(
                    f"Missing required columns: {missing}"
                )

            print("    Rows:", metadata.num_rows)
            print("    Row groups:", metadata.num_row_groups)
            print("    Columns:", schema.names)

            table_rows.append({
                "shard_index": position,
                "repository_file": filename,
                "row_count": metadata.num_rows,
                "row_groups": metadata.num_row_groups,
                "schema_columns": ";".join(schema.names),
                "has_required_columns": True,
                "metadata_bytes": (
                    metadata.serialized_size
                ),
            })

            for rg_index in range(
                metadata.num_row_groups
            ):

                rg = metadata.row_group(rg_index)

                result = {
                    "shard_index": position,
                    "repository_file": filename,
                    "row_group": rg_index,
                    "row_count": rg.num_rows,
                    "total_byte_size": rg.total_byte_size,
                }

                for name in METADATA_COLUMNS:

                    column_index = (
                        schema.get_field_index(name)
                    )

                    column = rg.column(column_index)
                    stats = column.statistics

                    minimum = None
                    maximum = None
                    has_min_max = False

                    if (
                        stats is not None
                        and stats.has_min_max
                    ):
                        minimum = decode_stat(stats.min)
                        maximum = decode_stat(stats.max)
                        has_min_max = True

                    result[f"{name}_has_min_max"] = (
                        has_min_max
                    )
                    result[f"{name}_min"] = minimum
                    result[f"{name}_max"] = maximum

                group_rows.append(result)

    except Exception as exc:
        raise RuntimeError(
            "Remote Parquet metadata probe failed for:\n"
            f"{filename}\n"
            f"{type(exc).__name__}: {exc}\n"
            "No image dataset has been downloaded "
            "by this script."
        ) from exc

print("\n[3] Summarizing probe...", flush=True)

total_rows = sum(
    row["row_count"]
    for row in table_rows
)

total_groups = len(group_rows)

print("Probed shards:", len(table_rows))
print("Total probed rows:", total_rows)
print("Total row groups:", total_groups)

for name in METADATA_COLUMNS:
    with_stats = sum(
        bool(row[f"{name}_has_min_max"])
        for row in group_rows
    )

    print(
        f"{name}: min/max statistics "
        f"in {with_stats}/{total_groups} row groups"
    )

print("\nSample slide ID ranges:")

for row in group_rows[:8]:
    print(
        f"  Group {row['row_group']}: "
        f"{row['slide_id_min']} "
        f"to {row['slide_id_max']}"
    )

print("\n[4] Saving metadata audit...", flush=True)

OUTPUT_TABLE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_REPORT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

csv_write(
    OUTPUT_TABLE,
    table_rows,
    [
        "shard_index",
        "repository_file",
        "row_count",
        "row_groups",
        "schema_columns",
        "has_required_columns",
        "metadata_bytes",
    ],
)

group_columns = [
    "shard_index",
    "repository_file",
    "row_group",
    "row_count",
    "total_byte_size",
]

for name in METADATA_COLUMNS:
    group_columns.extend([
        f"{name}_has_min_max",
        f"{name}_min",
        f"{name}_max",
    ])

csv_write(
    OUTPUT_GROUPS,
    group_rows,
    group_columns,
)

report = {
    "phase": "4",
    "step": "4.1.4",
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset": DATASET_ID,
    "pinned_revision": revision,
    "original_inventory_sha256": sha256(INPUT),
    "total_published_shards": len(parquet_files),
    "probed_shards": len(table_rows),
    "probed_rows": total_rows,
    "probed_row_groups": total_groups,
    "required_schema_columns_verified": True,
    "image_column_values_read": False,
    "tile_level_overlap_verified": False,
    "independent_specimens_verified": False,
    "historical_research_modified": False,
    "next_step": (
        "Audit metadata columns across selected "
        "acquisitions and verify exact shared tile IDs."
    ),
}

with OUTPUT_REPORT.open(
    "x",
    encoding="utf-8",
) as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 70)
print("PHASE 4.1.4 - PARQUET PROBE COMPLETED")
print("=" * 70)

print("Parquet shards probed:", len(table_rows))
print("Rows represented:", total_rows)
print("Row groups inspected:", total_groups)

print("\nSaved:")
print(OUTPUT_TABLE)
print(OUTPUT_GROUPS)
print(OUTPUT_REPORT)

print("\nPNG column values: NOT READ")
print("Original images: UNCHANGED")
print("Phase 3B: UNCHANGED")
print("Biological independence: UNVERIFIED")
print("\nSTEP 4.1.4 COMPLETED")
