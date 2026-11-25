
"""
PHASE 4.1.5 - FULL PLISM PARQUET ROW-GROUP INDEX

Objective:
Find row groups that could contain the 31 selected
PLISM acquisitions using Parquet footer statistics.

No PNG columns are read.
No image tiles are downloaded.
No Phase 3B files are changed.

A row group is a CANDIDATE for a slide when:
    slide_id_min <= selected_slide_id <= slide_id_max

This is a metadata-range test, not proof that a
slide or tile exists within that row group.

Resumable:
One small JSON checkpoint is saved per shard.
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

TABLES = BASE / "outputs/tables"
REPORTS = BASE / "outputs/reports"
MANIFESTS = BASE / "datasets/manifests"

PUBLISHED = TABLES / "01_plism_published_file_inventory.csv"
INVENTORY = REPORTS / "01_phase4_dataset_inventory.json"
SELECTED = MANIFESTS / "03_candidate_acquisitions.csv"

CHECKPOINTS = (
    BASE / "datasets/metadata/05_rowgroup_checkpoints"
)

ALL_GROUPS = TABLES / "05_all_parquet_rowgroups.csv"
CANDIDATES = TABLES / "05_candidate_acquisition_rowgroups.csv"
SUMMARY = TABLES / "05_acquisition_rowgroup_summary.csv"
REPORT = REPORTS / "05_parquet_rowgroup_index.json"

DATASET_ID = "owkin/plism-dataset-tiles"

FINAL_OUTPUTS = (
    ALL_GROUPS,
    CANDIDATES,
    SUMMARY,
    REPORT,
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        return list(csv.DictReader(stream))


def write_csv_new(path, records, fieldnames):
    with path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(records)


def stat_text(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return str(value)


def get_stats(rowgroup, schema, column):
    index = schema.get_field_index(column)

    if index < 0:
        raise RuntimeError(f"Missing Parquet column: {column}")

    statistics = rowgroup.column(index).statistics

    if statistics is None or not statistics.has_min_max:
        return None, None

    return (
        stat_text(statistics.min),
        stat_text(statistics.max),
    )


def inspect_shard(fs, filename, revision):
    url = (
        f"https://huggingface.co/datasets/"
        f"{DATASET_ID}/resolve/{revision}/"
        f"{quote(filename, safe='/')}"
    )

    records = []
    with fs.open(
        url,
        mode="rb",
        block_size=256 * 1024,
    ) as stream:
        parquet = pq.ParquetFile(stream)
        metadata = parquet.metadata
        schema = parquet.schema_arrow

        required = {
            "slide_id", "tile_id", "scanner",
            "stainer", "png",
        }

        if not required.issubset(set(schema.names)):
            raise RuntimeError(
                f"Unexpected schema in {filename}"
            )

        for rg_id in range(metadata.num_row_groups):
            group = metadata.row_group(rg_id)

            slide_min, slide_max = get_stats(
                group, schema, "slide_id"
            )

            if slide_min is None or slide_max is None:
                raise RuntimeError(
                    f"No slide_id range: {filename} rg={rg_id}"
                )

            tile_min, tile_max = get_stats(
                group, schema, "tile_id"
            )

            scanner_min, scanner_max = get_stats(
                group, schema, "scanner"
            )

            stainer_min, stainer_max = get_stats(
                group, schema, "stainer"
            )

            records.append({
                "repository_file": filename,
                "row_group": rg_id,
                "row_count": group.num_rows,
                "slide_id_min": slide_min,
                "slide_id_max": slide_max,
                "tile_id_min": tile_min,
                "tile_id_max": tile_max,
                "scanner_min": scanner_min,
                "scanner_max": scanner_max,
                "stainer_min": stainer_min,
                "stainer_max": stainer_max,
            })

        if sum(r["row_count"] for r in records) != metadata.num_rows:
            raise RuntimeError("Row-group row-count mismatch.")

    return records


print("=" * 70, flush=True)
print("PHASE 4.1.5 - FULL PLISM ROW-GROUP INDEX", flush=True)
print("=" * 70, flush=True)

print("\n[1] Verifying research inputs...", flush=True)

for source in (PUBLISHED, INVENTORY, SELECTED):
    if not source.is_file():
        raise RuntimeError(f"Missing input: {source}")

for path in FINAL_OUTPUTS:
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite final output: {path}"
        )

with INVENTORY.open(encoding="utf-8") as stream:
    previous = json.load(stream)

revision = previous["remote_revision"]

if not revision:
    raise RuntimeError("Missing remote revision.")

publication = read_csv(PUBLISHED)
selected = read_csv(SELECTED)

files = sorted({
    row["repository_file"]
    for row in publication
    if row["repository_file"].endswith(".parquet")
})

if len(files) != 293:
    raise RuntimeError("Expected exactly 293 Parquet shards.")

if len(selected) != 31:
    raise RuntimeError("Expected 31 selected acquisitions.")

slide_ids = sorted({
    row["slide_id"] for row in selected
})

if len(slide_ids) != 31:
    raise RuntimeError("Duplicate selected acquisitions.")

publication_hash = digest(PUBLISHED)
selection_hash = digest(SELECTED)

CHECKPOINTS.mkdir(parents=True, exist_ok=True)

print("Pinned dataset revision:", revision)
print("Parquet shards:", len(files))
print("Selected acquisitions:", len(slide_ids))

print("\n[2] Inspecting metadata footers...", flush=True)

filesystem = fsspec.filesystem(
    "http",
    block_size=256 * 1024,
    timeout=90,
)

all_records = []
resumed = 0

for index, filename in enumerate(files):
    checkpoint = CHECKPOINTS / f"shard_{index:03d}.json"

    if checkpoint.exists():
        with checkpoint.open(encoding="utf-8") as stream:
            saved = json.load(stream)

        if (
            saved.get("repository_file") != filename
            or saved.get("revision") != revision
            or saved.get("publication_sha256") != publication_hash
        ):
            raise RuntimeError(
                f"Stale or incompatible checkpoint: {checkpoint}"
            )

        shard_records = saved["rowgroups"]
        resumed += 1

    else:
        try:
            shard_records = inspect_shard(
                filesystem, filename, revision
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed at shard {index + 1}/293: {filename}\n"
                f"{type(exc).__name__}: {exc}\n"
                "Previously saved checkpoints are preserved. "
                "Rerun the same script to resume."
            ) from exc

        checkpoint_data = {
            "repository_file": filename,
            "revision": revision,
            "publication_sha256": publication_hash,
            "rowgroups": shard_records,
        }

        # Exclusive creation prevents accidental replacement.
        with checkpoint.open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(checkpoint_data, stream)

    all_records.extend(shard_records)

    if (index + 1) % 20 == 0 or index + 1 == len(files):
        print(
            f"  Processed {index + 1}/{len(files)} shards; "
            f"row groups indexed: {len(all_records)}",
            flush=True,
        )


print("\n[3] Selecting candidate row groups...", flush=True)

candidate_records = []
counts = {slide_id: 0 for slide_id in slide_ids}
candidate_rows = {slide_id: 0 for slide_id in slide_ids}

for row in all_records:
    lower = row["slide_id_min"]
    upper = row["slide_id_max"]

    if lower > upper:
        raise RuntimeError("Invalid slide ID statistics.")

    for slide_id in slide_ids:
        if lower <= slide_id <= upper:
            candidate_records.append({
                "slide_id": slide_id,
                "repository_file": row["repository_file"],
                "row_group": row["row_group"],
                "row_group_rows": row["row_count"],
                "slide_id_min": lower,
                "slide_id_max": upper,
                "is_exact_single_slide_group": lower == upper == slide_id,
            })

            counts[slide_id] += 1
            candidate_rows[slide_id] += row["row_count"]

summary_records = []

for entry in selected:
    slide_id = entry["slide_id"]

    summary_records.append({
        "slide_id": slide_id,
        "source_condition": entry["source_condition"],
        "scanner_token": entry["scanner_token"],
        "candidate_rowgroups": counts[slide_id],
        "candidate_rowgroup_total_rows": candidate_rows[slide_id],
        "exact_tile_count_verified": False,
    })

missing = [
    slide_id
    for slide_id, count in counts.items()
    if count == 0
]

if missing:
    print("\nWARNING: Selected slides with no candidate row groups:")
    for slide_id in missing:
        print(" ", slide_id)

print("Total row groups indexed:", len(all_records))
print("Candidate slide/row-group mappings:", len(candidate_records))
print("Selected slides with candidates:", len(slide_ids) - len(missing))

print("\nCandidate row-group counts by acquisition:")
for entry in summary_records:
    print(
        f"  {entry['slide_id']}: "
        f"{entry['candidate_rowgroups']} groups",
        flush=True,
    )


print("\n[4] Saving full metadata index...", flush=True)

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

group_fields = [
    "repository_file",
    "row_group",
    "row_count",
    "slide_id_min",
    "slide_id_max",
    "tile_id_min",
    "tile_id_max",
    "scanner_min",
    "scanner_max",
    "stainer_min",
    "stainer_max",
]

write_csv_new(
    ALL_GROUPS,
    all_records,
    group_fields,
)

write_csv_new(
    CANDIDATES,
    candidate_records,
    [
        "slide_id",
        "repository_file",
        "row_group",
        "row_group_rows",
        "slide_id_min",
        "slide_id_max",
        "is_exact_single_slide_group",
    ],
)

write_csv_new(
    SUMMARY,
    summary_records,
    [
        "slide_id",
        "source_condition",
        "scanner_token",
        "candidate_rowgroups",
        "candidate_rowgroup_total_rows",
        "exact_tile_count_verified",
    ],
)

report = {
    "phase": "4",
    "step": "4.1.5",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset_revision": revision,
    "published_inventory_sha256": publication_hash,
    "selected_acquisition_sha256": selection_hash,
    "shards_indexed": len(files),
    "shards_loaded_from_checkpoints": resumed,
    "total_row_groups": len(all_records),
    "candidate_slide_rowgroup_mappings": len(candidate_records),
    "selected_acquisitions": len(slide_ids),
    "selected_acquisitions_with_candidates": len(slide_ids) - len(missing),
    "missing_candidate_slides": missing,
    "image_column_values_read": False,
    "exact_tile_overlap_verified": False,
    "biological_independence_verified": False,
    "phase3b_modified": False,
    "next_step": (
        "Read only metadata columns from candidate row groups "
        "to verify actual scanner/slide IDs, tile coordinates "
        "and exact intersection sizes."
    ),
}

with REPORT.open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)

print("\n" + "=" * 70, flush=True)
print("PHASE 4.1.5 - ROW-GROUP INDEX COMPLETED", flush=True)
print("=" * 70, flush=True)

print("Shards indexed:", len(files))
print("Row groups indexed:", len(all_records))
print("Selected acquisitions:", len(slide_ids))
print("Acquisitions with candidate groups:", len(slide_ids) - len(missing))

print("\nSaved:")
print(ALL_GROUPS)
print(CANDIDATES)
print(SUMMARY)
print(REPORT)

print("\nPNG images downloaded: NONE")
print("Tile-coordinate intersections: NOT YET VERIFIED")
print("Phase 3B: UNCHANGED")
print("Biological independence: UNVERIFIED")
print("\nSTEP 4.1.5 COMPLETED")
