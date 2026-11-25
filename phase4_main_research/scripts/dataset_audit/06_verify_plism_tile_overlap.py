
"""
PHASE 4.1.6 - EXACT PLISM TILE OVERLAP

Read metadata columns only from the candidate row groups
identified in Step 4.1.5.

Checks:
- Actual slide IDs and tile IDs
- Scanner and stainer values
- Duplicate tile identifiers
- Coordinate parsing
- Seven-scanner GIVH intersection
- AT2/GT450 intersection for all 13 conditions
- Overlap with frozen Phase 3B tile identifiers

Saves exact row-group and row-position locations for
subsequent selective PNG retrieval.

Resumable: one JSON checkpoint per Parquet shard.

No PNG column is explicitly read.
No new images are saved.
No previous research outputs are modified.
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
from urllib.parse import quote
import csv
import hashlib
import json
import re

import fsspec
import pyarrow.parquet as pq


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "phase4_main_research"

TABLES = BASE / "outputs/tables"
REPORTS = BASE / "outputs/reports"
MANIFESTS = BASE / "datasets/manifests"
METADATA = BASE / "datasets/metadata"

CANDIDATES = TABLES / "05_candidate_acquisition_rowgroups.csv"
SELECTED = MANIFESTS / "03_candidate_acquisitions.csv"
PREVIOUS = REPORTS / "05_parquet_rowgroup_index.json"

OLD_MANIFEST = (
    ROOT / "multi_region_research/outputs/validated/"
    "phase3b_frozen_tile_manifest.csv"
)

CHECKPOINTS = METADATA / "06_tile_metadata_checkpoints"

EXACT_ROWS = TABLES / "06_exact_plism_tile_locations.csv"
ACQUISITION_SUMMARY = TABLES / "06_acquisition_tile_summary.csv"
OVERLAPS = TABLES / "06_tile_overlap_summary.csv"
REPORT = REPORTS / "06_exact_tile_overlap_audit.json"

DATASET = "owkin/plism-dataset-tiles"

COLUMNS = ["slide_id", "tile_id", "scanner", "stainer"]
TILE_PATTERN = re.compile(r"^tile_(\d+)_(-?\d+)_(-?\d+)$")

OUTPUTS = (
    EXACT_ROWS, ACQUISITION_SUMMARY, OVERLAPS, REPORT
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path):
    with path.open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with path.open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)


print("=" * 70, flush=True)
print("PHASE 4.1.6 - EXACT PLISM TILE OVERLAP", flush=True)
print("=" * 70, flush=True)

print("\n[1] Verifying inputs...", flush=True)

for path in (
    CANDIDATES, SELECTED, PREVIOUS, OLD_MANIFEST
):
    if not path.is_file():
        raise RuntimeError(f"Missing input: {path}")

for path in OUTPUTS:
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {path}"
        )

with PREVIOUS.open(encoding="utf-8") as stream:
    previous = json.load(stream)

revision = previous["dataset_revision"]

selected = load_csv(SELECTED)
candidate_rows = load_csv(CANDIDATES)

if len(selected) != 31:
    raise RuntimeError("Expected 31 acquisitions.")

selected_ids = {r["slide_id"] for r in selected}

if len(selected_ids) != 31:
    raise RuntimeError("Duplicate selected slide IDs.")

selected_lookup = {
    r["slide_id"]: r
    for r in selected
}

groups_by_file = defaultdict(set)
expected_files = defaultdict(set)

for row in candidate_rows:
    slide_id = row["slide_id"]

    if slide_id not in selected_ids:
        raise RuntimeError(
            f"Unexpected slide: {slide_id}"
        )

    filename = row["repository_file"]
    rg = int(row["row_group"])

    groups_by_file[filename].add(rg)
    expected_files[filename].add(slide_id)

if len(candidate_rows) != 5120:
    raise RuntimeError(
        "Expected 5120 candidate mappings."
    )

with OLD_MANIFEST.open(
    "r", encoding="utf-8", newline=""
) as stream:
    old_rows = list(csv.DictReader(stream))

if len(old_rows) != 360:
    raise RuntimeError(
        "Unexpected historical cohort size."
    )

old_tile_ids = {
    r["tile_id"] for r in old_rows
}

input_hashes = {
    "candidate_index": sha256(CANDIDATES),
    "selection": sha256(SELECTED),
    "previous_report": sha256(PREVIOUS),
    "phase3b_manifest": sha256(OLD_MANIFEST),
}

CHECKPOINTS.mkdir(parents=True, exist_ok=True)

print("Selected acquisitions:", len(selected_ids))
print("Candidate mappings:", len(candidate_rows))
print("Shards requiring metadata reads:", len(groups_by_file))
print("Revision:", revision)


# ============================================================
# STEP 2: FETCH ONLY METADATA COLUMNS
# ============================================================

print("\n[2] Reading exact metadata...", flush=True)

filesystem = fsspec.filesystem(
    "http",
    block_size=256 * 1024,
    timeout=90,
)

filenames = sorted(groups_by_file)
checkpoint_reused = 0

for file_number, filename in enumerate(filenames, 1):
    checkpoint = CHECKPOINTS / (
        hashlib.sha256(
            filename.encode("utf-8")
        ).hexdigest()[:24] + ".json"
    )

    if checkpoint.exists():
        with checkpoint.open(
            encoding="utf-8"
        ) as stream:
            saved = json.load(stream)

        if (
            saved.get("filename") != filename
            or saved.get("revision") != revision
            or saved.get("input_hashes") != input_hashes
            or saved.get("rowgroups")
                != sorted(groups_by_file[filename])
        ):
            raise RuntimeError(
                f"Incompatible checkpoint: {checkpoint}"
            )

        checkpoint_reused += 1
        continue

    url = (
        "https://huggingface.co/datasets/"
        f"{DATASET}/resolve/{revision}/"
        f"{quote(filename, safe='/')}"
    )

    extracted = []

    try:
        with filesystem.open(
            url,
            mode="rb",
            block_size=256 * 1024,
        ) as stream:
            parquet = pq.ParquetFile(stream)

            if not set(COLUMNS).issubset(
                set(parquet.schema_arrow.names)
            ):
                raise RuntimeError(
                    "Missing required metadata columns."
                )

            for rg in sorted(groups_by_file[filename]):
                table = parquet.read_row_group(
                    rg,
                    columns=COLUMNS,
                )

                columns = table.to_pydict()
                n = table.num_rows

                for row_index in range(n):
                    slide_id = columns[
                        "slide_id"
                    ][row_index]

                    if slide_id not in selected_ids:
                        continue

                    tile_id = columns[
                        "tile_id"
                    ][row_index]

                    if not isinstance(tile_id, str):
                        raise RuntimeError(
                            "Invalid tile_id type."
                        )

                    match = TILE_PATTERN.fullmatch(tile_id)

                    if match is None:
                        raise RuntimeError(
                            f"Unrecognized tile ID: {tile_id}"
                        )

                    level, x, y = map(
                        int, match.groups()
                    )

                    extracted.append({
                        "slide_id": slide_id,
                        "tile_id": tile_id,
                        "level": level,
                        "x": x,
                        "y": y,
                        "scanner": columns[
                            "scanner"
                        ][row_index],
                        "stainer": columns[
                            "stainer"
                        ][row_index],
                        "repository_file": filename,
                        "row_group": rg,
                        "row_in_group": row_index,
                    })

    except Exception as exc:
        raise RuntimeError(
            f"Metadata read failed: {filename}\n"
            f"{type(exc).__name__}: {exc}\n"
            "Completed shard checkpoints remain intact. "
            "Rerun to resume."
        ) from exc

    payload = {
        "filename": filename,
        "revision": revision,
        "input_hashes": input_hashes,
        "rowgroups": sorted(groups_by_file[filename]),
        "records": extracted,
    }

    # Write checkpoint only after all selected row groups
    # from this shard have been successfully processed.
    with checkpoint.open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(payload, stream)

    if file_number % 20 == 0 or file_number == len(filenames):
        print(
            f"  Completed {file_number}/{len(filenames)} "
            "shards",
            flush=True,
        )


# ============================================================
# STEP 3: BUILD EXACT ACQUISITION INVENTORY
# ============================================================

print("\n[3] Verifying exact tile sets...", flush=True)

all_records = []
tile_sets = defaultdict(set)
tile_counts = Counter()
scanners = defaultdict(set)
stainers = defaultdict(set)
duplicates = Counter()
seen_locations = set()

for filename in filenames:
    checkpoint = CHECKPOINTS / (
        hashlib.sha256(
            filename.encode("utf-8")
        ).hexdigest()[:24] + ".json"
    )

    with checkpoint.open(encoding="utf-8") as stream:
        payload = json.load(stream)

    if (
        payload.get("filename") != filename
        or payload.get("revision") != revision
        or payload.get("input_hashes") != input_hashes
    ):
        raise RuntimeError(
            f"Checkpoint integrity mismatch: {checkpoint}"
        )

    for record in payload["records"]:
        slide = record["slide_id"]
        tile = record["tile_id"]

        location = (
            filename,
            record["row_group"],
            record["row_in_group"],
        )

        if location in seen_locations:
            raise RuntimeError(
                "Duplicate physical row location."
            )

        seen_locations.add(location)

        tile_counts[slide] += 1
        scanners[slide].add(record["scanner"])
        stainers[slide].add(record["stainer"])

        if tile in tile_sets[slide]:
            duplicates[slide] += 1

        tile_sets[slide].add(tile)
        all_records.append(record)

missing = sorted(
    selected_ids - set(tile_counts)
)

if missing:
    raise RuntimeError(
        "Selected acquisitions without exact records:\n"
        + "\n".join(missing)
    )

summary = []

for slide_id in sorted(selected_ids):
    entry = selected_lookup[slide_id]

    summary.append({
        "slide_id": slide_id,
        "source_condition": entry["source_condition"],
        "scanner_token": entry["scanner_token"],
        "total_rows": tile_counts[slide_id],
        "unique_tile_ids": len(tile_sets[slide_id]),
        "duplicate_tile_ids": duplicates[slide_id],
        "scanner_values": ";".join(
            sorted(map(str, scanners[slide_id]))
        ),
        "stainer_values": ";".join(
            sorted(map(str, stainers[slide_id]))
        ),
        "overlap_with_phase3b_positions": len(
            tile_sets[slide_id] & old_tile_ids
        ),
    })

print("Exact tile records:", len(all_records))
print("Acquisitions with exact records:", len(tile_sets))
print("Duplicate tile IDs:", sum(duplicates.values()))

print("\nPer-acquisition verification:")
for record in summary:
    print(
        f"  {record['slide_id']}: "
        f"{record['unique_tile_ids']} unique tiles, "
        f"{record['duplicate_tile_ids']} duplicates",
        flush=True,
    )


# ============================================================
# STEP 4: CALCULATE ACQUISITION INTERSECTIONS
# ============================================================

print("\n[4] Calculating exact tile intersections...", flush=True)

overlap_rows = []

def add_overlap(label, slide_ids):
    shared = set.intersection(
        *(tile_sets[s] for s in slide_ids)
    )

    overlap_rows.append({
        "comparison": label,
        "acquisition_count": len(slide_ids),
        "slide_ids": ";".join(sorted(slide_ids)),
        "shared_tile_positions": len(shared),
        "shared_with_phase3b": len(
            shared & old_tile_ids
        ),
    })

    return len(shared)


# Seven scanners within GIVH.
givh = [
    r["slide_id"]
    for r in selected
    if r["source_condition"] == "GIVH"
]

if len(givh) != 7:
    raise RuntimeError(
        "Expected seven GIVH acquisitions."
    )

seven_way = add_overlap(
    "GIVH_all_7_scanners",
    givh,
)

print(
    "GIVH seven-scanner shared positions:",
    seven_way,
)


# AT2 and GT450 within each condition.
conditions = sorted({
    r["source_condition"]
    for r in selected
})

for condition in conditions:
    pair = [
        r["slide_id"]
        for r in selected
        if (
            r["source_condition"] == condition
            and r["scanner_token"] in {"AT2", "GT450"}
        )
    ]

    if len(pair) != 2:
        raise RuntimeError(
            f"Missing AT2/GT450 pair: {condition}"
        )

    shared = add_overlap(
        f"{condition}_AT2_GT450",
        pair,
    )

    print(
        f"  {condition} AT2/GT450: "
        f"{shared} shared positions",
        flush=True,
    )


# Shared positions across all 31 acquisitions.
all_31 = add_overlap(
    "all_31_acquisitions",
    sorted(selected_ids),
)

print(
    "All 31 acquisitions shared positions:",
    all_31,
)


# ============================================================
# STEP 5: SAVE EXACT INDEX
# ============================================================

print("\n[5] Saving verified metadata...", flush=True)

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

write_csv(
    EXACT_ROWS,
    all_records,
    [
        "slide_id",
        "tile_id",
        "level",
        "x",
        "y",
        "scanner",
        "stainer",
        "repository_file",
        "row_group",
        "row_in_group",
    ],
)

write_csv(
    ACQUISITION_SUMMARY,
    summary,
    [
        "slide_id",
        "source_condition",
        "scanner_token",
        "total_rows",
        "unique_tile_ids",
        "duplicate_tile_ids",
        "scanner_values",
        "stainer_values",
        "overlap_with_phase3b_positions",
    ],
)

write_csv(
    OVERLAPS,
    overlap_rows,
    [
        "comparison",
        "acquisition_count",
        "slide_ids",
        "shared_tile_positions",
        "shared_with_phase3b",
    ],
)

report = {
    "phase": "4",
    "step": "4.1.6",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset": DATASET,
    "dataset_revision": revision,
    "selected_acquisitions": len(selected_ids),
    "exact_tile_records": len(all_records),
    "duplicate_tile_ids": sum(duplicates.values()),
    "acquisitions_with_exact_records": len(tile_sets),
    "givh_seven_scanner_shared_positions": seven_way,
    "all_31_shared_positions": all_31,
    "previous_index_sha256": sha256(CANDIDATES),
    "selection_sha256": sha256(SELECTED),
    "phase3b_manifest_sha256": sha256(OLD_MANIFEST),
    "shard_checkpoints_reused": checkpoint_reused,
    "image_column_values_requested": False,
    "original_images_downloaded": 0,
    "biological_independence_verified": False,
    "final_train_validation_test_split_frozen": False,
    "next_step": (
        "Choose spatial regions using verified tile "
        "intersections, exclude frozen pilot positions "
        "where appropriate, establish leakage-aware "
        "evaluation partitions, then download original PNGs."
    ),
}

with REPORT.open(
    "x", encoding="utf-8"
) as stream:
    json.dump(report, stream, indent=2)

print("\n" + "=" * 70, flush=True)
print("PHASE 4.1.6 - EXACT TILE OVERLAP COMPLETED", flush=True)
print("=" * 70, flush=True)

print("Exact tile records:", len(all_records))
print("Acquisitions verified:", len(tile_sets))
print("GIVH seven-scanner intersection:", seven_way)
print("All-31 acquisition intersection:", all_31)

print("\nSaved:")
print(EXACT_ROWS)
print(ACQUISITION_SUMMARY)
print(OVERLAPS)
print(REPORT)

print("\nPNG column explicitly read: NO")
print("Original images downloaded: NONE")
print("Phase 3B data: UNCHANGED")
print("Biological independence: UNVERIFIED")
print("\nSTEP 4.1.6 COMPLETED")
