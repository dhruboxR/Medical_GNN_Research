
"""
PHASE 4.1.8 - STAGED PLISM IMAGE DOWNLOAD MANIFEST

Prepare exact source locations for the first Phase 4
image download, using existing verified Parquet metadata.

Stage A:
    Condition: GIVH
    Scanners: AT2, GT450
    Regions: 60
    Positions: 2,160
    Expected original PNGs: 4,320

Preserve spatial train/validation/test assignments.

This script:
- Validates the provisional region manifest
- Validates the selected acquisitions
- Resolves every requested original image to one
  verified Parquet shard, row group and row position
- Records the download destination and split
- Checks for duplicate records and missing pairs
- Saves manifest and audit report

It does NOT read PNG payloads or download images.
It does NOT freeze the final evaluation protocol.
"""

from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "phase4_main_research"

REGIONS = (
    BASE / "datasets/manifests"
    / "07_provisional_phase4_regions.csv"
)

ACQUISITIONS = (
    BASE / "datasets/manifests"
    / "03_candidate_acquisitions.csv"
)

LOCATIONS = (
    BASE / "outputs/tables"
    / "06_exact_plism_tile_locations.csv"
)

INVENTORY = (
    BASE / "outputs/reports"
    / "01_phase4_dataset_inventory.json"
)

MANIFEST = (
    BASE / "datasets/manifests"
    / "08_stage_a_original_png_download_manifest.csv"
)

SUMMARY = (
    BASE / "outputs/tables"
    / "08_stage_a_download_summary.csv"
)

REPORT = (
    BASE / "outputs/reports"
    / "08_stage_a_download_manifest_audit.json"
)

CONDITION = "GIVH"
SCANNERS = ("AT2", "GT450")

EXPECTED_REGIONS = 60
EXPECTED_POSITIONS = 2160
EXPECTED_IMAGES = 4320

EXPECTED_SPLITS = {
    "train": 40,
    "validation": 10,
    "test": 10,
}

OUTPUTS = (MANIFEST, SUMMARY, REPORT)


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_csv(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, columns):
    with path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(rows)


print("=" * 70, flush=True)
print(
    "PHASE 4.1.8 - STAGE A DOWNLOAD MANIFEST",
    flush=True,
)
print("=" * 70, flush=True)

print("\n[1] Validating previous research metadata...", flush=True)

for path in (REGIONS, ACQUISITIONS, LOCATIONS, INVENTORY):
    if not path.is_file():
        raise RuntimeError(f"Missing input: {path}")

for path in OUTPUTS:
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite previous output: {path}"
        )

regions = read_csv(REGIONS)
acquisitions = read_csv(ACQUISITIONS)

with INVENTORY.open(encoding="utf-8") as stream:
    inventory = json.load(stream)

revision = inventory["remote_revision"]

if not revision:
    raise RuntimeError("Missing pinned dataset revision.")

if len(regions) != EXPECTED_POSITIONS:
    raise RuntimeError("Unexpected region manifest size.")

region_ids = {
    r["region_id"] for r in regions
}

if len(region_ids) != EXPECTED_REGIONS:
    raise RuntimeError("Expected 60 regions.")

region_split = {}
position_lookup = {}

for row in regions:
    region_id = row["region_id"]
    split = row["spatial_split"]
    tile_id = row["tile_id"]

    if split not in EXPECTED_SPLITS:
        raise RuntimeError(f"Unexpected split: {split}")

    if (
        region_id in region_split
        and region_split[region_id] != split
    ):
        raise RuntimeError(
            "Region assigned to multiple splits."
        )

    region_split[region_id] = split

    if tile_id in position_lookup:
        raise RuntimeError(
            f"Duplicate selected tile: {tile_id}"
        )

    expected_id = (
        f"tile_{row['level']}_{row['x']}_{row['y']}"
    )

    if tile_id != expected_id:
        raise RuntimeError(
            f"Tile coordinate mismatch: {tile_id}"
        )

    position_lookup[tile_id] = row


split_counts = Counter(region_split.values())

if dict(split_counts) != EXPECTED_SPLITS:
    raise RuntimeError(
        f"Unexpected spatial split counts: {split_counts}"
    )

region_sizes = Counter(
    r["region_id"] for r in regions
)

if any(count != 36 for count in region_sizes.values()):
    raise RuntimeError("Every region must have 36 tiles.")

if len(acquisitions) != 31:
    raise RuntimeError("Expected 31 candidate acquisitions.")

selected_slides = {}

for scanner in SCANNERS:
    matches = [
        row["slide_id"]
        for row in acquisitions
        if row["source_condition"] == CONDITION
        and row["scanner_token"] == scanner
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one GIVH acquisition for {scanner}."
        )

    selected_slides[scanner] = matches[0]

print("Regions:", len(region_ids))
print("Tile positions:", len(position_lookup))
print("Scanners:", ", ".join(SCANNERS))
print("Pinned revision:", revision)


print("\n[2] Resolving exact original PNG locations...", flush=True)

required = set(selected_slides.values())

locations = {}
matched_rows = Counter()

with LOCATIONS.open(
    newline="",
    encoding="utf-8",
) as stream:
    reader = csv.DictReader(stream)

    required_columns = {
        "slide_id",
        "tile_id",
        "repository_file",
        "row_group",
        "row_in_group",
        "scanner",
        "stainer",
    }

    if not required_columns.issubset(reader.fieldnames or []):
        raise RuntimeError(
            "Missing expected exact-location columns."
        )

    for record in reader:
        slide_id = record["slide_id"]
        tile_id = record["tile_id"]

        if slide_id not in required:
            continue

        if tile_id not in position_lookup:
            continue

        key = (slide_id, tile_id)

        if key in locations:
            raise RuntimeError(
                f"Duplicate original PNG location: {key}"
            )

        if not record["repository_file"].endswith(".parquet"):
            raise RuntimeError(
                "Invalid Parquet source filename."
            )

        if (
            int(record["row_group"]) < 0
            or int(record["row_in_group"]) < 0
        ):
            raise RuntimeError(
                "Invalid Parquet row location."
            )

        locations[key] = record
        matched_rows[slide_id] += 1

if len(locations) != EXPECTED_IMAGES:
    raise RuntimeError(
        f"Expected {EXPECTED_IMAGES} image locations, "
        f"found {len(locations)}."
    )

for scanner, slide_id in selected_slides.items():
    if matched_rows[slide_id] != EXPECTED_POSITIONS:
        raise RuntimeError(
            f"Incomplete {scanner} source locations."
        )

print("Exact original PNG locations:", len(locations))


print("\n[3] Preparing partition-aware download records...", flush=True)

records = []

for tile_id, region in sorted(
    position_lookup.items(),
    key=lambda item: (
        item[1]["region_id"],
        int(item[1]["x"]),
        int(item[1]["y"]),
    ),
):
    for scanner in SCANNERS:
        slide_id = selected_slides[scanner]
        source = locations[(slide_id, tile_id)]

        # Distinct paths prevent confusion with Phase 3B.
        relative_path = (
            f"datasets/expanded_plism/stage_a/"
            f"{region['spatial_split']}/"
            f"{scanner}/{tile_id}.png"
        )

        records.append({
            "region_id": region["region_id"],
            "spatial_split": region["spatial_split"],
            "tile_id": tile_id,
            "level": region["level"],
            "x": region["x"],
            "y": region["y"],
            "source_condition": CONDITION,
            "scanner": scanner,
            "slide_id": slide_id,
            "repository_file": source["repository_file"],
            "row_group": int(source["row_group"]),
            "row_in_group": int(source["row_in_group"]),
            "dataset_revision": revision,
            "destination_relative_path": relative_path,
            "download_status": "planned",
        })

if len(records) != EXPECTED_IMAGES:
    raise RuntimeError("Incorrect download record count.")

paths = [
    row["destination_relative_path"]
    for row in records
]

if len(set(paths)) != len(paths):
    raise RuntimeError("Duplicate destination paths.")

summary_rows = []

for split in EXPECTED_SPLITS:
    subset = [
        row for row in records
        if row["spatial_split"] == split
    ]

    summary_rows.append({
        "spatial_split": split,
        "regions": EXPECTED_SPLITS[split],
        "positions": EXPECTED_SPLITS[split] * 36,
        "acquisitions": len(SCANNERS),
        "planned_images": len(subset),
    })

print("\nPlanned images by split:")

for row in summary_rows:
    print(
        f"  {row['spatial_split']}: "
        f"{row['planned_images']} images",
        flush=True,
    )


print("\n[4] Saving verified download manifest...", flush=True)

MANIFEST.parent.mkdir(parents=True, exist_ok=True)
SUMMARY.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

write_csv(
    MANIFEST,
    records,
    [
        "region_id",
        "spatial_split",
        "tile_id",
        "level",
        "x",
        "y",
        "source_condition",
        "scanner",
        "slide_id",
        "repository_file",
        "row_group",
        "row_in_group",
        "dataset_revision",
        "destination_relative_path",
        "download_status",
    ],
)

write_csv(
    SUMMARY,
    summary_rows,
    [
        "spatial_split",
        "regions",
        "positions",
        "acquisitions",
        "planned_images",
    ],
)

report = {
    "phase": "4",
    "step": "4.1.8",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset": "owkin/plism-dataset-tiles",
    "dataset_revision": revision,
    "condition": CONDITION,
    "scanners": list(SCANNERS),
    "region_count": len(region_ids),
    "unique_positions": len(position_lookup),
    "planned_original_pngs": len(records),
    "split_summary": summary_rows,
    "source_region_manifest_sha256": sha256(REGIONS),
    "source_acquisition_manifest_sha256": sha256(ACQUISITIONS),
    "source_exact_locations_sha256": sha256(LOCATIONS),
    "download_manifest_sha256": sha256(MANIFEST),
    "selection_status": "provisional",
    "final_evaluation_protocol_frozen": False,
    "image_downloads_performed": 0,
    "biological_independence_verified": False,
    "phase3b_modified": False,
    "next_step": (
        "Download a small original-PNG pilot using exact "
        "Parquet row-group locations, verify PNG hashes, "
        "then expand to the full Stage A manifest."
    ),
}

with REPORT.open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)

print("\n" + "=" * 70, flush=True)
print("PHASE 4.1.8 - DOWNLOAD MANIFEST COMPLETED")
print("=" * 70)

print("Selected regions:", len(region_ids))
print("Matched positions:", len(position_lookup))
print("Original PNGs planned:", len(records))

print("\nSaved:")
print(MANIFEST)
print(SUMMARY)
print(REPORT)

print("\nImages downloaded: NONE")
print("Phase 3B: UNCHANGED")
print("Final evaluation protocol: NOT YET FROZEN")
print("\nSTEP 4.1.8 COMPLETED")
