
"""
PHASE 4.1.7 - SPATIAL REGION SELECTION

Create a reproducible, provisional Phase 4 spatial
cohort using the verified 31-acquisition tile index.

Targets:
    60 regions
    6x6 tiles per region
    40 train / 10 validation / 10 test regions
    2-tile minimum gap between region bounding boxes

Exclude:
    Frozen Phase 3B tile positions
    Locations within a 2-tile buffer of Phase 3B
    Incomplete 6x6 regions
    Duplicate or mismatched tile-coordinate records

All acquisitions use the same tile positions.

IMPORTANT:
    This is a PROVISIONAL spatial partition, not
    independently validated biological splitting.

No images are downloaded.
No old research results are modified.
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import hashlib
import json
import random
import csv
import re

ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "phase4_main_research"

EXACT = BASE / "outputs/tables/06_exact_plism_tile_locations.csv"
SELECTED = BASE / "datasets/manifests/03_candidate_acquisitions.csv"
OLD = ROOT / (
    "multi_region_research/outputs/validated/"
    "phase3b_frozen_tile_manifest.csv"
)

OUT = BASE / "datasets/manifests/07_provisional_phase4_regions.csv"
SUMMARY = BASE / "outputs/tables/07_phase4_region_summary.csv"
REPORT = BASE / "outputs/reports/07_phase4_region_selection.json"

REGION_SIZE = 6
BUFFER = 2
TARGET_REGIONS = 60
SEED = 42
LEVEL = 16

SPLIT_COUNTS = {
    "train": 40,
    "validation": 10,
    "test": 10,
}

PATTERN = re.compile(r"^tile_(\d+)_(-?\d+)_(-?\d+)$")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def box_conflict(a, b, gap=BUFFER):
    """
    Each box is represented by its top-left corner.

    Require at least 'gap' unused grid positions
    separating rectangles in x or y.
    """
    ax, ay = a
    bx, by = b

    reach = REGION_SIZE - 1 + gap

    return (
        abs(ax - bx) <= reach
        and abs(ay - by) <= reach
    )


def point_near_box(point, box, buffer=BUFFER):
    px, py = point
    bx, by = box

    return (
        bx - buffer <= px <= bx + REGION_SIZE - 1 + buffer
        and by - buffer <= py <= by + REGION_SIZE - 1 + buffer
    )


print("=" * 70, flush=True)
print("PHASE 4.1.7 - SPATIAL REGION SELECTION", flush=True)
print("=" * 70, flush=True)

print("\n[1] Verifying metadata and historical cohort...", flush=True)

for path in (EXACT, SELECTED, OLD):
    if not path.is_file():
        raise RuntimeError(f"Missing input: {path}")

for path in (OUT, SUMMARY, REPORT):
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite: {path}")

selected = load_csv(SELECTED)
old_rows = load_csv(OLD)

slide_ids = {r["slide_id"] for r in selected}

if len(slide_ids) != 31:
    raise RuntimeError("Expected 31 unique acquisitions.")

if len(old_rows) != 360:
    raise RuntimeError("Unexpected frozen pilot cohort.")

pilot_coords = set()

for row in old_rows:
    if int(row["level"]) != LEVEL:
        raise RuntimeError("Unexpected Phase 3B tile level.")

    pilot_coords.add((int(row["x"]), int(row["y"])))

if len(pilot_coords) != 360:
    raise RuntimeError("Duplicate Phase 3B coordinates.")

# Stream the verified index without loading its
# half-million rows as Python dictionaries.
positions = defaultdict(set)
counts = Counter()
invalid = 0

with EXACT.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for record in reader:
        slide = record["slide_id"]

        if slide not in slide_ids:
            raise RuntimeError("Unexpected acquisition.")

        level = int(record["level"])
        x = int(record["x"])
        y = int(record["y"])

        match = PATTERN.fullmatch(record["tile_id"])

        if match is None:
            invalid += 1
            continue

        if tuple(map(int, match.groups())) != (level, x, y):
            invalid += 1
            continue

        position = (level, x, y)

        if position in positions[slide]:
            raise RuntimeError(
                f"Duplicate tile in acquisition: {slide}"
            )

        positions[slide].add(position)
        counts[slide] += 1

if invalid:
    raise RuntimeError(f"Invalid tile metadata: {invalid}")

if any(counts[s] != 16278 for s in slide_ids):
    raise RuntimeError("Unexpected per-acquisition tile counts.")

reference = positions[sorted(slide_ids)[0]]

if any(positions[s] != reference for s in slide_ids):
    raise RuntimeError(
        "Coordinate sets differ between selected acquisitions."
    )

common = {
    (x, y)
    for level, x, y in reference
    if level == LEVEL
}

print("Verified acquisitions:", len(slide_ids))
print("Shared level-16 positions:", len(common))
print("Frozen pilot coordinates:", len(pilot_coords))

if not common:
    raise RuntimeError("No level-16 coordinates.")


print("\n[2] Finding complete 6x6 candidate regions...", flush=True)

anchors = []

for x, y in sorted(common):
    cells = {
        (x + dx, y + dy)
        for dx in range(REGION_SIZE)
        for dy in range(REGION_SIZE)
    }

    if not cells.issubset(common):
        continue

    if any(
        point_near_box(point, (x, y))
        for point in pilot_coords
    ):
        continue

    anchors.append((x, y))

print("Eligible complete 6x6 anchors:", len(anchors))

if len(anchors) < TARGET_REGIONS:
    raise RuntimeError(
        "Too few eligible 6x6 anchors for the requested cohort."
    )


print("\n[3] Selecting spatially separated regions...", flush=True)

rng = random.Random(SEED)
rng.shuffle(anchors)

chosen = []

# Greedy farthest-first selection encourages geographic
# spread while enforcing nonoverlapping buffered boxes.
while len(chosen) < TARGET_REGIONS:

    eligible = [
        anchor for anchor in anchors
        if all(
            not box_conflict(anchor, selected_anchor)
            for selected_anchor in chosen
        )
    ]

    if not eligible:
        raise RuntimeError(
            f"Only {len(chosen)} separated regions found. "
            "No partial manifest was saved. "
            "Review the region target or buffer."
        )

    if not chosen:
        next_anchor = eligible[0]
    else:
        next_anchor = max(
            eligible,
            key=lambda anchor: min(
                (anchor[0] - other[0]) ** 2
                + (anchor[1] - other[1]) ** 2
                for other in chosen
            )
        )

    chosen.append(next_anchor)

    if len(chosen) % 10 == 0:
        print(
            f"  Selected {len(chosen)}/{TARGET_REGIONS} regions",
            flush=True,
        )

# Check spatial separation independently.
for i, a in enumerate(chosen):
    for b in chosen[i + 1:]:
        if box_conflict(a, b):
            raise RuntimeError("Spatial gap validation failed.")


print("\n[4] Assigning provisional spatial partitions...", flush=True)

# Shuffle region IDs independently of spatial selection.
assignment_order = list(range(TARGET_REGIONS))
rng.shuffle(assignment_order)

split_by_index = {}

offset = 0

for split, count in SPLIT_COUNTS.items():
    for idx in assignment_order[offset:offset + count]:
        split_by_index[idx] = split
    offset += count

if len(split_by_index) != TARGET_REGIONS:
    raise RuntimeError("Incomplete split assignment.")

manifest_rows = []
summary_rows = []
all_selected_positions = set()

for i, (x, y) in enumerate(chosen):
    region_id = f"PHASE4_REGION_{i + 1:03d}"
    split = split_by_index[i]

    region_positions = set()

    for dx in range(REGION_SIZE):
        for dy in range(REGION_SIZE):
            px = x + dx
            py = y + dy

            pos = (px, py)

            if pos not in common:
                raise RuntimeError("Incomplete selected region.")

            if pos in all_selected_positions:
                raise RuntimeError("Duplicate selected position.")

            if pos in pilot_coords:
                raise RuntimeError("Phase 3B coordinate reused.")

            region_positions.add(pos)
            all_selected_positions.add(pos)

            manifest_rows.append({
                "region_id": region_id,
                "spatial_split": split,
                "tile_id": f"tile_{LEVEL}_{px}_{py}",
                "level": LEVEL,
                "x": px,
                "y": py,
                "region_origin_x": x,
                "region_origin_y": y,
                "selection_status": "provisional",
            })

    if len(region_positions) != 36:
        raise RuntimeError("Invalid region size.")

    summary_rows.append({
        "region_id": region_id,
        "spatial_split": split,
        "origin_x": x,
        "origin_y": y,
        "tile_count": len(region_positions),
        "phase3b_overlap": len(
            region_positions & pilot_coords
        ),
    })

assert len(manifest_rows) == TARGET_REGIONS * 36

print("Selected regions:", len(summary_rows))
print("Selected positions:", len(all_selected_positions))
print("Overlap with Phase 3B:", len(all_selected_positions & pilot_coords))

print("Provisional split:")
for split in SPLIT_COUNTS:
    count = sum(
        r["spatial_split"] == split
        for r in summary_rows
    )
    print(f"  {split}: {count} regions")


print("\n[5] Saving provisional region plan...", flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
SUMMARY.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

write_csv(
    OUT,
    manifest_rows,
    [
        "region_id",
        "spatial_split",
        "tile_id",
        "level",
        "x",
        "y",
        "region_origin_x",
        "region_origin_y",
        "selection_status",
    ],
)

write_csv(
    SUMMARY,
    summary_rows,
    [
        "region_id",
        "spatial_split",
        "origin_x",
        "origin_y",
        "tile_count",
        "phase3b_overlap",
    ],
)

report = {
    "phase": "4",
    "step": "4.1.7",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "selection_seed": SEED,
    "region_size": REGION_SIZE,
    "minimum_spatial_buffer_tiles": BUFFER,
    "selected_regions": len(summary_rows),
    "unique_positions": len(all_selected_positions),
    "acquisitions_available": len(slide_ids),
    "maximum_proposed_pngs": len(all_selected_positions) * len(slide_ids),
    "spatial_splits": SPLIT_COUNTS,
    "phase3b_overlap": len(
        all_selected_positions & pilot_coords
    ),
    "source_exact_index_sha256": digest(EXACT),
    "source_acquisition_selection_sha256": digest(SELECTED),
    "source_phase3b_manifest_sha256": digest(OLD),
    "manifest_sha256": digest(OUT),
    "selection_status": "provisional",
    "image_downloads": 0,
    "biological_independence_verified": False,
    "scanner_condition_holdout_finalized": False,
    "limitations": [
        "Spatial separation does not establish specimen independence",
        "The same tissue material appears in all selected acquisitions",
        "Region coverage may include uninformative tissue",
        "Scanner-condition evaluation plan must be finalized",
        "Manual registration and tissue-quality review pending",
    ],
}

with REPORT.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 70)
print("PHASE 4.1.7 - REGION SELECTION COMPLETED")
print("=" * 70)

print("Selected regions:", len(summary_rows))
print("Selected tile positions:", len(all_selected_positions))
print("Provisional spatial splits:", SPLIT_COUNTS)

print("\nSaved:")
print(OUT)
print(SUMMARY)
print(REPORT)

print("\nPhase 3B original data: UNCHANGED")
print("Images downloaded: NONE")
print("Biological independence: UNVERIFIED")
print("Final training protocol: NOT YET FROZEN")
print("\nSTEP 4.1.7 COMPLETED")
