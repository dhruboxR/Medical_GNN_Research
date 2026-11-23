
"""
PHASE 3B - STEP 3B.5
Owkin Region Separation and Resampling

Inputs:
    Step 3B.4 API matched-tile inventory

Outputs:
    Revised spatially separated region shortlist
    Exact region-to-tile manifest
    Pairwise spatial distance audit
    Sampling quality report

Uses local metadata only.
No images, downloads, or remote requests.
"""

from pathlib import Path
from itertools import combinations
from datetime import datetime, timezone
import json
import math

import pandas as pd


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "multi_region_research"

TABLES = BASE / "outputs/tables"
REPORTS = BASE / "outputs/reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

INPUT = TABLES / "plism_api_matched_tile_inventory.csv"

WINDOW = 6
MIN_MATCHED = 24
MAX_REGIONS = 12
MIN_SPATIAL_SEPARATION = 6.0

EXPECTED_AT2 = "GIVH_AT2_to_GMH_S60"
EXPECTED_GT450 = "GIVH_GT450_to_GMH_S60"


def rectangle_distance(a, b):
    """
    Minimum Euclidean grid distance between
    two inclusive rectangular footprints.
    """
    dx = max(
        0,
        a["x_start"] - b["x_end"],
        b["x_start"] - a["x_end"],
    )

    dy = max(
        0,
        a["y_start"] - b["y_end"],
        b["y_start"] - a["y_end"],
    )

    return math.hypot(dx, dy)


def point_to_rectangle_distance(x, y, r):
    dx = max(
        r["x_start"] - x,
        0,
        x - r["x_end"]
    )

    dy = max(
        r["y_start"] - y,
        0,
        y - r["y_end"]
    )

    return math.hypot(dx, dy)


print("=" * 70)
print("PHASE 3B - REGION SEPARATION AND RESAMPLING")
print("=" * 70)

# ============================================================
# STEP 1: LOAD MATCHED TILE METADATA
# ============================================================

print("\n[1] Loading matched tile inventory...")

if not INPUT.is_file():
    raise SystemExit(f"Missing input: {INPUT}")

df = pd.read_csv(INPUT, dtype={
    "tile_id": str,
    "at2_slide_id": str,
    "gt450_slide_id": str,
})

required = {
    "tile_id", "level", "x", "y",
    "at2_slide_id", "gt450_slide_id"
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if df.empty:
    raise ValueError("Matched tile inventory is empty.")

if df["tile_id"].duplicated().any():
    raise ValueError("Duplicate matched tile IDs.")

for row in df.itertuples(index=False):
    expected_id = (
        f"tile_{int(row.level)}_{int(row.x)}_{int(row.y)}"
    )
    if row.tile_id != expected_id:
        raise ValueError(f"Tile coordinate mismatch: {row.tile_id}")

if (
    (df["at2_slide_id"] != EXPECTED_AT2).any()
    or (df["gt450_slide_id"] != EXPECTED_GT450).any()
):
    raise ValueError(
        "Unexpected acquisition IDs. "
        "Inspect the scanner provenance before continuing."
    )

levels = sorted(df["level"].unique().tolist())

print(f"Matched positions: {len(df)}")
print(f"Levels: {levels}")
print("Scanner acquisition IDs: VERIFIED")

# ============================================================
# STEP 2: RECONSTRUCT ELIGIBLE WINDOWS
# ============================================================

print("\n[2] Reconstructing candidate windows...")

positions_by_level = {
    int(level): {
        (int(row.x), int(row.y))
        for row in group.itertuples(index=False)
    }
    for level, group in df.groupby("level")
}

candidates = []

for level, positions in positions_by_level.items():

    for x0, y0 in sorted(positions):

        x1 = x0 + WINDOW - 1
        y1 = y0 + WINDOW - 1

        # Exclude any position overlapping the Phase 1 pilot.
        if level == 16 and (
            x0 <= 109 and x1 >= 100
            and y0 <= 135 and y1 >= 126
        ):
            continue

        matched = {
            (x, y)
            for x in range(x0, x1 + 1)
            for y in range(y0, y1 + 1)
            if (x, y) in positions
        }

        if len(matched) < MIN_MATCHED:
            continue

        candidates.append({
            "level": level,
            "x_start": x0,
            "y_start": y0,
            "x_end": x1,
            "y_end": y1,
            "matched_positions": len(matched),
            "coverage_fraction": len(matched) / 36,
            "_matched": matched,
        })

print(f"Eligible windows: {len(candidates)}")

if not candidates:
    raise SystemExit("No eligible windows were found.")

# ============================================================
# STEP 3: SPATIALLY DISPERSED GREEDY SELECTION
# ============================================================

print("\n[3] Selecting spatially separated regions...")

# Prefer fully populated regions.
full_candidates = [
    c for c in candidates
    if c["matched_positions"] == WINDOW * WINDOW
]

pool = full_candidates if full_candidates else candidates

print(f"Fully populated candidates: {len(full_candidates)}")
print(f"Sampling pool: {len(pool)}")

selected = []

while len(selected) < MAX_REGIONS:

    eligible = []

    for candidate in pool:

        if candidate in selected:
            continue

        if selected:
            nearest = min(
                rectangle_distance(candidate, chosen)
                if candidate["level"] == chosen["level"]
                else float("inf")
                for chosen in selected
            )

            if nearest < MIN_SPATIAL_SEPARATION:
                continue
        else:
            nearest = float("inf")

        eligible.append((candidate, nearest))

    if not eligible:
        break

    if not selected:
        # Start with a complete candidate distant from
        # the provisional pilot grid.
        chosen = max(
            (item[0] for item in eligible),
            key=lambda c: (
                point_to_rectangle_distance(104.5, 130.5, c),
                c["matched_positions"],
                -c["x_start"],
                -c["y_start"],
            )
        )
    else:
        # Farthest-first selection across the observed area.
        chosen = max(
            eligible,
            key=lambda item: (
                item[1],
                item[0]["matched_positions"],
                -item[0]["x_start"],
                -item[0]["y_start"],
            )
        )[0]

    selected.append(chosen)

print(f"Regions selected: {len(selected)}")

# ============================================================
# STEP 4: CREATE EXACT TILE MANIFEST
# ============================================================

print("\n[4] Building exact region-to-tile manifest...")

region_records = []
manifest_records = []

for index, region in enumerate(selected, 1):

    region_id = f"OWKIN_REGION_{index:03d}"

    region_records.append({
        "region_id": region_id,
        "level": region["level"],
        "x_start": region["x_start"],
        "y_start": region["y_start"],
        "x_end": region["x_end"],
        "y_end": region["y_end"],
        "matched_positions": region["matched_positions"],
        "coverage_fraction": region["coverage_fraction"],
        "tissue_type": "UNVERIFIED",
        "specimen_id": "UNVERIFIED",
    })

    for x, y in sorted(region["_matched"]):

        manifest_records.append({
            "region_id": region_id,
            "tile_id": f"tile_{region['level']}_{x}_{y}",
            "level": region["level"],
            "x": x,
            "y": y,
            "at2_slide_id": EXPECTED_AT2,
            "gt450_slide_id": EXPECTED_GT450,
        })

regions_df = pd.DataFrame(region_records)
manifest_df = pd.DataFrame(manifest_records)

if manifest_df["tile_id"].duplicated().any():
    raise RuntimeError("Tile overlap between selected regions.")

print(f"Selected matched tile positions: {len(manifest_df)}")
print(f"Expected paired images: {len(manifest_df) * 2}")

# ============================================================
# STEP 5: DISTANCE VERIFICATION
# ============================================================

print("\n[5] Verifying pairwise region distances...")

distance_records = []

for (idx_a, a), (idx_b, b) in combinations(
    enumerate(selected, 1), 2
):

    distance = (
        rectangle_distance(a, b)
        if a["level"] == b["level"]
        else None
    )

    if distance is not None and distance < MIN_SPATIAL_SEPARATION:
        raise RuntimeError("Spatial separation requirement failed.")

    distance_records.append({
        "region_a": f"OWKIN_REGION_{idx_a:03d}",
        "region_b": f"OWKIN_REGION_{idx_b:03d}",
        "same_level": a["level"] == b["level"],
        "minimum_grid_distance": distance,
    })

distances_df = pd.DataFrame(
    distance_records,
    columns=[
        "region_a",
        "region_b",
        "same_level",
        "minimum_grid_distance"
    ],
)

finite_distances = (
    distances_df["minimum_grid_distance"].dropna()
    if not distances_df.empty
    else pd.Series(dtype=float)
)

minimum_distance = (
    float(finite_distances.min())
    if not finite_distances.empty else None
)

print(f"Region pairs: {len(distances_df)}")
print(f"Minimum same-level distance: {minimum_distance}")

# ============================================================
# STEP 6: SAVE RESULTS
# ============================================================

print("\n[6] Saving revised sampling outputs...")

regions_path = (
    TABLES / "plism_spatially_separated_regions.csv"
)

manifest_path = (
    TABLES / "plism_spatially_separated_tile_manifest.csv"
)

distances_path = (
    TABLES / "plism_spatial_region_distances.csv"
)

report_path = (
    REPORTS / "05_owkin_region_resampling_audit.json"
)

regions_df.to_csv(regions_path, index=False)
manifest_df.to_csv(manifest_path, index=False)
distances_df.to_csv(distances_path, index=False)

report = {
    "phase": "3B",
    "step": "3B.5",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "input_matched_tiles": len(df),
    "candidate_windows": len(candidates),
    "fully_populated_candidates": len(full_candidates),
    "selected_regions": len(selected),
    "selected_matched_tile_positions": len(manifest_df),
    "expected_paired_images": len(manifest_df) * 2,
    "minimum_required_grid_separation": MIN_SPATIAL_SEPARATION,
    "minimum_observed_grid_separation": minimum_distance,
    "original_wsi_mapping_verified": False,
    "tissue_type_verified": False,
    "biological_independence_verified": False,
    "source_inventory_complete": False,
    "images_downloaded": 0,
    "sampling_limitations": [
        "Only the initial 2000 API rows per scanner were considered",
        "Spatially separated regions may share a biological specimen",
        "Tissue labels are unavailable for these Owkin-native regions",
        "Farthest-first selection is deterministic, not randomized",
        "Grid distance is not a verified physical distance",
    ],
}

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3B.5 - REGION RESAMPLING COMPLETED")
print("=" * 70)

print(f"Candidate windows: {len(candidates)}")
print(f"Selected regions: {len(selected)}")
print(f"Selected matched positions: {len(manifest_df)}")
print(f"Potential paired images: {len(manifest_df) * 2}")
print(f"Minimum grid separation: {minimum_distance}")

if not regions_df.empty:
    print("\nRevised region shortlist:")
    print(regions_df.to_string(index=False))

print("\nSaved:")
print(regions_path)
print(manifest_path)
print(distances_path)
print(report_path)

print("\nImage downloads: NONE")
print("Biological independence: UNVERIFIED")
print("Source inventory complete: False")
print("\nSTEP 3B.5 COMPLETED")
