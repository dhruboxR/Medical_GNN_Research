
"""
PHASE 3B - STEP 3B.3
Original PLISM to Owkin Coordinate Mapping Investigation

Inputs:
- Phase 1 DINOv2 feature inventory
- Phase 3B provisional region shortlist
- Locally saved Phase 1 CSV metadata

Goals:
1. Verify original WSI coordinate ranges.
2. Inspect Owkin tile identifiers and grid coordinates.
3. Check tile-ID and grid-coordinate consistency.
4. Inventory locally available provenance fields.
5. Identify what is still required to verify a mapping.

No downloads or remote requests.
Does not claim a coordinate transformation.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import re

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"

FEATURE_FILE = (
    ROOT / "dataset_exploration/outputs/features/"
    "plism_dinov2_features.npz"
)

SHORTLIST_FILE = (
    ROOT / "multi_region_research/outputs/tables/"
    "plism_provisional_region_shortlist.csv"
)

PILOT_OUTPUTS = ROOT / "dataset_exploration/outputs"

OUTPUT = ROOT / "multi_region_research/outputs"

TABLES = OUTPUT / "tables"
REPORTS = OUTPUT / "reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

ORIGINAL_GRID_STEP = 1024

TILE_PATTERN = re.compile(
    r"^tile_(\d+)_(\d+)_(\d+)$"
)

PROVENANCE_TERMS = (
    "tile_id",
    "slide_id",
    "scanner",
    "stainer",
    "stain",
    "coordinate",
    "grid_row",
    "grid_col",
    "tissue",
    "specimen",
    "patient",
    "core",
    "registration",
    "transform",
)


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

print("=" * 70)
print("PHASE 3B - COORDINATE MAPPING INVESTIGATION")
print("=" * 70)

print("\n[1] Loading existing research metadata...")

for path in (FEATURE_FILE, SHORTLIST_FILE):
    if not path.is_file():
        raise SystemExit(f"Missing input: {path}")

with np.load(FEATURE_FILE) as features:

    required_keys = {
        "tile_ids",
        "grid_rows",
        "grid_cols",
        "at2",
        "gt450",
    }

    missing = required_keys - set(features.files)

    if missing:
        raise ValueError(
            f"Missing feature keys: {sorted(missing)}"
        )

    tile_ids = features["tile_ids"].astype(str)
    grid_rows = features["grid_rows"].astype(int)
    grid_cols = features["grid_cols"].astype(int)

    at2_shape = features["at2"].shape
    gt450_shape = features["gt450"].shape

shortlist = pd.read_csv(SHORTLIST_FILE)

required_region_columns = {
    "region_id",
    "tissue_type",
    "grid_x_start",
    "grid_y_start",
    "grid_x_end",
    "grid_y_end",
}

missing = required_region_columns - set(shortlist.columns)

if missing:
    raise ValueError(
        f"Missing shortlist columns: {sorted(missing)}"
    )

if at2_shape != gt450_shape:
    raise ValueError("Scanner feature shapes differ.")

if at2_shape[0] != len(tile_ids):
    raise ValueError("Feature/node count mismatch.")

print(f"Pilot matched tiles: {len(tile_ids)}")
print(f"Feature dimensions: {at2_shape[1]}")
print(f"Selected original WSI regions: {len(shortlist)}")


# ============================================================
# STEP 2: AUDIT OWKIN TILE IDS
# ============================================================

print("\n[2] Examining Owkin tile identifiers...")

tile_records = []
invalid_ids = []
levels = set()

for tile_id, row, col in zip(
    tile_ids,
    grid_rows,
    grid_cols,
):

    match = TILE_PATTERN.match(tile_id)

    if match is None:
        invalid_ids.append(tile_id)
        continue

    level, parsed_a, parsed_b = map(
        int, match.groups()
    )

    levels.add(level)

    tile_records.append({
        "tile_id": tile_id,
        "tile_level": level,
        "id_coordinate_1": parsed_a,
        "id_coordinate_2": parsed_b,
        "saved_grid_row": int(row),
        "saved_grid_col": int(col),
        "direct_id_grid_match": (
            parsed_a == int(row)
            and parsed_b == int(col)
        ),
        "swapped_id_grid_match": (
            parsed_a == int(col)
            and parsed_b == int(row)
        ),
    })

tile_df = pd.DataFrame(tile_records)

if invalid_ids:
    raise ValueError(
        f"Invalid tile IDs found: {invalid_ids[:5]}"
    )

if len(set(tile_ids)) != len(tile_ids):
    raise ValueError("Duplicate tile IDs detected.")

if len(np.unique(
    np.column_stack([grid_rows, grid_cols]),
    axis=0,
)) != len(tile_ids):
    raise ValueError("Duplicate pilot grid coordinates.")

direct_matches = int(
    tile_df["direct_id_grid_match"].sum()
)

swapped_matches = int(
    tile_df["swapped_id_grid_match"].sum()
)

print(f"Tile ID levels: {sorted(levels)}")
print(f"Direct ID/grid matches: {direct_matches}")
print(f"Swapped ID/grid matches: {swapped_matches}")
print(f"Pilot row range: {grid_rows.min()} to {grid_rows.max()}")
print(f"Pilot col range: {grid_cols.min()} to {grid_cols.max()}")

print(
    "\nNote: Matching saved grid fields to tile IDs "
    "does not establish their source WSI coordinates."
)


# ============================================================
# STEP 3: EXAMINE ORIGINAL WSI REGIONS
# ============================================================

print("\n[3] Examining original annotation grid regions...")

region_records = []

for record in shortlist.itertuples(index=False):

    x0 = int(record.grid_x_start)
    y0 = int(record.grid_y_start)
    x1 = int(record.grid_x_end)
    y1 = int(record.grid_y_end)

    region_records.append({
        "region_id": str(record.region_id),
        "tissue_type": str(record.tissue_type),
        "original_grid_x_start": x0,
        "original_grid_y_start": y0,
        "original_grid_x_end": x1,
        "original_grid_y_end": y1,
        "original_coordinate_x_start":
            x0 * ORIGINAL_GRID_STEP,
        "original_coordinate_y_start":
            y0 * ORIGINAL_GRID_STEP,
        "original_coordinate_x_end":
            x1 * ORIGINAL_GRID_STEP,
        "original_coordinate_y_end":
            y1 * ORIGINAL_GRID_STEP,
        "mapping_to_owkin_verified": False,
    })

region_df = pd.DataFrame(region_records)

print(
    "Original annotation grid X range: "
    f"{region_df['original_grid_x_start'].min()} to "
    f"{region_df['original_grid_x_end'].max()}"
)

print(
    "Original annotation grid Y range: "
    f"{region_df['original_grid_y_start'].min()} to "
    f"{region_df['original_grid_y_end'].max()}"
)

print(
    "Original coordinate system: "
    "1024-unit grid inferred from annotation values"
)


# ============================================================
# STEP 4: INVENTORY LOCAL PHASE 1 METADATA
# ============================================================

print("\n[4] Inspecting saved Phase 1 metadata schemas...")

inventory = []

for path in sorted(PILOT_OUTPUTS.rglob("*.csv")):

    relative = path.relative_to(ROOT).as_posix()

    try:
        columns = list(
            pd.read_csv(path, nrows=0).columns
        )

        matching = [
            col
            for col in columns
            if any(
                term in col.lower()
                for term in PROVENANCE_TERMS
            )
        ]

        inventory.append({
            "file": relative,
            "column_count": len(columns),
            "provenance_columns": "; ".join(matching),
            "has_provenance_columns": bool(matching),
            "inspection_status": "OK",
        })

    except (OSError, ValueError, UnicodeError) as exc:

        inventory.append({
            "file": relative,
            "column_count": 0,
            "provenance_columns": "",
            "has_provenance_columns": False,
            "inspection_status": (
                f"UNREADABLE: {type(exc).__name__}"
            ),
        })

inventory_df = pd.DataFrame(
    inventory,
    columns=[
        "file",
        "column_count",
        "provenance_columns",
        "has_provenance_columns",
        "inspection_status",
    ],
)

relevant = inventory_df[
    inventory_df["has_provenance_columns"]
] if not inventory_df.empty else inventory_df

print(f"CSV metadata files inspected: {len(inventory_df)}")
print(f"Files with relevant columns: {len(relevant)}")

if not relevant.empty:

    print("\nPotential provenance sources:")

    print(
        relevant[
            ["file", "provenance_columns"]
        ].to_string(index=False, max_colwidth=90)
    )


# ============================================================
# STEP 5: SAVE AUDIT TABLES
# ============================================================

print("\n[5] Saving coordinate investigation tables...")

tile_path = TABLES / "plism_owkin_tile_id_audit.csv"

region_path = (
    TABLES / "plism_original_region_coordinate_audit.csv"
)

inventory_path = (
    TABLES / "plism_phase1_metadata_inventory.csv"
)

tile_df.to_csv(tile_path, index=False)
region_df.to_csv(region_path, index=False)
inventory_df.to_csv(inventory_path, index=False)


# ============================================================
# STEP 6: MAPPING READINESS ASSESSMENT
# ============================================================

print("\n[6] Assessing coordinate mapping readiness...")

report = {
    "phase": "3B",
    "step": "3B.3",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "pilot_tile_count": len(tile_ids),
    "pilot_tile_levels": sorted(levels),
    "direct_tile_id_grid_matches": direct_matches,
    "swapped_tile_id_grid_matches": swapped_matches,
    "pilot_grid_row_range": [
        int(grid_rows.min()),
        int(grid_rows.max()),
    ],
    "pilot_grid_col_range": [
        int(grid_cols.min()),
        int(grid_cols.max()),
    ],
    "original_wsi_regions": len(region_df),
    "original_annotation_grid_step":
        ORIGINAL_GRID_STEP,
    "phase1_csv_files_inspected":
        len(inventory_df),
    "phase1_files_with_provenance_columns":
        len(relevant),
    "mapping_status": "UNVERIFIED",
    "requirements_before_mapping": [
        "Official Owkin tile coordinate definitions",
        "Relationship between registered tiles and original WSI",
        "Slide registration and coordinate transformations",
        "Verification using known corresponding image landmarks",
        "Tissue-label attribution in the registered tile grid",
    ],
    "ready_for_targeted_image_extraction": False,
    "biological_specimen_independence_verified": False,
    "interpretation": (
        "This audit documents coordinate metadata and "
        "possible provenance sources; it does not infer "
        "or validate a coordinate transformation."
    ),
}

report_path = (
    REPORTS / "03_coordinate_mapping_investigation.json"
)

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3B.3 - COORDINATE INVESTIGATION COMPLETED")
print("=" * 70)

print(f"Pilot nodes: {len(tile_ids)}")
print(f"Original WSI candidate regions: {len(region_df)}")
print(f"Provenance-related CSV files: {len(relevant)}")

print("\nOriginal-to-Owkin mapping: UNVERIFIED")
print("Independent biological specimens: UNVERIFIED")
print("Ready for targeted image extraction: False")

print("\nSaved:")
print(tile_path)
print(region_path)
print(inventory_path)
print(report_path)

print("\nSTEP 3B.3 COMPLETED")
