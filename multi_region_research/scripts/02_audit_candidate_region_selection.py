
"""
PHASE 3B - STEP 3B.2
Candidate Region Selection and Provenance Audit

Inputs:
    plism_region_candidates.csv
    plism_provisional_region_shortlist.csv

Checks:
    1. Window completeness
    2. Spatial footprint overlap
    3. Minimum separation between regions
    4. Tissue-label diversity
    5. Selection coverage and deterministic bias
    6. Provenance and image-extraction readiness

No network requests or image downloads.
"""

from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
import json
import math

import pandas as pd


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"

BASE = ROOT / "multi_region_research"
TABLES = BASE / "outputs" / "tables"
REPORTS = BASE / "outputs" / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)

CANDIDATES = TABLES / "plism_region_candidates.csv"
SHORTLIST = TABLES / "plism_provisional_region_shortlist.csv"

WINDOW_SIZE = 6
MIN_ANNOTATED_POSITIONS = 24


# ========================================================
# STEP 1: LOAD DATA
# ========================================================

print("=" * 70)
print("PHASE 3B - REGION SELECTION AND PROVENANCE AUDIT")
print("=" * 70)

print("\n[1] Loading region inventories...")

for path in [CANDIDATES, SHORTLIST]:
    if not path.is_file():
        raise SystemExit(f"Missing input: {path}")

candidates = pd.read_csv(CANDIDATES)
selected = pd.read_csv(SHORTLIST)

required = {
    "tissue_type",
    "grid_x_start",
    "grid_y_start",
    "grid_x_end",
    "grid_y_end",
    "annotated_positions",
    "window_positions",
    "coverage_fraction",
}

missing = required - set(selected.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if "region_id" not in selected.columns:
    raise ValueError("Missing region_id.")

if selected.empty:
    raise SystemExit("No selected regions to audit.")

print(f"Candidate windows: {len(candidates)}")
print(f"Selected regions: {len(selected)}")


# ========================================================
# STEP 2: VALIDATE REGION WINDOWS
# ========================================================

print("\n[2] Validating region dimensions and coverage...")

validation = []
footprints = {}

for row in selected.itertuples(index=False):

    rid = str(row.region_id)

    x0 = int(row.grid_x_start)
    y0 = int(row.grid_y_start)
    x1 = int(row.grid_x_end)
    y1 = int(row.grid_y_end)

    width = x1 - x0 + 1
    height = y1 - y0 + 1

    annotated = int(row.annotated_positions)
    capacity = int(row.window_positions)

    expected_coverage = annotated / capacity

    valid = (
        width == WINDOW_SIZE
        and height == WINDOW_SIZE
        and capacity == WINDOW_SIZE ** 2
        and MIN_ANNOTATED_POSITIONS <= annotated <= capacity
        and math.isclose(
            float(row.coverage_fraction),
            expected_coverage,
            abs_tol=1e-9
        )
    )

    footprint = {
        (x, y)
        for x in range(x0, x1 + 1)
        for y in range(y0, y1 + 1)
    }

    footprints[rid] = footprint

    validation.append({
        "region_id": rid,
        "tissue_type": str(row.tissue_type),
        "width": width,
        "height": height,
        "annotated_positions": annotated,
        "coverage_fraction": expected_coverage,
        "valid_window": valid,
    })

validation_df = pd.DataFrame(validation)

if not validation_df["valid_window"].all():
    raise RuntimeError(
        "Invalid region dimensions or coverage detected."
    )

print("Window validation: PASSED")
print(
    "Fully populated regions:",
    int((validation_df["coverage_fraction"] == 1).sum())
)


# ========================================================
# STEP 3: CHECK OVERLAP AND DISTANCES
# ========================================================

print("\n[3] Measuring spatial separation...")

pairs = []

for a, b in combinations(sorted(footprints), 2):

    fa = footprints[a]
    fb = footprints[b]

    overlap = len(fa & fb)

    # Minimum Euclidean separation between
    # the two window footprints, in original
    # annotation-grid coordinate units.
    minimum_distance = min(
        math.hypot(ax - bx, ay - by)
        for ax, ay in fa
        for bx, by in fb
    )

    pairs.append({
        "region_a": a,
        "region_b": b,
        "overlapping_grid_positions": overlap,
        "minimum_grid_distance": minimum_distance,
    })

pairs_df = pd.DataFrame(pairs)

overlapping_pairs = int(
    (pairs_df["overlapping_grid_positions"] > 0).sum()
) if not pairs_df.empty else 0

minimum_separation = (
    float(pairs_df["minimum_grid_distance"].min())
    if not pairs_df.empty else None
)

print(f"Region pairs checked: {len(pairs_df)}")
print(f"Overlapping pairs: {overlapping_pairs}")
print(f"Minimum footprint distance: {minimum_separation}")

if overlapping_pairs:
    raise RuntimeError("Overlapping regions detected.")


# ========================================================
# STEP 4: TISSUE DIVERSITY AND SELECTION
# ========================================================

print("\n[4] Evaluating tissue diversity...")

candidate_tissues = set(
    candidates["tissue_type"].dropna().astype(str)
)

selected_tissues = set(
    selected["tissue_type"].dropna().astype(str)
)

total_selected_coordinates = int(
    selected["annotated_positions"].sum()
)

unique_footprint_positions = len(
    set().union(*footprints.values())
)

tissue_coverage = (
    len(selected_tissues) / len(candidate_tissues)
    if candidate_tissues else 0.0
)

print(f"Tissues with candidate windows: {len(candidate_tissues)}")
print(f"Selected tissue labels: {len(selected_tissues)}")
print(f"Candidate tissue-label coverage: {tissue_coverage:.2%}")
print(f"Selected annotated positions: {total_selected_coordinates}")
print(f"Unique window-grid positions: {unique_footprint_positions}")

# Selection-bias note:
# The preceding selection script prioritizes
# window occupancy and then tissue names in
# deterministic lexical order.
selection_method = (
    "Greedy maximum-coverage selection with "
    "lexicographic tie-breaking and tissue diversity preference"
)


# ========================================================
# STEP 5: PROVENANCE AND READINESS
# ========================================================

print("\n[5] Assessing sampling readiness...")

checks = {
    "window_dimensions_valid":
        bool(validation_df["valid_window"].all()),
    "selected_footprints_nonoverlapping":
        overlapping_pairs == 0,
    "distinct_tissue_labels":
        len(selected_tissues) == len(selected),
    "original_coordinate_label_pairing":
        "Verified in Phase 3B.1",
    "owkin_coordinate_transform_verified":
        False,
    "biological_specimen_independence_verified":
        False,
}

ready_to_download = (
    checks["window_dimensions_valid"]
    and checks["selected_footprints_nonoverlapping"]
    and checks["owkin_coordinate_transform_verified"]
)

print("Window geometry: VERIFIED")
print("Spatial non-overlap: VERIFIED")
print("Owkin coordinate mapping: UNVERIFIED")
print("Biological independence: UNVERIFIED")
print("Ready for targeted Owkin tile extraction:", ready_to_download)


# ========================================================
# STEP 6: SAVE TABLES AND REPORT
# ========================================================

print("\n[6] Saving audit outputs...")

validation_path = (
    TABLES / "plism_region_selection_validation.csv"
)

distance_path = (
    TABLES / "plism_region_pairwise_distances.csv"
)

validation_df.to_csv(validation_path, index=False)
pairs_df.to_csv(distance_path, index=False)

report = {
    "phase": "3B",
    "step": "3B.2",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "candidate_windows": len(candidates),
    "selected_regions": len(selected),
    "candidate_tissue_labels": len(candidate_tissues),
    "selected_tissue_labels": len(selected_tissues),
    "candidate_label_coverage_fraction": tissue_coverage,
    "selected_annotation_locations": total_selected_coordinates,
    "unique_window_grid_positions": unique_footprint_positions,
    "overlapping_region_pairs": overlapping_pairs,
    "minimum_footprint_grid_distance": minimum_separation,
    "selection_method": selection_method,
    "checks": checks,
    "ready_for_targeted_owkin_extraction":
        ready_to_download,
    "limitations": [
        "Tissue labels do not establish independent specimens",
        "Region independence is not biological independence",
        "Original WSI coordinates are not verified Owkin tile coordinates",
        "Selection is deterministic but not randomized",
        "Original annotation-grid distance is not a verified physical distance"
    ]
}

report_path = (
    REPORTS / "02_region_selection_provenance_audit.json"
)

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)


# ========================================================
# FINAL SUMMARY
# ========================================================

print("\n" + "=" * 70)
print("PHASE 3B.2 - REGION AUDIT COMPLETED")
print("=" * 70)

print(f"Selected regions: {len(selected)}")
print(f"Distinct selected tissue labels: {len(selected_tissues)}")
print(f"Overlapping region pairs: {overlapping_pairs}")
print(f"Minimum grid separation: {minimum_separation}")
print(f"Unique window positions: {unique_footprint_positions}")
print(f"Targeted image extraction ready: {ready_to_download}")

print("\nSaved:")
print(validation_path)
print(distance_path)
print(report_path)

print("\nSTEP 3B.2 COMPLETED")
