
"""
PHASE 3B - STEP 3B.8
Validate and Freeze Distributed Sampling Manifest

Checks:
1. Exactly 36 matched positions per selected region
2. No duplicate tile IDs across regions
3. Complete 6x6 grid geometry
4. Correct scanner acquisition provenance
5. No overlap with Phase 1 pilot
6. Minimum spatial separation
7. Consistency with discovered tile inventory
8. Stable SHA256 fingerprint for reproducibility

Does not download images.
Does not modify earlier sampling results.
"""

from pathlib import Path
from itertools import combinations
from datetime import datetime, timezone
import hashlib
import json
import math

import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "multi_region_research"

TABLES = BASE / "outputs/tables"
REPORTS = BASE / "outputs/reports"
VALIDATED = BASE / "outputs/validated"

VALIDATED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

INVENTORY_FILE = TABLES / "plism_distributed_matched_tiles.csv"
REGIONS_FILE = TABLES / "plism_distributed_region_shortlist.csv"
MANIFEST_FILE = TABLES / "plism_distributed_tile_manifest.csv"

FROZEN_FILE = VALIDATED / "phase3b_frozen_tile_manifest.csv"
REPORT_FILE = REPORTS / "08_sampling_manifest_validation.json"

WINDOW = 6
MIN_SEPARATION = 6.0

EXPECTED_SLIDES = {
    "AT2": "GIVH_AT2_to_GMH_S60",
    "GT450": "GIVH_GT450_to_GMH_S60",
}

REQUIRED_MANIFEST = {
    "region_id", "band", "tile_id",
    "level", "x", "y",
    "at2_slide_id", "gt450_slide_id",
}


def footprint(row):
    return {
        (int(row.level), x, y)
        for x in range(int(row.x_start), int(row.x_end) + 1)
        for y in range(int(row.y_start), int(row.y_end) + 1)
    }


def rectangle_distance(a, b):
    dx = max(
        0,
        int(a.x_start) - int(b.x_end),
        int(b.x_start) - int(a.x_end),
    )
    dy = max(
        0,
        int(a.y_start) - int(b.y_end),
        int(b.y_start) - int(a.y_end),
    )
    return math.hypot(dx, dy)


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


# --------------------------------------------------
# STEP 1: LOAD FILES
# --------------------------------------------------

print("=" * 70)
print("PHASE 3B - SAMPLING MANIFEST VALIDATION")
print("=" * 70)

print("\n[1] Loading sampling files...")

for path in (INVENTORY_FILE, REGIONS_FILE, MANIFEST_FILE):
    if not path.is_file():
        raise SystemExit(f"Missing file: {path}")

inventory = pd.read_csv(INVENTORY_FILE)
regions = pd.read_csv(REGIONS_FILE)
manifest = pd.read_csv(MANIFEST_FILE)

if REQUIRED_MANIFEST - set(manifest.columns):
    raise ValueError("Manifest is missing required columns.")

if regions.empty or manifest.empty or inventory.empty:
    raise ValueError("One or more input tables are empty.")

if inventory["tile_id"].duplicated().any():
    raise ValueError("Duplicate tile IDs in source inventory.")

if regions["region_id"].duplicated().any():
    raise ValueError("Duplicate region IDs.")

print(f"Inventory positions: {len(inventory)}")
print(f"Regions: {len(regions)}")
print(f"Manifest positions: {len(manifest)}")


# --------------------------------------------------
# STEP 2: VERIFY SCANNER PAIRING
# --------------------------------------------------

print("\n[2] Checking acquisition provenance...")

for scanner, column in (
    ("AT2", "at2_slide_id"),
    ("GT450", "gt450_slide_id"),
):
    if (
        manifest[column].astype(str)
        != EXPECTED_SLIDES[scanner]
    ).any():
        raise ValueError(
            f"Unexpected {scanner} slide acquisition."
        )

source = inventory.set_index("tile_id")

for row in manifest.itertuples(index=False):
    if row.tile_id not in source.index:
        raise ValueError(
            f"Tile missing from inventory: {row.tile_id}"
        )

    original = source.loc[row.tile_id]

    for field in ("level", "x", "y"):
        if int(getattr(row, field)) != int(original[field]):
            raise ValueError(
                f"Coordinate mismatch: {row.tile_id}"
            )

    for field in ("at2_slide_id", "gt450_slide_id"):
        if str(getattr(row, field)) != str(original[field]):
            raise ValueError(
                f"Provenance mismatch: {row.tile_id}"
            )

    expected = f"tile_{int(row.level)}_{int(row.x)}_{int(row.y)}"

    if str(row.tile_id) != expected:
        raise ValueError(f"Invalid tile ID: {row.tile_id}")

print("All selected tiles found in paired inventory.")
print("Scanner acquisition provenance: PASSED")


# --------------------------------------------------
# STEP 3: VALIDATE COMPLETE REGIONS
# --------------------------------------------------

print("\n[3] Validating region geometry...")

all_positions = set()
region_footprints = {}
region_validation = []

for region in regions.itertuples(index=False):

    rid = str(region.region_id)

    group = manifest[
        manifest["region_id"] == rid
    ]

    expected_positions = footprint(region)

    actual_positions = {
        (int(row.level), int(row.x), int(row.y))
        for row in group.itertuples(index=False)
    }

    if int(region.matched_positions) != WINDOW ** 2:
        raise ValueError(f"Incomplete region: {rid}")

    if len(group) != WINDOW ** 2:
        raise ValueError(f"Incorrect tile count: {rid}")

    if actual_positions != expected_positions:
        raise ValueError(f"Invalid region footprint: {rid}")

    if (
        int(region.x_end) - int(region.x_start) + 1 != WINDOW
        or int(region.y_end) - int(region.y_start) + 1 != WINDOW
    ):
        raise ValueError(f"Invalid window size: {rid}")

    if not (group["band"].astype(str) == str(region.band)).all():
        raise ValueError(f"Band attribution mismatch: {rid}")

    if all_positions & actual_positions:
        raise ValueError(f"Overlapping tile positions: {rid}")

    all_positions.update(actual_positions)
    region_footprints[rid] = expected_positions

    region_validation.append({
        "region_id": rid,
        "band": str(region.band),
        "tile_count": len(group),
        "complete_6x6": True,
        "unique_positions": len(actual_positions),
    })

if set(manifest["region_id"]) != set(regions["region_id"]):
    raise ValueError("Region IDs disagree between files.")

if manifest["tile_id"].duplicated().any():
    raise ValueError("Duplicate tiles in selected manifest.")

print(f"Complete regions: {len(region_validation)}")
print(f"Unique matched positions: {len(all_positions)}")


# --------------------------------------------------
# STEP 4: PILOT OVERLAP AND SPATIAL SEPARATION
# --------------------------------------------------

print("\n[4] Checking pilot exclusion and separation...")

pilot_positions = {
    (16, x, y)
    for x in range(100, 110)
    for y in range(126, 136)
}

if all_positions & pilot_positions:
    raise ValueError("Selected positions overlap pilot grid.")

# Check entire rectangular footprints, not only saved rows.
for rid, positions in region_footprints.items():
    if positions & pilot_positions:
        raise ValueError(f"Pilot footprint overlap: {rid}")

distances = []

for a, b in combinations(
    list(regions.itertuples(index=False)), 2
):
    if int(a.level) != int(b.level):
        continue

    distance = rectangle_distance(a, b)
    distances.append(distance)

    if distance < MIN_SEPARATION:
        raise ValueError(
            f"Separation violation: "
            f"{a.region_id}, {b.region_id}"
        )

minimum_distance = min(distances) if distances else None

print("Pilot overlap: 0")
print(f"Minimum region separation: {minimum_distance:.4f}")


# --------------------------------------------------
# STEP 5: CREATE REPRODUCIBLE FROZEN MANIFEST
# --------------------------------------------------

print("\n[5] Creating frozen sampling manifest...")

frozen = manifest.sort_values(
    ["region_id", "level", "x", "y", "tile_id"]
).reset_index(drop=True)

content = frozen.to_csv(
    index=False,
    lineterminator="\n"
).encode("utf-8")

fingerprint = sha256_bytes(content)

if FROZEN_FILE.exists():
    previous_content = FROZEN_FILE.read_bytes()

    if sha256_bytes(previous_content) != fingerprint:
        raise RuntimeError(
            "A different frozen manifest already exists. "
            "Refusing to overwrite it."
        )

    print("Existing frozen manifest matches.")

else:
    FROZEN_FILE.write_bytes(content)
    print("Frozen manifest created.")

print(f"SHA256: {fingerprint}")


# --------------------------------------------------
# STEP 6: SAVE VALIDATION REPORT
# --------------------------------------------------

print("\n[6] Saving validation report...")

report = {
    "phase": "3B",
    "step": "3B.8",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "regions": len(regions),
    "matched_tile_positions": len(manifest),
    "expected_images": len(manifest) * 2,
    "window_size": WINDOW,
    "minimum_region_separation": minimum_distance,
    "pilot_overlap": False,
    "scanner_pairing_verified": True,
    "all_regions_complete": True,
    "region_validation": region_validation,
    "frozen_manifest": str(FROZEN_FILE.relative_to(ROOT)),
    "manifest_sha256": fingerprint,
    "source_inventory_complete": False,
    "tissue_labels_verified": False,
    "biological_independence_verified": False,
    "original_wsi_mapping_verified": False,
    "images_downloaded": 0,
    "limitations": [
        "Sampling uses selected API bands, not full dataset coverage",
        "Spatial separation does not establish biological independence",
        "Tissue labels remain unverified",
        "Original WSI coordinates are not mapped to Owkin tiles",
        "The manifest is frozen for exploratory experiments only",
    ],
}

if REPORT_FILE.exists():
    existing = json.loads(REPORT_FILE.read_text(encoding="utf-8"))

    if existing.get("manifest_sha256") != fingerprint:
        raise RuntimeError(
            "Existing validation report has a different "
            "manifest fingerprint."
        )

REPORT_FILE.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)


# --------------------------------------------------
# FINAL
# --------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 3B.8 - MANIFEST VALIDATION COMPLETED")
print("=" * 70)

print(f"Regions validated: {len(regions)}")
print(f"Matched tile positions: {len(manifest)}")
print(f"Potential images: {len(manifest) * 2}")
print(f"Minimum separation: {minimum_distance:.4f}")
print("Scanner pairing: VERIFIED")
print("Complete region geometry: VERIFIED")
print("Pilot overlap: NONE")

print("\nFrozen manifest:")
print(FROZEN_FILE)

print("\nManifest SHA256:")
print(fingerprint)

print("\nValidation report:")
print(REPORT_FILE)

print("\nBiological independence: UNVERIFIED")
print("Tissue attribution: UNVERIFIED")
print("Image downloads: NONE")

print("\nSTEP 3B.8 COMPLETED")
