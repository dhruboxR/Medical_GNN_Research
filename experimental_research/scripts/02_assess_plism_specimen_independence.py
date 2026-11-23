
"""
Phase 3A - Step 3.2
PLISM Biological Specimen Independence Assessment

Purpose:
- Audit available per-tile metadata fields
- Inspect scanner-specific acquisition identifiers
- Verify matched tile correspondence
- Determine whether independent specimens can be
  identified from existing metadata

This script uses LOCAL metadata only.
No remote downloads or image loading.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd


# ==========================================================
# CONFIGURATION
# ==========================================================

ROOT = Path.cwd()

PHASE1 = ROOT / "dataset_exploration" / "outputs"
PHASE3 = ROOT / "experimental_research" / "outputs"

TABLE_DIR = PHASE3 / "tables"
REPORT_DIR = PHASE3 / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SPATIAL_FILE = (
    PHASE1 / "tables"
    / "plism_spatial_region_manifest.csv"
)

DOWNLOAD_FILE = (
    PHASE1 / "tables"
    / "plism_spatial_region_download_manifest.csv"
)

METADATA_FILE = (
    PHASE1 / "tables"
    / "plism_spatial_region_metadata.csv"
)

EXPLICIT_SPECIMEN_FIELDS = [
    "specimen_id",
    "patient_id",
    "patientid",
    "tissue_core_id",
    "core_id",
    "biological_sample_id",
    "donor_id",
]

print("=" * 70)
print("PHASE 3A - PLISM SPECIMEN INDEPENDENCE ASSESSMENT")
print("=" * 70)


# ==========================================================
# STEP 1: LOAD AVAILABLE METADATA
# ==========================================================

print("\n[1] Loading local metadata...", flush=True)

if not SPATIAL_FILE.exists():
    raise SystemExit(f"Missing: {SPATIAL_FILE}")

if not DOWNLOAD_FILE.exists():
    raise SystemExit(f"Missing: {DOWNLOAD_FILE}")

spatial = pd.read_csv(SPATIAL_FILE, dtype=str)
download = pd.read_csv(DOWNLOAD_FILE, dtype=str)

if METADATA_FILE.exists():
    metadata = pd.read_csv(METADATA_FILE, dtype=str)
else:
    metadata = pd.DataFrame()

print(f"Spatial records: {len(spatial)}")
print(f"Download records: {len(download)}")
print(f"Additional metadata records: {len(metadata)}")

print("\nSpatial metadata columns:")
print(", ".join(spatial.columns))

if not metadata.empty:
    print("\nAdditional metadata columns:")
    print(", ".join(metadata.columns))


# ==========================================================
# STEP 2: CHECK EXPLICIT SPECIMEN IDENTIFIERS
# ==========================================================

print("\n[2] Checking biological specimen identifiers...")

all_columns = set(
    column.lower()
    for column in (
        list(spatial.columns)
        + list(download.columns)
        + list(metadata.columns)
    )
)

found_specimen_fields = [
    field
    for field in EXPLICIT_SPECIMEN_FIELDS
    if field in all_columns
]

if found_specimen_fields:
    print("Candidate specimen fields found:")
    print(found_specimen_fields)
    print("Their semantics still require verification.")
else:
    print("No explicit specimen identifier found.")

# The schema alone never proves biological independence.
independence_verified = False


# ==========================================================
# STEP 3: INSPECT SCANNER ACQUISITION GROUPS
# ==========================================================

print("\n[3] Inspecting scanner-specific slide IDs...")

required = {
    "tile_id",
    "scanner_a",
    "scanner_b",
    "slide_id_a",
    "slide_id_b",
    "stainer",
}

missing = required - set(spatial.columns)

if missing:
    raise SystemExit(
        f"Missing columns: {sorted(missing)}"
    )

scanner_records = []

for scanner_col, slide_col in [
    ("scanner_a", "slide_id_a"),
    ("scanner_b", "slide_id_b"),
]:
    subset = spatial[
        ["tile_id", "stainer", scanner_col, slide_col]
    ].copy()

    subset.columns = [
        "tile_id",
        "stainer",
        "scanner",
        "slide_id",
    ]

    scanner_records.append(subset)

acquisitions = pd.concat(
    scanner_records,
    ignore_index=True
)

group_summary = (
    acquisitions.groupby(
        ["scanner", "stainer", "slide_id"],
        dropna=False
    )
    .agg(observed_tiles=("tile_id", "nunique"))
    .reset_index()
)

print(group_summary.to_string(index=False))

groups_path = (
    TABLE_DIR / "plism_acquisition_group_assessment.csv"
)

group_summary.to_csv(groups_path, index=False)


# ==========================================================
# STEP 4: VERIFY SCANNER-PAIRED LOCATIONS
# ==========================================================

print("\n[4] Checking scanner-paired tile correspondence...")

if spatial["tile_id"].duplicated().any():
    raise SystemExit(
        "Duplicate tile IDs in spatial manifest."
    )

if download["tile_id"].duplicated().any():
    raise SystemExit(
        "Duplicate tile IDs in download manifest."
    )

pair_check = spatial[
    [
        "tile_id",
        "stainer",
        "scanner_a",
        "scanner_b",
        "slide_id_a",
        "slide_id_b",
        "grid_row",
        "grid_col",
    ]
].merge(
    download[["tile_id", "complete_pair"]],
    on="tile_id",
    how="left",
    validate="one_to_one"
)

pair_check["complete_pair"] = (
    pair_check["complete_pair"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("true")
)

complete_pairs = int(
    pair_check["complete_pair"].sum()
)

same_scanner = (
    pair_check["scanner_a"]
    == pair_check["scanner_b"]
)

valid_cross_scanner_pairs = int(
    (
        pair_check["complete_pair"]
        & ~same_scanner
    ).sum()
)

print(f"Matched tile locations: {len(pair_check)}")
print(f"Complete image pairs: {complete_pairs}")
print(f"Valid cross-scanner pairs: {valid_cross_scanner_pairs}")

pair_check["specimen_identity_verified"] = False
pair_check["independence_group"] = "UNRESOLVED"

pair_path = (
    TABLE_DIR / "plism_specimen_pair_assessment.csv"
)

pair_check.to_csv(pair_path, index=False)


# ==========================================================
# STEP 5: DEFINE INDEPENDENCE REQUIREMENTS
# ==========================================================

print("\n[5] Building provenance requirements...")

requirements = [
    {
        "requirement": "Biological specimen identifier",
        "status": "UNRESOLVED",
        "reason": (
            "The observed tile schema does not contain "
            "a verified specimen ID."
        ),
    },
    {
        "requirement": "Patient or donor independence",
        "status": "UNRESOLVED",
        "reason": (
            "Distinct patient or donor identities cannot "
            "be determined from slide IDs."
        ),
    },
    {
        "requirement": "Tissue-core identifier",
        "status": "UNRESOLVED",
        "reason": (
            "Tile coordinates have not been linked to "
            "individual tissue microarray cores."
        ),
    },
    {
        "requirement": "Cross-scanner tile pairing",
        "status": (
            "VERIFIED_FOR_PILOT"
            if valid_cross_scanner_pairs == len(pair_check)
            else "INCOMPLETE"
        ),
        "reason": (
            "Corresponding tile IDs and complete images "
            "are available across both scanners."
        ),
    },
    {
        "requirement": "Physical coordinate semantics",
        "status": "PARTIALLY_VERIFIED",
        "reason": (
            "Tile ID suffixes agree with saved grid "
            "coordinates, but their physical interpretation "
            "requires external validation."
        ),
    },
    {
        "requirement": "Independent held-out evaluation",
        "status": "NOT_READY",
        "reason": (
            "Verified biological grouping information "
            "is required before specimen-level splitting."
        ),
    },
]

requirements_df = pd.DataFrame(requirements)

requirements_path = (
    TABLE_DIR / "plism_provenance_requirements.csv"
)

requirements_df.to_csv(
    requirements_path,
    index=False
)

print(
    requirements_df[
        ["requirement", "status"]
    ].to_string(index=False)
)


# ==========================================================
# STEP 6: COMPUTE FINAL ASSESSMENT
# ==========================================================

print("\n[6] Computing independence assessment...")

summary = {
    "dataset": "PLISM",
    "phase": "3A",
    "step": "3.2",
    "generated_utc": (
        datetime.now(timezone.utc).isoformat()
    ),
    "matched_tile_locations": int(len(pair_check)),
    "complete_image_pairs": complete_pairs,
    "valid_cross_scanner_pairs": valid_cross_scanner_pairs,
    "observed_scanner_groups": int(len(group_summary)),
    "observed_slide_ids": int(
        acquisitions["slide_id"].nunique()
    ),
    "explicit_specimen_fields_found": found_specimen_fields,
    "biological_independence_verified": independence_verified,
    "verified_independent_specimen_count": None,
    "pilot_pairing_suitable": (
        valid_cross_scanner_pairs == len(pair_check)
        and len(pair_check) > 0
    ),
    "specimen_level_inference_ready": False,
    "confirmatory_test_split_ready": False,
}

json_path = (
    REPORT_DIR / "02_specimen_independence_assessment.json"
)

json_path.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)


# ==========================================================
# STEP 7: WRITE MARKDOWN REPORT
# ==========================================================

print("\n[7] Generating research report...")

report_path = (
    REPORT_DIR / "02_specimen_independence_assessment.md"
)

report = f"""# PLISM Biological Specimen Independence Assessment

## Research Context

- Phase: 3A
- Step: 3.2
- Dataset: PLISM
- Scanner pair: AT2 and GT450
- Staining condition: GIVH

## Local Metadata Results

| Metric | Value |
|---|---:|
| Matched tile locations | {len(pair_check)} |
| Complete scanner pairs | {complete_pairs} |
| Valid cross-scanner pairs | {valid_cross_scanner_pairs} |
| Observed scanner groups | {len(group_summary)} |
| Distinct observed slide IDs | {acquisitions["slide_id"].nunique()} |

## Biological Independence

**Status: NOT VERIFIED**

The available metadata does not establish independently
identified biological specimens.

Slide IDs represent acquisition-related identifiers and
must not be counted as separate independent specimens
without verified provenance.

The dataset documentation describes repeated acquisitions
of shared tissue material across scanner and staining
conditions.

## Specimen Identifier Audit

Candidate explicit specimen fields observed:
{found_specimen_fields if found_specimen_fields else "None"}

Any discovered identifier would require verification
before it could be used for statistical grouping.

## Matched Tissue Locations

The pilot includes {valid_cross_scanner_pairs} valid
cross-scanner tile pairs.

This supports paired scanner-comparison experiments,
but not independent specimen-level statistical claims.

## Sampling Implications

1. Keep the Phase 1 pilot as development-only.
2. Do not use tile-level random train/test splitting.
3. Do not interpret scanner-specific slides as
   independent biological specimens.
4. Verify tissue-core mapping before assigning
   independent analysis groups.
5. Consider complementary datasets if independent
   specimen-level validation is not feasible.

## Research Readiness

| Research activity | Decision |
|---|---|
| Paired scanner feature comparison | Suitable |
| Paired graph topology comparison | Suitable |
| Specimen-level significance testing | Not ready |
| Independent held-out specimen testing | Not ready |
| Clinical classification claims | Not supported |

## Next Step

Step 3.3: Tissue-Core Mapping and Sampling Feasibility.

Investigate whether PLISM provides tissue-core annotations
or external metadata that can identify biologically
distinct sampling units.

No additional image downloading is required at this stage.

## References

- PLISM dataset:
  https://huggingface.co/datasets/owkin/plism-dataset-tiles

- Original PLISM publication:
  https://doi.org/10.1038/s41597-024-03122-5
"""

report_path.write_text(
    report,
    encoding="utf-8"
)


# ==========================================================
# STEP 8: PRINT FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("SPECIMEN INDEPENDENCE ASSESSMENT COMPLETED")
print("=" * 70)

print(
    f"Matched tile locations: "
    f"{summary['matched_tile_locations']}"
)

print(
    f"Valid scanner pairs: "
    f"{summary['valid_cross_scanner_pairs']}"
)

print(
    f"Observed slide IDs: "
    f"{summary['observed_slide_ids']}"
)

print(
    "Biological independence verified: "
    f"{summary['biological_independence_verified']}"
)

print(
    "Independent specimen count: UNKNOWN"
)

print("\nSaved outputs:")
print(groups_path)
print(pair_path)
print(requirements_path)
print(json_path)
print(report_path)

print("\nSTEP 3.2 COMPLETED")
