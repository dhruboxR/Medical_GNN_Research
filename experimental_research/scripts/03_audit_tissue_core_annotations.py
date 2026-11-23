
"""
Phase 3A - Step 3.3
PLISM Tissue-Core Annotation and Provenance Audit

Local-only audit:
- Inventory candidate annotation files
- Inspect metadata column names
- Identify possible specimen/tissue identifiers
- Generate a 90-tile mapping template
- Report whether core-level mapping is currently feasible

No image downloads and no remote requests.
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path.cwd()

DATASET_ROOT = ROOT / "datasets"

MANIFEST = (
    ROOT
    / "dataset_exploration"
    / "outputs"
    / "tables"
    / "plism_spatial_region_manifest.csv"
)

OUTPUT_ROOT = (
    ROOT / "experimental_research" / "outputs"
)

TABLE_DIR = OUTPUT_ROOT / "tables"
REPORT_DIR = OUTPUT_ROOT / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXTENSIONS = {".csv", ".tsv", ".json"}

# Prevent accidentally reading massive metadata files.
MAX_INSPECT_BYTES = 20 * 1024 * 1024

IDENTIFIER_KEYWORDS = [
    "specimen",
    "patient",
    "donor",
    "core",
    "tissue",
    "sample",
    "organ",
    "coordinate",
    "slide",
]

print("=" * 70)
print("PHASE 3A - PLISM TISSUE-CORE ANNOTATION AUDIT")
print("=" * 70)


# ============================================================
# 2. LOAD MATCHED TILE MANIFEST
# ============================================================

print("\n[1] Loading Phase 1 tile manifest...", flush=True)

if not MANIFEST.is_file():
    raise SystemExit(
        f"Missing manifest: {MANIFEST}"
    )

manifest = pd.read_csv(MANIFEST, dtype=str)

required = {
    "tile_id",
    "stainer",
    "grid_row",
    "grid_col",
    "slide_id_a",
    "slide_id_b",
}

missing = required - set(manifest.columns)

if missing:
    raise SystemExit(
        f"Missing columns: {sorted(missing)}"
    )

if manifest["tile_id"].duplicated().any():
    raise SystemExit("Duplicate tile IDs detected.")

print(f"Matched tile locations: {len(manifest)}")


# ============================================================
# 3. INVENTORY LOCAL ANNOTATION FILES
# ============================================================

print("\n[2] Searching local annotation files...", flush=True)

candidates = []

if DATASET_ROOT.is_dir():

    for path in sorted(DATASET_ROOT.rglob("*")):

        if not path.is_file():
            continue

        if path.suffix.lower() not in EXTENSIONS:
            continue

        size = path.stat().st_size

        if size > MAX_INSPECT_BYTES:
            inspection = "SKIPPED_SIZE_LIMIT"
            columns = []
            error = ""
        else:
            try:
                if path.suffix.lower() == ".csv":
                    sample = pd.read_csv(
                        path,
                        nrows=5
                    )

                elif path.suffix.lower() == ".tsv":
                    sample = pd.read_csv(
                        path,
                        sep="\t",
                        nrows=5
                    )

                else:
                    with open(
                        path,
                        encoding="utf-8"
                    ) as f:
                        object_data = json.load(f)

                    if isinstance(object_data, dict):
                        columns = list(object_data.keys())

                    elif (
                        isinstance(object_data, list)
                        and object_data
                        and isinstance(object_data[0], dict)
                    ):
                        columns = list(object_data[0].keys())

                    else:
                        columns = []

                    sample = None

                if sample is not None:
                    columns = sample.columns.tolist()

                inspection = "INSPECTED"
                error = ""

            except Exception as exc:
                columns = []
                inspection = "READ_ERROR"
                error = str(exc)[:300]

        matching_columns = [
            column
            for column in columns
            if any(
                keyword in column.lower()
                for keyword in IDENTIFIER_KEYWORDS
            )
        ]

        candidates.append({
            "file_path": str(path.relative_to(ROOT)),
            "size_bytes": size,
            "inspection_status": inspection,
            "columns": "; ".join(columns),
            "candidate_identifier_columns":
                "; ".join(matching_columns),
            "error": error,
        })

inventory_columns = [
    "file_path",
    "size_bytes",
    "inspection_status",
    "columns",
    "candidate_identifier_columns",
    "error",
]

inventory = pd.DataFrame(
    candidates,
    columns=inventory_columns
)

inventory_path = (
    TABLE_DIR / "plism_local_annotation_inventory.csv"
)

inventory.to_csv(inventory_path, index=False)

print(f"Candidate metadata files: {len(inventory)}")

if not inventory.empty:
    print(
        inventory[
            ["file_path", "inspection_status"]
        ].to_string(index=False)
    )


# ============================================================
# 4. CREATE BIOLOGICAL MAPPING TEMPLATE
# ============================================================

print("\n[3] Creating tile-to-core mapping template...")

template = manifest[
    [
        "tile_id",
        "stainer",
        "grid_row",
        "grid_col",
        "slide_id_a",
        "slide_id_b",
    ]
].copy()

# Unknown values must remain unknown.
template["tissue_core_id"] = pd.NA
template["tissue_type"] = pd.NA
template["biological_specimen_id"] = pd.NA
template["annotation_source"] = pd.NA
template["mapping_verified"] = False

template_path = (
    TABLE_DIR / "plism_tile_to_core_mapping_template.csv"
)

template.to_csv(template_path, index=False)

print(f"Template rows: {len(template)}")


# ============================================================
# 5. ASSESS READINESS
# ============================================================

print("\n[4] Assessing annotation readiness...")

files_with_candidate_columns = int(
    inventory["candidate_identifier_columns"]
    .fillna("")
    .ne("")
    .sum()
)

# Candidate columns do not establish a valid spatial join.
verified_core_mappings = 0

report = {
    "phase": "3A",
    "step": "3.3",
    "generated_utc":
        datetime.now(timezone.utc).isoformat(),
    "matched_tile_locations": len(manifest),
    "local_metadata_files_found": len(inventory),
    "files_with_candidate_identifier_columns":
        files_with_candidate_columns,
    "verified_core_mappings": verified_core_mappings,
    "core_id_mapping_status": "UNRESOLVED",
    "coordinate_transform_status": "UNVERIFIED",
    "independent_specimen_status": "UNVERIFIED",
    "next_action": (
        "Find authoritative original PLISM annotations "
        "and verify their coordinate relationship to "
        "the Owkin registered tile grid."
    ),
}

report_json = (
    REPORT_DIR / "03_tissue_core_annotation_audit.json"
)

report_json.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

report_md = (
    REPORT_DIR / "03_tissue_core_annotation_audit.md"
)

markdown = f"""# PLISM Tissue-Core Annotation Audit

## Phase 3A, Step 3.3

### Local Inventory

- Matched tissue locations: {len(manifest)}
- Local metadata files found: {len(inventory)}
- Files with candidate identifier fields:
  {files_with_candidate_columns}

### Biological Provenance

**Tissue-core mapping: UNRESOLVED**

No mapping is accepted until the annotation source,
coordinate interpretation, and acquisition relationship
have been validated.

### Important Limitations

- Tile coordinates alone do not identify biological specimens.
- Distinct slide IDs do not prove independent specimens.
- Tissue-type labels do not necessarily establish donor identity.
- Candidate metadata columns are not verified join keys.
- No independent specimen count can be claimed yet.

### Next Action

Inspect original PLISM annotation resources, locate
tissue-core or tissue-type annotations, and verify whether
they can be mapped to the registered Owkin tile grid.

### Data Sources

- https://huggingface.co/datasets/owkin/plism-dataset-tiles
- https://doi.org/10.1038/s41597-024-03122-5
"""

report_md.write_text(
    markdown,
    encoding="utf-8"
)


# ============================================================
# 6. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TISSUE-CORE ANNOTATION AUDIT COMPLETED")
print("=" * 70)

print(f"Matched tile locations: {len(manifest)}")
print(f"Candidate metadata files: {len(inventory)}")
print(
    "Files with candidate identifier columns: "
    f"{files_with_candidate_columns}"
)
print("Verified tissue-core mappings: 0")
print("Biological independence: UNVERIFIED")

print("\nSaved outputs:")
print(inventory_path)
print(template_path)
print(report_json)
print(report_md)

print("\nSTEP 3.3 LOCAL AUDIT COMPLETED")

