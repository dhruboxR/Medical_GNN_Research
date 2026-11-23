
"""
Phase 3A - Step 3.6
Inspect PLISM WSI annotation sample values and coordinate formats.

Uses the local CSV created in Step 3.5.
No internet access or additional downloads.
"""

from pathlib import Path
import json
import pandas as pd

ROOT = Path.cwd()

INPUT = (
    ROOT / "experimental_research/outputs/tables/"
    "plism_annotation_sample_rows.csv"
)

OUTPUT_DIR = (
    ROOT / "experimental_research/outputs/reports"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PHASE 3A - PLISM ANNOTATION VALUE ANALYSIS")
print("=" * 70)

# ============================================================
# STEP 1: LOAD SAMPLES
# ============================================================

print("\n[1] Loading previously inspected annotation samples...")

if not INPUT.exists():
    raise SystemExit(f"Missing sample file: {INPUT}")

df = pd.read_csv(INPUT, dtype=str).fillna("")

required = {
    "filename", "stain", "device",
    "tissue_type", "coordinate", "path"
}

missing = required - set(df.columns)

if missing:
    raise SystemExit(f"Missing columns: {sorted(missing)}")

print(f"Total sample records: {len(df)}")

wsi = df[
    df["filename"] == "PLISM_wsi_en.csv"
].copy()

sm = df[
    df["filename"] == "PLISM_sm_en.csv"
].copy()

print(f"WSI sample records: {len(wsi)}")
print(f"Smartphone sample records: {len(sm)}")

# ============================================================
# STEP 2: INSPECT WSI ANNOTATION VALUES
# ============================================================

print("\n[2] WSI annotation sample values...")

for index, row in wsi.iterrows():

    print(f"\nWSI sample {row.get('sample_row', index + 1)}")
    print(f"  Stain: {row['stain']}")
    print(f"  Device: {row['device']}")
    print(f"  Tissue type: {row['tissue_type']}")
    print(f"  Coordinate: {row['coordinate']}")
    print(f"  Path: {row['path']}")

# ============================================================
# STEP 3: INSPECT SMARTPHONE ANNOTATIONS
# ============================================================

print("\n[3] Smartphone annotation sample values...")

for index, row in sm.iterrows():

    print(f"\nSmartphone sample {row.get('sample_row', index + 1)}")
    print(f"  Stain: {row['stain']}")
    print(f"  Device: {row['device']}")
    print(f"  Tissue type: {row['tissue_type']}")
    print(f"  Coordinate: {row['coordinate']}")
    print(f"  Path: {row['path']}")

# ============================================================
# STEP 4: CHECK FOR PILOT METADATA MATCHES
# ============================================================

print("\n[4] Checking pilot acquisition identifiers...")

pilot_terms = [
    "GIVH",
    "AT2",
    "GT450",
    "GMH_S60",
]

for term in pilot_terms:

    matches = wsi[
        wsi.apply(
            lambda row: any(
                term.lower() in str(value).lower()
                for value in row.values
            ),
            axis=1
        )
    ]

    print(f"{term}: {len(matches)} sampled WSI rows")

print(
    "\nNote: Zero matches in a three-row sample "
    "do not imply absence from the full dataset."
)

# ============================================================
# STEP 5: ASSESS MAPPING READINESS
# ============================================================

print("\n[5] Assessing coordinate mapping readiness...")

coordinate_examples = (
    wsi["coordinate"].drop_duplicates().tolist()
)

path_examples = (
    wsi["path"].drop_duplicates().tolist()
)

report = {
    "phase": "3A",
    "step": "3.6",
    "wsi_sample_rows": len(wsi),
    "smartphone_sample_rows": len(sm),
    "coordinate_examples": coordinate_examples,
    "path_examples": path_examples,
    "explicit_specimen_id": False,
    "coordinate_mapping_verified": False,
    "tissue_core_mapping_verified": False,
    "next_requirement": (
        "Determine original WSI coordinate conventions "
        "and registered Owkin tile-to-source transforms."
    )
}

report_path = (
    OUTPUT_DIR / "06_annotation_value_analysis.json"
)

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("ANNOTATION VALUE ANALYSIS COMPLETED")
print("=" * 70)

print(f"WSI samples analyzed: {len(wsi)}")
print(f"Coordinate formats observed: {len(coordinate_examples)}")
print("Specimen independence: UNVERIFIED")
print("Coordinate mapping: UNVERIFIED")
print(f"Saved report: {report_path}")

print("\nSTEP 3.6 COMPLETED")
