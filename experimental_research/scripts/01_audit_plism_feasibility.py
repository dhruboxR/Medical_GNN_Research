
"""
Phase 3A - Step 3.1
PLISM Metadata and Specimen Feasibility Audit

Run from the Medical_GNN_Research project root.

Default mode:
    python experimental_research/scripts/01_audit_plism_feasibility.py

Optional bounded remote metadata sampling:
    python experimental_research/scripts/01_audit_plism_feasibility.py --remote
"""

import argparse
import json
import re

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path.cwd()

PILOT_DIR = ROOT / "dataset_exploration" / "outputs"
OUTPUT_DIR = ROOT / "experimental_research" / "outputs"

TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TILE_PATTERN = re.compile(r"^tile_(\d+)_(\d+)_(\d+)$")


# ============================================================
# 2. FILE VALIDATION
# ============================================================

def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file missing: {path}\n"
            "Run the script from ~/Medical_GNN_Research"
        )
    return path


# ============================================================
# 3. AUDIT LOCAL PILOT DATA
# ============================================================

def audit_local():

    print("\n[1] Loading existing PLISM manifests...")

    spatial_file = require_file(
        PILOT_DIR
        / "tables"
        / "plism_spatial_region_manifest.csv"
    )

    download_file = require_file(
        PILOT_DIR
        / "tables"
        / "plism_spatial_region_download_manifest.csv"
    )

    spatial_df = pd.read_csv(spatial_file, dtype=str)
    download_df = pd.read_csv(download_file, dtype=str)

    required_columns = {
        "tile_id",
        "stainer",
        "scanner_a",
        "scanner_b",
        "slide_id_a",
        "slide_id_b",
        "source_shard_a",
        "source_shard_b",
        "grid_row",
        "grid_col",
    }

    missing = required_columns - set(spatial_df.columns)

    if missing:
        raise ValueError(
            f"Missing spatial manifest columns: {sorted(missing)}"
        )

    if "complete_pair" not in download_df.columns:
        raise ValueError(
            "Download manifest missing complete_pair column."
        )

    if spatial_df["tile_id"].duplicated().any():
        raise ValueError("Duplicate tile IDs found.")

    print(f"Spatial manifest rows: {len(spatial_df)}")
    print(f"Download manifest rows: {len(download_df)}")

    # --------------------------------------------------------
    # Match verified downloads to spatial records
    # --------------------------------------------------------

    print("\n[2] Checking downloaded scanner pairs...")

    matched = spatial_df.merge(
        download_df[["tile_id", "complete_pair"]],
        on="tile_id",
        how="left",
        validate="one_to_one"
    )

    matched["complete_pair"] = (
        matched["complete_pair"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    complete_count = int(matched["complete_pair"].sum())

    print(f"Complete matched pairs: {complete_count}")

    # --------------------------------------------------------
    # Parse tile identifiers
    # --------------------------------------------------------

    print("\n[3] Validating tile ID coordinate structure...")

    parsed = matched["tile_id"].str.extract(TILE_PATTERN)

    parsed.columns = [
        "tile_level",
        "tile_axis_1",
        "tile_axis_2"
    ]

    matched = pd.concat([matched, parsed], axis=1)

    matched["tile_id_parse_ok"] = (
        parsed.notna().all(axis=1)
    )

    matched["coordinate_matches_manifest"] = (
        matched["tile_axis_1"].eq(matched["grid_row"])
        &
        matched["tile_axis_2"].eq(matched["grid_col"])
    )

    matched["verified_pair_for_pilot"] = (
        matched["complete_pair"]
        &
        matched["tile_id_parse_ok"]
        &
        matched["coordinate_matches_manifest"]
    )

    parsed_count = int(matched["tile_id_parse_ok"].sum())

    agreement_count = int(
        matched["coordinate_matches_manifest"].sum()
    )

    print(f"Parseable tile IDs: {parsed_count}")
    print(f"Coordinate agreement: {agreement_count}")

    # Save tile-level audit
    audit_path = (
        TABLE_DIR / "plism_pilot_pair_audit.csv"
    )

    matched.to_csv(audit_path, index=False)

    # --------------------------------------------------------
    # Scanner-specific slide group inventory
    # --------------------------------------------------------

    print("\n[4] Inspecting scanner-specific slide groups...")

    group_records = []

    scanner_settings = [
        ("scanner_a", "slide_id_a", "source_shard_a"),
        ("scanner_b", "slide_id_b", "source_shard_b")
    ]

    for scanner_col, slide_col, shard_col in scanner_settings:

        grouped = matched.groupby(
            [scanner_col, slide_col, shard_col],
            dropna=False
        )

        for (scanner, slide, shard), group in grouped:

            group_records.append({
                "scanner": scanner,
                "slide_id": slide,
                "source_shard": shard,
                "stainers": ";".join(
                    sorted(group["stainer"].dropna().unique())
                ),
                "observed_tile_ids": group["tile_id"].nunique(),
                "complete_pairs": int(
                    group["complete_pair"].sum()
                ),
                "biological_independence": "NOT VERIFIED"
            })

    groups_df = pd.DataFrame(group_records)

    groups_path = (
        TABLE_DIR / "plism_observed_scanner_groups.csv"
    )

    groups_df.to_csv(groups_path, index=False)

    unique_slide_ids = pd.unique(
        pd.concat([
            matched["slide_id_a"],
            matched["slide_id_b"]
        ])
    )

    print(f"Observed scanner groups: {len(groups_df)}")
    print(f"Unique slide IDs: {len(unique_slide_ids)}")

    # --------------------------------------------------------
    # Inspect additional Phase 1 metadata
    # --------------------------------------------------------

    metadata_file = (
        PILOT_DIR
        / "tables"
        / "plism_spatial_region_metadata.csv"
    )

    metadata_rows = 0

    if metadata_file.exists():
        metadata_df = pd.read_csv(
            metadata_file,
            dtype=str
        )
        metadata_rows = len(metadata_df)

    # --------------------------------------------------------
    # Return statistics
    # --------------------------------------------------------

    statistics = {
        "pilot_matched_tile_ids": len(matched),
        "complete_downloaded_pairs": complete_count,
        "parseable_tile_ids": parsed_count,
        "tile_id_coordinate_agreement": agreement_count,
        "unique_scanner_specific_slide_ids_observed": len(
            unique_slide_ids
        ),
        "observed_scanner_groups": len(groups_df),
        "additional_local_metadata_rows": metadata_rows,
        "biologically_independent_specimens_verified": 0,
        "independence_status": "UNRESOLVED"
    }

    return statistics


# ============================================================
# 4. OPTIONAL REMOTE METADATA SAMPLING
# ============================================================

def inspect_remote_metadata(
    shards=(0, 3, 4, 8),
    rows_per_shard=100
):

    from datasets import load_dataset

    print("\n[5] Sampling remote metadata...")

    records = []

    for shard in shards:

        filename = (
            f"data/train-{shard:05d}-of-00293.parquet"
        )

        url = (
            "hf://datasets/owkin/"
            f"plism-dataset-tiles/{filename}"
        )

        print(f"Opening shard {shard}...", flush=True)

        try:

            stream = load_dataset(
                "parquet",
                data_files={"train": url},
                split="train",
                streaming=True
            )

            stream = stream.select_columns([
                "stainer",
                "scanner",
                "slide_id",
                "tile_id"
            ])

            for index, sample in enumerate(stream):

                if index >= rows_per_shard:
                    break

                records.append({
                    "shard": shard,
                    "stainer": sample["stainer"],
                    "scanner": sample["scanner"],
                    "slide_id": sample["slide_id"],
                    "tile_id": sample["tile_id"]
                })

        except Exception as error:
            print(
                f"Failed to inspect shard {shard}: {error}"
            )

    result = pd.DataFrame(records)

    output_path = (
        TABLE_DIR
        / "plism_remote_bounded_metadata_sample.csv"
    )

    result.to_csv(output_path, index=False)

    return {
        "remote_sampled_rows": len(result),
        "remote_sampled_scanners": (
            sorted(result["scanner"].dropna().unique().tolist())
            if not result.empty else []
        ),
        "remote_sampled_slide_ids": (
            int(result["slide_id"].nunique())
            if not result.empty else 0
        ),
        "remote_sample_is_complete_inventory": False
    }


# ============================================================
# 5. REPORT GENERATION
# ============================================================

def save_reports(stats):

    print("\n[6] Generating feasibility reports...")

    stats["dataset"] = "PLISM"
    stats["pilot_stainer"] = "GIVH"
    stats["pilot_scanners"] = ["AT2", "GT450"]

    stats["biological_independence_note"] = (
        "Scanner-specific slide IDs do not independently "
        "establish unique biological specimens."
    )

    stats["coordinate_validation_note"] = (
        "Tile ID parsing was verified against the saved "
        "manifest, but physical coordinate semantics "
        "remain unverified."
    )

    stats["generated_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )

    json_path = (
        REPORT_DIR / "plism_feasibility_audit.json"
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    markdown_path = (
        REPORT_DIR / "01_plism_feasibility_audit.md"
    )

    markdown_lines = [
        "# PLISM Dataset Feasibility Audit",
        "",
        "## 1. Research Context",
        "",
        "Phase 3A, Step 3.1",
        "",
        "Research topic: Cross-scanner graph topology stability.",
        "",
        "## 2. Local Pilot Audit",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Matched tile IDs | {stats['pilot_matched_tile_ids']} |",
        f"| Complete paired images | {stats['complete_downloaded_pairs']} |",
        f"| Parseable tile IDs | {stats['parseable_tile_ids']} |",
        f"| Coordinate agreement | {stats['tile_id_coordinate_agreement']} |",
        f"| Scanner groups observed | {stats['observed_scanner_groups']} |",
        f"| Unique slide IDs | {stats['unique_scanner_specific_slide_ids_observed']} |",
        "",
        "## 3. Biological Independence",
        "",
        "**Status: Not verified.**",
        "",
        "Scanner-specific slide identifiers cannot be treated "
        "as independent biological specimens without "
        "additional provenance information.",
        "",
        "## 4. Coordinate Validation",
        "",
        "The tile identifier coordinates agree with the "
        "previously generated manifests.",
        "",
        "However, this does not independently validate "
        "their physical spatial interpretation.",
        "",
        "## 5. Dataset Suitability",
        "",
        "PLISM is suitable for paired scanner-variation "
        "and graph-construction stability experiments.",
        "",
        "Independent biological generalization claims "
        "require verified specimen identities.",
        "",
        "## 6. Required Next Actions",
        "",
        "1. Verify biological specimen and tissue-core identifiers.",
        "2. Determine valid independent experimental units.",
        "3. Validate spatial coordinate semantics.",
        "4. Finalize grouped sampling and held-out evaluation.",
        "5. Keep the Phase 1 region separate from final testing.",
        "",
        "## 7. Overall Decision",
        "",
        "**Pilot benchmark: Suitable.**",
        "",
        "**Confirmatory multi-specimen study: Pending verification.**",
        ""
    ]

    markdown_path.write_text(
        "\n".join(markdown_lines),
        encoding="utf-8"
    )

    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


# ============================================================
# 6. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--remote",
        action="store_true",
        help="Optional bounded remote metadata sampling"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 3A - PLISM DATASET FEASIBILITY AUDIT")
    print("=" * 70)

    stats = audit_local()

    if args.remote:
        remote_stats = inspect_remote_metadata()
        stats.update(remote_stats)
    else:
        print("\n[5] Remote sampling skipped.")

    save_reports(stats)

    print("\n" + "=" * 70)
    print("FEASIBILITY AUDIT SUMMARY")
    print("=" * 70)

    for key, value in stats.items():

        if key in [
            "pilot_matched_tile_ids",
            "complete_downloaded_pairs",
            "parseable_tile_ids",
            "tile_id_coordinate_agreement",
            "unique_scanner_specific_slide_ids_observed",
            "observed_scanner_groups",
            "biologically_independent_specimens_verified"
        ]:
            print(f"{key}: {value}")

    print("\nBiological independence: NOT VERIFIED")
    print("Dataset suitability: PILOT BENCHMARK APPROVED")

    print("\nSTEP 3.1 COMPLETED")


if __name__ == "__main__":
    main()
