
from datasets import load_dataset
from pathlib import Path
import pandas as pd
import re
import time

# ==================================================
# CONFIGURATION
# ==================================================

REPO_ID = "owkin/plism-dataset-tiles"

STAINER = "GIVH"
LEVEL = 16

# Candidate 10 x 10 tile region
START_ROW = 100
START_COL = 126

GRID_ROWS = 10
GRID_COLS = 10

# Previously identified scanner shards
SCANNERS = {
    "AT2": [0],
    "GT450": [3],
}

TOTAL_SHARDS = 293

# Safety limit per shard
MAX_SCAN_RECORDS = 15000

OUTPUT_ROOT = Path("dataset_exploration/outputs")
TABLE_DIR = OUTPUT_ROOT / "tables"
REPORT_DIR = OUTPUT_ROOT / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PATTERN = re.compile(
    r"^tile_(\d+)_(\d+)_(\d+)$"
)

start_time = time.perf_counter()

print("=" * 70)
print("PLISM SPATIAL TILE REGION IDENTIFICATION")
print("=" * 70)

# ==================================================
# STEP 1: GENERATE CANDIDATE TILE IDS
# ==================================================

print("\n[1] Generating candidate 2D grid...")

target_ids = {}

for row in range(
    START_ROW,
    START_ROW + GRID_ROWS
):
    for col in range(
        START_COL,
        START_COL + GRID_COLS
    ):

        tile_id = f"tile_{LEVEL}_{row}_{col}"

        target_ids[tile_id] = {
            "level": LEVEL,
            "grid_row": row,
            "grid_col": col
        }

target_set = set(target_ids)

print(f"Grid dimensions: {GRID_ROWS} x {GRID_COLS}")
print(f"Candidate locations: {len(target_set)}")

print(
    f"Rows: {START_ROW} to "
    f"{START_ROW + GRID_ROWS - 1}"
)

print(
    f"Columns: {START_COL} to "
    f"{START_COL + GRID_COLS - 1}"
)

# ==================================================
# STEP 2: INSPECT SCANNER METADATA
# ==================================================

print("\n[2] Searching scanner metadata...")

all_records = []
scanner_results = {}

for scanner, shard_indices in SCANNERS.items():

    print(f"\nScanner: {scanner}", flush=True)

    found = {}

    for shard_index in shard_indices:

        filename = (
            f"data/train-{shard_index:05d}"
            f"-of-{TOTAL_SHARDS:05d}.parquet"
        )

        url = f"hf://datasets/{REPO_ID}/{filename}"

        print(
            f"  Opening shard {shard_index}...",
            flush=True
        )

        try:
            ds = load_dataset(
                "parquet",
                data_files={"train": url},
                split="train",
                streaming=True
            )

            # Metadata only: PNG is excluded
            ds = ds.select_columns([
                "stainer",
                "scanner",
                "slide_id",
                "tile_id"
            ])

            scanned = 0

            for sample in ds:

                scanned += 1

                if (
                    sample["scanner"] != scanner
                    or sample["stainer"] != STAINER
                ):
                    if scanned >= MAX_SCAN_RECORDS:
                        break
                    continue

                tile_id = sample["tile_id"]

                if tile_id in target_set:

                    if tile_id not in found:

                        info = target_ids[tile_id]

                        found[tile_id] = {
                            "scanner": scanner,
                            "stainer": STAINER,
                            "slide_id": sample["slide_id"],
                            "tile_id": tile_id,
                            "level": info["level"],
                            "grid_row": info["grid_row"],
                            "grid_col": info["grid_col"],
                            "source_shard": shard_index
                        }

                if scanned % 1000 == 0:
                    print(
                        f"  Scanned: {scanned} | "
                        f"Found: {len(found)}/"
                        f"{len(target_set)}",
                        flush=True
                    )

                if len(found) == len(target_set):
                    break

                if scanned >= MAX_SCAN_RECORDS:
                    break

            print(
                f"  Finished shard {shard_index}. "
                f"Scanned {scanned} records. "
                f"Found {len(found)} tiles.",
                flush=True
            )

        except Exception as exc:
            print(
                f"  Error reading shard "
                f"{shard_index}: {exc}",
                flush=True
            )

    scanner_results[scanner] = found
    all_records.extend(found.values())

    print(
        f"\n{scanner}: "
        f"{len(found)}/{len(target_set)} "
        f"candidate tiles found."
    )

# ==================================================
# STEP 3: FIND MATCHING LOCATIONS
# ==================================================

print("\n[3] Identifying shared tile IDs...")

at2_ids = set(scanner_results["AT2"])
gt450_ids = set(scanner_results["GT450"])

shared_ids = at2_ids & gt450_ids

print(f"AT2 tiles: {len(at2_ids)}")
print(f"GT450 tiles: {len(gt450_ids)}")
print(f"Complete matched locations: {len(shared_ids)}")

missing_at2 = target_set - at2_ids
missing_gt450 = target_set - gt450_ids

print(f"Missing from AT2: {len(missing_at2)}")
print(f"Missing from GT450: {len(missing_gt450)}")

# ==================================================
# STEP 4: BUILD DOWNLOAD MANIFEST
# ==================================================

print("\n[4] Building download manifest...")

manifest_records = []

for tile_id in sorted(
    shared_ids,
    key=lambda x: (
        target_ids[x]["grid_row"],
        target_ids[x]["grid_col"]
    )
):

    at2_record = scanner_results["AT2"][tile_id]
    gt450_record = scanner_results["GT450"][tile_id]

    manifest_records.append({
        "tile_id": tile_id,
        "stainer": STAINER,
        "level": target_ids[tile_id]["level"],
        "grid_row": target_ids[tile_id]["grid_row"],
        "grid_col": target_ids[tile_id]["grid_col"],
        "scanner_a": "AT2",
        "scanner_b": "GT450",
        "slide_id_a": at2_record["slide_id"],
        "slide_id_b": gt450_record["slide_id"],
        "source_shard_a": at2_record["source_shard"],
        "source_shard_b": gt450_record["source_shard"]
    })

manifest_columns = [
    "tile_id",
    "stainer",
    "level",
    "grid_row",
    "grid_col",
    "scanner_a",
    "scanner_b",
    "slide_id_a",
    "slide_id_b",
    "source_shard_a",
    "source_shard_b"
]

manifest_df = pd.DataFrame(
    manifest_records,
    columns=manifest_columns
)

manifest_path = (
    TABLE_DIR / "plism_spatial_region_manifest.csv"
)

manifest_df.to_csv(
    manifest_path,
    index=False
)

# Save individual scanner metadata
metadata_df = pd.DataFrame(
    all_records,
    columns=[
        "scanner", "stainer", "slide_id",
        "tile_id", "level", "grid_row",
        "grid_col", "source_shard"
    ]
)

metadata_path = (
    TABLE_DIR / "plism_spatial_region_metadata.csv"
)

metadata_df.to_csv(
    metadata_path,
    index=False
)

# ==================================================
# STEP 5: PRINT SPATIAL AVAILABILITY GRID
# ==================================================

print("\n[5] Spatial availability map:")

print("\nLegend:")
print("  X = Tile available in both scanners")
print("  A = Only AT2 available")
print("  G = Only GT450 available")
print("  . = Missing in both")

header = "Row\\Col " + " ".join(
    f"{c:3d}"
    for c in range(
        START_COL,
        START_COL + GRID_COLS
    )
)

print(header)

for row in range(
    START_ROW,
    START_ROW + GRID_ROWS
):

    symbols = []

    for col in range(
        START_COL,
        START_COL + GRID_COLS
    ):

        tile_id = f"tile_{LEVEL}_{row}_{col}"

        in_a = tile_id in at2_ids
        in_g = tile_id in gt450_ids

        if in_a and in_g:
            symbol = "X"
        elif in_a:
            symbol = "A"
        elif in_g:
            symbol = "G"
        else:
            symbol = "."

        symbols.append(f"{symbol:>3}")

    print(f"{row:7d} " + " ".join(symbols))

# ==================================================
# STEP 6: CHECK REGION COMPLETENESS
# ==================================================

print("\n[6] Checking grid completeness...")

complete_grid = len(shared_ids) == len(target_set)

if complete_grid:

    print("SUCCESS: Complete 10 x 10 matched grid found.")
    print("Available graph nodes per scanner: 100")

else:

    print("INCOMPLETE: Not all candidate tiles were found.")
    print(
        f"Shared locations: "
        f"{len(shared_ids)}/{len(target_set)}"
    )
    print(
        "The manifest contains only verified "
        "matching tile IDs."
    )

# ==================================================
# STEP 7: SAVE REPORT
# ==================================================

report_path = (
    REPORT_DIR / "plism_spatial_region_report.txt"
)

elapsed = time.perf_counter() - start_time

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("PLISM Spatial Region Report\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Stainer: {STAINER}\n")
    f.write("Scanners: AT2, GT450\n")

    f.write(
        f"Grid size: "
        f"{GRID_ROWS} x {GRID_COLS}\n"
    )

    f.write(
        f"Candidate locations: "
        f"{len(target_set)}\n"
    )

    f.write(f"AT2 locations: {len(at2_ids)}\n")
    f.write(f"GT450 locations: {len(gt450_ids)}\n")
    f.write(f"Matched locations: {len(shared_ids)}\n")
    f.write(f"Complete grid: {complete_grid}\n")

    f.write(
        f"Execution time: {elapsed:.2f}s\n"
    )

    f.write("\nMissing AT2 IDs:\n")
    f.write("\n".join(sorted(missing_at2)))

    f.write("\n\nMissing GT450 IDs:\n")
    f.write("\n".join(sorted(missing_gt450)))

    f.write(
        "\n\nNOTE: Tile ID coordinates are "
        "treated as provisional grid coordinates "
        "pending validation.\n"
    )

print("\n[7] Saved output files:")
print(manifest_path)
print(metadata_path)
print(report_path)

print("\n" + "=" * 70)
print("SPATIAL REGION IDENTIFICATION COMPLETED")
print(f"Matching locations: {len(shared_ids)}")
print(f"Complete grid: {complete_grid}")
print(f"Execution time: {elapsed:.2f} seconds")
print("=" * 70)
